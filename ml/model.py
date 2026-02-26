"""
Marjapussi hybrid model (Stream A + Stream B → Action Scorer).

Architecture:
    Stream A: Explicit symbolic state → 128-dim state_vec
    Stream B: Event token history via small Transformer → 64-dim history_vec
    Scorer:   Score each legal action given concat(action_emb, state_vec, history_vec)
"""

import math
import warnings
warnings.filterwarnings("ignore", ".*enable_nested_tensor is True.*")
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Vocabulary constants (must match src/ml/observation.rs tokens module) ──────
VOCAB_SIZE = 250
CARD_BASE = 70       # card tokens 70..105
NUM_CARDS = 36
ACTION_FEAT_DIM = 87
SUIT_BASE = 60       # suit tokens 60..63
NUM_SUITS = 4
NUM_ROLES = 5        # VH/MH/LH/RH/None
NUM_TRUMP_POSS = 3   # Own/Yours/Ours
MAX_SEQ_LEN = 1024


# ── Card feature builder ───────────────────────────────────────────────────────

def card_features(card_idx: int, is_trump: bool, is_higher_than_lead: bool) -> torch.Tensor:
    """Build 16-dim feature vector for a single card."""
    suit_idx = card_idx // 9
    value_idx = card_idx % 9
    point_vals = [0, 0, 0, 0, 2, 3, 4, 10, 11]

    suit_oh = F.one_hot(torch.tensor(suit_idx), 4).float()        # 4-dim
    value_oh = F.one_hot(torch.tensor(value_idx), 9).float()      # 9-dim
    point = torch.tensor([point_vals[value_idx] / 11.0])          # 1-dim
    trump_bit = torch.tensor([float(is_trump)])                    # 1-dim
    higher_bit = torch.tensor([float(is_higher_than_lead)])       # 1-dim
    return torch.cat([suit_oh, value_oh, point, trump_bit, higher_bit])  # 16-dim


def build_card_features_batch(obs_list: list[dict]) -> torch.Tensor:
    """Build [batch, 36, 16] tensor of card features for each observation."""
    batch = []
    for obs in obs_list:
        trump = obs.get('trump')
        # current_trick[0] is the lead card if trick has started
        lead_card_suit = None
        if obs['current_trick_indices']:
            lead_idx = obs['current_trick_indices'][0]
            lead_card_suit = lead_idx // 9

        feats = []
        for c in range(NUM_CARDS):
            is_trump = (c // 9 == trump) if trump is not None else False
            is_higher = False
            if lead_card_suit is not None:
                # Simplified: higher in same suit (or trump > non-trump)
                c_suit = c // 9
                c_val = c % 9
                if trump is not None and c_suit == trump and lead_card_suit != trump:
                    is_higher = True
                elif c_suit == lead_card_suit:
                    lead_val = (obs['current_trick_indices'][0] % 9) if obs['current_trick_indices'] else 0
                    is_higher = c_val > lead_val
            feats.append(card_features(c, is_trump, is_higher))
        batch.append(torch.stack(feats))  # [36, 16]
    return torch.stack(batch)  # [B, 36, 16]


# ── Card Embedding (shared) ────────────────────────────────────────────────────

class CardEmbedding(nn.Module):
    """Shared MLP: card features [16] → embedding [32]."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 16] → [..., 32]"""
        return self.net(x)


# ── Stream A: Explicit State Encoder ──────────────────────────────────────────

class StreamA(nn.Module):
    """
    Encodes all symbolic/computable state into a 128-dim vector.

    Inputs (from ObservationJson):
        card_feats:     [B, 36, 16] — card features for all 36 cards
        my_hand_mask:   [B, 36]     — 1 if card in hand
        poss_masks:     [B, 3, 36]  — possible cards per opponent
        conf_masks:     [B, 3, 36]  — confirmed cards per opponent
        trick_mask:     [B, 36]     — cards in current trick
        cards_rem:      [B, 4]      — cards remaining per player (normalized)
        trump_oh:       [B, 5]      — trump onehot (4 suits + none)
        trump_called:   [B, 4]      — which suits ever called
        trump_poss:     [B, 3]      — Own/Yours/Ours onehot
        role_oh:        [B, 5]      — VH/MH/LH/RH/None onehot
        trick_pos_oh:   [B, 4]      — position in current trick (0..3)
        trick_num:      [B, 1]      — trick number / 9
        pts_mine:       [B, 1]      — points my team / 120
        pts_opp:        [B, 1]      — points opp team / 120
        last_bonus:     [B, 1]      — last trick bonus live bit
    """

    def __init__(self, card_emb_dim=128, state_dim=512):
        super().__init__()
        self.card_emb = CardEmbedding(card_emb_dim)

        # Opponent belief encoder (per opponent, shared weights)
        # Input: possible(36) + confirmed(36) + card_count(1) = 73
        self.opp_enc = nn.Sequential(
            nn.Linear(73, card_emb_dim),
            nn.ReLU(),
        )

        # A1: hand (128) + A2: 3×opp (384) + A3: trick (128) + A3: trick_pos (4)
        # + A4: trump (5) + trump_called (4) + trump_poss (3) + role (5)
        #      + trick_num (1) + pts_mine (1) + pts_opp (1) + last_bonus (1)
        #      + cards_rem (4) + active_parity (2)
        # = 128 + 384 + 128 + 4 + 5 + 4 + 3 + 5 + 1 + 1 + 1 + 1 + 4 + 2 = 671
        raw_dim = card_emb_dim + (3 * card_emb_dim) + card_emb_dim + 4 + 5 + 4 + 3 + 5 + 1 + 1 + 1 + 1 + 4 + 2
        self.state_enc = nn.Sequential(
            nn.Linear(raw_dim, state_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        B = obs['my_hand_mask'].shape[0]

        card_feats = obs['card_feats'].to(device)        # [B, 36, 16]
        all_embs = self.card_emb(card_feats)              # [B, 36, 32]

        # A1: hand DeepSets sum
        hand_mask = obs['my_hand_mask'].unsqueeze(-1).to(device)   # [B, 36, 1]
        hand_emb = (all_embs * hand_mask).sum(dim=1)                # [B, 32]

        # A2: opponent belief
        opp_embs = []
        for i in range(3):
            poss = obs['poss_masks'][:, i, :].to(device)   # [B, 36]
            conf = obs['conf_masks'][:, i, :].to(device)   # [B, 36]
            cnt = obs['cards_rem'][:, i+1:i+2].to(device)  # [B, 1]
            inp = torch.cat([poss, conf, cnt], dim=-1)      # [B, 73]
            opp_embs.append(self.opp_enc(inp))              # [B, 32]
        opp_cat = torch.cat(opp_embs, dim=-1)               # [B, 96]

        # A3: current trick
        trick_mask = obs['trick_mask'].unsqueeze(-1).to(device)  # [B, 36, 1]
        trick_emb = (all_embs * trick_mask).sum(dim=1)            # [B, 32]

        # A4: context scalars
        trump_oh = obs['trump_oh'].to(device)            # [B, 5]
        trump_called = obs['trump_called'].to(device)    # [B, 4]
        trump_poss = obs['trump_poss'].to(device)        # [B, 3]
        role_oh = obs['role_oh'].to(device)              # [B, 5]
        trick_pos = obs['trick_pos_oh'].to(device)       # [B, 4]
        trick_num = obs['trick_num'].to(device)          # [B, 1]
        pts_mine = obs['pts_mine'].to(device)            # [B, 1]
        pts_opp = obs['pts_opp'].to(device)              # [B, 1]
        last_bonus = obs['last_bonus'].to(device)        # [B, 1]
        cards_rem = obs['cards_rem'].to(device)          # [B, 4]
        active_parity = obs['active_parity'].to(device)  # [B, 2]

        raw = torch.cat([
            hand_emb, opp_cat, trick_emb,
            trick_pos, trump_oh, trump_called, trump_poss, role_oh,
            trick_num, pts_mine, pts_opp, last_bonus, cards_rem, active_parity
        ], dim=-1)

        return self.state_enc(raw)   # [B, 128]


# ── Stream B: Event History Transformer ───────────────────────────────────────

class StreamB(nn.Module):
    """
    Medium transformer encoder over event token ids → 256-dim history_vec.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=256, n_heads=8,
                 n_layers=12, ff_dim=1024, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.emb_dim = emb_dim

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L] int64
        mask:      [B, L] bool  (True = padding, to be ignored)
        returns:   [B, 64]
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)  # [1, L]
        x = self.emb(token_ids) + self.pos_emb(pos)                  # [B, L, 64]
        x = self.transformer(x, src_key_padding_mask=mask)           # [B, L, 64]
        # Mean pool over non-padding positions
        not_pad = (~mask).unsqueeze(-1).float()                       # [B, L, 1]
        pooled = (x * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp(min=1)
        return pooled   # [B, 64]


# ── Action Scoring Head ────────────────────────────────────────────────────────

class ActionScorer(nn.Module):
    """
    Scores each legal action given the fused context.
    action_feats: [B, max_actions, D_action]
    fused:        [B, 192]
    → scores:     [B, max_actions]
    """

    def __init__(self, action_feat_dim=ACTION_FEAT_DIM, state_dim=512, hist_dim=256, action_emb_dim=128):
        super().__init__()
        self.action_enc = nn.Sequential(
            nn.Linear(action_feat_dim, action_emb_dim),
            nn.ReLU(),
        )
        fused_dim = action_emb_dim + state_dim + hist_dim
        self.scorer = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, action_feats: torch.Tensor,
                state_vec: torch.Tensor,
                history_vec: torch.Tensor,
                action_mask: torch.Tensor) -> torch.Tensor:
        """
        action_feats: [B, A, ACTION_FEAT_DIM]
        state_vec:    [B, 128]
        history_vec:  [B, 64]
        action_mask:  [B, A] bool — True = padding (no action)
        → logits:     [B, A]  (-inf for padding)
        """
        B, A, _ = action_feats.shape
        action_emb = self.action_enc(action_feats)   # [B, A, 128]
        sv = state_vec.unsqueeze(1).expand(-1, A, -1)  # [B, A, 512]
        hv = history_vec.unsqueeze(1).expand(-1, A, -1)  # [B, A, 256]
        fused = torch.cat([action_emb, sv, hv], dim=-1)   # [B, A, 896]
        logits = self.scorer(fused).squeeze(-1)            # [B, A]
        logits = logits.masked_fill(action_mask, -1e4)
        return logits


# ── Auxiliary Heads ────────────────────────────────────────────────────────────

class AuxHeads(nn.Module):
    """Auxiliary training heads sharing the fused context."""

    def __init__(self, state_dim=512, hist_dim=256):
        super().__init__()
        fused = state_dim + hist_dim
        # Predict which cards each opponent holds (3 × 36 binary)
        self.card_pred = nn.Linear(fused, 3 * NUM_CARDS)
        # Predict final points (2 scalars: my team, opp team, both in [0,1])
        self.points_pred = nn.Sequential(
            nn.Linear(fused, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        # Value head for PPO (State Value V(s))
        self.value_pred = nn.Sequential(
            nn.Linear(fused, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state_vec: torch.Tensor, history_vec: torch.Tensor):
        fused = torch.cat([state_vec, history_vec], dim=-1)  # [B, 192]
        card_logits = self.card_pred(fused).view(-1, 3, NUM_CARDS)   # [B, 3, 36]
        points = torch.sigmoid(self.points_pred(fused))               # [B, 2]
        value = self.value_pred(fused).squeeze(-1)                    # [B]
        return card_logits, points, value


# ── Full Model ─────────────────────────────────────────────────────────────────

class MarjapussiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream_a = StreamA(card_emb_dim=128, state_dim=512)
        self.stream_b = StreamB(emb_dim=256, n_heads=8, n_layers=12, ff_dim=1024)
        self.action_scorer = ActionScorer(state_dim=512, hist_dim=256, action_emb_dim=128)
        self.aux_heads = AuxHeads(state_dim=512, hist_dim=256)

    def forward(self, batch: dict):
        """
        batch keys:
            obs_a:          dict of Stream A tensors (see StreamA.forward)
            token_ids:      [B, L] int64
            token_mask:     [B, L] bool
            action_feats:   [B, A, ACTION_FEAT_DIM] float
            action_mask:    [B, A] bool

        returns:
            policy_logits:  [B, A]
            card_logits:    [B, 3, 36]
            point_preds:    [B, 2]
            value_pred:     [B]
        """
        state_vec = self.stream_a(batch['obs_a'])                 # [B, 128]
        history_vec = self.stream_b(batch['token_ids'],
                                    batch['token_mask'])           # [B, 64]
        logits = self.action_scorer(
            batch['action_feats'], state_vec, history_vec,
            batch['action_mask'])                                  # [B, A]
        card_logits, point_preds, value_pred = self.aux_heads(state_vec, history_vec)
        return logits, card_logits, point_preds, value_pred

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    import sys
    # Smoke test
    B, A, L = 2, 5, 32
    device = 'cpu'
    model = MarjapussiNet().to(device)
    print(f"Model parameters: {model.param_count():,}")

    # Fake batch
    obs_a = {
        'card_feats': torch.zeros((B, 36, 16)),
        'my_hand_mask': torch.zeros((B, 36)),
        'poss_masks': torch.zeros((B, 3, 36)),
        'conf_masks': torch.zeros((B, 3, 36)),
        'trick_mask': torch.zeros((B, 36)),
        'cards_rem': torch.zeros((B, 4)),
        'trump_oh': torch.zeros((B, 5)),
        'trump_called': torch.zeros((B, 4)),
        'trump_poss': torch.zeros((B, 3)),
        'role_oh': torch.zeros((B, 5)),
        'trick_pos_oh': torch.zeros((B, 4)),
        'trick_num': torch.zeros((B, 1)),
        'pts_mine': torch.zeros((B, 1)),
        'pts_opp': torch.zeros((B, 1)),
        'last_bonus': torch.zeros((B, 1)),
        'active_parity': torch.zeros((B, 2)),
    }
    batch = {
        'obs_a': obs_a,
        'token_ids': torch.zeros((B, L), dtype=torch.long),
        'token_mask': torch.zeros((B, L), dtype=torch.bool),
        'action_feats': torch.zeros((B, A, ACTION_FEAT_DIM)),
        'action_mask': torch.zeros((B, A), dtype=torch.bool),
    }
    logits, card_logits, point_preds, value_pred = model(batch)
    assert logits.shape == (B, A), f"Unexpected logits shape: {logits.shape}"
    assert card_logits.shape == (B, 3, 36)
    assert point_preds.shape == (B, 2)
    assert value_pred.shape == (B,)
    print("Smoke test passed. Output shapes correct.")
    sys.exit(0)
