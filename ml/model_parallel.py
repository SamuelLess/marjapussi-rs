"""
Parallel v2 Marjapussi model family.

Design goals:
1. Keep legacy model available while enabling a stronger phase-specialized variant.
2. Stay within strict parameter budget for practical training/inference.
3. Preserve output contract: (policy_logits, card_logits, point_preds, value_pred).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model import ACTION_FEAT_DIM, MAX_SEQ_LEN, NUM_CARDS, VOCAB_SIZE
except ModuleNotFoundError:
    from .model import ACTION_FEAT_DIM, MAX_SEQ_LEN, NUM_CARDS, VOCAB_SIZE


@dataclass
class ParallelModelConfig:
    # Safety gate
    max_params: int = 28_000_000

    # Core dimensions
    state_dim: int = 640
    belief_dim: int = 320
    history_dim: int = 320
    fusion_dim: int = 768
    action_emb_dim: int = 192
    phase_head_hidden: int = 384

    # Stream A: public state
    card_emb_dim: int = 160
    opp_emb_dim: int = 192
    state_hidden_dim: int = 1024

    # Stream B: history
    history_layers: int = 10
    history_heads: int = 8
    history_ff_dim: int = 1280
    history_dropout: float = 0.10
    max_seq_len: int = MAX_SEQ_LEN
    vocab_size: int = VOCAB_SIZE

    # Belief stream
    belief_seat_hidden: int = 256
    belief_hidden_dim: int = 512

    # Aux heads
    aux_hidden_dim: int = 512
    aux_value_hidden_dim: int = 256

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phase_to_head_idx(phase_oh: torch.Tensor) -> torch.Tensor:
    """
    Map coarse phase id [0..4] -> head index [0..3]:
      0 bidding -> bidding head
      1 passing -> passing head
      2 trick   -> trick head
      3/4 other -> info head
    """
    phase_idx = torch.argmax(phase_oh, dim=-1)
    return torch.where(phase_idx <= 2, phase_idx, torch.full_like(phase_idx, 3))


class PublicStateTower(nn.Module):
    """Encode deterministic public/game-state features."""

    def __init__(self, cfg: ParallelModelConfig):
        super().__init__()
        self.cfg = cfg

        self.card_emb = nn.Sequential(
            nn.Linear(16, cfg.card_emb_dim),
            nn.GELU(),
            nn.Linear(cfg.card_emb_dim, cfg.card_emb_dim),
            nn.GELU(),
        )

        # poss(36)+conf(36)+known(36)+count(1) = 109
        self.opp_enc = nn.Sequential(
            nn.Linear(109, cfg.opp_emb_dim),
            nn.GELU(),
            nn.Linear(cfg.opp_emb_dim, cfg.opp_emb_dim),
            nn.GELU(),
        )

        # context dims:
        # trick_pos(4)+trump(5)+trump_called(4)+trump_poss(3)+role(5)
        # trick_num(1)+pts_mine(1)+pts_opp(1)+last_bonus(1)+cards_rem(4)+active_parity(2)+phase_oh(5)
        context_dim = 4 + 5 + 4 + 3 + 5 + 1 + 1 + 1 + 1 + 4 + 2 + 5
        raw_dim = (
            cfg.card_emb_dim  # hand
            + cfg.card_emb_dim  # current trick
            + 3 * cfg.opp_emb_dim
            + context_dim
        )

        self.state_enc = nn.Sequential(
            nn.Linear(raw_dim, cfg.state_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.state_hidden_dim, cfg.state_dim),
            nn.GELU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        card_feats = obs["card_feats"]
        all_emb = self.card_emb(card_feats)  # [B, 36, C]

        hand_emb = (all_emb * obs["my_hand_mask"].unsqueeze(-1)).sum(dim=1)
        trick_emb = (all_emb * obs["trick_mask"].unsqueeze(-1)).sum(dim=1)

        known_masks = obs["conf_masks"]
        opp_embs = []
        for rel in range(3):
            poss = obs["poss_masks"][:, rel, :]
            conf = obs["conf_masks"][:, rel, :]
            known = known_masks[:, rel, :]
            rem = obs["cards_rem"][:, rel + 1 : rel + 2]
            inp = torch.cat([poss, conf, known, rem], dim=-1)
            opp_embs.append(self.opp_enc(inp))
        opp_cat = torch.cat(opp_embs, dim=-1)

        context = torch.cat(
            [
                obs["trick_pos_oh"],
                obs["trump_oh"],
                obs["trump_called"],
                obs["trump_poss"],
                obs["role_oh"],
                obs["trick_num"],
                obs["pts_mine"],
                obs["pts_opp"],
                obs["last_bonus"],
                obs["cards_rem"],
                obs["active_parity"],
                obs["phase_oh"],
            ],
            dim=-1,
        )

        raw = torch.cat([hand_emb, trick_emb, opp_cat, context], dim=-1)
        return self.state_enc(raw)


class BeliefTower(nn.Module):
    """Encode hidden-hand possibility/known/exclusivity structure."""

    def __init__(self, cfg: ParallelModelConfig):
        super().__init__()
        self.cfg = cfg
        self.seat_enc = nn.Sequential(
            nn.Linear(109, cfg.belief_seat_hidden),
            nn.GELU(),
            nn.Linear(cfg.belief_seat_hidden, cfg.belief_seat_hidden),
            nn.GELU(),
        )
        self.belief_mlp = nn.Sequential(
            nn.Linear(3 * cfg.belief_seat_hidden + 24, cfg.belief_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.belief_hidden_dim, cfg.belief_dim),
            nn.GELU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        poss = obs["poss_masks"]  # [B,3,36]
        conf = obs["conf_masks"]  # [B,3,36]
        known = conf
        cards_rem = obs["cards_rem"]  # [B,4]

        seat_vecs = []
        for rel in range(3):
            inp = torch.cat([poss[:, rel, :], conf[:, rel, :], known[:, rel, :], cards_rem[:, rel + 1 : rel + 2]], dim=-1)
            seat_vecs.append(self.seat_enc(inp))
        seat_cat = torch.cat(seat_vecs, dim=-1)

        # Global set-theory summaries.
        union_poss = poss.sum(dim=1).clamp(max=1.0)  # [B,36]
        singleton = (poss.sum(dim=1) == 1).float()
        impossible = (poss.sum(dim=1) == 0).float()
        known_total = conf.sum(dim=1).clamp(max=1.0)
        stats = torch.stack(
            [
                union_poss.mean(dim=-1),
                singleton.mean(dim=-1),
                impossible.mean(dim=-1),
                known_total.mean(dim=-1),
                cards_rem[:, 1:4].mean(dim=-1),
                cards_rem[:, 1:4].sum(dim=-1),
            ],
            dim=-1,
        )  # [B,6]
        # Repeat simple stats to keep fixed 24 dim feature block.
        stats_24 = stats.repeat(1, 4)
        return self.belief_mlp(torch.cat([seat_cat, stats_24], dim=-1))


class EventHistoryTower(nn.Module):
    def __init__(self, cfg: ParallelModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.history_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.history_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.history_dim,
            nhead=cfg.history_heads,
            dim_feedforward=cfg.history_ff_dim,
            dropout=cfg.history_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.history_layers)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = token_ids.shape
        if seqlen > self.cfg.max_seq_len:
            token_ids = token_ids[:, -self.cfg.max_seq_len :]
            token_mask = token_mask[:, -self.cfg.max_seq_len :]
            seqlen = token_ids.shape[1]
        pos = torch.arange(seqlen, device=token_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.emb(token_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=token_mask)
        valid = (~token_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return pooled


class PhaseActionHeads(nn.Module):
    def __init__(self, cfg: ParallelModelConfig):
        super().__init__()
        self.cfg = cfg
        self.action_enc = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM, cfg.action_emb_dim),
            nn.GELU(),
            nn.Linear(cfg.action_emb_dim, cfg.action_emb_dim),
            nn.GELU(),
        )
        in_dim = cfg.action_emb_dim + cfg.fusion_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, cfg.phase_head_hidden),
                    nn.GELU(),
                    nn.Linear(cfg.phase_head_hidden, 1),
                )
                for _ in range(4)
            ]
        )

    def forward(
        self,
        action_feats: torch.Tensor,
        action_mask: torch.Tensor,
        fused: torch.Tensor,
        phase_oh: torch.Tensor,
    ) -> torch.Tensor:
        bsz, act_count, _ = action_feats.shape
        action_emb = self.action_enc(action_feats)  # [B,A,E]
        fused_rep = fused.unsqueeze(1).expand(-1, act_count, -1)
        inp = torch.cat([action_emb, fused_rep], dim=-1)

        # Compute logits from each phase head, then route by phase id.
        per_head = torch.stack([h(inp).squeeze(-1) for h in self.heads], dim=1)  # [B,4,A]
        route = _phase_to_head_idx(phase_oh).view(bsz, 1, 1).expand(-1, 1, act_count)
        logits = per_head.gather(dim=1, index=route).squeeze(1)  # [B,A]
        return logits.masked_fill(action_mask, -1e4)


class ParallelAuxHeads(nn.Module):
    def __init__(self, cfg: ParallelModelConfig):
        super().__init__()
        self.hidden_head = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.aux_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.aux_hidden_dim, 3 * NUM_CARDS),
        )
        self.points_head = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.aux_value_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.aux_value_hidden_dim, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.aux_value_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.aux_value_hidden_dim, 1),
        )

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        card_logits = self.hidden_head(fused).view(-1, 3, NUM_CARDS)
        point_preds = torch.sigmoid(self.points_head(fused))
        value_pred = self.value_head(fused).squeeze(-1)
        return card_logits, point_preds, value_pred


class MarjapussiParallelNet(nn.Module):
    model_family = "parallel_v2"

    def __init__(self, cfg: ParallelModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ParallelModelConfig()

        self.public_tower = PublicStateTower(self.cfg)
        self.belief_tower = BeliefTower(self.cfg)
        self.history_tower = EventHistoryTower(self.cfg)

        self.fusion = nn.Sequential(
            nn.Linear(self.cfg.state_dim + self.cfg.belief_dim + self.cfg.history_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, self.cfg.fusion_dim),
            nn.GELU(),
        )
        self.phase_heads = PhaseActionHeads(self.cfg)
        self.aux_heads = ParallelAuxHeads(self.cfg)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_a = batch["obs_a"]
        state_vec = self.public_tower(obs_a)
        belief_vec = self.belief_tower(obs_a)
        history_vec = self.history_tower(batch["token_ids"], batch["token_mask"])
        fused = self.fusion(torch.cat([state_vec, belief_vec, history_vec], dim=-1))

        logits = self.phase_heads(
            action_feats=batch["action_feats"],
            action_mask=batch["action_mask"],
            fused=fused,
            phase_oh=obs_a["phase_oh"],
        )
        card_logits, point_preds, value_pred = self.aux_heads(fused)
        return logits, card_logits, point_preds, value_pred

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
