"""
Game environment wrapper.
Spawns the Rust ml_server binary as a subprocess and communicates via JSON lines.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from model import MarjapussiNet, build_card_features_batch, NUM_CARDS

BINARY_PATH = Path(__file__).parent.parent / "target" / "debug" / "ml_server.exe"
if not BINARY_PATH.exists():
    BINARY_PATH = Path(__file__).parent.parent / "target" / "debug" / "ml_server"


def _send(proc: subprocess.Popen, msg: dict) -> dict:
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    response = proc.stdout.readline()
    return json.loads(response)


class MarjapussiEnv:
    """
    Manages one game instance via the Rust ml_server subprocess.
    """

    def __init__(self, pov: int = 0):
        self.pov = pov
        self.proc: Optional[subprocess.Popen] = None
        self._obs: Optional[dict] = None
        self._done = False
        self._start_proc()

    def _start_proc(self):
        if not BINARY_PATH.exists():
            raise FileNotFoundError(
                f"ml_server binary not found at {BINARY_PATH}. "
                "Run: cargo build --bin ml_server"
            )
        self.proc = subprocess.Popen(
            [str(BINARY_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def reset(self, pov: Optional[int] = None) -> dict:
        if pov is not None:
            self.pov = pov
        resp = _send(self.proc, {"cmd": "new_game", "pov": self.pov})
        if resp.get("type") == "error":
            raise RuntimeError(resp["message"])
        self._obs = resp["obs"]
        self._done = resp.get("done", False)
        return self._obs

    def step(self, action_idx: int) -> tuple[dict, bool, dict]:
        """Apply action by index into legal_actions list. Returns (obs, done, info)."""
        resp = _send(self.proc, {"cmd": "step", "action_idx": action_idx})
        if resp.get("type") == "error":
            raise RuntimeError(resp["message"])
        self._obs = resp["obs"]
        self._done = resp.get("done", False)
        info = resp.get("outcome") or {}
        return self._obs, self._done, info

    def observe(self) -> dict:
        resp = _send(self.proc, {"cmd": "observe"})
        return resp["obs"]

    def run_to_end(self, policy: str = "heuristic") -> dict:
        resp = _send(self.proc, {"cmd": "run_to_end", "policy": policy})
        return resp.get("outcome", {})

    def try_all_actions(self, policy: str = "heuristic") -> list[dict]:
        resp = _send(self.proc, {"cmd": "try_all_actions", "policy": policy})
        return resp.get("branches", [])

    @property
    def obs(self) -> Optional[dict]:
        return self._obs

    @property
    def done(self) -> bool:
        return self._done

    @property
    def legal_actions(self) -> list[dict]:
        if self._obs is None:
            return []
        return self._obs.get("legal_actions", [])

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait()


# ── Observation → Tensor conversion ───────────────────────────────────────────

def obs_to_tensors(obs: dict) -> dict:
    """Convert a single raw observation dict to model-ready tensors (batch dim=1)."""

    def bitmask(key):
        return torch.tensor(obs[key], dtype=torch.float32).unsqueeze(0)  # [1, 36]

    # Trump one-hot [1, 5]: 4 suits + none
    trump_idx = obs.get('trump')
    trump_oh = torch.zeros(1, 5)
    if trump_idx is not None:
        trump_oh[0, trump_idx] = 1.0
    else:
        trump_oh[0, 4] = 1.0

    # Role one-hot [1, 5]
    role_oh = F.one_hot(torch.tensor([obs['my_role']]), 5).float()

    # Trick position one-hot [1, 4]
    trick_pos_oh = F.one_hot(torch.tensor([min(obs['trick_position'], 3)]), 4).float()

    # Trump possibilities one-hot [1, 3]
    trump_poss = F.one_hot(torch.tensor([obs['trump_possibilities']]), 3).float()

    # Trump announced [1, 4]
    trump_called = torch.tensor(obs['trump_announced'], dtype=torch.float32).unsqueeze(0)

    # Cards remaining (normalized) [1, 4]
    cards_rem = torch.tensor(obs['cards_remaining'], dtype=torch.float32).unsqueeze(0) / 9.0

    obs_a = {
        'card_feats': build_card_features_batch([obs]),   # [1, 36, 16]
        'my_hand_mask': bitmask('my_hand_bitmask'),
        'poss_masks': torch.tensor(obs['possible_bitmasks'], dtype=torch.float32).unsqueeze(0),  # [1, 3, 36]
        'conf_masks': torch.tensor(obs['confirmed_bitmasks'], dtype=torch.float32).unsqueeze(0),
        'trick_mask': trick_bitmask(obs),
        'cards_rem': cards_rem,
        'trump_oh': trump_oh,
        'trump_called': trump_called,
        'trump_poss': trump_poss,
        'role_oh': role_oh,
        'trick_pos_oh': trick_pos_oh,
        'trick_num': torch.tensor([[obs['trick_number'] / 9.0]]),
        'pts_mine': torch.tensor([[obs['points_my_team'] / 120.0]]),
        'pts_opp': torch.tensor([[obs['points_opp_team'] / 120.0]]),
        'last_bonus': torch.tensor([[float(obs['last_trick_bonus_live'])]]),
    }

    # Token sequence [1, L]
    tokens = obs['event_tokens']
    token_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    token_mask = torch.zeros(1, len(tokens), dtype=torch.bool)

    # Legal action features [1, A, 51]
    action_feats, action_mask = encode_legal_actions(obs['legal_actions'])

    return {
        'obs_a': obs_a,
        'token_ids': token_ids,
        'token_mask': token_mask,
        'action_feats': action_feats,
        'action_mask': action_mask,
    }


def trick_bitmask(obs: dict) -> torch.Tensor:
    """Build [1, 36] bitmask for cards currently in the trick."""
    mask = torch.zeros(1, NUM_CARDS)
    for idx in obs.get('current_trick_indices', []):
        mask[0, idx] = 1.0
    return mask


def encode_legal_actions(legal: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode legal actions as [1, A, 51] feature tensors."""
    A = max(len(legal), 1)
    feats = torch.zeros(1, A, 51)
    mask = torch.ones(1, A, dtype=torch.bool)  # True = ignored

    # Action type one-hot: 15 types
    ACTION_TOKENS = {40: 0, 41: 1, 42: 2, 43: 3, 44: 4,
                     45: 5, 46: 6, 47: 7, 48: 8, 49: 9, 50: 10}

    for i, la in enumerate(legal):
        mask[0, i] = False
        tok = la['action_token']
        type_idx = ACTION_TOKENS.get(tok, 11)
        feats[0, i, type_idx] = 1.0  # action type [0..14]

        # Card features [15..46] (32-dim placeholder, filled by model's card_emb)
        # We encode a simplified card one-hot here:
        if la.get('card_idx') is not None:
            c = la['card_idx']
            suit_oh = F.one_hot(torch.tensor(c // 9), 4).float()
            val_oh = F.one_hot(torch.tensor(c % 9), 9).float()
            feats[0, i, 15:19] = suit_oh
            feats[0, i, 19:28] = val_oh

        # Suit features [28..31]
        if la.get('suit_idx') is not None:
            feats[0, i, 28 + la['suit_idx']] = 1.0

        # Bid value [32] (normalized)
        if la.get('bid_value') is not None:
            feats[0, i, 32] = (la['bid_value'] - 120) / 300.0

    return feats, mask  # [1, A, 51], [1, A]


if __name__ == '__main__':
    # Quick environment test
    env = MarjapussiEnv(pov=0)
    obs = env.reset()
    print(f"Legal actions on start: {len(obs['legal_actions'])}")
    steps = 0
    while not env.done and steps < 200:
        la = env.legal_actions
        if not la:
            break
        obs, done, info = env.step(0)
        steps += 1
    print(f"Game completed in {steps} steps. Done: {env.done}")
    env.close()
    print("env.py test passed.")
