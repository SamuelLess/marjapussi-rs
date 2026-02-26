"""
Game environment wrapper.
Spawns the Rust ml_server binary as a subprocess and communicates via JSON lines.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from model import ACTION_FEAT_DIM, build_card_features_batch, NUM_CARDS

SUPPORTED_OBS_SCHEMA_VERSION = 1

def resolve_ml_server_binary(binary_path: Optional[str | Path] = None) -> Path:
    """Resolve the ml_server binary path with stable override semantics."""
    if binary_path is not None:
        p = Path(binary_path)
        if not p.exists():
            raise FileNotFoundError(f"ml_server binary not found at {p}")
        return p

    env_override = os.getenv("ML_SERVER_BIN")
    if env_override:
        p = Path(env_override)
        if not p.exists():
            raise FileNotFoundError(f"ML_SERVER_BIN points to missing binary: {p}")
        return p

    # Prefer release builds for better simulation throughput.
    release_candidates = [
        Path(__file__).parent.parent / "target" / "release" / "ml_server.exe",
        Path(__file__).parent.parent / "target" / "release" / "ml_server",
    ]
    for p in release_candidates:
        if p.exists():
            return p

    # Fall back to debug when release is not available.
    debug_candidates = [
        Path(__file__).parent.parent / "target" / "debug" / "ml_server.exe",
        Path(__file__).parent.parent / "target" / "debug" / "ml_server",
    ]
    for p in debug_candidates:
        if p.exists():
            return p

    # Default expected location (release) when neither exists yet.
    return release_candidates[0]


def _send(proc: subprocess.Popen, msg: dict) -> dict:
    try:
        line = json.dumps(msg) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()
        response = proc.stdout.readline()
        if not response:
            # Process probably died
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"ml_server died. Stderr: {err}")
        return json.loads(response)
    except Exception as e:
        if proc.poll() is not None:
             err = proc.stderr.read() if proc.stderr else "No stderr"
             raise RuntimeError(f"ml_server (PID {proc.pid}) exited with {proc.returncode}. Stderr: {err}") from e
        raise e


class MarjapussiEnv:
    """
    Manages one game instance via the Rust ml_server subprocess.
    """

    def __init__(
        self,
        pov: int = 0,
        binary_path: Optional[str | Path] = None,
        include_labels: bool = False,
    ):
        self.pov = pov
        self.proc: Optional[subprocess.Popen] = None
        self._obs: Optional[dict] = None
        self._labels: Optional[dict] = None
        self._done = False
        self.include_labels = include_labels
        self.binary_path = resolve_ml_server_binary(binary_path)
        self._spawn_binary_path: Optional[Path] = None
        self._start_proc()

    def _prepare_spawn_binary(self) -> Path:
        """
        On Windows, execute a per-process copy of ml_server.exe so Cargo can rebuild
        target/release/ml_server.exe while env processes are still running.
        """
        src = self.binary_path
        if os.name != "nt":
            return src

        runtime_dir = Path(__file__).parent / ".runtime_bins"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        # Opportunistic cleanup of stale runtime binaries.
        now = time.time()
        for p in runtime_dir.glob("ml_server_run_*.exe"):
            try:
                if now - p.stat().st_mtime > 24 * 3600:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

        dst = runtime_dir / f"ml_server_run_{os.getpid()}_{uuid.uuid4().hex}.exe"
        shutil.copy2(src, dst)
        self._spawn_binary_path = dst
        return dst

    def _start_proc(self):
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"ml_server binary not found at {self.binary_path}. "
                "Run: cargo build --release --bin ml_server"
            )
        spawn_binary = self._prepare_spawn_binary()
        self.proc = subprocess.Popen(
            [str(spawn_binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def reset(self, pov: Optional[int] = None, start_trick: Optional[int] = None, seed: Optional[int] = None) -> dict:
        if pov is not None:
            self.pov = pov
        cmd = {
            "cmd": "new_game",
            "pov": self.pov,
            "include_labels": self.include_labels,
        }
        if start_trick is not None:
            cmd["start_trick"] = start_trick
        if seed is not None:
            cmd["seed"] = seed
        resp = _send(self.proc, cmd)
        if resp.get("type") == "error":
            raise RuntimeError(resp["message"])
        if "obs" not in resp:
            raise RuntimeError(f"Invalid new_game response keys: {sorted(resp.keys())}")
        self._obs = resp["obs"]
        self._labels = resp.get("labels")
        self._done = resp.get("done", False)
        return self._obs

    def step(self, action_idx: int) -> tuple[dict, bool, dict]:
        """Apply action by index into legal_actions list. Returns (obs, done, info)."""
        resp = _send(self.proc, {"cmd": "step", "action_idx": action_idx})
        if resp.get("type") == "error":
            raise RuntimeError(resp["message"])
        if "obs" in resp:
            self._obs = resp["obs"]
        elif resp.get("type") == "done":
            # Backward compatibility for older server responses.
            # Keep last observation so callers can still render terminal state.
            if self._obs is None:
                self._obs = {}
        else:
            raise RuntimeError(f"Invalid step response keys: {sorted(resp.keys())}")
        self._labels = resp.get("labels")
        self._done = bool(resp.get("done", resp.get("type") == "done"))
        info = resp.get("outcome") or {}
        return self._obs, self._done, info

    def get_heuristic_action(self) -> int:
        """Ask the Rust backend what action the deterministic heuristic policy would take."""
        resp = _send(self.proc, {"cmd": "get_heuristic_action"})
        if resp.get("type") == "error":
            raise RuntimeError(resp["message"])
        return resp.get("action_idx", 0)

    def debug_pass(self, card_indices: list[int]) -> tuple[dict, bool, dict]:
        """Force a pass action via card indices for debug purposes"""
        resp = _send(self.proc, {"cmd": "debug_pass", "card_indices": card_indices})
        if resp.get("type") == "error":
            raise RuntimeError(resp["message"])
        if "obs" not in resp:
            raise RuntimeError(f"Invalid debug_pass response keys: {sorted(resp.keys())}")
        self._obs = resp["obs"]
        self._labels = resp.get("labels")
        self._done = resp.get("done", False)
        info = resp.get("outcome") or {}
        return self._obs, self._done, info

    def observe(self, pov: Optional[int] = None) -> dict:
        if pov is None:
            cmd = {"cmd": "observe"}
        else:
            cmd = {"cmd": "observe_pov", "pov": int(pov)}
        resp = _send(self.proc, cmd)
        if resp.get("type") == "error":
            raise RuntimeError(resp.get("message", "observe failed"))
        if "obs" not in resp:
            raise RuntimeError(f"Invalid observe response keys: {sorted(resp.keys())}")
        self._labels = resp.get("labels")
        return resp["obs"]

    def observe_pov(self, pov: int) -> dict:
        """Observe current state from an arbitrary seat POV without changing env.pov."""
        return self.observe(pov=pov)

    def observe_debug(self) -> dict:
        resp = _send(self.proc, {"cmd": "observe_debug"})
        if resp.get("type") == "error":
            raise RuntimeError(resp["message"])
        return resp.get("debug", {})

    def run_to_end(self, policy: str = "heuristic") -> dict:
        resp = _send(self.proc, {"cmd": "run_to_end", "policy": policy})
        return resp.get("outcome", {})

    def try_all_actions(self, policy: str = "heuristic", num_rollouts: int = 1) -> list[dict]:
        resp = _send(self.proc, {"cmd": "try_all_actions", "policy": policy, "num_rollouts": num_rollouts})
        return resp.get("branches", [])

    def get_advantages(self, policy: str = "heuristic", num_rollouts: int = 1) -> list[float]:
        resp = _send(self.proc, {"cmd": "get_advantages", "policy": policy, "num_rollouts": num_rollouts})
        return resp.get("advantages", [])

    @property
    def obs(self) -> Optional[dict]:
        return self._obs

    @property
    def done(self) -> bool:
        return self._done

    @property
    def labels(self) -> Optional[dict]:
        return self._labels

    @property
    def legal_actions(self) -> list[dict]:
        if self._obs is None:
            return []
        return self._obs.get("legal_actions", [])

    def close(self):
        if self.proc:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except Exception:
                pass

            if self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    try:
                        self.proc.kill()
                        self.proc.wait(timeout=0.8)
                    except Exception:
                        pass
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass

            try:
                if self.proc.stdout and not self.proc.stdout.closed:
                    self.proc.stdout.close()
            except Exception:
                pass
            try:
                if self.proc.stderr and not self.proc.stderr.closed:
                    self.proc.stderr.close()
            except Exception:
                pass
        if self._spawn_binary_path is not None:
            try:
                self._spawn_binary_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._spawn_binary_path = None


# ── Observation → Tensor conversion ───────────────────────────────────────────

def obs_to_tensors(obs: dict, labels: Optional[dict] = None) -> dict:
    """Convert a single raw observation dict to model-ready tensors (batch dim=1)."""

    schema_version = obs.get("schema_version")
    if schema_version is not None and int(schema_version) != SUPPORTED_OBS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported observation schema_version={schema_version}; "
            f"expected {SUPPORTED_OBS_SCHEMA_VERSION}. "
            "Update loader/model compatibility before continuing."
        )

    def bitmask(key):
        return torch.tensor(obs[key], dtype=torch.float32).unsqueeze(0)  # [1, 36]

    # Trump one-hot [1, 5]: 4 suits + none
    trump_idx = obs.get('trump')
    trump_oh = torch.zeros((1, 5))
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

    # Active Parity (one-hot team) [1, 2]
    # In 'obs', active_player is a relative index 0..3 (where 0=me, 1=next, 2=partner, 3=prev)
    # Parity 0 means my team (0 or 2), Parity 1 means opp team (1 or 3)
    parity_idx = obs['active_player'] % 2
    active_parity = F.one_hot(torch.tensor([parity_idx]), 2).float()
    phase_oh = encode_phase(obs.get("phase", ""))

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
        'pts_mine': torch.tensor([[obs['points_my_team'] / 420.0]]),
        'pts_opp': torch.tensor([[obs['points_opp_team'] / 420.0]]),
        'last_bonus': torch.tensor([[float(obs['last_trick_bonus_live'])]]),
        'active_parity': active_parity,
        'phase_oh': phase_oh,
    }

    # Token sequence [1, L]
    tokens = obs['event_tokens'][-1024:]  # Safely truncate to MAX_SEQ_LEN to prevent PyTorch pos_emb assert
    token_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    token_mask = torch.zeros((1, len(tokens)), dtype=torch.bool)

    # Legal action features [1, A, ACTION_FEAT_DIM]
    action_feats, action_mask = encode_legal_actions(
        obs['legal_actions'],
        phase_name=obs.get("phase", ""),
        pass_selection_indices=obs.get("pass_selection_indices", []),
        pass_selection_target=obs.get("pass_selection_target", 4),
    )

    # Hidden-hand supervision targets (relative opponents: left, partner, right).
    hidden_target = torch.zeros((1, 3, NUM_CARDS), dtype=torch.float32)
    hidden_hands = (labels or {}).get("hidden_hands", [])
    if hidden_hands:
        for rel_opp in range(min(3, len(hidden_hands))):
            for card_idx in hidden_hands[rel_opp]:
                if 0 <= card_idx < NUM_CARDS:
                    hidden_target[0, rel_opp, card_idx] = 1.0
    else:
        # Backward compatibility for legacy/offline records that still include all_hands.
        all_hands = obs.get('all_hands', [])
        for rel_opp in range(3):
            seat_idx = rel_opp + 1
            if seat_idx < len(all_hands):
                for card_idx in all_hands[seat_idx]:
                    if 0 <= card_idx < NUM_CARDS:
                        hidden_target[0, rel_opp, card_idx] = 1.0

    hidden_possible = torch.tensor(
        obs.get('possible_bitmasks', [[False] * NUM_CARDS for _ in range(3)]),
        dtype=torch.float32
    ).unsqueeze(0)
    hidden_known = torch.tensor(
        obs.get('confirmed_bitmasks', [[False] * NUM_CARDS for _ in range(3)]),
        dtype=torch.float32
    ).unsqueeze(0)
    # Keep known cards feasible even if upstream possible-masks lag behind constraints.
    hidden_possible = torch.maximum(hidden_possible, hidden_known)

    return {
        'obs_a': obs_a,
        'token_ids': token_ids,
        'token_mask': token_mask,
        'action_feats': action_feats,
        'action_mask': action_mask,
        'hidden_target': hidden_target,
        'hidden_possible': hidden_possible,
        'hidden_known': hidden_known,
    }


def trick_bitmask(obs: dict) -> torch.Tensor:
    """Build [1, 36] bitmask for cards currently in the trick."""
    mask = torch.zeros((1, NUM_CARDS))
    for idx in obs.get('current_trick_indices', []):
        mask[0, idx] = 1.0
    return mask


def encode_phase(phase_name: str) -> torch.Tensor:
    """
    Encode coarse game phase for robustness across future phase refactors.
    Order: bidding, passing, trick, answering, terminal/other.
    """
    name = str(phase_name)
    if name in {"Bidding", "Raising"}:
        idx = 0
    elif name in {"PassingForth", "PassingBack"}:
        idx = 1
    elif name in {"StartTrick", "Trick"}:
        idx = 2
    elif name.startswith("AnsweringPair") or name.startswith("AnsweringHalf"):
        idx = 3
    else:
        idx = 4
    return F.one_hot(torch.tensor([idx]), 5).float()


def encode_legal_actions(
    legal: list[dict],
    phase_name: str = "",
    pass_selection_indices: list[int] | None = None,
    pass_selection_target: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode legal actions as [1, A, ACTION_FEAT_DIM] feature tensors."""
    A = max(len(legal), 1)
    feats = torch.zeros((1, A, ACTION_FEAT_DIM))
    mask = torch.ones((1, A), dtype=torch.bool)  # True = ignored
    pass_phase = str(phase_name)
    pass_is_forth = 1.0 if pass_phase == "PassingForth" else 0.0
    pass_is_back = 1.0 if pass_phase == "PassingBack" else 0.0
    selected_cards = [
        int(c) for c in (pass_selection_indices or [])
        if isinstance(c, int) and 0 <= int(c) < NUM_CARDS
    ]
    pass_target = max(1, int(pass_selection_target or 4))
    selected_count = min(len(selected_cards), pass_target)
    remaining_count = max(0, pass_target - selected_count)
    selected_suit_hist = torch.zeros(4)
    selected_val_hist = torch.zeros(9)
    for c in selected_cards:
        selected_suit_hist[c // 9] += 1.0
        selected_val_hist[c % 9] += 1.0
    hist_denom = float(max(1, pass_target))
    selected_suit_hist /= hist_denom
    selected_val_hist /= hist_denom

    # Action type one-hot: 15 types
    ACTION_TOKENS = {
        40: 0,   # play
        41: 1,   # bid
        42: 2,   # stop bidding
        43: 3,   # pass 4 cards (legacy direct pass encoding)
        44: 4,   # trump
        45: 5,   # ask pair
        46: 6,   # ask half
        47: 7,   # yes pair
        48: 8,   # no pair
        49: 9,   # yes half
        50: 10,  # no half
        52: 11,  # sequential pass-pick card
    }

    for i, la in enumerate(legal):
        mask[0, i] = False
        tok = la['action_token']
        type_idx = ACTION_TOKENS.get(tok, 12)
        feats[0, i, type_idx] = 1.0  # action type [0..14]

        # Card/summarized action features.
        if la.get('card_idx') is not None:
            c = la['card_idx']
            suit_oh = F.one_hot(torch.tensor(c // 9), 4).float()
            val_oh = F.one_hot(torch.tensor(c % 9), 9).float()
            feats[0, i, 15:19] = suit_oh
            feats[0, i, 19:28] = val_oh
        elif la.get('pass_cards'):
            # Exact pass-set encoding: retain full card identity information.
            # Layout: [33..68] -> 36-bit selected-card mask.
            cards = la['pass_cards']
            for c in cards:
                if 0 <= c < NUM_CARDS:
                    feats[0, i, 33 + c] = 1.0

            # Keep compact suit/value summaries as auxiliary inductive features.
            suit_hist = torch.zeros(4)
            val_hist = torch.zeros(9)
            denom = float(max(len(cards), 1))
            for c in cards:
                suit_hist[c // 9] += 1.0
                val_hist[c % 9] += 1.0
            feats[0, i, 15:19] = suit_hist / denom
            feats[0, i, 19:28] = val_hist / denom

        # Suit features [28..31]
        if la.get('suit_idx') is not None:
            feats[0, i, 28 + la['suit_idx']] = 1.0

        # Bid value [32] (normalized)
        if la.get('bid_value') is not None:
            feats[0, i, 32] = (la['bid_value'] - 120) / 300.0

        # Passing context features [69..86] to avoid sequential-pass aliasing:
        # - role asymmetry (forth/back),
        # - current selection progress,
        # - selected-card suit/value composition so token-52 picks are conditioned
        #   on what has already been selected.
        if tok in (43, 52):
            feats[0, i, 69] = pass_is_forth
            feats[0, i, 70] = pass_is_back
            feats[0, i, 71] = selected_count / 4.0
            feats[0, i, 72] = min(1.0, pass_target / 4.0)
            feats[0, i, 73] = remaining_count / 4.0
            feats[0, i, 74:78] = selected_suit_hist
            feats[0, i, 78:87] = selected_val_hist

    return feats, mask  # [1, A, ACTION_FEAT_DIM], [1, A]


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
