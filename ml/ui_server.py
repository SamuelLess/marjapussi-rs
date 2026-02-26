"""
WebSocket + HTTP server for the Marjapussi debug/play UI.
Run:  python ml/ui_server.py [--checkpoint CKPT] [--port 8765]
Open: http://localhost:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web

ROOT = Path(__file__).parent.parent
ML = Path(__file__).parent
UI = ML / "ui"
CHECKPOINT_DIR = ML / "checkpoints"
RUNS_DIR = ML / "runs"

sys.path.insert(0, str(ML))

from env import MarjapussiEnv, obs_to_tensors

try:
    import torch
    import torch.nn.functional as F

    from model import MarjapussiNet

    TORCH_OK = True
except ImportError as e:
    TORCH_OK = False
    print(f"[warn] PyTorch not available: {e} - model controllers disabled")


ACTION_LABELS = {
    40: "Play",
    41: "Bid",
    42: "Pass",
    43: "PassCards",
    44: "Trump",
    45: "AskPair",
    46: "AskHalf",
    47: "YesPair",
    48: "NoPair",
    49: "YesHalf",
    50: "NoHalf",
    52: "PickPassCard",
}
SUITS = ["Gruen", "Eichel", "Schellen", "Herz"]
VALS = ["6", "7", "8", "9", "U", "O", "K", "10", "A"]
CONTROLLER_MODES = {"human", "heuristic", "model"}


def action_label(la: dict[str, Any]) -> str:
    tok = int(la.get("action_token", 40))
    base = ACTION_LABELS.get(tok, f"Act{tok}")
    if la.get("card_idx") is not None:
        c = int(la["card_idx"])
        return f"{base} {VALS[c % 9]} {SUITS[c // 9]}"
    if la.get("bid_value") is not None:
        return f"{base} {la['bid_value']}"
    if la.get("suit_idx") is not None:
        return f"{base} {SUITS[int(la['suit_idx'])]}"
    if la.get("pass_cards"):
        cards = ",".join(str(int(c)) for c in la["pass_cards"])
        return f"{base} [{cards}]"
    return base


def rel_hidden_abs_seat(pov_abs_seat: int, rel_hidden_idx: int) -> int:
    # rel hidden order: left(+1), partner(+2), right(+3)
    return (pov_abs_seat + rel_hidden_idx + 1) % 4


@dataclass
class SeatController:
    mode: str
    checkpoint: Optional[str] = None
    error: Optional[str] = None


class GameManager:
    def __init__(self, checkpoint: Optional[str] = None):
        self.env: Optional[MarjapussiEnv] = None
        self.obs: Optional[dict[str, Any]] = None
        self.done = True
        self.info: dict[str, Any] = {}
        self.last_seed: Optional[int] = None
        self.view_seat: int = 0

        self.model_cache: dict[str, Any] = {}
        self.seat_models: dict[int, Any] = {}
        self.seat_debug: dict[int, dict[str, Any]] = {}

        self.default_checkpoint = checkpoint
        self.controllers = [
            SeatController(mode="human", checkpoint=None),
            SeatController(mode="heuristic", checkpoint=checkpoint),
            SeatController(mode="heuristic", checkpoint=checkpoint),
            SeatController(mode="heuristic", checkpoint=checkpoint),
        ]

        if TORCH_OK:
            for seat in (1, 2, 3):
                self._try_enable_model_for_seat(seat, checkpoint)

    def close(self) -> None:
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass

    def _resolve_checkpoint(self, checkpoint: Optional[str]) -> Optional[Path]:
        if checkpoint is None or checkpoint == "" or checkpoint == "latest":
            p = CHECKPOINT_DIR / "latest.pt"
            return p if p.exists() else None

        raw = Path(checkpoint)
        if raw.exists():
            return raw

        # Allow paths relative to repository ml/ root, e.g. runs/<name>/checkpoints/latest.pt
        rel = ML / raw
        if rel.exists():
            return rel

        if raw.suffix == "":
            p = CHECKPOINT_DIR / f"{checkpoint}.pt"
            return p if p.exists() else None

        p = CHECKPOINT_DIR / raw.name
        return p if p.exists() else None

    def available_checkpoints(self) -> list[str]:
        roots: list[Path] = [CHECKPOINT_DIR]
        if RUNS_DIR.exists():
            roots.extend([p / "checkpoints" for p in RUNS_DIR.iterdir() if p.is_dir()])

        files: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            files.extend(list(root.glob("*.pt")))

        # De-duplicate by absolute path, newest first.
        dedup: dict[str, Path] = {}
        for p in files:
            dedup[str(p.resolve())] = p
        files = sorted(dedup.values(), key=lambda p: p.stat().st_mtime, reverse=True)

        names: list[str] = []
        for p in files:
            if p.parent.resolve() == CHECKPOINT_DIR.resolve():
                names.append(p.name)
            else:
                try:
                    names.append(str(p.resolve().relative_to(ML.resolve())))
                except Exception:
                    names.append(str(p))

        # Prefer canonical latest first if present.
        if "latest.pt" in names:
            names.remove("latest.pt")
            names.insert(0, "latest.pt")
        return names

    def _load_model(
        self,
        checkpoint: Optional[str],
        force: bool = False,
    ) -> tuple[Optional[Any], Optional[str], Optional[str]]:
        if not TORCH_OK:
            return None, checkpoint, "PyTorch unavailable"

        path = self._resolve_checkpoint(checkpoint)
        if path is None:
            return None, checkpoint, f"Checkpoint not found: {checkpoint or 'latest.pt'}"

        key = str(path.resolve())
        if force:
            self.model_cache.pop(key, None)
        if key in self.model_cache:
            return self.model_cache[key], path.name, None

        try:
            model = MarjapussiNet()
            state = torch.load(str(path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]

            model_state = model.state_dict()
            compatible = {}
            skipped = 0
            for k, v in state.items():
                if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
                    compatible[k] = v
                else:
                    skipped += 1

            if not compatible:
                return None, path.name, (
                    f"Incompatible checkpoint {path.name}: no matching tensor shapes for current model."
                )

            model.load_state_dict(compatible, strict=False)
            model.eval()
            self.model_cache[key] = model
            warn = None
            if skipped > 0:
                warn = (
                    f"Partially loaded {path.name}: "
                    f"{len(compatible)} tensors loaded, {skipped} skipped (model/version mismatch)."
                )
            return model, path.name, warn
        except Exception as e:
            return None, path.name, f"Failed to load checkpoint {path.name}: {e}"

    def _try_enable_model_for_seat(self, seat: int, checkpoint: Optional[str]) -> None:
        model, resolved, err = self._load_model(checkpoint)
        ctrl = self.controllers[seat]
        ctrl.checkpoint = resolved or checkpoint
        ctrl.error = err
        if model is None:
            ctrl.mode = "heuristic"
            self.seat_models.pop(seat, None)
            return
        ctrl.mode = "model"
        self.seat_models[seat] = model

    def _effective_mode(self, seat: int) -> str:
        ctrl = self.controllers[seat]
        if ctrl.mode != "model":
            return ctrl.mode
        if seat not in self.seat_models:
            return "heuristic"
        return "model"

    def controller_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for seat in range(4):
            ctrl = self.controllers[seat]
            out[str(seat)] = {
                "mode": ctrl.mode,
                "effective_mode": self._effective_mode(seat),
                "checkpoint": ctrl.checkpoint,
                "error": ctrl.error,
            }
        return out

    def human_seats(self) -> list[int]:
        return [s for s in range(4) if self._effective_mode(s) == "human"]

    def active_seat(self) -> int:
        if not self.obs:
            return -1
        return int(self.obs.get("active_player", -1))

    def new_game(self, seed: Optional[int] = None) -> None:
        self.close()
        self.env = MarjapussiEnv(pov=0)
        self.obs = self.env.reset(seed=seed)
        self.done = self.env.done
        self.info = {}
        self.last_seed = seed
        self.seat_debug.clear()

    def set_view_seat(self, seat: int) -> None:
        if seat < 0 or seat > 3:
            raise ValueError(f"view seat out of range: {seat}")
        self.view_seat = int(seat)

    def set_controller(self, seat: int, mode: str, checkpoint: Optional[str] = None) -> None:
        if seat < 0 or seat > 3:
            raise ValueError(f"seat out of range: {seat}")
        mode = (mode or "").strip().lower()
        if mode not in CONTROLLER_MODES:
            raise ValueError(f"invalid controller mode: {mode}")

        ctrl = self.controllers[seat]
        ctrl.mode = mode

        if mode == "model":
            wanted = checkpoint or ctrl.checkpoint or self.default_checkpoint
            model, resolved, err = self._load_model(wanted)
            ctrl.checkpoint = resolved or wanted
            ctrl.error = err
            if model is None:
                self.seat_models.pop(seat, None)
            else:
                self.seat_models[seat] = model
            return

        if mode in ("human", "heuristic"):
            ctrl.error = None
        self.seat_models.pop(seat, None)

    def reload_model(self, seat: Optional[int], checkpoint: Optional[str] = None) -> None:
        seats = range(4) if seat is None else [seat]
        for s in seats:
            if s < 0 or s > 3:
                continue
            ctrl = self.controllers[s]
            wanted = checkpoint or ctrl.checkpoint or self.default_checkpoint
            model, resolved, err = self._load_model(wanted, force=True)
            ctrl.checkpoint = resolved or wanted
            ctrl.error = err
            if ctrl.mode == "model":
                if model is None:
                    self.seat_models.pop(s, None)
                else:
                    self.seat_models[s] = model

    def _policy_preview(
        self,
        seat: int,
        seat_obs: dict[str, Any],
        model: Any,
        include_hidden: bool = False,
    ) -> dict[str, Any]:
        legal = seat_obs.get("legal_actions", [])
        if not legal:
            return {
                "probs": [],
                "entropy": 0.0,
                "chosen_action_pos": 0,
                "chosen_action_list_idx": None,
                "chosen_label": None,
            }

        tensors = obs_to_tensors(seat_obs)
        with torch.no_grad():
            logits, card_logits, _, _ = model(tensors)
        probs_t = F.softmax(logits[0], dim=-1)
        probs = probs_t.tolist()
        entropy = float(-(probs_t * torch.log(probs_t + 1e-9)).sum().item())
        chosen_pos = int(torch.argmax(probs_t).item())
        chosen_pos = min(chosen_pos, len(legal) - 1)

        prob_rows: list[dict[str, Any]] = []
        for i, la in enumerate(legal):
            if i >= len(probs):
                break
            prob_rows.append(
                {
                    "label": action_label(la),
                    "prob": float(probs[i]),
                    "action_list_idx": int(la.get("action_list_idx", i)),
                    "action_token": int(la.get("action_token", 0)),
                }
            )
        prob_rows.sort(key=lambda p: p["prob"], reverse=True)

        chosen_la = legal[chosen_pos]
        out: dict[str, Any] = {
            "probs": prob_rows[:8],
            "entropy": entropy,
            "chosen_action_pos": chosen_pos,
            "chosen_action_list_idx": int(chosen_la.get("action_list_idx", chosen_pos)),
            "chosen_label": action_label(chosen_la),
        }

        if include_hidden:
            out["hidden_summary"] = self._hidden_summary(seat, seat_obs, card_logits)
        return out

    def _hidden_summary(self, seat: int, seat_obs: dict[str, Any], card_logits: Any) -> dict[str, Any]:
        if self.env is None:
            return {}
        try:
            debug_obs = self.env.observe_debug()
            all_hands = debug_obs.get("all_hands", [])
        except Exception:
            all_hands = []

        probs = torch.sigmoid(card_logits[0]).tolist()  # [3][36]
        cards_remaining = seat_obs.get("cards_remaining", [0, 0, 0, 0])
        possible = seat_obs.get("possible_bitmasks", [[False] * 36 for _ in range(3)])
        seats = ["left", "partner", "right"]
        summary: dict[str, Any] = {}

        pos_losses: list[float] = []
        imp_losses: list[float] = []
        pos_hits: list[float] = []
        imp_probs: list[float] = []

        for rel in range(3):
            row = probs[rel]
            want = int(cards_remaining[rel + 1]) if rel + 1 < len(cards_remaining) else 0
            want = max(0, min(9, want))
            ranked = sorted(range(36), key=lambda i: row[i], reverse=True)[:want]
            abs_seat = rel_hidden_abs_seat(seat, rel)
            truth = set(all_hands[abs_seat]) if abs_seat < len(all_hands) else set()

            imp_vals = [row[i] for i in range(36) if not bool(possible[rel][i])]
            impossible_mass = (sum(imp_vals) / len(imp_vals)) if imp_vals else 0.0

            top_cards = [
                {"card_idx": int(ci), "prob": float(row[ci]), "possible": bool(possible[rel][ci])}
                for ci in ranked
            ]
            summary[seats[rel]] = {
                "abs_seat": abs_seat,
                "impossible_mass": float(impossible_mass),
                "top_cards": top_cards,
            }

            eps = 1e-8
            for ci in range(36):
                p = min(max(float(row[ci]), eps), 1.0 - eps)
                if ci in truth:
                    pos_losses.append(-math.log(p))
                    pos_hits.append(1.0 if p >= 0.5 else 0.0)
                elif not bool(possible[rel][ci]):
                    imp_losses.append(-math.log(1.0 - p))
                    imp_probs.append(p)

        summary["hidden_loss"] = {
            "pos_bce": (sum(pos_losses) / len(pos_losses)) if pos_losses else 0.0,
            "impossible_bce": (sum(imp_losses) / len(imp_losses)) if imp_losses else 0.0,
            "pos_acc": (sum(pos_hits) / len(pos_hits)) if pos_hits else 0.0,
            "impossible_mass": (sum(imp_probs) / len(imp_probs)) if imp_probs else 0.0,
        }
        summary["hidden_loss"]["total"] = (
            summary["hidden_loss"]["pos_bce"] + 2.0 * summary["hidden_loss"]["impossible_bce"]
        )
        return summary

    def _step_action_idx(self, action_list_idx: int) -> None:
        if self.env is None:
            raise RuntimeError("No game in progress")
        self.obs, self.done, self.info = self.env.step(action_list_idx)

    def human_step(self, action_list_idx: int) -> None:
        self._step_action_idx(action_list_idx)

    def proceed_one(self) -> None:
        if self.done or self.obs is None or self.env is None:
            return

        active = self.active_seat()
        if active < 0:
            return

        mode = self._effective_mode(active)
        if mode == "human":
            return

        seat_obs: dict[str, Any]
        observe_err: Optional[str] = None
        try:
            seat_obs = self.env.observe_pov(active)
        except Exception as e:
            observe_err = str(e)
            # Fallback for robustness: if POV observation fails, keep game progressing.
            # For active P0 we can still use root observation; otherwise fall back to heuristic action.
            if active == 0 and isinstance(self.obs, dict):
                seat_obs = self.obs
            else:
                try:
                    pos = int(self.env.get_heuristic_action())
                    legal_root = (self.obs or {}).get("legal_actions", [])
                    if legal_root:
                        pos = max(0, min(pos, len(legal_root) - 1))
                        chosen = legal_root[pos]
                        self.seat_debug[active] = {
                            "probs": [],
                            "entropy": None,
                            "chosen_action_pos": pos,
                            "chosen_action_list_idx": int(chosen.get("action_list_idx", 0)),
                            "chosen_label": action_label(chosen),
                            "status": "fallback_heuristic",
                            "error": f"observe_pov failed: {observe_err}",
                        }
                        self._step_action_idx(int(chosen.get("action_list_idx", 0)))
                except Exception as he:
                    self.seat_debug[active] = {
                        "probs": [],
                        "entropy": None,
                        "chosen_action_pos": 0,
                        "chosen_action_list_idx": None,
                        "chosen_label": None,
                        "status": "observe_error",
                        "error": f"observe_pov failed: {observe_err}; heuristic fallback failed: {he}",
                    }
                return

        legal = seat_obs.get("legal_actions", [])
        if not legal:
            return

        if mode == "heuristic":
            pos = int(self.env.get_heuristic_action())
            pos = max(0, min(pos, len(legal) - 1))
            chosen = legal[pos]
            self.seat_debug[active] = {
                "probs": [],
                "entropy": None,
                "chosen_action_pos": pos,
                "chosen_action_list_idx": int(chosen.get("action_list_idx", 0)),
                "chosen_label": action_label(chosen),
                "status": "heuristic",
            }
            self._step_action_idx(int(chosen.get("action_list_idx", 0)))
            return

        model = self.seat_models.get(active)
        if model is None:
            pos = int(self.env.get_heuristic_action())
            pos = max(0, min(pos, len(legal) - 1))
            chosen = legal[pos]
            self.seat_debug[active] = {
                "probs": [],
                "entropy": None,
                "chosen_action_pos": pos,
                "chosen_action_list_idx": int(chosen.get("action_list_idx", 0)),
                "chosen_label": action_label(chosen),
                "status": "fallback_heuristic",
            }
            self._step_action_idx(int(chosen.get("action_list_idx", 0)))
            return

        preview = self._policy_preview(active, seat_obs, model, include_hidden=True)
        self.seat_debug[active] = {
            **preview,
            "status": "model",
        }
        if observe_err:
            self.seat_debug[active]["error"] = f"observe_pov recovered: {observe_err}"
        chosen_idx = int(preview.get("chosen_action_list_idx", 0))
        self._step_action_idx(chosen_idx)

    def ai_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        active = self.active_seat()

        for seat in range(4):
            ctrl = self.controllers[seat]
            row: dict[str, Any] = {
                "controller": ctrl.mode,
                "effective_controller": self._effective_mode(seat),
                "checkpoint": ctrl.checkpoint,
                "error": ctrl.error,
                "status": "idle",
            }
            cached = self.seat_debug.get(seat)
            if cached:
                row.update(cached)

            if seat == active and self.env is not None and self.obs is not None:
                mode = self._effective_mode(seat)
                if mode == "heuristic":
                    try:
                        seat_obs = self.env.observe_pov(seat)
                        legal = seat_obs.get("legal_actions", [])
                        if legal:
                            pos = int(self.env.get_heuristic_action())
                            pos = max(0, min(pos, len(legal) - 1))
                            row["next_label"] = action_label(legal[pos])
                            row["status"] = "heuristic_ready"
                    except Exception as e:
                        row["status"] = "heuristic_error"
                        row["error"] = str(e)
                elif mode == "model":
                    model = self.seat_models.get(seat)
                    if model is None:
                        row["status"] = "model_missing"
                    else:
                        try:
                            seat_obs = self.env.observe_pov(seat)
                            preview = self._policy_preview(seat, seat_obs, model, include_hidden=True)
                            row.update(preview)
                            row["status"] = "model_ready"
                        except Exception as e:
                            row["status"] = "model_error"
                            row["error"] = str(e)
                elif mode == "human":
                    row["status"] = "waiting_human"

            info[str(seat)] = row

        return info

    def seat_views(self) -> dict[str, Any]:
        views: dict[str, Any] = {}
        if self.env is None or self.obs is None:
            return views

        seats: set[int] = set()
        if self.view_seat > 0:
            seats.add(self.view_seat)

        # Ensure a human-controlled active non-P0 seat is always viewable for input.
        active = self.active_seat()
        if active > 0 and self._effective_mode(active) == "human":
            seats.add(active)

        for seat in sorted(seats):
            try:
                views[str(seat)] = self.env.observe_pov(seat)
            except Exception as e:
                views[str(seat)] = {"error": str(e)}
        return views

    def state(self) -> dict[str, Any]:
        return {
            "obs": self.obs,
            "done": self.done,
            "info": self.info,
            "human_seats": self.human_seats(),
            "controllers": self.controller_payload(),
            "last_seed": self.last_seed,
            "active_seat": self.active_seat(),
            "view_seat": self.view_seat,
            "seat_views": self.seat_views(),
            "checkpoints": self.available_checkpoints(),
            "torch_ok": TORCH_OK,
        }

    def debug_state(self) -> dict[str, Any]:
        result: dict[str, Any] = {"hands": {}, "tricks": []}
        if self.obs is None or self.env is None:
            return result

        debug_obs: dict[str, Any] = {}
        try:
            debug_obs = self.env.observe_debug()
        except Exception:
            debug_obs = {}

        hands = debug_obs.get("all_hands", [])
        if hands:
            result["hands"] = {str(i): hand for i, hand in enumerate(hands)}
        if self.info:
            result["tricks"] = self.info.get("tricks", [])

        result["confirmed_bitmasks"] = self.obs.get("confirmed_bitmasks", [])
        result["possible_bitmasks"] = self.obs.get("possible_bitmasks", [])
        result["event_tokens"] = self.obs.get("event_tokens", [])
        result["seat_policy"] = self.ai_info()

        cards_remaining = self.obs.get("cards_remaining", [0, 0, 0, 0])
        possible = self.obs.get("possible_bitmasks", [[False] * 36 for _ in range(3)])
        confirmed = self.obs.get("confirmed_bitmasks", [[False] * 36 for _ in range(3)])

        stats = {}
        for rel_opp in range(3):
            poss_count = sum(1 for i in range(36) if bool(possible[rel_opp][i]))
            conf_count = sum(1 for i in range(36) if bool(confirmed[rel_opp][i]))
            need = int(cards_remaining[rel_opp + 1]) if (rel_opp + 1) < len(cards_remaining) else 0
            stats[str(rel_opp + 1)] = {
                "possible": poss_count,
                "confirmed": conf_count,
                "need": need,
                "slack": max(0, poss_count - need),
            }

        singleton_cards = 0
        for card_idx in range(36):
            cands = sum(1 for rel_opp in range(3) if bool(possible[rel_opp][card_idx]))
            if cands == 1:
                singleton_cards += 1

        true_possible_violations = 0
        wrong_confirmed = 0
        for rel_opp in range(3):
            seat_idx = rel_opp + 1
            truth = set(hands[seat_idx]) if seat_idx < len(hands) else set()
            for card_idx in truth:
                if 0 <= card_idx < 36 and not bool(possible[rel_opp][card_idx]):
                    true_possible_violations += 1
            for card_idx in range(36):
                if bool(confirmed[rel_opp][card_idx]) and card_idx not in truth:
                    wrong_confirmed += 1

        result["inference_stats"] = {
            **stats,
            "singleton_cards": singleton_cards,
            "true_possible_violations": true_possible_violations,
            "wrong_confirmed": wrong_confirmed,
        }

        debug_model = None
        debug_model_seat = None
        for seat in range(4):
            if self._effective_mode(seat) == "model" and seat in self.seat_models:
                debug_model = self.seat_models[seat]
                debug_model_seat = seat
                break

        if TORCH_OK and debug_model is not None:
            try:
                tensors = obs_to_tensors(self.obs)
                with torch.no_grad():
                    _, card_logits, _, _ = debug_model(tensors)
                probs = torch.sigmoid(card_logits[0]).tolist()
                result["debug_model_seat"] = debug_model_seat

                predicted_hands = {}
                impossible_mass = {}
                for rel_opp in range(3):
                    want = int(cards_remaining[rel_opp + 1]) if (rel_opp + 1) < len(cards_remaining) else 0
                    want = max(0, min(9, want))
                    row = probs[rel_opp]
                    ranked = sorted(range(36), key=lambda i: row[i], reverse=True)
                    predicted_hands[str(rel_opp + 1)] = ranked[:want]

                    imp_vals = [row[i] for i in range(36) if not bool(possible[rel_opp][i])]
                    impossible_mass[str(rel_opp + 1)] = (sum(imp_vals) / len(imp_vals)) if imp_vals else 0.0

                result["predicted_hands"] = predicted_hands
                result["predicted_card_probs"] = probs
                result["predicted_impossible_mass"] = impossible_mass

                eps = 1e-8
                pos_losses = []
                imp_losses = []
                pos_hits = []
                imp_probs = []
                for rel_opp in range(3):
                    seat_idx = rel_opp + 1
                    truth = set(hands[seat_idx]) if seat_idx < len(hands) else set()
                    row = probs[rel_opp]
                    for card_idx in range(36):
                        p = min(max(float(row[card_idx]), eps), 1.0 - eps)
                        is_true = card_idx in truth
                        is_possible = bool(possible[rel_opp][card_idx])
                        if is_true:
                            pos_losses.append(-math.log(p))
                            pos_hits.append(1.0 if p >= 0.5 else 0.0)
                        elif not is_possible:
                            imp_losses.append(-math.log(1.0 - p))
                            imp_probs.append(p)

                pos_bce = (sum(pos_losses) / len(pos_losses)) if pos_losses else 0.0
                imp_bce = (sum(imp_losses) / len(imp_losses)) if imp_losses else 0.0
                pos_acc = (sum(pos_hits) / len(pos_hits)) if pos_hits else 0.0
                imp_mass = (sum(imp_probs) / len(imp_probs)) if imp_probs else 0.0
                result["predicted_hidden_loss"] = {
                    "pos_bce": pos_bce,
                    "impossible_bce": imp_bce,
                    "total": pos_bce + 2.0 * imp_bce,
                    "pos_acc": pos_acc,
                    "impossible_mass": imp_mass,
                }
            except Exception as e:
                result["prediction_error"] = str(e)

        return result

    def game_message(self) -> dict[str, Any]:
        return {"type": "game_state", "data": self.state(), "ai_info": self.ai_info()}


game: Optional[GameManager] = None
clients: set[web.WebSocketResponse] = set()


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)

    if game is not None and game.obs is not None:
        await ws.send_str(json.dumps(game.game_message()))

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                await handle(ws, json.loads(msg.data))
            except Exception as e:
                await ws.send_str(json.dumps({"type": "error", "message": str(e)}))
        elif msg.type == aiohttp.WSMsgType.ERROR:
            break

    clients.discard(ws)
    return ws


async def handle(ws: web.WebSocketResponse, cmd: dict[str, Any]) -> None:
    if game is None:
        raise RuntimeError("Game manager not initialized")

    act = cmd.get("cmd")
    if act in ("new_game", "reset_game"):
        seed = cmd.get("seed")
        game.new_game(seed=seed if seed is None else int(seed))
    elif act in ("proceed", "step_ai"):
        if not game.done:
            game.proceed_one()
    elif act == "human_action":
        game.human_step(int(cmd.get("action_list_idx", 0)))
    elif act == "debug_pass":
        card_indices = cmd.get("card_indices", [])
        if not isinstance(card_indices, list):
            raise ValueError("card_indices must be a list")
        if game.env is None:
            raise RuntimeError("No game in progress")
        game.obs, game.done, game.info = game.env.debug_pass([int(c) for c in card_indices])
    elif act == "set_seat":
        seat = int(cmd.get("seat", 0))
        human = bool(cmd.get("human"))
        mode = "human" if human else "heuristic"
        game.set_controller(seat, mode)
    elif act == "set_controller":
        seat = int(cmd.get("seat", 0))
        mode = str(cmd.get("mode", "heuristic"))
        checkpoint = cmd.get("checkpoint")
        game.set_controller(seat, mode, checkpoint)
    elif act == "set_view_seat":
        seat = int(cmd.get("seat", 0))
        game.set_view_seat(seat)
    elif act == "reload_model":
        seat = cmd.get("seat")
        checkpoint = cmd.get("checkpoint")
        game.reload_model(None if seat is None else int(seat), checkpoint)
    elif act == "load_checkpoint":
        checkpoint = cmd.get("checkpoint")
        seat = cmd.get("seat")
        if seat is None:
            for s in range(4):
                if game.controllers[s].mode == "model":
                    game.set_controller(s, "model", checkpoint)
        else:
            game.set_controller(int(seat), "model", checkpoint)
    elif act == "list_checkpoints":
        await ws.send_str(json.dumps({"type": "checkpoints", "items": game.available_checkpoints()}))
        return
    elif act == "debug_state":
        ds = game.debug_state()
        await ws.send_str(json.dumps({"type": "debug_state", **ds}))
        return
    else:
        raise ValueError(f"Unknown command: {act}")

    await broadcast()


async def broadcast() -> None:
    if game is None:
        return
    msg = json.dumps(game.game_message())
    dead: set[web.WebSocketResponse] = set()
    for c in clients:
        try:
            await c.send_str(msg)
        except Exception:
            dead.add(c)
    clients.difference_update(dead)


async def watch_log(_app: web.Application) -> None:
    asyncio.create_task(_watch_log())


async def _watch_log() -> None:
    while True:
        p = RUNS_DIR / "latest" / "log.jsonl"
        if p.exists():
            try:
                lines = p.read_text().splitlines()
                for line in reversed(lines[-20:]):
                    entry = json.loads(line)
                    if entry.get("event") in ("update", "eval"):
                        stats = {
                            "game": entry.get("game"),
                            "stage": entry.get("stage", 0),
                            "loss": entry.get("losses", {}).get("total"),
                            "win_rate": entry.get("win_rate"),
                        }
                        msg = json.dumps({"type": "train_stats", "data": stats})
                        for c in list(clients):
                            try:
                                await c.send_str(msg)
                            except Exception:
                                clients.discard(c)
                        break
            except Exception:
                pass
        await asyncio.sleep(2)


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(UI / "index.html")


async def cleanup(_app: web.Application) -> None:
    if game is not None:
        game.close()


def main() -> None:
    global game
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    game = GameManager(args.checkpoint)
    try:
        game.new_game()
        print("[game] started OK")
    except Exception as e:
        print(f"[warn] Could not start game: {e}")
        print("[warn] Build ml_server first: cargo build --release --bin ml_server")

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", index)
    app.router.add_static("/ui", UI)
    app.router.add_static("/suits", UI / "suits")
    app.on_startup.append(watch_log)
    app.on_cleanup.append(cleanup)

    print(f"\nMarjapussi UI -> http://localhost:{args.port}")
    print(f"PyTorch: {'yes' if TORCH_OK else 'no (heuristic mode)'}")
    print()
    web.run_app(app, port=args.port, print=False)


if __name__ == "__main__":
    main()
