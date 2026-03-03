"""
Evaluate checkpoints on fixed Marjapussi deals with readable trick/action logs.

Examples:
  python ml/eval_fixed_deals.py --list-checkpoints
  python ml/eval_fixed_deals.py --all-checkpoint latest --max-cases 5 --echo
  python ml/eval_fixed_deals.py --interactive-select --echo
  python ml/eval_fixed_deals.py --suite ml/eval/fixed_deals_100.json --all-checkpoint 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
ML = Path(__file__).parent
CHECKPOINT_DIR = ML / "checkpoints"
COMPLETED_CHECKPOINT_DIR = CHECKPOINT_DIR / "completed"
RUNS_DIR = ML / "runs"

sys.path.insert(0, str(ML))

from env import MarjapussiEnv, obs_to_tensors
from model_factory import create_model, load_state_compatible, parse_checkpoint


VALS = ["6", "7", "8", "9", "U", "O", "K", "10", "A"]
SUIT_SHORT = ["g", "e", "s", "r"]  # Green, Acorns, Bells, Red
SUIT_SYMBOL = ["\u2663", "\u2660", "\u2666", "\u2665"]
SUIT_ANSI = ["\033[92m", "\033[33m", "\033[96m", "\033[91m"]
ANSI_RESET = "\033[0m"

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


@dataclass
class LoadedModel:
    path: Path
    model: Any
    family: str
    loaded_tensors: int
    skipped_tensors: int


class LogSink:
    def __init__(self, echo: bool) -> None:
        self.echo = bool(echo)
        self.lines: list[str] = []

    def write(self, line: str) -> None:
        self.lines.append(line)
        if self.echo:
            print(line)


def card_idx_to_text(card_idx: int, ansi: bool = False) -> str:
    suit_idx = int(card_idx) // 9
    val_idx = int(card_idx) % 9
    v = VALS[val_idx]
    sym = SUIT_SYMBOL[suit_idx]
    if ansi:
        return f"{SUIT_ANSI[suit_idx]}{v}{sym}{ANSI_RESET}"
    return f"{v}{sym}"


def parse_card_spec(raw: Any) -> int:
    if isinstance(raw, int):
        if 0 <= raw < 36:
            return raw
        raise ValueError(f"card index out of range: {raw}")

    if not isinstance(raw, str):
        raise ValueError(f"invalid card spec type: {type(raw)}")

    s = raw.strip().upper().replace("_", "").replace(" ", "").replace("-", "")
    if not s:
        raise ValueError(f"invalid card spec: {raw!r}")

    # Also accept raw numeric strings ("0".."35").
    if s.isdigit():
        ci = int(s)
        if 0 <= ci < 36:
            return ci
        raise ValueError(f"card index out of range: {raw!r}")

    suit_map = {
        # Canonical core-game notation (see src/game/cards.rs):
        # g/e/s/r
        "G": 0,
        "E": 1,
        "S": 2,
        "R": 3,
        # Compatibility aliases:
        "C": 0,  # clubs/leaves
        "L": 0,
        "B": 2,  # bells
        "A": 2,  # old custom alias for bells in previous eval files
        "H": 3,  # hearts alias for red
    }
    val_map = {
        "6": 0,
        "7": 1,
        "8": 2,
        "9": 3,
        "U": 4,
        "J": 4,
        "O": 5,
        "Q": 5,
        "K": 6,
        "Z": 7,
        "10": 7,
        "T": 7,
        # Ace/Sau aliases
        "A": 8,
        "S": 8,
    }

    def _parse_suit_first(token: str) -> int | None:
        if len(token) < 2:
            return None
        suit = token[0]
        value = token[1:]
        if suit not in suit_map or value not in val_map:
            return None
        return suit_map[suit] * 9 + val_map[value]

    def _parse_value_first(token: str) -> int | None:
        if len(token) < 2:
            return None
        suit = token[-1]
        value = token[:-1]
        if suit not in suit_map or value not in val_map:
            return None
        return suit_map[suit] * 9 + val_map[value]

    ci = _parse_suit_first(s)
    if ci is not None:
        return ci
    ci = _parse_value_first(s)
    if ci is not None:
        return ci
    raise ValueError(f"invalid card spec: {raw!r}")


def parse_hand_spec(raw: Any) -> list[int]:
    if isinstance(raw, list):
        return [parse_card_spec(c) for c in raw]
    if isinstance(raw, str):
        text = (
            raw.replace("[", " ")
            .replace("]", " ")
            .replace(";", " ")
            .replace("|", " ")
            .replace("\n", " ")
        )
        tokens = [t for t in re.split(r"[,\s]+", text) if t]
        return [parse_card_spec(t) for t in tokens]
    raise ValueError(f"hand must be list or string, got: {type(raw)}")


def normalize_fixed_hands(raw_hands: Any) -> list[list[int]]:
    hands_by_seat: list[Any]
    if isinstance(raw_hands, list):
        if len(raw_hands) != 4:
            raise ValueError("hands must be a list of 4 seat hands")
        hands_by_seat = raw_hands
    elif isinstance(raw_hands, dict):
        keys = ["p0_hand", "p1_hand", "p2_hand", "p3_hand"]
        missing = [k for k in keys if k not in raw_hands]
        if missing:
            raise ValueError(f"hands object missing keys: {', '.join(missing)}")
        hands_by_seat = [raw_hands[k] for k in keys]
    else:
        raise ValueError("hands must be either a list[4] or object with p0_hand..p3_hand")

    out: list[list[int]] = []
    seen: set[int] = set()
    for seat_idx, hand in enumerate(hands_by_seat):
        parsed = parse_hand_spec(hand)
        if len(parsed) != 9:
            raise ValueError(f"hands[{seat_idx}] must contain exactly 9 cards, got {len(parsed)}")
        for c in parsed:
            if c in seen:
                raise ValueError(f"duplicate card index in fixed hands: {c}")
            seen.add(c)
        out.append(parsed)
    if len(seen) != 36:
        missing = sorted(set(range(36)) - seen)
        raise ValueError(f"fixed hands must cover all 36 cards exactly once, missing: {missing}")
    return out


def format_hand(cards: list[int], ansi: bool = False) -> str:
    vals = sorted(int(c) for c in cards)
    return "[" + " ".join(card_idx_to_text(c, ansi=ansi) for c in vals) + "]"


def format_action_label(la: dict[str, Any], ansi: bool = False) -> str:
    tok = int(la.get("action_token", 0))
    base = ACTION_LABELS.get(tok, f"Act{tok}")
    if la.get("card_idx") is not None:
        return f"{base} {card_idx_to_text(int(la['card_idx']), ansi=ansi)}"
    if la.get("bid_value") is not None:
        return f"{base} {int(la['bid_value'])}"
    if la.get("suit_idx") is not None:
        si = int(la["suit_idx"])
        if 0 <= si < len(SUIT_SYMBOL):
            suit_txt = SUIT_SYMBOL[si]
            if ansi:
                suit_txt = f"{SUIT_ANSI[si]}{suit_txt}{ANSI_RESET}"
            return f"{base} {suit_txt}"
    if la.get("pass_cards"):
        cards = " ".join(card_idx_to_text(int(ci), ansi=ansi) for ci in la["pass_cards"])
        return f"{base} [{cards}]"
    return base


def discover_checkpoints() -> list[tuple[str, Path]]:
    roots: list[Path] = [CHECKPOINT_DIR, COMPLETED_CHECKPOINT_DIR]
    if RUNS_DIR.exists():
        roots.extend([p / "checkpoints" for p in RUNS_DIR.iterdir() if p.is_dir()])

    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("*.pt"):
            if p.name.startswith("epoch_"):
                continue
            found[str(p.resolve())] = p

    files = sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[tuple[str, Path]] = []
    for p in files:
        try:
            label = str(p.resolve().relative_to(ROOT.resolve()))
        except Exception:
            label = str(p)
        out.append((label, p.resolve()))
    return out


def print_checkpoint_list(entries: list[tuple[str, Path]]) -> None:
    print("Available checkpoints:")
    for i, (label, _path) in enumerate(entries, start=1):
        print(f"  {i:3d}. {label}")


def resolve_checkpoint_spec(spec: str | None, entries: list[tuple[str, Path]]) -> Path | None:
    if spec is None or str(spec).strip() == "" or str(spec).strip().lower() in {"none", "heuristic"}:
        return None
    s = str(spec).strip()

    p = Path(s)
    if p.exists():
        return p.resolve()
    p_rel = ROOT / s
    if p_rel.exists():
        return p_rel.resolve()

    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(entries):
            return entries[idx - 1][1]
        raise ValueError(f"checkpoint index out of range: {s}")

    s_l = s.lower()
    exact = [path for label, path in entries if label.lower() == s_l or path.name.lower() == s_l]
    if exact:
        return exact[0]
    contains = [path for label, path in entries if s_l in label.lower() or s_l in path.name.lower()]
    if contains:
        return contains[0]

    raise ValueError(f"could not resolve checkpoint spec: {spec!r}")


def choose_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def tensor_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=(device.type == "cuda"))
    if isinstance(x, dict):
        return {k: tensor_to_device(v, device) for k, v in x.items()}
    return x


def load_model(path: Path, device: torch.device, strict_param_budget: int) -> LoadedModel:
    state_dict, ck_meta, _ = parse_checkpoint(path, map_location="cpu")
    family = str(ck_meta.get("model_family", "parallel_v2"))
    model_cfg = ck_meta.get("model_config_path")
    model, _meta = create_model(
        model_family=family,
        model_config_path=model_cfg,
        strict_param_budget=strict_param_budget,
    )
    loaded, skipped = load_state_compatible(model, state_dict)
    model.to(device)
    model.eval()
    return LoadedModel(
        path=path,
        model=model,
        family=family,
        loaded_tensors=loaded,
        skipped_tensors=skipped,
    )


def choose_action_pos(model: Any, seat_obs: dict[str, Any], device: torch.device) -> tuple[int, float]:
    legal = seat_obs.get("legal_actions", [])
    if not legal:
        return 0, 0.0

    tensors = obs_to_tensors(seat_obs)
    tensors = tensor_to_device(tensors, device)
    with torch.no_grad():
        logits, _, _, _ = model(tensors)
        logits = logits[0, : len(legal)]
        probs = F.softmax(logits, dim=-1)
        pos = int(torch.argmax(probs).item())
        conf = float(probs[pos].item())
    return pos, conf


def load_suite(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        cases = raw.get("cases", [])
    elif isinstance(raw, list):
        cases = raw
    else:
        raise ValueError("suite JSON must be an object with 'cases' or a list of cases")

    out: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"case #{i} is not an object")
        case_id = str(case.get("id", f"case_{i:03d}"))
        seed = case.get("seed")
        hands = case.get("hands")
        if seed is None and hands is None:
            raise ValueError(f"{case_id}: either 'seed' or 'hands' must be provided")
        fixed_hands = None
        if hands is not None:
            fixed_hands = normalize_fixed_hands(hands)
        out.append(
            {
                "id": case_id,
                "seed": (None if seed is None else int(seed)),
                "hands": fixed_hands,
                "notes": str(case.get("notes", "")),
            }
        )
    return out


def seat_checkpoint_specs_from_args(args: argparse.Namespace) -> list[str | None]:
    specs = [args.p0, args.p1, args.p2, args.p3]
    if args.all_checkpoint:
        specs = [spec if spec is not None else args.all_checkpoint for spec in specs]
    return specs


def interactive_fill_specs(specs: list[str | None], entries: list[tuple[str, Path]]) -> list[str | None]:
    print_checkpoint_list(entries)
    print("Enter checkpoint index/name/path per seat. Empty keeps current.")
    out = list(specs)
    for seat in range(4):
        cur = out[seat] if out[seat] is not None else "none"
        v = input(f"Seat P{seat} checkpoint [{cur}]: ").strip()
        if v:
            out[seat] = v
    return out


def run_case(
    case: dict[str, Any],
    seat_models: list[LoadedModel | None],
    device: torch.device,
    log: LogSink,
    ansi: bool = False,
    max_steps: int = 300,
) -> dict[str, Any]:
    env = MarjapussiEnv(pov=0, include_labels=False)
    info: dict[str, Any] = {}
    try:
        obs = env.reset(pov=0, seed=case["seed"], fixed_hands=case["hands"])
        done = False
        step_no = 0
        trick_cards: list[tuple[int, int]] = []
        trick_no = 0

        log.write("")
        log.write(f"=== {case['id']} ===")
        if case["notes"]:
            log.write(f"notes: {case['notes']}")
        if case["seed"] is not None:
            log.write(f"seed: {case['seed']}")
        if case["hands"] is not None:
            for s in range(4):
                log.write(f"P{s} start: {format_hand(case['hands'][s], ansi=ansi)}")

        while not done and step_no < max_steps:
            step_no += 1
            active = int(obs.get("active_player", 0)) % 4
            seat_obs = env.observe_pov(active)
            legal = seat_obs.get("legal_actions", [])
            if not legal:
                break

            model_slot = seat_models[active]
            if model_slot is None:
                pos = 0
                conf = 0.0
            else:
                pos, conf = choose_action_pos(model_slot.model, seat_obs, device)
                pos = max(0, min(pos, len(legal) - 1))
            chosen = legal[pos]
            action_idx = int(chosen.get("action_list_idx", pos))
            action_token = int(chosen.get("action_token", 0))
            before_pts = (int(obs.get("points_my_team", 0)), int(obs.get("points_opp_team", 0)))

            obs, done, info = env.step(action_idx)

            if action_token == 40 and chosen.get("card_idx") is not None:
                trick_cards.append((active, int(chosen["card_idx"])))
                if len(trick_cards) == 4:
                    trick_no += 1
                    winner = int(obs.get("active_player", -1))
                    d_pts = (
                        int(obs.get("points_my_team", 0)) - before_pts[0],
                        int(obs.get("points_opp_team", 0)) - before_pts[1],
                    )
                    debug = env.observe_debug()
                    hands = debug.get("all_hands", [])
                    plays = " ".join(
                        f"P{p}:{card_idx_to_text(ci, ansi=ansi)}" for (p, ci) in trick_cards
                    )
                    hands_txt = " | ".join(
                        f"P{s}:{format_hand(hands[s], ansi=ansi)}" if s < len(hands) else f"P{s}:[?]"
                        for s in range(4)
                    )
                    log.write(
                        f"T{trick_no:02d} {plays} -> W P{winner} | dPts {d_pts[0]}:{d_pts[1]} | {hands_txt}"
                    )
                    trick_cards.clear()
            else:
                label = format_action_label(chosen, ansi=ansi)
                log.write(f"A  P{active} {label} (p={conf:.2f})")

        if not done and (not info or "team_points" not in info):
            info = env.run_to_end("heuristic")
        team_points = info.get("team_points", [0, 0]) or [0, 0]
        playing_party = info.get("playing_party")
        log.write(
            "END "
            f"playing_party={playing_party} | "
            f"won={info.get('won')} | schwarz={info.get('schwarz')} | "
            f"value={info.get('game_value')} | bid={info.get('highest_bid')} | "
            f"pts={team_points[0]}:{team_points[1]} | "
            f"contract_made={info.get('contract_made', info.get('won'))}"
        )
        return info
    finally:
        env.close()


def summarize_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(outcomes)
    if n == 0:
        return {"games": 0}

    pass_games = 0
    contracts = 0
    made = 0
    bid_sum = 0.0
    tricks_play_sum = 0.0
    tricks_def_sum = 0.0
    tricks_n = 0
    pair_called = 0
    pair_possible = 0
    margin_sum = 0.0
    margin_n = 0
    for info in outcomes:
        no_one = bool(info.get("no_one_played", False))
        if no_one:
            pass_games += 1
        else:
            contracts += 1
            bid_sum += float(info.get("highest_bid", info.get("game_value", 115)))
            if bool(info.get("contract_made", info.get("won", False))):
                made += 1
        pt = info.get("playing_party_tricks")
        dt = info.get("defending_party_tricks")
        if pt is not None and dt is not None:
            tricks_play_sum += float(pt)
            tricks_def_sum += float(dt)
            tricks_n += 1
        cp = info.get("playing_called_trumps")
        pp = info.get("playing_possible_pairs")
        if cp is not None and pp is not None:
            pair_called += int(cp)
            pair_possible += int(pp)
        playing_party = info.get("playing_party")
        team_points = info.get("team_points", [0, 0]) or [0, 0]
        if playing_party is not None and isinstance(team_points, (list, tuple)) and len(team_points) == 2:
            p = int(playing_party) % 2
            margin_sum += float(team_points[p]) - float(team_points[1 - p])
            margin_n += 1

    return {
        "games": n,
        "pass_game_rate": pass_games / max(1, n),
        "contract_made_rate": made / max(1, contracts),
        "avg_highest_bid": bid_sum / max(1, contracts),
        "avg_playing_tricks": tricks_play_sum / max(1, tricks_n),
        "avg_defending_tricks": tricks_def_sum / max(1, tricks_n),
        "pair_call_rate": (pair_called / max(1, pair_possible)) if pair_possible > 0 else 0.0,
        "pair_called": pair_called,
        "pair_possible": pair_possible,
        "avg_playing_margin_points": margin_sum / max(1, margin_n),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate checkpoints on fixed deal suites.")
    p.add_argument("--suite", default="ml/eval/fixed_deals_100.json")
    p.add_argument("--output", default="")
    p.add_argument("--max-cases", type=int, default=0, help="0 means all")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--strict-param-budget", type=int, default=28_000_000)
    p.add_argument("--all-checkpoint", default=None, help="Checkpoint spec for all seats")
    p.add_argument("--p0", default=None, help="Checkpoint spec for seat P0")
    p.add_argument("--p1", default=None, help="Checkpoint spec for seat P1")
    p.add_argument("--p2", default=None, help="Checkpoint spec for seat P2")
    p.add_argument("--p3", default=None, help="Checkpoint spec for seat P3")
    p.add_argument("--list-checkpoints", action="store_true")
    p.add_argument("--interactive-select", action="store_true")
    p.add_argument("--echo", action="store_true", help="Also print trick/action log to console")
    p.add_argument("--ansi", action="store_true", help="Emit ANSI colors in logs")
    args = p.parse_args()

    entries = discover_checkpoints()
    if args.list_checkpoints:
        print_checkpoint_list(entries)
        return

    specs = seat_checkpoint_specs_from_args(args)
    if args.interactive_select:
        specs = interactive_fill_specs(specs, entries)

    resolved_paths: list[Path | None] = []
    for spec in specs:
        resolved_paths.append(resolve_checkpoint_spec(spec, entries))

    device = choose_device(args.device)
    print(f"[INFO] device={device}")

    model_cache: dict[str, LoadedModel] = {}
    seat_models: list[LoadedModel | None] = []
    for seat, path in enumerate(resolved_paths):
        if path is None:
            seat_models.append(None)
            print(f"[INFO] P{seat}: no checkpoint (fallback first legal action)")
            continue
        key = str(path.resolve())
        if key not in model_cache:
            model_cache[key] = load_model(path, device, strict_param_budget=args.strict_param_budget)
        seat_models.append(model_cache[key])
        m = model_cache[key]
        print(
            f"[INFO] P{seat}: {path} | family={m.family} | loaded={m.loaded_tensors} skipped={m.skipped_tensors}"
        )

    suite_path = Path(args.suite)
    if not suite_path.exists():
        raise FileNotFoundError(f"suite file not found: {suite_path}")
    cases = load_suite(suite_path)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    print(f"[INFO] cases={len(cases)} suite={suite_path}")

    out_path = Path(args.output) if args.output else (
        ML / "eval" / "logs" / f"fixed_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sink = LogSink(echo=args.echo)

    outcomes: list[dict[str, Any]] = []
    for case in cases:
        info = run_case(case, seat_models, device=device, log=sink, ansi=args.ansi)
        outcomes.append(info)

    summary = summarize_outcomes(outcomes)
    sink.write("")
    sink.write("=== SUMMARY ===")
    sink.write(
        f"games={summary.get('games', 0)} | "
        f"made={summary.get('contract_made_rate', 0.0):.1%} | "
        f"pass={summary.get('pass_game_rate', 0.0):.1%} | "
        f"avg_bid={summary.get('avg_highest_bid', 0.0):.1f}"
    )
    sink.write(
        f"tricks(play:def)={summary.get('avg_playing_tricks', 0.0):.2f}:{summary.get('avg_defending_tricks', 0.0):.2f} | "
        f"trump_calls={summary.get('pair_called', 0)}/{summary.get('pair_possible', 0)} "
        f"({summary.get('pair_call_rate', 0.0):.1%}) | "
        f"avg_margin={summary.get('avg_playing_margin_points', 0.0):+.1f}"
    )

    out_path.write_text("\n".join(sink.lines) + "\n", encoding="utf-8")
    print(f"[SUCCESS] wrote log: {out_path}")


if __name__ == "__main__":
    main()
