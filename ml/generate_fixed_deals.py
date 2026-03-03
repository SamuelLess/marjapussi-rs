"""
Generate editable fixed-deal evaluation suites via the Rust game engine.

This uses ml_server through MarjapussiEnv:
- deal generation happens in the real game backend
- output stores explicit 4x9 hands so users can edit cases directly

Example:
  python ml/generate_fixed_deals.py --count 100 --output ml/eval/fixed_deals_random_100.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from env import MarjapussiEnv

# Canonical core-game notation from src/game/cards.rs (g/e/s/r + 6..A).
SUIT_CODE = ["G", "E", "S", "R"]  # Green, Acorns, Bells, Red
VALUE_CODE = ["6", "7", "8", "9", "U", "O", "K", "Z", "A"]


def _validate_hands(hands: list[list[int]]) -> None:
    if len(hands) != 4:
        raise ValueError(f"expected 4 hands, got {len(hands)}")
    seen: set[int] = set()
    for i, hand in enumerate(hands):
        if len(hand) != 9:
            raise ValueError(f"seat {i} has {len(hand)} cards, expected 9")
        for c in hand:
            ci = int(c)
            if ci < 0 or ci >= 36:
                raise ValueError(f"invalid card index {ci}")
            if ci in seen:
                raise ValueError(f"duplicate card index {ci}")
            seen.add(ci)
    if len(seen) != 36:
        raise ValueError(f"hands do not cover deck, only {len(seen)} unique cards")


def _card_to_compact(card_idx: int) -> str:
    suit = int(card_idx) // 9
    value = int(card_idx) % 9
    return f"{SUIT_CODE[suit]}{VALUE_CODE[value]}"


def _hand_to_compact_line(cards: list[int]) -> str:
    return " ".join(_card_to_compact(c) for c in sorted(int(ci) for ci in cards))


def generate_suite(count: int, master_seed: int) -> dict:
    rng = random.Random(master_seed)
    env = MarjapussiEnv(pov=0, include_labels=False)
    cases: list[dict] = []
    used_seeds: set[int] = set()

    try:
        for i in range(1, count + 1):
            # Keep seeds stable for reproducibility and prevent duplicates in one run.
            seed = rng.randrange(1, 2**63 - 1)
            while seed in used_seeds:
                seed = rng.randrange(1, 2**63 - 1)
            used_seeds.add(seed)

            env.reset(seed=seed, pov=0)
            dbg = env.observe_debug()
            hands = dbg.get("all_hands", [])
            if not isinstance(hands, list):
                raise ValueError(f"bad observe_debug payload for seed={seed}: no all_hands list")
            fixed_hands = [[int(c) for c in hand] for hand in hands]
            _validate_hands(fixed_hands)

            cases.append(
                {
                    "id": f"random_{i:04d}",
                    "seed": int(seed),
                    "notes": "generated via ml_server shuffle, explicit hands for editing",
                    "hands": {
                        "p0_hand": _hand_to_compact_line(fixed_hands[0]),
                        "p1_hand": _hand_to_compact_line(fixed_hands[1]),
                        "p2_hand": _hand_to_compact_line(fixed_hands[2]),
                        "p3_hand": _hand_to_compact_line(fixed_hands[3]),
                    },
                }
            )
    finally:
        env.close()

    return {
        "version": 1,
        "description": "Random fixed deals generated via ml_server; edit compact hand strings freely.",
        "card_encoding": "Compact token SUIT+VALUE, canonical suits: G=Green, E=Acorns, S=Bells, R=Red; values: 6,7,8,9,U,O,K,Z(10),A(Ace).",
        "generator": {
            "source": "ml/generate_fixed_deals.py",
            "master_seed": int(master_seed),
            "generated_at_unix": int(time.time()),
            "count": int(count),
        },
        "cases": cases,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate fixed-deal suite JSON via ml_server.")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--master-seed", type=int, default=20260303)
    p.add_argument("--output", default="ml/eval/fixed_deals_random_100.json")
    args = p.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be > 0")

    out = generate_suite(args.count, args.master_seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[SUCCESS] wrote {args.count} cases to {out_path}")


if __name__ == "__main__":
    main()
