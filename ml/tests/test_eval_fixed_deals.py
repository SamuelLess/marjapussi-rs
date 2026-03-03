import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_fixed_deals import normalize_fixed_hands, parse_card_spec, summarize_outcomes


class EvalFixedDealsHelpersTest(unittest.TestCase):
    def test_parse_card_spec_accepts_int_and_symbol(self):
        self.assertEqual(parse_card_spec(0), 0)
        self.assertEqual(parse_card_spec("g-6"), 0)
        self.assertEqual(parse_card_spec("r-a"), 35)
        self.assertEqual(parse_card_spec("s10"), 25)
        self.assertEqual(parse_card_spec("RK"), 33)
        self.assertEqual(parse_card_spec("SZ"), 25)
        self.assertEqual(parse_card_spec("SA"), 26)
        # Legacy alias compatibility (older custom fixed-deal files)
        self.assertEqual(parse_card_spec("AZ"), 25)

    def test_normalize_fixed_hands_requires_full_unique_deck(self):
        hands = [
            list(range(0, 9)),
            list(range(9, 18)),
            list(range(18, 27)),
            list(range(27, 36)),
        ]
        norm = normalize_fixed_hands(hands)
        self.assertEqual(len(norm), 4)
        self.assertEqual(sum(len(h) for h in norm), 36)

    def test_normalize_fixed_hands_accepts_compact_object_format(self):
        hands = {
            "p0_hand": "G6 G7 G8 G9 GU GO GK GZ GS",
            "p1_hand": "E6 E7 E8 E9 EU EO EK EZ ES",
            "p2_hand": "S6 S7 S8 S9 SU SO SK SZ SA",
            "p3_hand": "R6 R7 R8 R9 RU RO RK RZ RA",
        }
        norm = normalize_fixed_hands(hands)
        self.assertEqual(len(norm), 4)
        self.assertEqual(sum(len(h) for h in norm), 36)

    def test_summarize_outcomes_basic(self):
        out = summarize_outcomes(
            [
                {
                    "no_one_played": False,
                    "contract_made": True,
                    "highest_bid": 140,
                    "playing_party_tricks": 6,
                    "defending_party_tricks": 3,
                    "playing_called_trumps": 2,
                    "playing_possible_pairs": 3,
                    "playing_party": 0,
                    "team_points": [160, 80],
                },
                {
                    "no_one_played": True,
                    "playing_party": None,
                    "team_points": [90, 70],
                },
            ]
        )
        self.assertEqual(out["games"], 2)
        self.assertAlmostEqual(out["pass_game_rate"], 0.5)
        self.assertAlmostEqual(out["contract_made_rate"], 1.0)
        self.assertAlmostEqual(out["pair_call_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
