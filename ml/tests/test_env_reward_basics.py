import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import encode_legal_actions, encode_phase
from model import ACTION_FEAT_DIM
from train.reward import RewardConfig, contract_reward_from_pov


class EnvRewardBasicsTest(unittest.TestCase):
    def test_pass_action_uses_exact_card_mask(self):
        cards = [0, 5, 12, 35]
        legal = [{
            "action_token": 43,
            "card_idx": None,
            "pass_cards": cards,
            "suit_idx": None,
            "bid_value": None,
        }]
        feats, mask = encode_legal_actions(legal)
        self.assertEqual(tuple(feats.shape), (1, 1, ACTION_FEAT_DIM))
        self.assertFalse(bool(mask[0, 0]))
        for c in cards:
            self.assertEqual(float(feats[0, 0, 33 + c]), 1.0)
        self.assertEqual(int(feats[0, 0, 33:69].sum().item()), len(cards))

    def test_encode_phase_answering_half(self):
        phase = encode_phase("AnsweringHalf(Bells)")
        self.assertEqual(tuple(phase.shape), (1, 5))
        self.assertEqual(int(phase.argmax(dim=-1).item()), 3)

    def test_pass_pick_action_uses_single_card_features(self):
        legal = [{
            "action_token": 52,
            "card_idx": 17,
            "pass_cards": None,
            "suit_idx": None,
            "bid_value": None,
        }]
        feats, _ = encode_legal_actions(legal)
        self.assertEqual(float(feats[0, 0, 11]), 1.0)
        self.assertEqual(float(feats[0, 0, 15 + (17 // 9)]), 1.0)
        self.assertEqual(float(feats[0, 0, 19 + (17 % 9)]), 1.0)

    def test_pass_pick_action_gets_phase_and_selection_context(self):
        legal = [{
            "action_token": 52,
            "card_idx": 12,
            "pass_cards": None,
            "suit_idx": None,
            "bid_value": None,
        }]
        feats, _ = encode_legal_actions(
            legal,
            phase_name="PassingBack",
            pass_selection_indices=[0, 9],
            pass_selection_target=4,
        )
        # Asymmetry flag: backpass vs forthpass
        self.assertEqual(float(feats[0, 0, 69]), 0.0)
        self.assertEqual(float(feats[0, 0, 70]), 1.0)
        # Progress: 2 selected out of 4, 2 remaining
        self.assertAlmostEqual(float(feats[0, 0, 71]), 0.5, places=6)
        self.assertAlmostEqual(float(feats[0, 0, 73]), 0.5, places=6)
        # Suit histogram sees one card in suit 0 and one in suit 1
        self.assertAlmostEqual(float(feats[0, 0, 74]), 0.25, places=6)
        self.assertAlmostEqual(float(feats[0, 0, 75]), 0.25, places=6)

    def test_contract_reward_passgame_and_contract_mode(self):
        cfg = RewardConfig()
        passgame_info = {"team_points": [160, 120], "tricks": []}
        reward, _, _, playing_party = contract_reward_from_pov(passgame_info, pov_party=0, cfg=cfg)
        self.assertIsNone(playing_party)
        self.assertGreater(reward, 0.0)

        contract_info = {
            "team_points": [130, 90],
            "tricks": [{"winner": 0}, {"winner": 1}],
            "playing_party": 0,
            "game_value": 140,
            "won": True,
            "schwarz": False,
        }
        reward, _, _, playing_party = contract_reward_from_pov(contract_info, pov_party=0, cfg=cfg)
        self.assertEqual(playing_party, 0)
        self.assertAlmostEqual(reward, 140.0 / 420.0, places=6)


if __name__ == "__main__":
    unittest.main()
