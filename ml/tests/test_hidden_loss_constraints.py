import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.loss import (
    _hidden_exclusive_assignment_loss,
    _hidden_known_positive_loss,
)


class HiddenLossConstraintsTest(unittest.TestCase):
    def test_known_positive_loss_tracks_confirmed_cards(self):
        card_logits = torch.full((1, 3, 36), -6.0)
        hidden_known = torch.zeros((1, 3, 36))
        hidden_known[0, 1, 5] = 1.0

        loss_bad, acc_bad = _hidden_known_positive_loss(card_logits, hidden_known)
        self.assertGreater(float(loss_bad.item()), 4.0)
        self.assertLess(float(acc_bad.item()), 0.5)

        card_logits[0, 1, 5] = 8.0
        loss_good, acc_good = _hidden_known_positive_loss(card_logits, hidden_known)
        self.assertLess(float(loss_good.item()), 0.1)
        self.assertGreater(float(acc_good.item()), 0.99)

    def test_exclusive_assignment_enforces_single_seat_per_card(self):
        card_logits = torch.full((1, 3, 36), -5.0)
        hidden_target = torch.zeros((1, 3, 36))
        hidden_possible = torch.zeros((1, 3, 36))

        # Three hidden cards, all seats possible -> model must choose correct seat.
        for c in (0, 1, 2):
            hidden_possible[0, :, c] = 1.0
        hidden_target[0, 0, 0] = 1.0
        hidden_target[0, 1, 1] = 1.0
        hidden_target[0, 2, 2] = 1.0

        card_logits[0, 0, 0] = 9.0
        card_logits[0, 1, 1] = 9.0
        card_logits[0, 2, 2] = 9.0

        loss, acc = _hidden_exclusive_assignment_loss(card_logits, hidden_target, hidden_possible)
        self.assertLess(float(loss.item()), 0.1)
        self.assertGreater(float(acc.item()), 0.99)


if __name__ == "__main__":
    unittest.main()
