import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.loss import _blend_policy_advantage


class LossMetaBlendTest(unittest.TestCase):
    def test_no_blend_outside_full_game(self):
        base = torch.tensor([1.0, -1.0])
        meta = torch.tensor([10.0, 10.0])
        out = _blend_policy_advantage(base, meta, "trick", 0.9)
        self.assertTrue(torch.equal(out, base))

    def test_blend_in_full_game(self):
        base = torch.tensor([1.0, -1.0])
        meta = torch.tensor([3.0, 5.0])
        out = _blend_policy_advantage(base, meta, "full_game", 0.75)
        expected = 0.25 * base + 0.75 * meta
        self.assertTrue(torch.allclose(out, expected))

    def test_handles_none_meta(self):
        base = torch.tensor([1.0, -1.0])
        out = _blend_policy_advantage(base, None, "full_game", 0.75)
        self.assertTrue(torch.equal(out, base))


if __name__ == "__main__":
    unittest.main()

