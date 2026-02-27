import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_online import _select_curriculum


class TrainOnlineCurriculumTest(unittest.TestCase):
    def test_curriculum_boundaries_match_50_10_40_split(self):
        # first 50% => trick-play
        phase, start = _select_curriculum(0.00, 0)
        self.assertEqual(phase, "trick")
        self.assertTrue(1 <= int(start) <= 9)

        phase, start = _select_curriculum(0.49, 5)
        self.assertEqual(phase, "trick")
        self.assertTrue(1 <= int(start) <= 9)

        # next 5% => passing
        phase, start = _select_curriculum(0.50, 0)
        self.assertEqual(phase, "passing")
        self.assertEqual(start, 0)

        phase, start = _select_curriculum(0.549, 0)
        self.assertEqual(phase, "passing")
        self.assertEqual(start, 0)

        # next 5% => bidding
        phase, start = _select_curriculum(0.55, 0)
        self.assertEqual(phase, "bidding_prop")
        self.assertEqual(start, -1)

        phase, start = _select_curriculum(0.599, 0)
        self.assertEqual(phase, "bidding_prop")
        self.assertEqual(start, -1)

        # last 40% => full game sequence
        phase, start = _select_curriculum(0.60, 0)
        self.assertEqual(phase, "full_game")
        self.assertIsNone(start)

        phase, start = _select_curriculum(0.95, 0)
        self.assertEqual(phase, "full_game")
        self.assertIsNone(start)

    def test_trick_target_cycles_through_1_to_9(self):
        seen = set()
        for rnd in range(18):
            phase, start = _select_curriculum(0.25, rnd)
            self.assertEqual(phase, "trick")
            seen.add(int(start))
        self.assertEqual(seen, set(range(1, 10)))


if __name__ == "__main__":
    unittest.main()
