import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.utils import Transition
from train_online import (
    _apply_round_meta_advantages,
    _apply_series_meta_to_episodes,
    _compose_meta_signal,
    _compute_series_signals_norm,
    _normalize_signal,
)


class TrainOnlineMetaSignalTest(unittest.TestCase):
    def test_normalize_signal_zscore(self):
        vals = [1.0, 2.0, 3.0]
        out = _normalize_signal(vals)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(sum(out) / len(out), 0.0, places=6)
        std = math.sqrt(sum(v * v for v in out) / len(out))
        self.assertAlmostEqual(std, 1.0, places=6)

    def test_normalize_signal_constant(self):
        out = _normalize_signal([7.0, 7.0, 7.0])
        self.assertEqual(out, [0.0, 0.0, 0.0])

    def test_apply_round_meta_advantages_updates_transition_field(self):
        transitions = [
            Transition({}, 0, 0.0, 0.0, 0.0, 0.0, 0, -1.0, True, 0.0, -2.0),
            Transition({}, 0, 0.0, 0.0, 0.0, 0.0, 0, -1.0, True, 0.0, 0.0),
            Transition({}, 0, 0.0, 0.0, 0.0, 0.0, 0, -1.0, True, 0.0, 2.0),
        ]
        out = _apply_round_meta_advantages(transitions)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(sum(t.meta_advantage for t in out) / len(out), 0.0, places=6)

    def test_compose_meta_signal_adds_margin_term(self):
        out = _compose_meta_signal(
            contract_outcome=0.4,
            pts_my_norm=0.7,
            pts_opp_norm=0.5,
            margin_weight=0.2,
            margin_clip=1.0,
        )
        self.assertAlmostEqual(out, 0.44, places=6)

    def test_compose_meta_signal_clips_margin(self):
        out = _compose_meta_signal(
            contract_outcome=-0.3,
            pts_my_norm=1.5,
            pts_opp_norm=0.0,
            margin_weight=0.5,
            margin_clip=0.4,
        )
        self.assertAlmostEqual(out, -0.3 + 0.5 * 0.4, places=6)

    def test_compute_series_signals_target_close(self):
        # target normalized = 500 / 100 = 5.0
        totals = [(2.0, 1.0), (2.5, 1.0), (1.0, 3.0)]
        out = _compute_series_signals_norm(
            totals,
            series_target_points_norm=5.0,
            series_max_games=8,
            series_diff_bonus_frac=0.20,
            series_total_weight=0.0,
            series_diff_weight=1.0,
        )
        # First two games close at target (my=4.5 -> no, after second my=4.5, still no target)
        # Third is last in input and closes segment with diff=(5.5-5.0)=0.5.
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0], out[1], places=6)

    def test_compute_series_signals_applies_bonus_on_max_games_close(self):
        totals = [(1.0, 2.0), (2.0, 1.0)]  # close by max_games=2, no target
        out = _compute_series_signals_norm(
            totals,
            series_target_points_norm=99.0,
            series_max_games=2,
            series_diff_bonus_frac=0.20,
            series_total_weight=0.0,
            series_diff_weight=1.0,
        )
        # Cum is tie (3,3) so bonus doesn't apply; diff signal is 0.
        self.assertEqual(out, [0.0, 0.0])

    def test_apply_series_meta_to_episodes_blends_signal(self):
        eps = [
            [Transition({}, 0, 0.0, 0.8, 0.6, 0.0, 0, -1.0, False, 0.0, 0.2)],
            [Transition({}, 0, 0.0, 0.7, 0.5, 0.0, 0, -1.0, False, 0.0, -0.1)],
        ]
        out = _apply_series_meta_to_episodes(
            eps,
            points_normalizer=100.0,
            series_target_points=500.0,
            series_max_games=2,
            series_diff_bonus_frac=0.20,
            series_total_weight=0.0,
            series_diff_weight=1.0,
            series_blend_weight=1.0,
        )
        # With full blend=1.0, both episode transitions get same segment-level series signal.
        self.assertAlmostEqual(out[0][0].meta_advantage, out[1][0].meta_advantage, places=6)


if __name__ == "__main__":
    unittest.main()
