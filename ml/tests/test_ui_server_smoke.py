import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ui_server


class UiServerSmokeTest(unittest.TestCase):
    def test_action_label_formats_card_and_bid(self):
        self.assertTrue(ui_server.action_label({"action_token": 40, "card_idx": 8}).startswith("Play "))
        self.assertEqual(ui_server.action_label({"action_token": 41, "bid_value": 130}), "Bid 130")

    def test_relative_hidden_seat_mapping(self):
        self.assertEqual(ui_server.rel_hidden_abs_seat(0, 0), 1)
        self.assertEqual(ui_server.rel_hidden_abs_seat(0, 1), 2)
        self.assertEqual(ui_server.rel_hidden_abs_seat(0, 2), 3)
        self.assertEqual(ui_server.rel_hidden_abs_seat(3, 0), 0)

    def test_game_manager_controller_payload_shape(self):
        gm = ui_server.GameManager(checkpoint=None)
        payload = gm.controller_payload()
        self.assertEqual(set(payload.keys()), {"0", "1", "2", "3"})
        self.assertEqual(payload["0"]["mode"], "human")
        gm.close()

    def test_checkpoint_resolve_handles_missing(self):
        gm = ui_server.GameManager(checkpoint=None)
        missing = gm._resolve_checkpoint("definitely_missing_checkpoint_name")
        self.assertIsNone(missing)
        latest = gm._resolve_checkpoint("latest")
        if latest is not None:
            self.assertEqual(Path(latest).name, "latest.pt")
        gm.close()

    def test_seat_views_returns_only_active_human_seat(self):
        class DummyEnv:
            def observe_pov(self, seat: int):
                return {"seat": seat}

        gm = ui_server.GameManager(checkpoint=None)
        gm.env = DummyEnv()
        gm.obs = {"active_player": 2}
        gm.set_controller(1, "human")
        gm.set_controller(2, "human")
        gm.set_controller(3, "human")

        views = gm.seat_views()
        self.assertEqual(set(views.keys()), {"2"})
        self.assertEqual(views["2"]["seat"], 2)
        gm.close()


if __name__ == "__main__":
    unittest.main()
