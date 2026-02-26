import sys
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import ui_server


class UiServerSmokeTest(unittest.TestCase):
    def test_ui_loads_emoji_mapper(self):
        html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "ui" / "emoji_map.js").read_text(encoding="utf-8")
        self.assertIn("/ui/emoji_map.js", html)
        self.assertIn("UI_GLYPHS", js)
        self.assertIn('id="view-seat"', html)
        self.assertIn('id="handle-main"', html)
        self.assertIn('id="handle-side"', html)

    def test_action_label_formats_card_and_bid(self):
        self.assertTrue(ui_server.action_label({"action_token": 40, "card_idx": 8}).startswith("Play "))
        self.assertEqual(ui_server.action_label({"action_token": 41, "bid_value": 130}), "Bid 130")

    def test_relative_hidden_seat_mapping(self):
        self.assertEqual(ui_server.rel_hidden_abs_seat(0, 0), 1)
        self.assertEqual(ui_server.rel_hidden_abs_seat(0, 1), 2)
        self.assertEqual(ui_server.rel_hidden_abs_seat(0, 2), 3)
        self.assertEqual(ui_server.rel_hidden_abs_seat(3, 0), 0)

    def test_hidden_assignment_is_unique_and_possible(self):
        probs = [
            [0.9] + [0.01] * 35,
            [0.8] + [0.7] + [0.01] * 34,
            [0.6] + [0.5] + [0.4] + [0.01] * 33,
        ]
        possible = [[False] * 36 for _ in range(3)]
        for s in range(3):
            for c in range(3):
                possible[s][c] = True
        assigned = ui_server.assign_hidden_cards_unique(probs, possible, wants=[1, 1, 1])
        flat = [c for seat_cards in assigned for c in seat_cards]
        self.assertEqual(len(flat), 3)
        self.assertEqual(len(set(flat)), 3)
        for s, seat_cards in enumerate(assigned):
            for c in seat_cards:
                self.assertTrue(possible[s][c])

    def test_game_manager_controller_payload_shape(self):
        gm = ui_server.GameManager(checkpoint=None)
        payload = gm.controller_payload()
        self.assertEqual(set(payload.keys()), {"0", "1", "2", "3"})
        self.assertEqual(payload["0"]["mode"], "human")
        self.assertIn("model_family", payload["0"])
        gm.close()

    def test_state_contains_view_seat(self):
        gm = ui_server.GameManager(checkpoint=None)
        st = gm.state()
        self.assertIn("view_seat", st)
        self.assertEqual(st["view_seat"], 0)
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

    def test_seat_views_include_configured_view_seat(self):
        class DummyEnv:
            def observe_pov(self, seat: int):
                return {"seat": seat}

        gm = ui_server.GameManager(checkpoint=None)
        gm.env = DummyEnv()
        gm.obs = {"active_player": 0}
        gm.set_view_seat(3)
        views = gm.seat_views()
        self.assertEqual(set(views.keys()), {"3"})
        self.assertEqual(views["3"]["seat"], 3)
        gm.close()

    def test_reload_model_keeps_checkpoint_choice(self):
        gm = ui_server.GameManager(checkpoint=None)
        gm.set_controller(1, "model", "definitely_missing_checkpoint_name")
        gm.reload_model(1, "definitely_missing_checkpoint_name")
        payload = gm.controller_payload()
        self.assertIn("checkpoint", payload["1"])
        self.assertTrue(payload["1"]["checkpoint"] in ("definitely_missing_checkpoint_name", None))
        gm.close()

    def test_resolve_checkpoint_from_ml_relative_path(self):
        tag = f".ui_test_{uuid.uuid4().hex}"
        rel_path = Path("runs") / tag / "checkpoints" / "tmp.pt"
        abs_path = ROOT / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(b"x")
        try:
            gm = ui_server.GameManager(checkpoint=None)
            resolved = gm._resolve_checkpoint(str(rel_path))
            self.assertIsNotNone(resolved)
            self.assertEqual(Path(resolved).resolve(), abs_path.resolve())
            gm.close()
        finally:
            try:
                abs_path.unlink(missing_ok=True)
                abs_path.parent.rmdir()
                abs_path.parent.parent.rmdir()
            except Exception:
                pass

    def test_load_model_partial_fallback(self):
        if not ui_server.TORCH_OK:
            self.skipTest("PyTorch unavailable")
        gm = ui_server.GameManager(checkpoint=None)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "partial.pt"
            base = ui_server.MarjapussiNet()
            sd = base.state_dict()
            first_key = next(iter(sd.keys()))
            partial = {
                first_key: sd[first_key].clone(),
                "totally_wrong.weight": torch.randn(1, 1),
            }
            torch.save(partial, p)
            model, _name, warn = gm._load_model(str(p), force=True)
            self.assertIsNotNone(model)
            self.assertTrue(warn is None or "Partially loaded" in warn)
        gm.close()


if __name__ == "__main__":
    unittest.main()
