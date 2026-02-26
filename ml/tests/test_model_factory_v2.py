import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from model_factory import (
    build_checkpoint_payload,
    create_model,
    load_state_compatible,
    parse_checkpoint,
)


class ModelFactoryV2Test(unittest.TestCase):
    def test_create_legacy_model(self):
        model, meta = create_model("legacy")
        self.assertEqual(meta["model_family"], "legacy")
        self.assertGreater(model.param_count(), 0)

    def test_create_parallel_model_with_budget(self):
        model, meta = create_model("parallel_v2")
        self.assertEqual(meta["model_family"], "parallel_v2")
        self.assertLessEqual(model.param_count(), 28_000_000)

    def test_budget_gate_raises(self):
        with self.assertRaises(ValueError):
            create_model("parallel_v2", strict_param_budget=1_000_000)

    def test_checkpoint_roundtrip_with_metadata(self):
        model, meta = create_model("parallel_v2")
        payload = build_checkpoint_payload(
            model,
            metadata={**meta, "schema_version": 1, "action_encoding_version": 1},
            extra_metadata={"checkpoint_kind": "unit_test"},
        )

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "tmp_parallel.pt"
            torch.save(payload, ckpt)
            state, parsed_meta, _ = parse_checkpoint(ckpt)
            self.assertEqual(parsed_meta.get("model_family"), "parallel_v2")

            model2, _ = create_model("parallel_v2")
            loaded, skipped = load_state_compatible(model2, state)
            self.assertGreater(loaded, 0)
            self.assertEqual(skipped, 0)

    def test_parse_legacy_state_dict_checkpoint(self):
        model, _ = create_model("legacy")
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "legacy_state_only.pt"
            torch.save(model.state_dict(), ckpt)
            state, meta, _ = parse_checkpoint(ckpt)
            self.assertEqual(meta, {})
            model2, _ = create_model("legacy")
            loaded, skipped = load_state_compatible(model2, state)
            self.assertGreater(loaded, 0)
            self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()
