import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import SUPPORTED_OBS_SCHEMA_VERSION, resolve_ml_server_binary
from model import ACTION_FEAT_DIM
from model_factory import build_checkpoint_payload, create_model, parse_checkpoint
from train_online import train_online


class TrainOnlineEpisodeSmokeTest(unittest.TestCase):
    def test_two_round_online_training_from_artificial_checkpoint(self):
        ml_server = resolve_ml_server_binary()
        if not ml_server.exists():
            self.skipTest("ml_server binary not found; build ml_server before running this test")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ckpt_dir = td_path / "checkpoints"
            runs_dir = td_path / "runs"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            # Artificial seed checkpoint to validate resume/load path as part of the smoke run.
            seed_model, seed_meta = create_model(model_family="legacy")
            seed_payload = build_checkpoint_payload(
                seed_model,
                metadata={
                    **seed_meta,
                    "schema_version": SUPPORTED_OBS_SCHEMA_VERSION,
                    "action_encoding_version": 1,
                    "action_feat_dim": ACTION_FEAT_DIM,
                },
                extra_metadata={"checkpoint_kind": "unit_test_seed"},
            )
            seed_ckpt = td_path / "seed.pt"
            torch.save(seed_payload, seed_ckpt)

            old_ml_server_bin = os.environ.get("ML_SERVER_BIN")
            os.environ["ML_SERVER_BIN"] = str(ml_server)
            try:
                train_online(
                    rounds=2,                 # tiny 2-episode smoke window
                    games_per_round=2,        # tiny simulation load
                    workers=1,                # deterministic, lightweight
                    mc_rollouts=0,            # disable branch rollouts for speed
                    train_batch=16,
                    lr=1e-4,
                    device="cpu",
                    checkpoint=str(seed_ckpt),
                    eval_every=999,           # skip expensive eval loop
                    ppo_epochs=1,
                    min_ppo_epochs=1,
                    max_ppo_epochs=1,
                    max_adv_calls_per_episode=0,
                    max_pass_adv_calls_per_episode=0,
                    max_info_adv_calls_per_episode=0,
                    named_checkpoint="unit_smoke_final.pt",
                    checkpoints_dir=ckpt_dir,
                    runs_dir=runs_dir,
                )
            finally:
                if old_ml_server_bin is None:
                    os.environ.pop("ML_SERVER_BIN", None)
                else:
                    os.environ["ML_SERVER_BIN"] = old_ml_server_bin

            latest_ckpt = ckpt_dir / "latest.pt"
            named_ckpt = ckpt_dir / "unit_smoke_final.pt"
            self.assertTrue(latest_ckpt.exists(), "latest checkpoint should be written")
            self.assertTrue(named_ckpt.exists(), "named checkpoint should be written")

            # Checkpoint should be parseable and carry expected metadata fields.
            _state, meta, _ = parse_checkpoint(latest_ckpt, map_location="cpu")
            self.assertEqual(meta.get("schema_version"), SUPPORTED_OBS_SCHEMA_VERSION)
            self.assertEqual(meta.get("action_feat_dim"), ACTION_FEAT_DIM)

            run_dirs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("online_")]
            self.assertGreaterEqual(len(run_dirs), 1, "at least one run directory should be created")
            log_path = run_dirs[0] / "log.jsonl"
            self.assertTrue(log_path.exists(), "run log should exist")
            with log_path.open("r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(any(e.get("event") == "update" for e in events), "training should emit update events")


if __name__ == "__main__":
    unittest.main()
