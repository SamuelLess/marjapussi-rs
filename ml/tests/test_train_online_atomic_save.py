import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_online import atomic_torch_save


class TrainOnlineAtomicSaveTest(unittest.TestCase):
    def test_atomic_torch_save_writes_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "latest.pt"
            atomic_torch_save({"w": torch.tensor([1.0, 2.0])}, dst)
            self.assertTrue(dst.exists())
            ckpt = torch.load(dst, map_location="cpu")
            self.assertEqual(ckpt["w"].shape[0], 2)

            # Overwrite should stay loadable.
            atomic_torch_save({"w": torch.tensor([3.0])}, dst)
            ckpt2 = torch.load(dst, map_location="cpu")
            self.assertEqual(float(ckpt2["w"][0]), 3.0)


if __name__ == "__main__":
    unittest.main()
