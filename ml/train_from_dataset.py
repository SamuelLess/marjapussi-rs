"""
GPU-accelerated training from generated NDJSON dataset.

Features:
  - Streaming IterableDataset with shuffle buffer (constant RAM)
  - Multi-worker DataLoader + pin_memory for GPU throughput
  - torch.cuda.amp mixed precision (fp16 for GTX 1080 / RTX 4090)
  - MC advantage as primary loss signal when available, outcome as fallback
  - Cosine LR schedule

Usage:
  python ml/train_from_dataset.py --data ml/data/dataset.ndjson --device cuda --epochs 3
  python ml/train_from_dataset.py --data ml/data/dataset.ndjson --device cuda --batch 1024 --workers 4
"""

import argparse, json, math, random, sys, time
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from model import MarjapussiNet
from env import obs_to_tensors

CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────

class NdJsonDataset(IterableDataset):
    """
    Streams NDJSON records with a shuffle buffer.
    Multi-worker safe: each worker reads every Nth line starting at its id.
    """
    def __init__(self, path: str, shuffle_buf: int = 50_000, epochs: int = 1):
        self.path = path
        self.shuffle_buf = shuffle_buf
        self.epochs = epochs
        self.worker_id = 0
        self.num_workers = 1

    def __iter__(self):
        for _ in range(self.epochs):
            buf = []
            with open(self.path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    # Shard across workers
                    if i % self.num_workers != self.worker_id:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        buf.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(buf) >= self.shuffle_buf:
                        random.shuffle(buf)
                        yield from buf
                        buf.clear()
            if buf:
                random.shuffle(buf)
                yield from buf


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.worker_id = worker_id
    ds.num_workers = info.num_workers


# ── Collation ─────────────────────────────────────────────────────────────────

def collate(records: list[dict]):
    """Convert a batch of raw records to model tensors. Returns None on empty."""
    valid = []
    for r in records:
        try:
            t = obs_to_tensors(r["obs"])
            a = r.get("action_taken", 0)
            # Prefer MC advantage for chosen action; fall back to outcome pts
            if "chosen_advantage" in r:
                adv = float(r["chosen_advantage"])
            else:
                my_pts = r.get("outcome_pts_my_team", 60) / 120.0
                adv = my_pts * 2.0 - 1.0   # map [0,1] → [-1,1]
            pts_my  = r.get("outcome_pts_my_team", 60) / 120.0
            pts_opp = r.get("outcome_pts_opp", 60) / 120.0
            valid.append((t, a, adv, pts_my, pts_opp))
        except Exception:
            continue
    if not valid:
        return None

    tensors, actions, advs, pts_my, pts_opp = zip(*valid)
    B = len(tensors)

    max_seq = max(t["token_ids"].shape[1] for t in tensors)
    max_act = max(t["action_feats"].shape[1] for t in tensors)

    obs_a_keys = list(tensors[0]["obs_a"].keys())
    obs_a = {k: torch.cat([t["obs_a"][k] for t in tensors], 0) for k in obs_a_keys}

    tok = torch.zeros(B, max_seq, dtype=torch.long)
    tok_mask = torch.ones(B, max_seq, dtype=torch.bool)
    for i, t in enumerate(tensors):
        L = t["token_ids"].shape[1]
        tok[i, :L] = t["token_ids"][0]
        tok_mask[i, :L] = t["token_mask"][0]

    af = torch.zeros(B, max_act, 51)
    am = torch.ones(B, max_act, dtype=torch.bool)
    for i, t in enumerate(tensors):
        A = t["action_feats"].shape[1]
        af[i, :A] = t["action_feats"][0]
        am[i, :A] = t["action_mask"][0]

    ai = torch.tensor([min(a, tensors[i]["action_feats"].shape[1]-1) for i, a in enumerate(actions)], dtype=torch.long)
    return {
        "obs_a": obs_a, "token_ids": tok, "token_mask": tok_mask,
        "action_feats": af, "action_mask": am,
        "action_idx": ai,
        "advantage": torch.tensor(advs, dtype=torch.float32),
        "pts_target": torch.tensor([[pm, po] for pm, po in zip(pts_my, pts_opp)]),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path: str, epochs: int = 3, batch: int = 1024, lr: float = 3e-4,
          device: str = "cpu", ckpt: str | None = None, workers: int = 4,
          log_every: int = 500, amp: bool = True):

    model = MarjapussiNet().to(device)
    print(f"Model: {model.param_count():,} params  Device: {device}")
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)); print(f"Loaded: {ckpt}")

    use_amp = amp and device.startswith("cuda")
    scaler  = GradScaler(enabled=use_amp)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Count lines for scheduler estimate
    try:
        n_lines = sum(1 for _ in open(data_path))
        steps_per_epoch = n_lines // batch
        print(f"Dataset: ~{n_lines:,} records → ~{steps_per_epoch:,} steps/epoch")
    except:
        steps_per_epoch = 100_000

    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    for epoch in range(epochs):
        print(f"\n── Epoch {epoch+1}/{epochs} ──────────────────────────")
        ds = NdJsonDataset(data_path, shuffle_buf=min(100_000, batch * 200), epochs=1)
        loader = DataLoader(ds, batch_size=batch, collate_fn=collate,
                            num_workers=workers, worker_init_fn=worker_init,
                            pin_memory=(device != "cpu"),
                            prefetch_factor=2 if workers > 0 else None,
                            persistent_workers=(workers > 0))
        model.train()
        t0 = time.time(); step = 0; sum_loss = sum_bc = sum_pts = 0.0

        for batch_data in loader:
            if batch_data is None: continue

            # Move to GPU (non-blocking for pinned memory)
            def to(x):
                return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) \
                    else {k: v.to(device, non_blocking=True) for k, v in x.items()}

            obs_a   = to(batch_data["obs_a"])
            tok     = batch_data["token_ids"].to(device, non_blocking=True)
            tok_mask= batch_data["token_mask"].to(device, non_blocking=True)
            af      = batch_data["action_feats"].to(device, non_blocking=True)
            am      = batch_data["action_mask"].to(device, non_blocking=True)
            ai      = batch_data["action_idx"].to(device, non_blocking=True)
            adv     = batch_data["advantage"].to(device, non_blocking=True)
            pts_tgt = batch_data["pts_target"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits, _, pts_pred, _ = model({"obs_a": obs_a, "token_ids": tok,
                    "token_mask": tok_mask, "action_feats": af, "action_mask": am})

                # Behavior cloning (cross-entropy) weighted by advantage
                log_p   = F.log_softmax(logits, dim=-1)
                chosen_lp = log_p.gather(1, ai.unsqueeze(1)).squeeze(1)
                # Advantage-weighted BC: good moves get stronger gradient
                weights = F.relu(adv) + 0.1   # always ≥ 0.1 to learn from all moves
                bc_loss = -(chosen_lp * weights).mean()

                # Points regression (auxiliary)
                pts_loss = F.mse_loss(pts_pred, pts_tgt) * 0.3

                loss = bc_loss + pts_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()

            sum_loss += loss.item(); sum_bc += bc_loss.item(); sum_pts += pts_loss.item()
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                rate = step * batch / elapsed
                print(f"  [{step:6,}] loss={sum_loss/step:.4f}  bc={sum_bc/step:.4f}"
                      f"  pts={sum_pts/step:.4f}  {rate:,.0f} samples/s"
                      f"  lr={sched.get_last_lr()[0]:.2e}")

        ckpt_path = CKPT_DIR / f"epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        torch.save(model.state_dict(), CKPT_DIR / "latest.pt")
        print(f"  → Saved {ckpt_path}")

    print("\nTraining done. Checkpoint: ml/checkpoints/latest.pt")
    print("Next: python ml/train.py --checkpoint ml/checkpoints/latest.pt  (self-play fine-tune)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True,    help="Path to NDJSON dataset")
    p.add_argument("--epochs",   type=int,   default=3)
    p.add_argument("--batch",    type=int,   default=1024,    help="Batch size (1024 for GTX 1080, 4096 for RTX 4090)")
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers",  type=int,   default=4,       help="DataLoader workers")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--log-every", type=int,  default=500)
    p.add_argument("--no-amp",   action="store_true",         help="Disable mixed precision")
    args = p.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3} GB")

    train(data_path=args.data, epochs=args.epochs, batch=args.batch,
          lr=args.lr, device=args.device, ckpt=args.checkpoint,
          workers=args.workers, log_every=args.log_every, amp=not args.no_amp)
