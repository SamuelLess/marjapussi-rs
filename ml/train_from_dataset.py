"""
GPU-accelerated training from generated NDJSON dataset.

Features:
  - Streaming IterableDataset with shuffle buffer (constant RAM)
  - Multi-worker DataLoader + pin_memory for GPU throughput
  - torch.amp mixed precision (fp16/bf16 depending on hardware)
  - MC advantage as primary loss signal when available, outcome as fallback
  - Configurable LR schedule (plateau/cosine/constant) with warmup + LR floor

Usage:
  python ml/train_from_dataset.py --data ml/data/dataset.ndjson --device cuda --epochs 3
  python ml/train_from_dataset.py --data ml/data/dataset.ndjson --device cuda --batch 1024 --workers 4
"""

from __future__ import annotations

import argparse, json, math, os, random, sys, time
from functools import partial
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from model import ACTION_FEAT_DIM
from env import SUPPORTED_OBS_SCHEMA_VERSION, obs_to_tensors
from model_factory import (
    build_checkpoint_payload,
    create_model,
    load_state_compatible,
    parse_checkpoint,
)
from train.utils import Log

DEFAULT_CKPT_DIR = Path(__file__).parent / "checkpoints"


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def configure_torch_runtime(device: str, workers: int) -> None:
    """Apply safe runtime settings for better training throughput."""
    torch.set_float32_matmul_precision("high")

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    cpu_count = os.cpu_count() or 4
    if workers > 0:
        torch.set_num_threads(max(1, min(8, cpu_count // max(1, workers))))

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

def collate(records: list[dict], bid_weight: float = 1.0, pass_weight: float = 1.0):
    """Convert a batch of raw records to model tensors. Returns None on empty."""
    valid = []
    for r in records:
        try:
            t = obs_to_tensors(r["obs"])
            a = r.get("action_taken", 0)
            legal = r.get("obs", {}).get("legal_actions", [])
            clamped_a = min(a, len(legal) - 1) if legal else 0
            # Prefer MC advantage for chosen action; fall back to outcome pts
            if "chosen_advantage" in r:
                adv = float(r["chosen_advantage"])
            else:
                my_pts = r.get("outcome_pts_my_team", 210) / 420.0
                adv = my_pts * 2.0 - 1.0   # map [0,1] to [-1,1]
            pts_my = r.get("outcome_pts_my_team", 210) / 420.0
            pts_opp = r.get("outcome_pts_opp", 210) / 420.0

            # Extra supervised emphasis on bidding and pass-pick actions.
            w = 1.0
            feats = t["action_feats"]
            if feats.shape[1] > 0:
                idx = max(0, min(clamped_a, int(feats.shape[1]) - 1))
                chosen = feats[0, idx]
                is_bid = float(chosen[1].item()) > 0.5
                is_pass = (float(chosen[3].item()) > 0.5) or (float(chosen[11].item()) > 0.5)
                if is_bid:
                    w *= max(1.0, float(bid_weight))
                if is_pass:
                    w *= max(1.0, float(pass_weight))
            valid.append((t, clamped_a, adv, pts_my, pts_opp, w))
        except Exception:
            continue
    if not valid:
        return None

    tensors, actions, advs, pts_my, pts_opp, phase_boost = zip(*valid)
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

    af = torch.zeros(B, max_act, ACTION_FEAT_DIM)
    am = torch.ones(B, max_act, dtype=torch.bool)
    for i, t in enumerate(tensors):
        A = t["action_feats"].shape[1]
        af[i, :A] = t["action_feats"][0]
        am[i, :A] = t["action_mask"][0]

    ai = torch.tensor(
        [min(a, tensors[i]["action_feats"].shape[1] - 1) for i, a in enumerate(actions)],
        dtype=torch.long,
    )
    return {
        "obs_a": obs_a,
        "token_ids": tok,
        "token_mask": tok_mask,
        "action_feats": af,
        "action_mask": am,
        "action_idx": ai,
        "advantage": torch.tensor(advs, dtype=torch.float32),
        "pts_target": torch.tensor([[pm, po] for pm, po in zip(pts_my, pts_opp)]),
        "phase_boost": torch.tensor(phase_boost, dtype=torch.float32),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path: str, epochs: int = 3, batch: int = 1024, lr: float = 3e-4,
          device: str = "cpu", ckpt: str | None = None, workers: int = 4,
          log_every: int = 500, amp: bool = True, max_steps: int = 0,
          bid_weight: float = 1.0, pass_weight: float = 1.0,
          model_family: str = "legacy", model_config: str | None = None,
          strict_param_budget: int | None = None,
          lr_schedule: str = "plateau", lr_min: float = 1e-5,
          lr_warmup_steps: int = 128,
          lr_plateau_patience: int = 1, lr_plateau_factor: float = 0.5,
          lr_plateau_threshold: float = 1e-3, lr_plateau_cooldown: int = 0,
          checkpoints_dir: str | Path = DEFAULT_CKPT_DIR):
    configure_torch_runtime(device=device, workers=workers)
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, model_meta = create_model(
        model_family=model_family,
        model_config_path=model_config,
        strict_param_budget=strict_param_budget,
    )
    model = model.to(device)
    Log.success(
        f"Supervised pretraining | epochs={epochs} | batch={batch} | workers={workers} | device={device} | family={model_meta.get('model_family')}"
    )
    Log.info(
        f"Phase/action emphasis: bid_weight={bid_weight:.2f}, pass_weight={pass_weight:.2f}"
    )
    Log.info(f"Model params: {model.param_count():,}")
    if ckpt and Path(ckpt).exists():
        state_dict, ckpt_meta, _ = parse_checkpoint(ckpt, map_location=device)
        loaded, skipped = load_state_compatible(model, state_dict)
        family_note = ckpt_meta.get("model_family")
        if family_note and family_note != model_meta.get("model_family"):
            Log.warn(
                f"Checkpoint family mismatch: ckpt={family_note}, requested={model_meta.get('model_family')}. "
                f"Loaded compatible tensors only ({loaded} loaded, {skipped} skipped)."
            )
        else:
            Log.info(f"Loaded checkpoint: {ckpt} ({loaded} tensors, {skipped} skipped)")

    use_amp = amp and device.startswith("cuda")
    scaler  = GradScaler("cuda", enabled=use_amp)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Count lines for scheduler estimate
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        steps_per_epoch = max(1, n_lines // max(1, batch))
        Log.info(f"Dataset: ~{n_lines:,} records -> ~{steps_per_epoch:,} steps/epoch")
    except Exception:
        steps_per_epoch = max(1, 100_000 // max(1, batch))
        Log.warn("Could not count dataset lines; using fallback scheduler horizon.")

    lr_schedule = str(lr_schedule).lower()
    if lr_schedule not in ("plateau", "cosine", "constant"):
        raise ValueError(f"Unsupported lr_schedule={lr_schedule!r} (expected plateau|cosine|constant)")

    cosine_steps = max(1, (epochs * steps_per_epoch) - max(0, int(lr_warmup_steps)))
    sched = None
    if lr_schedule == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=cosine_steps,
            eta_min=lr_min,
        )
    elif lr_schedule == "plateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            threshold=lr_plateau_threshold,
            cooldown=lr_plateau_cooldown,
            min_lr=lr_min,
        )

    Log.info(
        f"LR control: schedule={lr_schedule}, base_lr={lr:.2e}, min_lr={lr_min:.2e}, "
        f"warmup_steps={max(0, int(lr_warmup_steps))}"
    )
    progress_interval = max(1, min(log_every, max(1, steps_per_epoch // 8)))

    global_step = 0
    for epoch in range(epochs):
        Log.phase(f"Epoch {epoch+1}/{epochs}: Pretraining")
        ds = NdJsonDataset(data_path, shuffle_buf=min(100_000, batch * 200), epochs=1)
        collate_fn = partial(collate, bid_weight=bid_weight, pass_weight=pass_weight)
        loader = DataLoader(ds, batch_size=batch, collate_fn=collate_fn,
                            num_workers=workers, worker_init_fn=worker_init,
                            pin_memory=(device != "cpu"),
                            prefetch_factor=2 if workers > 0 else None,
                            persistent_workers=(workers > 0))
        model.train()
        t0 = time.time(); step = 0; sum_loss = sum_bc = sum_pts = 0.0

        stop_early = False
        for batch_data in loader:
            if batch_data is None: continue

            # Optional linear warmup to avoid unstable first updates.
            if lr_warmup_steps > 0 and global_step < lr_warmup_steps:
                warm_t = float(global_step + 1) / float(max(1, lr_warmup_steps))
                warm_lr = lr_min + warm_t * (lr - lr_min)
                for g in opt.param_groups:
                    g["lr"] = float(warm_lr)
            elif lr_warmup_steps > 0 and global_step == lr_warmup_steps and lr_schedule in ("plateau", "constant"):
                # Plateau/constant schedules should continue from base LR after warmup.
                for g in opt.param_groups:
                    g["lr"] = float(lr)

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

            with autocast("cuda", enabled=use_amp):
                logits, _, pts_pred, _ = model({"obs_a": obs_a, "token_ids": tok,
                    "token_mask": tok_mask, "action_feats": af, "action_mask": am})

                # Behavior cloning (cross-entropy) weighted by advantage
                log_p   = F.log_softmax(logits, dim=-1)
                chosen_lp = log_p.gather(1, ai.unsqueeze(1)).squeeze(1)
                # Advantage-weighted BC: good moves get stronger gradient
                weights = F.relu(adv) + 0.1   # always >= 0.1 to learn from all moves
                phase_boost = batch_data.get("phase_boost")
                if phase_boost is not None:
                    weights = weights * phase_boost.to(device, non_blocking=True)
                bc_loss = -(chosen_lp * weights).mean()

                # Points regression (auxiliary)
                pts_loss = F.mse_loss(pts_pred, pts_tgt) * 0.3

                loss = bc_loss + pts_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if sched is not None and lr_schedule == "cosine" and global_step >= lr_warmup_steps:
                sched.step()
            global_step += 1

            sum_loss += loss.item(); sum_bc += bc_loss.item(); sum_pts += pts_loss.item()
            step += 1

            if step % progress_interval == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / max(elapsed, 1e-6)
                sample_rate = steps_per_sec * batch
                eta = (max(0, steps_per_epoch - step)) / max(steps_per_sec, 1e-6)
                Log.opt(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Step {step}/{steps_per_epoch} | "
                    f"Loss: {sum_loss/step:.4f} | "
                    f"BC: {sum_bc/step:.4f} | "
                    f"Pts: {sum_pts/step:.4f} | "
                    f"{sample_rate:,.0f} samples/s | "
                    f"ETA: {_format_eta(eta)} | "
                    f"LR: {opt.param_groups[0]['lr']:.2e}",
                    end=""
                )

            if max_steps > 0 and global_step >= max_steps:
                Log.warn(f"Reached max_steps={max_steps}, stopping pretraining early.")
                stop_early = True
                break

        if step > 0:
            print()
        epoch_secs = time.time() - t0
        avg_loss = sum_loss / max(1, step)
        avg_bc = sum_bc / max(1, step)
        avg_pts = sum_pts / max(1, step)
        payload = build_checkpoint_payload(
            model,
            metadata={
                **model_meta,
                "schema_version": SUPPORTED_OBS_SCHEMA_VERSION,
                "action_encoding_version": 1,
                "action_feat_dim": ACTION_FEAT_DIM,
            },
            extra_metadata={
                "checkpoint_kind": "pretrain",
                "epoch": epoch + 1,
                "global_step": global_step,
            },
        )
        if sched is not None and lr_schedule == "plateau" and step > 0:
            lr_before = float(opt.param_groups[0]["lr"])
            sched.step(avg_loss)
            lr_after = float(opt.param_groups[0]["lr"])
            if lr_after < lr_before:
                Log.info(
                    f"LR reduced on plateau: {lr_before:.2e} -> {lr_after:.2e} "
                    f"(epoch_loss={avg_loss:.4f})"
                )
        ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
        torch.save(payload, ckpt_path)
        torch.save(payload, ckpt_dir / "latest.pt")
        Log.success(f"Epoch {epoch+1} Summary:")
        print(f"  - Steps:    {step}")
        print(f"  - Losses:   Total: {avg_loss:.4f} | BC: {avg_bc:.4f} | Pts: {avg_pts:.4f}")
        print(f"  - Time:     {epoch_secs:.1f}s")
        print(f"  - Saved:    {ckpt_path}")
        if stop_early:
            break

    Log.success("Pretraining complete.")
    Log.info(f"Checkpoint: {ckpt_dir / 'latest.pt'}")
    Log.info(f"Next: python ml/train.py --checkpoint {ckpt_dir / 'latest.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True,    help="Path to NDJSON dataset")
    p.add_argument("--epochs",   type=int,   default=3)
    p.add_argument("--batch",    type=int,   default=1024,    help="Batch size (1024 for GTX 1080, 4096 for RTX 4090)")
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers",  type=int,   default=4,       help="DataLoader workers")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model-family", choices=["legacy", "parallel_v2"], default="legacy",
                   help="Model family to train")
    p.add_argument("--model-config", default=None,
                   help="Optional model config path (used by parallel_v2)")
    p.add_argument("--strict-param-budget", type=int, default=None,
                   help="Optional strict max-parameter gate")
    p.add_argument("--log-every", type=int,  default=500)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Optional hard cap on optimizer steps (0 = disabled)")
    p.add_argument("--bid-weight", type=float, default=1.0,
                   help="Extra supervised emphasis multiplier for bidding actions")
    p.add_argument("--pass-weight", type=float, default=1.0,
                   help="Extra supervised emphasis multiplier for passing/pick-pass-card actions")
    p.add_argument("--lr-schedule", choices=["plateau", "cosine", "constant"], default="plateau",
                   help="Learning-rate control strategy")
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="Lower LR floor for schedule and warmup")
    p.add_argument("--lr-warmup-steps", type=int, default=128,
                   help="Linear warmup steps before main LR schedule")
    p.add_argument("--lr-plateau-patience", type=int, default=1,
                   help="Epochs with no meaningful loss improvement before LR decay")
    p.add_argument("--lr-plateau-factor", type=float, default=0.5,
                   help="LR multiplier applied on plateau (e.g. 0.5 halves LR)")
    p.add_argument("--lr-plateau-threshold", type=float, default=1e-3,
                   help="Minimum relative improvement to avoid plateau trigger")
    p.add_argument("--lr-plateau-cooldown", type=int, default=0,
                   help="Cooldown epochs after each plateau LR reduction")
    p.add_argument("--no-amp",   action="store_true",         help="Disable mixed precision")
    p.add_argument("--checkpoints-dir", default=str(DEFAULT_CKPT_DIR),
                   help="Directory for pretraining checkpoints")
    args = p.parse_args()

    Log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        Log.info(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3} GB"
        )

    train(data_path=args.data, epochs=args.epochs, batch=args.batch,
          lr=args.lr, device=args.device, ckpt=args.checkpoint,
          model_family=args.model_family, model_config=args.model_config,
          strict_param_budget=args.strict_param_budget,
          workers=args.workers, log_every=args.log_every,
          amp=not args.no_amp, max_steps=args.max_steps,
          bid_weight=args.bid_weight, pass_weight=args.pass_weight,
          lr_schedule=args.lr_schedule, lr_min=args.lr_min,
          lr_warmup_steps=args.lr_warmup_steps,
          lr_plateau_patience=args.lr_plateau_patience,
          lr_plateau_factor=args.lr_plateau_factor,
          lr_plateau_threshold=args.lr_plateau_threshold,
          lr_plateau_cooldown=args.lr_plateau_cooldown,
          checkpoints_dir=args.checkpoints_dir)
