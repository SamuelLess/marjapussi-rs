"""
Dataset loader for Rust-generated training data.
Reads NDJSON files produced by ml_generate --output dataset.ndjson

Usage:
    python ml/train_from_dataset.py --data ml/data/dataset.ndjson --epochs 3

This is the high-throughput pre-training path:
  1. Run ml_generate to produce millions of (obs, action, outcome) records at 500-2000 games/sec
  2. Train the model on this dataset for behavior cloning + points regression
  3. Run self-play fine-tuning with train.py --games 50000

Throughput for 10M games:
  - Rust generator (8 threads): ~1000-2000 games/sec → 10M games in 1.5-3 hours
  - ~50 decision points per game → ~500M transitions
  - At batch_size=512, ~1M training steps → a few hours per epoch on CPU, ~30min on GPU
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from model import MarjapussiNet
from env import obs_to_tensors, encode_legal_actions

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ── Dataset reader ────────────────────────────────────────────────────────────

def stream_records(path: str, shuffle_buffer: int = 50_000) -> Iterator[dict]:
    """Stream NDJSON records with a shuffle buffer for SGD randomness."""
    buf = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                buf.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(buf) >= shuffle_buffer:
                random.shuffle(buf)
                yield from buf
                buf.clear()
    if buf:
        random.shuffle(buf)
        yield from buf


def record_to_tensors(record: dict):
    """Convert one NDJSON record to (model_input, action_idx, pts_target)."""
    obs = record['obs']
    action_taken = record['action_taken']
    pts_my = record.get('outcome_pts_my_team', 0) / 120.0
    pts_opp = record.get('outcome_pts_opp', 0) / 120.0

    tensors = obs_to_tensors(obs)
    return tensors, action_taken, pts_my, pts_opp


def collate_batch(records: list[dict]):
    """Build a mini-batch from a list of records."""
    valid = []
    for r in records:
        try:
            t, a, pm, po = record_to_tensors(r)
            valid.append((t, a, pm, po))
        except Exception:
            continue
    if not valid:
        return None

    tensors_list, actions, pts_my, pts_opp = zip(*valid)
    B = len(tensors_list)

    # Find max lengths
    max_seq = max(t['token_ids'].shape[1] for t in tensors_list)
    max_act = max(t['action_feats'].shape[1] for t in tensors_list)

    obs_a_keys = list(tensors_list[0]['obs_a'].keys())
    stacked_obs_a = {}
    for k in obs_a_keys:
        stacked_obs_a[k] = torch.cat([t['obs_a'][k] for t in tensors_list], dim=0)

    padded_tokens = torch.zeros(B, max_seq, dtype=torch.long)
    padded_mask = torch.ones(B, max_seq, dtype=torch.bool)
    for i, t in enumerate(tensors_list):
        L = t['token_ids'].shape[1]
        padded_tokens[i, :L] = t['token_ids'][0]
        padded_mask[i, :L] = t['token_mask'][0]

    padded_actions = torch.zeros(B, max_act, 51)
    padded_action_mask = torch.ones(B, max_act, dtype=torch.bool)
    for i, t in enumerate(tensors_list):
        A = t['action_feats'].shape[1]
        padded_actions[i, :A] = t['action_feats'][0]
        padded_action_mask[i, :A] = t['action_mask'][0]

    # Clamp action indices to legal range
    action_indices = torch.tensor([
        min(a, tensors_list[i]['action_feats'].shape[1] - 1)
        for i, a in enumerate(actions)
    ], dtype=torch.long)

    pts_target = torch.tensor([[pm, po] for pm, po in zip(pts_my, pts_opp)])

    return {
        'obs_a': stacked_obs_a,
        'token_ids': padded_tokens,
        'token_mask': padded_mask,
        'action_feats': padded_actions,
        'action_mask': padded_action_mask,
        'action_indices': action_indices,
        'pts_target': pts_target,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_from_dataset(
    data_path: str,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = 'cpu',
    checkpoint: str | None = None,
    log_every: int = 1000,
):
    model = MarjapussiNet().to(device)
    print(f"Model parameters: {model.param_count():,}")

    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 100_000)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        model.train()

        buf: list[dict] = []
        step = 0
        total_loss = 0.0
        total_bc_loss = 0.0
        t_start = time.time()

        for record in stream_records(data_path):
            buf.append(record)
            if len(buf) < batch_size:
                continue

            batch = collate_batch(buf)
            buf.clear()

            if batch is None:
                continue

            # Move to device
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else
                     {kk: vv.to(device) for kk, vv in v.items()})
                 for k, v in batch.items()
                 if k not in ('action_indices', 'pts_target')}
            action_indices = batch['action_indices'].to(device)
            pts_target = batch['pts_target'].to(device)

            optimizer.zero_grad()
            logits, card_logits, point_preds = model({
                'obs_a': b['obs_a'],
                'token_ids': b['token_ids'],
                'token_mask': b['token_mask'],
                'action_feats': b['action_feats'],
                'action_mask': b['action_mask'],
            })

            # Behavior cloning loss: cross-entropy on chosen action
            bc_loss = F.cross_entropy(logits, action_indices)

            # Points regression loss
            pts_loss = F.mse_loss(point_preds, pts_target) * 0.5

            loss = bc_loss + pts_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_bc_loss += bc_loss.item()
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t_start
                rate = step * batch_size / elapsed
                avg_loss = total_loss / step
                avg_bc = total_bc_loss / step
                print(f"  Step {step:,} | Loss: {avg_loss:.4f} | BC: {avg_bc:.4f} | "
                      f"{rate:,.0f} samples/sec")

        # Save checkpoint after each epoch
        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")
        print(f"  → Saved checkpoint: {ckpt_path}")

    print("\nPre-training from dataset complete.")
    print(f"Run self-play fine-tuning with: python ml/train.py --checkpoint ml/checkpoints/latest.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Marjapussi model from generated dataset')
    parser.add_argument('--data', required=True, help='Path to NDJSON dataset file')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--log-every', type=int, default=1000)
    args = parser.parse_args()

    train_from_dataset(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint=args.checkpoint,
        log_every=args.log_every,
    )
