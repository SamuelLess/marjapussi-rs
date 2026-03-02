"""
Benchmark one or more checkpoints on the human NDJSON dataset.

Metrics are imitation-style (next-action prediction quality):
  - top1 / top3 accuracy
  - mean chosen-action probability
  - mean negative log-likelihood
  - subgroup top1 for bid / pass / play actions
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from model_factory import create_model, load_state_compatible, parse_checkpoint
from train.utils import Log
from train_from_dataset import NdJsonDataset, collate, worker_init


@dataclass
class Bucket:
    n: int = 0
    top1: int = 0

    def add(self, hits: torch.Tensor) -> None:
        if hits.numel() == 0:
            return
        self.n += int(hits.numel())
        self.top1 += int(hits.sum().item())

    def ratio(self) -> float:
        return float(self.top1) / float(self.n) if self.n else 0.0


@dataclass
class Metrics:
    n: int = 0
    top1: int = 0
    top3: int = 0
    sum_nll: float = 0.0
    sum_chosen_prob: float = 0.0
    bid: Bucket = field(default_factory=Bucket)
    ppass: Bucket = field(default_factory=Bucket)
    play: Bucket = field(default_factory=Bucket)

    def add_batch(
        self,
        pred: torch.Tensor,
        top3_idx: torch.Tensor,
        target: torch.Tensor,
        chosen_prob: torch.Tensor,
        chosen_feat: torch.Tensor,
    ) -> None:
        b = int(target.numel())
        if b == 0:
            return

        hit1 = pred.eq(target)
        hit3 = top3_idx.eq(target.unsqueeze(1)).any(dim=1)

        self.n += b
        self.top1 += int(hit1.sum().item())
        self.top3 += int(hit3.sum().item())
        self.sum_chosen_prob += float(chosen_prob.sum().item())
        self.sum_nll += float((-torch.log(chosen_prob.clamp_min(1e-12))).sum().item())

        is_bid = chosen_feat[:, 1] > 0.5
        is_pass = (chosen_feat[:, 3] > 0.5) | (chosen_feat[:, 11] > 0.5)
        is_play = chosen_feat[:, 0] > 0.5

        self.bid.add(hit1[is_bid].float())
        self.ppass.add(hit1[is_pass].float())
        self.play.add(hit1[is_play].float())

    def summary(self) -> dict[str, float | int]:
        if self.n == 0:
            return {
                "samples": 0,
                "top1": 0.0,
                "top3": 0.0,
                "chosen_prob": 0.0,
                "nll": math.inf,
                "bid_top1": 0.0,
                "pass_top1": 0.0,
                "play_top1": 0.0,
                "bid_n": 0,
                "pass_n": 0,
                "play_n": 0,
            }
        return {
            "samples": self.n,
            "top1": self.top1 / self.n,
            "top3": self.top3 / self.n,
            "chosen_prob": self.sum_chosen_prob / self.n,
            "nll": self.sum_nll / self.n,
            "bid_top1": self.bid.ratio(),
            "pass_top1": self.ppass.ratio(),
            "play_top1": self.play.ratio(),
            "bid_n": self.bid.n,
            "pass_n": self.ppass.n,
            "play_n": self.play.n,
        }


def resolve_model_spec(
    ckpt_path: Path,
    override_family: str | None,
    override_config: str | None,
) -> tuple[str, str | None]:
    _, ck_meta, _ = parse_checkpoint(ckpt_path, map_location="cpu")
    family = override_family or str(ck_meta.get("model_family", "parallel_v2"))
    config = override_config if override_config is not None else ck_meta.get("model_config_path")
    return family, config


def load_model_for_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    model_family: str | None,
    model_config: str | None,
    strict_param_budget: int,
) -> tuple[Any, dict[str, Any], int, int]:
    family, config = resolve_model_spec(ckpt_path, model_family, model_config)
    model, meta = create_model(
        model_family=family,
        model_config_path=config,
        strict_param_budget=strict_param_budget,
    )
    state_dict, _, _ = parse_checkpoint(ckpt_path, map_location="cpu")
    loaded, skipped = load_state_compatible(model, state_dict)
    model.to(device)
    model.eval()
    return model, meta, loaded, skipped


def benchmark_checkpoint(
    ckpt_path: Path,
    data_path: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    max_batches: int,
    model_family: str | None,
    model_config: str | None,
    strict_param_budget: int,
) -> dict[str, Any]:
    model, meta, loaded, skipped = load_model_for_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        model_family=model_family,
        model_config=model_config,
        strict_param_budget=strict_param_budget,
    )

    ds = NdJsonDataset(str(data_path), shuffle_buf=1, epochs=1)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
        worker_init_fn=worker_init if workers > 0 else None,
    )

    out = Metrics()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch is None:
                continue
            if 0 <= max_batches <= bi:
                break

            obs_a = {k: v.to(device, non_blocking=True) for k, v in batch["obs_a"].items()}
            tok = batch["token_ids"].to(device, non_blocking=True)
            tok_mask = batch["token_mask"].to(device, non_blocking=True)
            af = batch["action_feats"].to(device, non_blocking=True)
            am = batch["action_mask"].to(device, non_blocking=True)
            ai = batch["action_idx"].to(device, non_blocking=True)

            logits, _, _, _ = model(
                {
                    "obs_a": obs_a,
                    "token_ids": tok,
                    "token_mask": tok_mask,
                    "action_feats": af,
                    "action_mask": am,
                }
            )
            probs = F.softmax(logits, dim=-1)
            chosen_prob = probs.gather(1, ai.unsqueeze(1)).squeeze(1)
            pred = torch.argmax(logits, dim=-1)
            top3_idx = torch.topk(logits, k=min(3, logits.shape[1]), dim=-1).indices
            chosen_feat = af[torch.arange(ai.shape[0], device=device), ai]
            out.add_batch(pred, top3_idx, ai, chosen_prob, chosen_feat)

    summary = out.summary()
    summary.update(
        {
            "checkpoint": str(ckpt_path),
            "model_family": meta.get("model_family"),
            "model_params": int(meta.get("param_count", 0)),
            "loaded_tensors": int(loaded),
            "skipped_tensors": int(skipped),
        }
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark checkpoints on human NDJSON dataset")
    p.add_argument("--data", default="ml/data/human_dataset.ndjson")
    p.add_argument("--checkpoint", action="append", required=True, help="Path to checkpoint (.pt). Repeatable.")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=-1, help="Limit number of batches for quick smoke.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-family", default=None, help="Override model family (default: infer from checkpoint metadata)")
    p.add_argument("--model-config", default=None, help="Override model config path")
    p.add_argument("--strict-param-budget", type=int, default=28_000_000)
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    device = torch.device(args.device)
    Log.info(f"Benchmark device: {device}")
    rows: list[dict[str, Any]] = []
    for ck in args.checkpoint:
        ckpt_path = Path(ck)
        if not ckpt_path.exists():
            Log.warn(f"Skipping missing checkpoint: {ckpt_path}")
            continue
        Log.phase(f"Benchmark: {ckpt_path.name}")
        row = benchmark_checkpoint(
            ckpt_path=ckpt_path,
            data_path=data_path,
            batch_size=args.batch,
            workers=args.workers,
            device=device,
            max_batches=args.max_batches,
            model_family=args.model_family,
            model_config=args.model_config,
            strict_param_budget=args.strict_param_budget,
        )
        rows.append(row)
        Log.success(
            f"{ckpt_path.name} | top1={row['top1']:.3f} top3={row['top3']:.3f} "
            f"nll={row['nll']:.4f} chosen_p={row['chosen_prob']:.3f} "
            f"bid={row['bid_top1']:.3f} pass={row['pass_top1']:.3f} play={row['play_top1']:.3f}"
        )

    if not rows:
        Log.warn("No benchmark rows produced.")
        return

    print("\n=== BENCHMARK SUMMARY ===")
    print("checkpoint,top1,top3,nll,chosen_prob,bid_top1,pass_top1,play_top1,samples")
    for r in rows:
        print(
            f"{Path(str(r['checkpoint'])).name},"
            f"{r['top1']:.6f},{r['top3']:.6f},{r['nll']:.6f},{r['chosen_prob']:.6f},"
            f"{r['bid_top1']:.6f},{r['pass_top1']:.6f},{r['play_top1']:.6f},{r['samples']}"
        )


if __name__ == "__main__":
    main()
