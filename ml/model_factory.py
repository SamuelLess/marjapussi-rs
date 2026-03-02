from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

try:
    from model import MarjapussiNet
    from model_parallel import MarjapussiParallelNet, ParallelModelConfig
except ModuleNotFoundError:
    from .model import MarjapussiNet
    from .model_parallel import MarjapussiParallelNet, ParallelModelConfig

ML_DIR = Path(__file__).parent
DEFAULT_PARALLEL_CONFIG = ML_DIR / "config" / "model_parallel_v2.toml"
MODEL_FAMILIES = {"legacy", "parallel_v2"}


def _stable_hash_dict(data: dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_parallel_config(config_path: str | Path | None = None) -> tuple[ParallelModelConfig, str, str]:
    path = Path(config_path) if config_path else DEFAULT_PARALLEL_CONFIG
    cfg_dict: dict[str, Any] = {}
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid model config format: {path}")
        cfg_dict = raw

    cfg = ParallelModelConfig(**cfg_dict)
    cfg_hash = _stable_hash_dict(asdict(cfg))
    return cfg, str(path), cfg_hash


def create_model(
    model_family: str = "parallel_v2",
    model_config_path: str | Path | None = None,
    strict_param_budget: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    family = (model_family or "parallel_v2").strip().lower()
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_family={model_family!r}. Expected one of {sorted(MODEL_FAMILIES)}")

    if family == "legacy":
        model = MarjapussiNet()
        meta = {
            "model_family": "legacy",
            "model_config_path": None,
            "model_config_hash": None,
            "param_count": int(model.param_count()),
        }
        if strict_param_budget is not None and model.param_count() > int(strict_param_budget):
            raise ValueError(
                f"Model exceeds strict_param_budget: {model.param_count()} > {int(strict_param_budget)}"
            )
        return model, meta

    cfg, resolved_cfg_path, cfg_hash = load_parallel_config(model_config_path)
    model = MarjapussiParallelNet(cfg)
    param_count = int(model.param_count())
    budget = int(strict_param_budget if strict_param_budget is not None else cfg.max_params)
    if param_count > budget:
        raise ValueError(
            f"parallel_v2 param budget exceeded: {param_count} > {budget}. "
            f"Adjust model_parallel_v2 config."
        )

    meta = {
        "model_family": "parallel_v2",
        "model_config_path": resolved_cfg_path,
        "model_config_hash": cfg_hash,
        "param_count": param_count,
    }
    return model, meta


def build_checkpoint_payload(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    md = dict(metadata or {})
    if extra_metadata:
        md.update(extra_metadata)
    return {"state_dict": model.state_dict(), "metadata": md}


def parse_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    ckpt_path = Path(path)
    raw = torch.load(str(ckpt_path), map_location=map_location)
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        state_dict = raw["state_dict"]
        metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), dict) else {}
        return state_dict, metadata, raw
    if isinstance(raw, dict):
        # legacy raw state dict
        return raw, {}, raw
    raise ValueError(f"Unsupported checkpoint payload at {ckpt_path}")


def load_state_compatible(model: torch.nn.Module, state_dict: dict[str, Any]) -> tuple[int, int]:
    model_state = model.state_dict()
    compatible: dict[str, Any] = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped += 1
    if not compatible:
        raise ValueError("No compatible tensors found for model architecture.")
    model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped
