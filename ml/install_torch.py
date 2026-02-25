#!/usr/bin/env python3
"""
Install a suitable PyTorch build for the current Python runtime.

Behavior:
- If CUDA GPU is detected, prefer CUDA wheels from PyTorch index.
- Works with current interpreter (no Python version hardcoding).
- Falls back to CPU wheel only when CUDA wheel install/validation fails.

Env overrides:
- TORCH_CHANNELS="cu128,cu126,cu124" to control candidate CUDA channels.
- FORCE_CPU_TORCH=1 to force CPU wheel.
- FORCE_CUDA_TORCH=1 to fail instead of CPU fallback.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import json
from typing import Iterable


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, text=True, check=check)


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def detect_cuda_version_from_nvidia_smi() -> str | None:
    proc = run_capture(["nvidia-smi"])
    if proc.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", proc.stdout)
    return match.group(1) if match else None


def default_channels(cuda_version: str | None) -> list[str]:
    if cuda_version is None:
        return ["cu128", "cu126", "cu124"]
    try:
        major, minor = cuda_version.split(".")
        key = int(major) * 10 + int(minor)
    except Exception:
        return ["cu128", "cu126", "cu124"]

    if key >= 128:
        return ["cu128", "cu126", "cu124"]
    if key >= 126:
        return ["cu126", "cu124", "cu121"]
    if key >= 124:
        return ["cu124", "cu121"]
    return ["cu121"]


def requested_channels(cuda_version: str | None) -> list[str]:
    override = os.getenv("TORCH_CHANNELS", "").strip()
    if override:
        chans = [c.strip() for c in override.split(",") if c.strip()]
        if chans:
            return chans
    return default_channels(cuda_version)


def installed_torch_info() -> tuple[str | None, str | None, bool]:
    code = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda, 'ok': torch.cuda.is_available()}))\n"
        "except Exception:\n"
        "    print(json.dumps({'version': None, 'cuda': None, 'ok': False}))\n"
    )
    proc = run_capture([sys.executable, "-c", code])
    if proc.returncode != 0:
        return (None, None, False)
    text = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not text:
        return (None, None, False)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return (None, None, False)
    return (obj.get("version"), obj.get("cuda"), bool(obj.get("ok")))


def pip_install(args: Iterable[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *args])


def ensure_torch() -> int:
    force_cpu = os.getenv("FORCE_CPU_TORCH", "").strip() == "1"
    force_cuda = os.getenv("FORCE_CUDA_TORCH", "").strip() == "1"
    cuda_version = detect_cuda_version_from_nvidia_smi()
    has_gpu = cuda_version is not None
    channels = requested_channels(cuda_version)

    ver, cuda_build, cuda_ok = installed_torch_info()
    print(
        f"Current torch: version={ver}, build_cuda={cuda_build}, cuda_available={cuda_ok}, "
        f"gpu_detected={has_gpu}, gpu_cuda={cuda_version}"
    )

    if force_cpu:
        pip_install(["--upgrade", "torch"])
        return 0

    if has_gpu and cuda_ok and cuda_build:
        print("Torch CUDA build already working; nothing to do.")
        return 0

    if has_gpu:
        for ch in channels:
            index = f"https://download.pytorch.org/whl/{ch}"
            print(f"Trying CUDA channel: {ch}")
            proc = run_capture(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--index-url", index, "torch"]
            )
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr)
                continue

            ver, cuda_build, cuda_ok = installed_torch_info()
            print(
                f"Post-install torch: version={ver}, build_cuda={cuda_build}, cuda_available={cuda_ok}"
            )
            if cuda_ok and cuda_build:
                print(f"Using CUDA torch from channel {ch}.")
                return 0

        if force_cuda:
            print("ERROR: CUDA torch was required but could not be installed/validated.")
            return 1

    print("Falling back to CPU torch.")
    pip_install(["--upgrade", "torch"])
    ver, cuda_build, cuda_ok = installed_torch_info()
    print(
        f"Final torch: version={ver}, build_cuda={cuda_build}, cuda_available={cuda_ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(ensure_torch())
