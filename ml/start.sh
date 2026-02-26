#!/usr/bin/env bash
# Quick-start script for the Marjapussi AI pipeline.
# Run from the repo root: bash ml/start.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ML="$ROOT/ml"
DEFAULT_PYTHON="${PYTHON:-python}"

if [ ! -d "$ROOT/.venv" ]; then
  echo "=== 0. Creating virtual environment (.venv) ==="
  "$DEFAULT_PYTHON" -m venv "$ROOT/.venv"
fi

if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  VPY="$ROOT/.venv/Scripts/python.exe"
else
  VPY="$ROOT/.venv/bin/python"
fi

echo "=== 1. Building optimized Rust ml_server ==="
cd "$ROOT"
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server

if [ -x "$ROOT/target/release/ml_server.exe" ]; then
  export ML_SERVER_BIN="$ROOT/target/release/ml_server.exe"
else
  export ML_SERVER_BIN="$ROOT/target/release/ml_server"
fi

echo ""
echo "=== 2. Installing Python dependencies ==="
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -q -e "$ROOT[dev]"
"$VPY" "$ML/install_torch.py"

echo ""
echo "=== 3. Model smoke test ==="
"$VPY" "$ML/model.py"

echo ""
echo "=== 4. (Optional) Start training in background ==="
echo "    Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \"$VPY\" ml/train_online.py --device cuda"

echo ""
echo "=== 5. Starting UI server ==="
echo "    Open: http://localhost:8765"
"$VPY" "$ML/ui_server.py"
