#!/usr/bin/env bash
# Quick-start script for the Marjapussi AI pipeline.
# Run from the repo root: bash ml/start.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ML="$ROOT/ml"

echo "=== 1. Building Rust ml_server ==="
cd "$ROOT"
cargo build --bin ml_server

echo ""
echo "=== 2. Checking Python dependencies ==="
pip install -q -r "$ML/requirements.txt"

echo ""
echo "=== 3. Model smoke test ==="
python "$ML/model.py" --smoke-test

echo ""
echo "=== 4. (Optional) Start training in background ==="
echo "    Run: python ml/train.py --phase 1 --games 500"

echo ""
echo "=== 5. Starting UI server ==="
echo "    Open: http://localhost:8765"
python "$ML/ui_server.py"
