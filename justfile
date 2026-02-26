# Default list target
default:
    @just --list

os := os()
python := if os == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }
ml_server_bin := if os == "windows" { "target/release/ml_server.exe" } else { "target/release/ml_server" }
ml_convert_bin := if os == "windows" { "target/release/ml_convert_legacy.exe" } else { "target/release/ml_convert_legacy" }

# Ensure virtual environment exists.
ensure-venv:
    @if [ ! -f ".venv/Scripts/python.exe" ] && [ ! -f ".venv/bin/python" ]; then \
        echo "Creating virtual environment at .venv"; \
        if command -v python >/dev/null 2>&1; then \
            python -m venv .venv; \
        elif command -v python3 >/dev/null 2>&1; then \
            python3 -m venv .venv; \
        elif command -v py >/dev/null 2>&1; then \
            py -3 -m venv .venv; \
        else \
            echo "No Python interpreter found in PATH."; \
            exit 1; \
        fi; \
    fi

# Install Python dependencies into .venv.
install-ml-deps: ensure-venv
    @echo "Preparing Python environment using: {{python}}"
    {{python}} -m pip install --upgrade pip
    {{python}} -m pip install -r requirements.txt -r ml/requirements.txt
    {{python}} ml/install_torch.py

# Create venv, install deps, and build optimized Rust backend for ML.
setup-ml: install-ml-deps build-ml-server-release

# Build optimized Rust ML backend.
build-ml-server-release:
    RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server

# Convert legacy human games into decision-point NDJSON.
build-human-dataset input="ml/dataset/games.json" output="ml/data/human_dataset.ndjson": build-ml-server-release
    RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_convert_legacy
    {{ml_convert_bin}} --input {{input}} --output {{output}}

# Supervised pretraining on converted human data.
pretrain-human data="ml/data/human_dataset.ndjson" epochs="4" max_steps="512":
    @echo "Pretraining from human dataset {{data}} using virtual environment python: {{python}}"
    {{python}} ml/train_from_dataset.py --data {{data}} --epochs {{epochs}} --batch 1024 --workers 4 --device cuda --max-steps {{max_steps}}

# Train the model with 512 rounds, 128 games per round, and checkpoint every 16 rounds (total 65,536 games)
train-65k: setup-ml
    @echo "Running training using virtual environment python: {{python}}"
    ML_SERVER_BIN={{ml_server_bin}} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3

# Alias for explicit scratch baseline naming.
train-65k-scratch: train-65k

# Human-first training profile:
# 1) convert legacy logs, 2) supervised warm-start (~50% upfront step budget), 3) RL fine-tune.
train-65k-human run_name="human_first_65k": setup-ml build-human-dataset
    @echo "Running human pretraining warm-start (max_steps=512) before RL fine-tune..."
    {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 4 --batch 1024 --workers 4 --device cuda --max-steps 512
    cp -f ml/checkpoints/latest.pt ml/checkpoints/{{run_name}}_pretrain_final.pt
    @echo "Saved pretraining final checkpoint: ml/checkpoints/{{run_name}}_pretrain_final.pt"
    ML_SERVER_BIN={{ml_server_bin}} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint ml/checkpoints/{{run_name}}_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --named-checkpoint {{run_name}}_selfplay_final.pt
    cp -f ml/checkpoints/best.pt ml/checkpoints/{{run_name}}_selfplay_best.pt
    @echo "Saved self-play final checkpoint: ml/checkpoints/{{run_name}}_selfplay_final.pt"
    @echo "Saved self-play best checkpoint:  ml/checkpoints/{{run_name}}_selfplay_best.pt"

# 128k-game run profile: 256 games/round, 24 workers (optimized for this machine).
train-128k: setup-ml
    @echo "Running 128k profile using virtual environment python: {{python}}"
    ML_SERVER_BIN={{ml_server_bin}} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{python}} ml/train_online.py --rounds 512 --games-per-round 256 --workers 24 --mc-rollouts 4 --device cuda --eval-every 16 --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --hidden-loss-weight 0.10 --impossible-penalty-weight 2.00 --named-checkpoint set_theory_128k.pt

# Same 128k profile but with user-defined named checkpoint (auto-resume if it exists).
train-128k-named run_name="set_theory_128k": setup-ml
    @echo "Running named 128k profile checkpoint={{run_name}}.pt using virtual environment python: {{python}}"
    ML_SERVER_BIN={{ml_server_bin}} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{python}} ml/train_online.py --rounds 512 --games-per-round 256 --workers 24 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint ml/checkpoints/{{run_name}}.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --hidden-loss-weight 0.10 --impossible-penalty-weight 2.00 --named-checkpoint {{run_name}}.pt

# Resume training from a specific checkpoint and start round
resume-train run_name="latest" start_round="10": setup-ml
    @echo "Resuming training using virtual environment python: {{python}}"
    ML_SERVER_BIN={{ml_server_bin}} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint ml/checkpoints/{{run_name}}.pt --start-round {{start_round}} --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3

# Build and run the UI server (Ctrl+C kills both Python and the Rust engine)
ui: setup-ml
    @echo "Starting UI server (http://localhost:8765)..."
    ML_SERVER_BIN={{ml_server_bin}} {{python}} ml/ui_server.py
