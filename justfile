# Default list target
default:
    @just --list

os := os()
python := if os == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }
ml_server_bin := if os == "windows" { "target/release/ml_server.exe" } else { "target/release/ml_server" }
ui_ml_server_bin := if os == "windows" { "target/ui_runtime/release/ml_server.exe" } else { "target/ui_runtime/release/ml_server" }
ml_convert_bin := if os == "windows" { "target/release/ml_convert_legacy.exe" } else { "target/release/ml_convert_legacy" }
cuda_alloc_env := if os == "windows" { "" } else { "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" }

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
    {{python}} -m pip install -e ".[dev]"
    {{python}} ml/install_torch.py

# Create venv, install deps, and build optimized Rust backend for ML.
setup-ml: install-ml-deps ensure-ml-server-release

# Build optimized Rust ML backend.
build-ml-server-release:
    RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server

# Build optimized Rust ML backend in an isolated target dir (safe while other processes use target/release).
build-ml-server-ui-runtime:
    CARGO_TARGET_DIR="target/ui_runtime" RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server

# Ensure release ml_server exists; avoid rebuild if already present (helps parallel runs on Windows).
ensure-ml-server-release:
    @if [ ! -f "{{ml_server_bin}}" ]; then \
        echo "Release ml_server missing, building once..."; \
        RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server; \
    else \
        echo "Using existing release ml_server: {{ml_server_bin}}"; \
    fi

# Ensure isolated UI runtime ml_server exists (never touches target/release/ml_server).
ensure-ml-server-ui-runtime:
    @if [ ! -f "{{ui_ml_server_bin}}" ]; then \
        echo "Isolated UI ml_server missing, building once in target/ui_runtime..."; \
        CARGO_TARGET_DIR="target/ui_runtime" RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server; \
    else \
        echo "Using isolated UI ml_server: {{ui_ml_server_bin}}"; \
    fi

# Ensure release ml_convert_legacy exists; avoid rebuild if already present.
ensure-ml-convert-release:
    @if [ ! -f "{{ml_convert_bin}}" ]; then \
        echo "Release ml_convert_legacy missing, building once..."; \
        RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_convert_legacy; \
    else \
        echo "Using existing release ml_convert_legacy: {{ml_convert_bin}}"; \
    fi

# Convert legacy human games into decision-point NDJSON.
build-human-dataset input="ml/dataset/games.ndjson" output="ml/data/human_dataset.ndjson": ensure-ml-server-release ensure-ml-convert-release
    {{ml_convert_bin}} --input {{input}} --output {{output}}

# Supervised pretraining on converted human data.
pretrain-human data="ml/data/human_dataset.ndjson" epochs="4" max_steps="512" checkpoints_dir="ml/checkpoints":
    @echo "Pretraining from human dataset {{data}} using virtual environment python: {{python}}"
    {{python}} ml/train_from_dataset.py --data {{data}} --epochs {{epochs}} --batch 1024 --workers 4 --device cuda --max-steps {{max_steps}} --checkpoints-dir {{checkpoints_dir}}

# Train the model with 512 rounds, 128 games per round, and checkpoint every 16 rounds (total 65,536 games)
train-65k run_name="scratch_65k": setup-ml
    @echo "Running training using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$$run_label" in run_name=*) run_label="$${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$$run_label"; \
      run_ckpt="$$run_root/checkpoints"; \
      run_logs="$$run_root/logs"; \
      run_bin_dir="$$run_root/bin"; \
      run_bin="$$run_bin_dir/$$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$$run_ckpt" "$$run_logs" "$$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$$run_bin"; \
      echo "Isolated run root: $$run_root"; \
      ML_SERVER_BIN="$$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --checkpoints-dir "$$run_ckpt" --runs-dir "$$run_logs" --named-checkpoint "$$run_label"_selfplay_final.pt

# Fast 4k profile: 64 rounds x 64 games (4,096 total self-play games)
train-4k run_name="scratch_4k": setup-ml
    @echo "Running fast 4k training profile using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$$run_label" in run_name=*) run_label="$${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$$run_label"; \
      run_ckpt="$$run_root/checkpoints"; \
      run_logs="$$run_root/logs"; \
      run_bin_dir="$$run_root/bin"; \
      run_bin="$$run_bin_dir/$$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$$run_ckpt" "$$run_logs" "$$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$$run_bin"; \
      echo "Isolated run root: $$run_root"; \
      ML_SERVER_BIN="$$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 64 --games-per-round 64 --workers 16 --mc-rollouts 2 --device cuda --eval-every 8 --ppo-epochs 2 --min-ppo-epochs 1 --max-ppo-epochs 3 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.20 --max-adv-calls-per-episode 2 --checkpoints-dir "$$run_ckpt" --runs-dir "$$run_logs" --named-checkpoint "$$run_label"_selfplay_final.pt

# Alias for explicit scratch baseline naming.
train-65k-scratch: train-65k

# Human-first training profile:
# 1) convert legacy logs, 2) supervised warm-start (~50% upfront step budget), 3) RL fine-tune.
train-65k-human run_name="human_first_65k": setup-ml build-human-dataset
    @echo "Running human pretraining warm-start (max_steps=512) before RL fine-tune..."
    @run_label="{{run_name}}"; \
      case "$$run_label" in run_name=*) run_label="$${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$$run_label"; \
      run_ckpt="$$run_root/checkpoints"; \
      run_logs="$$run_root/logs"; \
      run_bin_dir="$$run_root/bin"; \
      run_bin="$$run_bin_dir/$$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$$run_ckpt" "$$run_logs" "$$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$$run_bin"; \
      {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 4 --batch 1024 --workers 4 --device cuda --max-steps 512 --checkpoints-dir "$$run_ckpt"; \
      cp -f "$$run_ckpt/latest.pt" "$$run_ckpt/$$run_label"_pretrain_final.pt; \
      echo "Saved pretraining final checkpoint: $$run_ckpt/$$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$$run_ckpt/$$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --named-checkpoint "$$run_label"_selfplay_final.pt --checkpoints-dir "$$run_ckpt" --runs-dir "$$run_logs"; \
      cp -f "$$run_ckpt/best.pt" "$$run_ckpt/$$run_label"_selfplay_best.pt; \
      echo "Saved self-play final checkpoint: $$run_ckpt/$$run_label"_selfplay_final.pt; \
      echo "Saved self-play best checkpoint:  $$run_ckpt/$$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $$run_root"

# Small human-first smoke profile (1,024 self-play games total):
# - quick verification that end-to-end human-pretrain + self-play pipeline works
# - keeps explicit checkpoint labels for pretrain/final/best comparison
train-1k-human run_name="human_smoke_1k": setup-ml build-human-dataset
    @echo "Running small human pretraining warm-start (max_steps=1024) before short RL smoke run..."
    @run_label="{{run_name}}"; \
      case "$$run_label" in run_name=*) run_label="$${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$$run_label"; \
      run_ckpt="$$run_root/checkpoints"; \
      run_logs="$$run_root/logs"; \
      run_bin_dir="$$run_root/bin"; \
      run_bin="$$run_bin_dir/$$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$$run_ckpt" "$$run_logs" "$$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 6 --batch 256 --workers 2 --device cuda --max-steps 1024 --checkpoints-dir "$$run_ckpt"; \
      cp -f "$$run_ckpt/latest.pt" "$$run_ckpt/$$run_label"_pretrain_final.pt; \
      echo "Saved pretraining final checkpoint: $$run_ckpt/$$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 32 --games-per-round 32 --workers 8 --mc-rollouts 2 --device cuda --eval-every 8 --checkpoint "$$run_ckpt/$$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 1 --max-ppo-epochs 3 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.20 --max-adv-calls-per-episode 2 --named-checkpoint "$$run_label"_selfplay_final.pt --checkpoints-dir "$$run_ckpt" --runs-dir "$$run_logs"; \
      cp -f "$$run_ckpt/best.pt" "$$run_ckpt/$$run_label"_selfplay_best.pt; \
      echo "Saved self-play final checkpoint: $$run_ckpt/$$run_label"_selfplay_final.pt; \
      echo "Saved self-play best checkpoint:  $$run_ckpt/$$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $$run_root"

# Alias matching requested naming style.
just-1k-human: train-1k-human

# 128k-game run profile: 256 games/round, 24 workers (optimized for this machine).
train-128k run_name="set_theory_128k": setup-ml
    @echo "Running 128k profile using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$$run_label" in run_name=*) run_label="$${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$$run_label"; \
      run_ckpt="$$run_root/checkpoints"; \
      run_logs="$$run_root/logs"; \
      run_bin_dir="$$run_root/bin"; \
      run_bin="$$run_bin_dir/$$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$$run_ckpt" "$$run_logs" "$$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$$run_bin"; \
      ML_SERVER_BIN="$$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 256 --workers 24 --mc-rollouts 4 --device cuda --eval-every 16 --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --hidden-loss-weight 0.10 --impossible-penalty-weight 2.00 --named-checkpoint "$$run_label".pt --checkpoints-dir "$$run_ckpt" --runs-dir "$$run_logs"; \
      echo "Artifacts root: $$run_root"

# Same 128k profile but with user-defined named checkpoint (auto-resume if it exists).
train-128k-named run_name="set_theory_128k": setup-ml
    @echo "Running named 128k profile checkpoint={{run_name}}.pt using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$$run_label" in run_name=*) run_label="$${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$$run_label"; \
      run_ckpt="$$run_root/checkpoints"; \
      run_logs="$$run_root/logs"; \
      run_bin_dir="$$run_root/bin"; \
      run_bin="$$run_bin_dir/$$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$$run_ckpt" "$$run_logs" "$$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$$run_bin"; \
      ML_SERVER_BIN="$$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 256 --workers 24 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$$run_ckpt/$$run_label".pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --hidden-loss-weight 0.10 --impossible-penalty-weight 2.00 --named-checkpoint "$$run_label".pt --checkpoints-dir "$$run_ckpt" --runs-dir "$$run_logs"; \
      echo "Artifacts root: $$run_root"

# Resume training from a specific checkpoint and start round
resume-train run_name="scratch_65k" checkpoint_name="latest" start_round="10": setup-ml
    @echo "Resuming training using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$$run_label" in run_name=*) run_label="$${run_label#run_name=}" ;; esac; \
      checkpoint_label="{{checkpoint_name}}"; \
      case "$$checkpoint_label" in checkpoint_name=*) checkpoint_label="$${checkpoint_label#checkpoint_name=}" ;; esac; \
      run_root="ml/runs/$$run_label"; \
      run_ckpt="$$run_root/checkpoints"; \
      run_logs="$$run_root/logs"; \
      run_bin_dir="$$run_root/bin"; \
      run_bin="$$run_bin_dir/$$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$$run_ckpt" "$$run_logs" "$$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$$run_bin"; \
      ML_SERVER_BIN="$$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$$run_ckpt/$$checkpoint_label".pt --start-round {{start_round}} --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --checkpoints-dir "$$run_ckpt" --runs-dir "$$run_logs" --named-checkpoint "$$run_label"_selfplay_final.pt; \
      echo "Artifacts root: $$run_root"

# Build and run the UI server (Ctrl+C kills both Python and the Rust engine)
# Example:
#   just ui
#   just ui checkpoint=latest port=8765
#   just ui checkpoint=my_run_best.pt port=18765
ui checkpoint="latest" port="8765": install-ml-deps ensure-ml-server-ui-runtime
    @ui_ckpt="{{checkpoint}}"; \
      case "$ui_ckpt" in checkpoint=*) ui_ckpt="${ui_ckpt#checkpoint=}" ;; esac; \
      ui_port="{{port}}"; \
      case "$ui_port" in port=*) ui_port="${ui_port#port=}" ;; esac; \
      echo "Starting UI server (http://localhost:$ui_port) with checkpoint=$ui_ckpt ..."; \
      ML_SERVER_BIN="{{ui_ml_server_bin}}" {{python}} ml/ui_server.py --port "$ui_port" --checkpoint "$ui_ckpt"
