# Default list target
default:
    @just --list

os := os()
python := if os == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }
ml_server_bin := if os == "windows" { "target/release/ml_server.exe" } else { "target/release/ml_server" }
ui_ml_server_bin := if os == "windows" { "target/ui_runtime/release/ml_server.exe" } else { "target/ui_runtime/release/ml_server" }
ml_convert_bin := if os == "windows" { "target/release/ml_convert_legacy.exe" } else { "target/release/ml_convert_legacy" }
cuda_alloc_env := if os == "windows" { "" } else { "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" }
v2_model_family := "parallel_v2"
v2_model_config := "ml/config/model_parallel_v2.toml"
v2_model_args := "--model-family parallel_v2 --model-config ml/config/model_parallel_v2.toml --strict-param-budget 28000000"

# Ensure virtual environment exists.
ensure-venv:
    @if [ ! -f ".venv/Scripts/python.exe" ] && [ ! -f ".venv/bin/python" ]; then \
        echo "Creating virtual environment at .venv"; \
        if command -v py >/dev/null 2>&1; then \
            selected=""; \
            for v in 3.13 3.12 3.11 3.10 3.9; do \
                if py -$v -c "import sys; print(sys.version)" >/dev/null 2>&1; then \
                    selected="$v"; \
                    break; \
                fi; \
            done; \
            if [ -n "$selected" ]; then \
                echo "Using Python $selected via py launcher"; \
                py -$selected -m venv .venv; \
            else \
                py -3 -m venv .venv; \
            fi; \
        elif command -v python3 >/dev/null 2>&1; then \
            python3 -m venv .venv; \
        elif command -v python >/dev/null 2>&1; then \
            python -m venv .venv; \
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
    @if [ ! -f "{{ml_server_bin}}" ] || [ -n "$(find src Cargo.toml Cargo.lock -type f -newer "{{ml_server_bin}}" 2>/dev/null | head -n 1)" ]; then \
        echo "Building fresh release ml_server..."; \
        if RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server; then \
            true; \
        elif [ -f "{{ml_server_bin}}" ]; then \
            echo "WARN: ml_server rebuild failed (likely Windows file lock). Falling back to existing binary: {{ml_server_bin}}"; \
        else \
            echo "ERROR: ml_server build failed and no existing binary is available."; \
            exit 1; \
        fi; \
    else \
        echo "Using up-to-date release ml_server: {{ml_server_bin}}"; \
    fi

# Ensure isolated UI runtime ml_server exists (never touches target/release/ml_server).
ensure-ml-server-ui-runtime:
    @if [ ! -f "{{ui_ml_server_bin}}" ] || [ -n "$(find src Cargo.toml Cargo.lock -type f -newer "{{ui_ml_server_bin}}" 2>/dev/null | head -n 1)" ]; then \
        echo "Building fresh isolated UI ml_server in target/ui_runtime..."; \
        CARGO_TARGET_DIR="target/ui_runtime" RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_server; \
    else \
        echo "Using up-to-date isolated UI ml_server: {{ui_ml_server_bin}}"; \
    fi

# Ensure release ml_convert_legacy exists; avoid rebuild if already present.
ensure-ml-convert-release:
    @if [ ! -f "{{ml_convert_bin}}" ] || [ -n "$(find src Cargo.toml Cargo.lock -type f -newer "{{ml_convert_bin}}" 2>/dev/null | head -n 1)" ]; then \
        echo "Building fresh release ml_convert_legacy..."; \
        if RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ml_convert_legacy; then \
            true; \
        elif [ -f "{{ml_convert_bin}}" ]; then \
            echo "WARN: ml_convert_legacy rebuild failed (likely Windows file lock). Falling back to existing binary: {{ml_convert_bin}}"; \
        else \
            echo "ERROR: ml_convert_legacy build failed and no existing binary is available."; \
            exit 1; \
        fi; \
    else \
        echo "Using up-to-date release ml_convert_legacy: {{ml_convert_bin}}"; \
    fi

# Convert legacy human games into decision-point NDJSON.
build-human-dataset input="ml/dataset/games.ndjson" output="ml/data/human_dataset.ndjson": ensure-ml-server-release ensure-ml-convert-release
    {{ml_convert_bin}} --input {{input}} --output {{output}}
    @{{python}} -c "import json,collections,sys; p='{{output}}'; c=collections.Counter(); n=0; f=open(p,'r',encoding='utf-8'); [c.update([(lambda r: (r['obs'].get('legal_actions',[{}])[min(max(0,int(r.get('action_taken',0))), max(0,len(r['obs'].get('legal_actions',[]))-1))].get('action_token') if r.get('obs',{}).get('legal_actions') else None))(json.loads(line))]) or (n:=n+1) for line in f if line.strip()]; f.close(); print(f'Dataset rows={n} pass_pick(token52)={c.get(52,0)} direct_pass(token43)={c.get(43,0)}'); sys.exit(0 if c.get(52,0)>0 and c.get(43,0)==0 else 2)"

# Convert legacy logs and keep only rows from players with winrate > threshold.
build-human-dataset-win55 input="ml/dataset/games.ndjson" output="ml/data/human_dataset_win55.ndjson" min_winrate="0.55": ensure-ml-server-release ensure-ml-convert-release
    {{ml_convert_bin}} --input {{input}} --output {{output}} --min-player-winrate {{min_winrate}}
    @{{python}} -c "import json,collections,sys; p='{{output}}'; c=collections.Counter(); players=collections.Counter(); n=0; f=open(p,'r',encoding='utf-8'); [players.update([json.loads(line).get('pov_player','?')]) or c.update([(lambda r: (r['obs'].get('legal_actions',[{}])[min(max(0,int(r.get('action_taken',0))), max(0,len(r['obs'].get('legal_actions',[]))-1))].get('action_token') if r.get('obs',{}).get('legal_actions') else None))(json.loads(line))]) or (n:=n+1) for line in f if line.strip()]; f.close(); print(f'Dataset rows={n} unique_players={len(players)} pass_pick(token52)={c.get(52,0)} direct_pass(token43)={c.get(43,0)}'); print('Top players by rows:', ', '.join([f'{k}:{v}' for k,v in players.most_common(8)])); sys.exit(0 if n>0 and c.get(52,0)>0 and c.get(43,0)==0 else 2)"

# Supervised pretraining on converted human data.
pretrain-human data="ml/data/human_dataset.ndjson" epochs="8" max_steps="2048" checkpoints_dir="ml/checkpoints":
    @echo "Pretraining from human dataset {{data}} using virtual environment python: {{python}}"
    {{python}} ml/train_from_dataset.py --data {{data}} --epochs {{epochs}} --batch 1024 --workers 4 --device cuda --max-steps {{max_steps}} --bid-weight 2.0 --pass-weight 2.5 --checkpoints-dir {{checkpoints_dir}}

# Train the model with 512 rounds, 128 games per round, and checkpoint every 16 rounds (total 65,536 games)
train-65k run_name="scratch_65k": setup-ml
    @echo "Running training using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      echo "Isolated run root: $run_root"; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs" --named-checkpoint "$run_label"_selfplay_final.pt

# Fast 4k profile: 64 rounds x 64 games (4,096 total self-play games)
train-4k run_name="scratch_4k": setup-ml
    @echo "Running fast 4k training profile using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      echo "Isolated run root: $run_root"; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 64 --games-per-round 64 --workers 16 --mc-rollouts 2 --device cuda --eval-every 8 --ppo-epochs 2 --min-ppo-epochs 1 --max-ppo-epochs 3 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.20 --max-adv-calls-per-episode 2 --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs" --named-checkpoint "$run_label"_selfplay_final.pt

# Human-first 4k profile:
# 1) convert legacy logs, 2) supervised warm-start, 3) 64x64 (4,096 games) RL self-play.
train-4k-human run_name="human_smoke_4k": setup-ml build-human-dataset
    @echo "Running stronger human pretraining warm-start (max_steps=3072) before 4k RL run..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 10 --batch 256 --workers 2 --device cuda --max-steps 3072 --bid-weight 2.0 --pass-weight 2.5 --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      echo "Saved pretraining final checkpoint: $run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 64 --games-per-round 64 --workers 16 --mc-rollouts 2 --device cuda --eval-every 8 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 1 --max-ppo-epochs 3 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.20 --max-adv-calls-per-episode 2 --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Saved self-play final checkpoint: $run_ckpt/$run_label"_selfplay_final.pt; \
      echo "Saved self-play best checkpoint:  $run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# Alias for explicit scratch baseline naming.
train-65k-scratch: train-65k

# Human-first training profile:
# 1) convert legacy logs, 2) supervised warm-start (~50% upfront step budget), 3) RL fine-tune.
train-65k-human run_name="human_first_65k": setup-ml build-human-dataset
    @echo "Running adaptive human pretraining warm-start (target BC loss) before RL fine-tune..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 32 --batch 512 --workers 4 --device cuda --max-steps 0 --min-epochs 8 --target-bc-loss 0.240 --target-bc-streak 2 --bid-weight 2.0 --pass-weight 2.5 --lr-schedule plateau --lr 3e-4 --lr-min 8e-5 --lr-warmup-steps 256 --lr-plateau-patience 2 --lr-plateau-factor 0.75 --lr-plateau-threshold 6e-4 --lr-plateau-cooldown 1 --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      echo "Saved pretraining final checkpoint: $run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Saved self-play final checkpoint: $run_ckpt/$run_label"_selfplay_final.pt; \
      echo "Saved self-play best checkpoint:  $run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# Small human-first smoke profile (1,024 self-play games total):
# - quick verification that end-to-end human-pretrain + self-play pipeline works
# - keeps explicit checkpoint labels for pretrain/final/best comparison
train-1k-human run_name="human_smoke_1k": setup-ml build-human-dataset
    @echo "Running small human pretraining warm-start (max_steps=1024) before short RL smoke run..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 6 --batch 256 --workers 2 --device cuda --max-steps 1024 --bid-weight 2.0 --pass-weight 2.5 --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      echo "Saved pretraining final checkpoint: $run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 32 --games-per-round 32 --workers 8 --mc-rollouts 2 --device cuda --eval-every 8 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 1 --max-ppo-epochs 3 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.20 --max-adv-calls-per-episode 2 --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Saved self-play final checkpoint: $run_ckpt/$run_label"_selfplay_final.pt; \
      echo "Saved self-play best checkpoint:  $run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# Alias matching requested naming style.
just-1k-human: train-1k-human

# Small human-first v2 smoke profile (1,024 self-play games total).
train-1k-human-v2 run_name="v2_human_smoke_1k": setup-ml build-human-dataset
    @echo "Running small human-first v2 smoke run..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 6 --batch 256 --workers 2 --device cuda --max-steps 1024 --bid-weight 2.2 --pass-weight 2.8 {{v2_model_args}} --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 32 --games-per-round 32 --workers 8 --mc-rollouts 2 --device cuda --eval-every 8 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 1 --max-ppo-epochs 3 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.20 --max-adv-calls-per-episode 2 {{v2_model_args}} --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# Fast 4k v2 profile with human-first warm start.
train-4k-human-v2 run_name="v2_human_smoke_4k": setup-ml build-human-dataset
    @echo "Running human-first 4k v2 training profile..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 10 --batch 256 --workers 2 --device cuda --max-steps 3072 --bid-weight 2.2 --pass-weight 2.8 {{v2_model_args}} --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 64 --games-per-round 64 --workers 16 --mc-rollouts 2 --device cuda --eval-every 8 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 1 --max-ppo-epochs 3 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.20 --max-adv-calls-per-episode 2 {{v2_model_args}} --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# Full 65k v2 profile with stronger human pretraining.
train-65k-human-v2 run_name="v2_human_first_65k": setup-ml build-human-dataset
    @echo "Running full 65k human-first v2 profile..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 48 --batch 256 --workers 2 --device cuda --max-steps 0 --min-epochs 12 --target-bc-loss 0.235 --target-bc-streak 2 --bid-weight 2.5 --pass-weight 3.0 --lr-schedule plateau --lr 3e-4 --lr-min 6e-5 --lr-warmup-steps 384 --lr-plateau-patience 2 --lr-plateau-factor 0.7 --lr-plateau-threshold 8e-4 --lr-plateau-cooldown 1 {{v2_model_args}} --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 {{v2_model_args}} --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# Full 65k v2 profile trained only on perspectives of players with winrate > 55%.
train-65k-human55-v2 run_name="v2_human55_first_65k": setup-ml build-human-dataset-win55
    @echo "Running full 65k human-first v2 profile on >55% winrate player perspectives..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset_win55.ndjson --epochs 48 --batch 256 --workers 2 --device cuda --max-steps 0 --min-epochs 12 --target-bc-loss 0.235 --target-bc-streak 2 --play-weight 1.5 --bid-weight 1.5 --pass-weight 1.8 --phase-balance-strength 1.0 --lr-schedule plateau --lr 3e-4 --lr-min 6e-5 --lr-warmup-steps 384 --lr-plateau-patience 2 --lr-plateau-factor 0.7 --lr-plateau-threshold 8e-4 --lr-plateau-cooldown 1 {{v2_model_args}} --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 {{v2_model_args}} --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# Full ~129k-game v2 profile (~128,064 games with staged curriculum defaults):
# 1) pretrain on all human data, 2) pretrain on >55% data, 3) staged curriculum RL.
# Note: `just` identifiers cannot start with a digit, so this is `train-129k-v2`.
train-129k-v2 run_name="v2_human_first_129k": setup-ml build-human-dataset build-human-dataset-win55
    @echo "Running full ~129k human-first v2 profile (two-stage pretraining + staged curriculum RL)..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      echo "[PRETRAIN] Stage 1/2: all_games"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --stage-label all_games --save-every-epochs 16 --keep-epoch-checkpoints 6 --data ml/data/human_dataset.ndjson --epochs 48 --batch 256 --workers 2 --device cuda --max-steps 0 --min-epochs 12 --target-bc-loss 0.235 --target-bc-streak 2 --play-weight 1.5 --bid-weight 1.5 --pass-weight 1.8 --phase-balance-strength 1.0 --lr-schedule plateau --lr 3e-4 --lr-min 6e-5 --lr-warmup-steps 384 --lr-plateau-patience 2 --lr-plateau-factor 0.7 --lr-plateau-threshold 8e-4 --lr-plateau-cooldown 1 {{v2_model_args}} --checkpoints-dir "$run_ckpt"; \
      if [ -f "$run_ckpt/best.pt" ]; then cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_pretrain_all_best.pt; fi; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_all_complete.pt; \
      echo "[PRETRAIN] Stage 2/2: win55_good_players"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --stage-label win55_good_players --save-every-epochs 16 --keep-epoch-checkpoints 6 --data ml/data/human_dataset_win55.ndjson --checkpoint "$run_ckpt/$run_label"_pretrain_all_complete.pt --epochs 48 --batch 256 --workers 2 --device cuda --max-steps 0 --min-epochs 12 --target-bc-loss 0.225 --target-bc-streak 2 --play-weight 1.5 --bid-weight 1.6 --pass-weight 1.9 --phase-balance-strength 1.0 --lr-schedule plateau --lr 2.5e-4 --lr-min 5e-5 --lr-warmup-steps 256 --lr-plateau-patience 2 --lr-plateau-factor 0.7 --lr-plateau-threshold 8e-4 --lr-plateau-cooldown 1 {{v2_model_args}} --checkpoints-dir "$run_ckpt"; \
      if [ -f "$run_ckpt/best.pt" ]; then cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_pretrain_good_best.pt; fi; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_good_complete.pt; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 645 --games-per-round 256 --workers 24 --mc-rollouts 4 --train-batch 256 --device cuda --eval-every 16 --save-latest-every 16 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --trick-phase-frac 0.20 --passing-phase-frac 0.05 --bidding-phase-frac 0.05 --phase-trick-games-per-round 64 --phase-passing-games-per-round 64 --phase-bidding-games-per-round 64 --phase-full-games-per-round 256 --phase-trick-train-batch 384 --phase-passing-train-batch 384 --phase-bidding-train-batch 384 --phase-full-train-batch 256 --phase-loss-patience 3 --phase-loss-min-delta 0.005 --phase-min-rounds-frac 0.40 {{v2_model_args}} --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      if [ -f "$run_ckpt/phase_gameplay_complete.pt" ]; then cp -f "$run_ckpt/phase_gameplay_complete.pt" "$run_ckpt/$run_label"_posttrain_gameplay_complete.pt; fi; \
      if [ -f "$run_ckpt/phase_passing_complete.pt" ]; then cp -f "$run_ckpt/phase_passing_complete.pt" "$run_ckpt/$run_label"_posttrain_passing_complete.pt; fi; \
      if [ -f "$run_ckpt/phase_bidding_complete.pt" ]; then cp -f "$run_ckpt/phase_bidding_complete.pt" "$run_ckpt/$run_label"_posttrain_bidding_complete.pt; fi; \
      if [ -f "$run_ckpt/phase_full_game_complete.pt" ]; then cp -f "$run_ckpt/phase_full_game_complete.pt" "$run_ckpt/$run_label"_full_game_complete.pt; fi; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# 128k-game human-first run profile:
# 1) convert legacy logs, 2) stronger supervised warm-start, 3) 512x256 (131,072 games) RL self-play.
train-128k run_name="set_theory_128k": setup-ml build-human-dataset
    @echo "Running 128k human-first profile with stronger pretraining warm-start..."
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      {{cuda_alloc_env}} {{python}} ml/train_from_dataset.py --data ml/data/human_dataset.ndjson --epochs 64 --batch 256 --workers 2 --device cuda --max-steps 0 --min-epochs 16 --target-bc-loss 0.230 --target-bc-streak 2 --bid-weight 2.5 --pass-weight 3.0 --lr-schedule plateau --lr 3e-4 --lr-min 6e-5 --lr-warmup-steps 384 --lr-plateau-patience 2 --lr-plateau-factor 0.7 --lr-plateau-threshold 8e-4 --lr-plateau-cooldown 1 --checkpoints-dir "$run_ckpt"; \
      cp -f "$run_ckpt/latest.pt" "$run_ckpt/$run_label"_pretrain_final.pt; \
      echo "Saved pretraining final checkpoint: $run_ckpt/$run_label"_pretrain_final.pt; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 256 --workers 24 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$run_ckpt/$run_label"_pretrain_final.pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --hidden-loss-weight 0.10 --impossible-penalty-weight 2.00 --named-checkpoint "$run_label"_selfplay_final.pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      cp -f "$run_ckpt/best.pt" "$run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Saved self-play final checkpoint: $run_ckpt/$run_label"_selfplay_final.pt; \
      echo "Saved self-play best checkpoint:  $run_ckpt/$run_label"_selfplay_best.pt; \
      echo "Artifacts root: $run_root"

# 128k scratch-only profile (no supervised warm-start).
train-128k-scratch run_name="set_theory_128k_scratch": setup-ml
    @echo "Running 128k scratch profile using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 256 --workers 24 --mc-rollouts 4 --device cuda --eval-every 16 --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --hidden-loss-weight 0.10 --impossible-penalty-weight 2.00 --named-checkpoint "$run_label".pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      echo "Artifacts root: $run_root"

# Resume/continue 128k from a named run checkpoint.
train-128k-named run_name="set_theory_128k": setup-ml
    @echo "Running named 128k profile checkpoint={{run_name}}.pt using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 256 --workers 24 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$run_ckpt/$run_label".pt --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --hidden-loss-weight 0.10 --impossible-penalty-weight 2.00 --named-checkpoint "$run_label".pt --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs"; \
      echo "Artifacts root: $run_root"

# Resume training from a specific checkpoint and start round
resume-train run_name="scratch_65k" checkpoint_name="latest" start_round="10": setup-ml
    @echo "Resuming training using virtual environment python: {{python}}"
    @run_label="{{run_name}}"; \
      case "$run_label" in run_name=*) run_label="${run_label#run_name=}" ;; esac; \
      checkpoint_label="{{checkpoint_name}}"; \
      case "$checkpoint_label" in checkpoint_name=*) checkpoint_label="${checkpoint_label#checkpoint_name=}" ;; esac; \
      run_root="ml/runs/$run_label"; \
      run_ckpt="$run_root/checkpoints"; \
      run_logs="$run_root/logs"; \
      run_bin_dir="$run_root/bin"; \
      run_bin="$run_bin_dir/$(basename "{{ml_server_bin}}")"; \
      mkdir -p "$run_ckpt" "$run_logs" "$run_bin_dir"; \
      cp -f "{{ml_server_bin}}" "$run_bin"; \
      ML_SERVER_BIN="$run_bin" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {{cuda_alloc_env}} {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint "$run_ckpt/$checkpoint_label".pt --start-round {{start_round}} --ppo-epochs 2 --min-ppo-epochs 2 --max-ppo-epochs 5 --target-kl 0.03 --max-clipfrac 0.40 --min-policy-improve 0.001 --opt-early-stop-patience 1 --adv-query-mode target_plus_stochastic --adv-non-target-prob 0.25 --max-adv-calls-per-episode 3 --checkpoints-dir "$run_ckpt" --runs-dir "$run_logs" --named-checkpoint "$run_label"_selfplay_final.pt; \
      echo "Artifacts root: $run_root"

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
      ML_SERVER_BIN="{{ui_ml_server_bin}}" exec {{python}} ml/ui_server.py --port "$ui_port" --checkpoint "$ui_ckpt"

# Evaluate checkpoints on deterministic fixed-deal suites.
# Example:
#   just eval-fixed checkpoint=latest max_cases=10
#   just eval-fixed checkpoint=1 echo=0
eval-fixed suite="ml/eval/fixed_deals_100.json" checkpoint="latest" max_cases="0" echo="1": install-ml-deps ensure-ml-server-release
    @echo_flag="{{echo}}"; \
      case "$echo_flag" in echo=*) echo_flag="${echo_flag#echo=}" ;; esac; \
      echo_opt=""; \
      if [ "$echo_flag" = "1" ]; then echo_opt="--echo"; fi; \
      ML_SERVER_BIN="{{ml_server_bin}}" {{python}} ml/eval_fixed_deals.py --suite "{{suite}}" --all-checkpoint "{{checkpoint}}" --max-cases "{{max_cases}}" $echo_opt

# Generate random fixed-deal suites (explicit 4x9 hands) via ml_server.
gen-fixed-deals count="100" output="ml/eval/fixed_deals_random_100.json" master_seed="20260303": install-ml-deps ensure-ml-server-release
    ML_SERVER_BIN="{{ml_server_bin}}" {{python}} ml/generate_fixed_deals.py --count "{{count}}" --master-seed "{{master_seed}}" --output "{{output}}"
