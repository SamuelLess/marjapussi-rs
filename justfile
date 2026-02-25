# Default list target
default:
    @just --list

os := os()
python := if os == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }

# Train the model with 512 rounds, 128 games per round, and checkpoint every 16 rounds (total 65,536 games)
train-65k:
    @echo "Running training using virtual environment python: {{python}}"
    {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16

# Resume training from a specific checkpoint and start round
resume-train run_name="latest" start_round="10":
    @echo "Resuming training using virtual environment python: {{python}}"
    {{python}} ml/train_online.py --rounds 512 --games-per-round 128 --workers 32 --mc-rollouts 4 --device cuda --eval-every 16 --checkpoint ml/checkpoints/{{run_name}}.pt --start-round {{start_round}}

# Build and run the UI server (Ctrl+C kills both Python and the Rust engine)
ui:
    @echo "Building Rust ML backend..."
    cargo build --bin ml_server
    @echo "Starting UI server (http://localhost:8765)..."
    {{python}} ml/ui_server.py
