"""
Iterative Online Training for Marjapussi AI
============================================
Each round:
  1. Play N games with the CURRENT model (quality improves every round)
  2. At each multi-choice decision: ask Rust for MC advantages (try_all_actions)
  3. Train on the batch (advantage-weighted BC + points regression)
  4. Save checkpoint → next round uses the improved policy

This means training data quality grows automatically — no stale 1M-game dumps.

Usage:
  python ml/train_online.py --rounds 50 --games-per-round 500 --workers 8 --mc-rollouts 8 --device cuda

Recommended starting point:
  python ml/train_online.py --rounds 100 --games-per-round 200 --workers 8 --mc-rollouts 4 --device cuda
"""

import argparse, collections, json, math, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from model import MarjapussiNet
from env import MarjapussiEnv, obs_to_tensors
from self_play import heuristic_select, compute_advantages

CKPT_DIR = Path(__file__).parent / "checkpoints"
RUNS_DIR  = Path(__file__).parent / "runs"
CKPT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


# ── Transition buffer ─────────────────────────────────────────────────────────

Transition = collections.namedtuple("Transition",
    ["obs", "action_idx", "advantage", "pts_my", "pts_opp"])


# ── Per-worker episode runner ─────────────────────────────────────────────────

def _model_select(model, obs, device):
    """Run inference (called from worker thread; model is read-only)."""
    try:
        tensors = obs_to_tensors(obs)
        with torch.no_grad():
            logits, _, _ = model(tensors)
        probs = F.softmax(logits[0], dim=-1)
        return int(torch.multinomial(probs, 1).item())
    except Exception:
        return 0


def run_episode(model, device, mc_rollouts: int, stage: int,
                lock: Lock) -> list[Transition]:
    """
    Play one complete game. Returns a list of Transitions.
    stage 0 = heuristic (bootstrap), 1 = model.
    """
    env = MarjapussiEnv(pov=0)
    transitions = []
    try:
        obs = env.reset()
        steps = 0
        while not env.done and steps < 300:
            steps += 1
            legal = obs.get("legal_actions", [])
            if not legal:
                break

            # Choose action
            if stage == 0 or len(legal) == 1:
                action_pos = heuristic_select(legal) if stage == 0 else 0
            else:
                with lock:   # model inference is not thread-safe without lock
                    action_pos = _model_select(model, obs, device)
                action_pos = min(action_pos, len(legal) - 1)

            # MC advantage at multi-choice points
            advantage = 0.0
            if len(legal) > 1 and mc_rollouts > 0:
                try:
                    branches = env.try_all_actions(policy="heuristic")
                    advs = compute_advantages(branches, pov_seat=0)
                    advantage = advs.get(action_pos, 0.0)
                    # Record all moves, not just chosen (for counterfactual learning)
                    for bidx, branch_adv in advs.items():
                        if bidx < len(legal):
                            # We can add alternate actions too — optional
                            pass
                except Exception:
                    pass

            t_before = obs_to_tensors(obs)
            obs, done, info = env.step(legal[action_pos]["action_list_idx"])

            # Accumulate transitions (only when MC or final outcome known)
            transitions.append(Transition(
                obs=t_before,
                action_idx=action_pos,
                advantage=advantage,
                pts_my=0.0,    # filled after game ends
                pts_opp=0.0,
            ))

        # Fill in final outcome for all transitions
        if info and "tricks" in info:
            my_pts  = sum(t["points"] for t in info["tricks"] if t["winner"] % 2 == 0) / 120.0
            opp_pts = 1.0 - my_pts
        else:
            my_pts = opp_pts = 0.5

        # Replace placeholder pts with real outcome
        transitions = [
            Transition(t.obs, t.action_idx,
                       t.advantage if abs(t.advantage) > 0.01 else (my_pts * 2 - 1.0),
                       my_pts, opp_pts)
            for t in transitions
        ]
    finally:
        env.close()

    return transitions


# ── Batch collation ───────────────────────────────────────────────────────────

def collate(transitions: list[Transition], device: str):
    if not transitions:
        return None

    B = len(transitions)
    max_seq = max(t.obs["token_ids"].shape[1] for t in transitions)
    max_act = max(t.obs["action_feats"].shape[1] for t in transitions)

    obs_a = {}
    for k in transitions[0].obs["obs_a"]:
        obs_a[k] = torch.cat([t.obs["obs_a"][k] for t in transitions], 0).to(device, non_blocking=True)

    tok = torch.zeros(B, max_seq, dtype=torch.long)
    tmask = torch.ones(B, max_seq, dtype=torch.bool)
    for i, t in enumerate(transitions):
        L = t.obs["token_ids"].shape[1]
        tok[i, :L] = t.obs["token_ids"][0]; tmask[i, :L] = t.obs["token_mask"][0]

    af = torch.zeros(B, max_act, 51)
    am = torch.ones(B, max_act, dtype=torch.bool)
    for i, t in enumerate(transitions):
        A = t.obs["action_feats"].shape[1]
        af[i, :A] = t.obs["action_feats"][0]; am[i, :A] = t.obs["action_mask"][0]

    ai = torch.tensor(
        [min(t.action_idx, transitions[i].obs["action_feats"].shape[1]-1)
         for i, t in enumerate(transitions)], dtype=torch.long)

    return {
        "obs_a":       obs_a,
        "token_ids":   tok.to(device, non_blocking=True),
        "token_mask":  tmask.to(device, non_blocking=True),
        "action_feats":af.to(device, non_blocking=True),
        "action_mask": am.to(device, non_blocking=True),
        "action_idx":  ai.to(device, non_blocking=True),
        "advantage":   torch.tensor([t.advantage for t in transitions]).to(device, non_blocking=True),
        "pts_target":  torch.tensor([[t.pts_my, t.pts_opp] for t in transitions]).to(device, non_blocking=True),
    }


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(model, opt, scaler, batch, use_amp: bool) -> dict:
    model.train()
    with autocast(enabled=use_amp):
        logits, _, pts_pred = model({
            "obs_a":        batch["obs_a"],
            "token_ids":    batch["token_ids"],
            "token_mask":   batch["token_mask"],
            "action_feats": batch["action_feats"],
            "action_mask":  batch["action_mask"],
        })
        log_p = F.log_softmax(logits, dim=-1)
        chosen_lp = log_p.gather(1, batch["action_idx"].unsqueeze(1)).squeeze(1)
        # Advantage-weighted BC: positive advantages get stronger reinforcement
        weights = F.relu(batch["advantage"]) + 0.2
        bc_loss  = -(chosen_lp * weights).mean()
        pts_loss = F.mse_loss(pts_pred, batch["pts_target"]) * 0.3
        loss     = bc_loss + pts_loss

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    return {"total": loss.item(), "bc": bc_loss.item(), "pts": pts_loss.item()}


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_win_rate(model, device, n=20) -> float:
    wins = 0
    for _ in range(n):
        lock = Lock()
        eps = run_episode(model, device, mc_rollouts=0, stage=1, lock=lock)
        # Look at last transition's pts_my > 0.5 ≈ win
        if eps and eps[-1].pts_my > 0.5:
            wins += 1
    return wins / n


# ── Main loop ─────────────────────────────────────────────────────────────────

def train_online(
    rounds: int        = 100,
    games_per_round: int = 200,
    workers: int        = 8,
    mc_rollouts: int    = 4,
    train_batch: int    = 256,
    lr: float           = 3e-4,
    device: str         = "cpu",
    checkpoint: str | None = None,
    eval_every: int     = 10,
):
    use_amp = device.startswith("cuda")
    model   = MarjapussiNet().to(device)
    scaler  = GradScaler(enabled=use_amp)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rounds)

    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")

    print(f"Online training | rounds={rounds} | games/round={games_per_round} "
          f"| workers={workers} | MC-rollouts={mc_rollouts} | device={device}")
    print(f"Params: {model.param_count():,}\n")

    run_dir  = RUNS_DIR / f"online_{int(time.time())}"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.jsonl"
    (RUNS_DIR / "latest").unlink(missing_ok=True)
    try: (RUNS_DIR / "latest").symlink_to(run_dir, target_is_directory=True)
    except: pass

    best_wr = 0.0
    lock    = Lock()  # shared lock for model inference across threads

    for rnd in range(rounds):
        t0 = time.time()
        stage = 0 if rnd < 3 else 1   # first 3 rounds: heuristic bootstrap

        # ── Collect games in parallel via threads ────────────────────────────
        model.eval()
        all_transitions: list[Transition] = []

        def collect_one(_):
            return run_episode(model, device, mc_rollouts, stage, lock)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(collect_one, range(games_per_round)))

        for eps in results:
            all_transitions.extend(eps)

        random.shuffle(all_transitions)
        n_trans = len(all_transitions)
        gen_time = time.time() - t0

        if n_trans == 0:
            print(f"[Round {rnd+1}] No transitions collected — skipping")
            continue

        # ── Train on collected batch ─────────────────────────────────────────
        model.train()
        t1 = time.time()
        loss_acc = collections.defaultdict(float)
        n_steps = 0

        for start in range(0, n_trans, train_batch):
            chunk = all_transitions[start:start + train_batch]
            batch = collate(chunk, device)
            if batch is None: continue
            losses = train_step(model, opt, scaler, batch, use_amp)
            for k, v in losses.items(): loss_acc[k] += v
            n_steps += 1

        sched.step()

        avg_loss = {k: v / max(n_steps, 1) for k, v in loss_acc.items()}
        train_time = time.time() - t1

        # ── Logging ──────────────────────────────────────────────────────────
        policy_label = "heuristic" if stage == 0 else f"model-v{rnd+1}"
        rate = games_per_round / gen_time
        print(f"[Round {rnd+1:3d}/{rounds}] policy={policy_label}  "
              f"games={games_per_round} ({rate:.0f}/s)  "
              f"transitions={n_trans}  "
              f"loss={avg_loss.get('total',0):.4f}  "
              f"gen={gen_time:.1f}s train={train_time:.1f}s")

        log_entry = {"round": rnd+1, "stage": stage, "n_games": games_per_round,
                     "n_transitions": n_trans, "losses": avg_loss,
                     "gen_secs": gen_time, "train_secs": train_time}

        # ── Evaluate + checkpoint ────────────────────────────────────────────
        if (rnd + 1) % eval_every == 0 and stage == 1:
            wr = eval_win_rate(model, device, n=20)
            print(f"           → Win rate (20 games): {wr:.0%}")
            log_entry["win_rate"] = wr
            torch.save(model.state_dict(), CKPT_DIR / f"online_round_{rnd+1}.pt")
            if wr > best_wr:
                best_wr = wr
                torch.save(model.state_dict(), CKPT_DIR / "best.pt")
                print(f"           → New best: {best_wr:.0%}")

        torch.save(model.state_dict(), CKPT_DIR / "latest.pt")
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "update", **log_entry}) + "\n")

    print(f"\nDone. Best win rate: {best_wr:.0%}")
    print(f"Checkpoint: {CKPT_DIR / 'latest.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Iterative online training for Marjapussi AI")
    p.add_argument("--rounds",           type=int,   default=100)
    p.add_argument("--games-per-round",  type=int,   default=200,
                   help="Games collected each round before training")
    p.add_argument("--workers",          type=int,   default=min(8, os.cpu_count() or 4),
                   help="Parallel game workers (each runs own ml_server)")
    p.add_argument("--mc-rollouts",      type=int,   default=4,
                   help="MC rollouts per action at decision points (0=disabled)")
    p.add_argument("--train-batch",      type=int,   default=256)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint",       default=None)
    p.add_argument("--eval-every",       type=int,   default=10,
                   help="Evaluate win rate every N rounds")
    args = p.parse_args()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_online(
        rounds=args.rounds,
        games_per_round=args.games_per_round,
        workers=args.workers,
        mc_rollouts=args.mc_rollouts,
        train_batch=args.train_batch,
        lr=args.lr,
        device=args.device,
        checkpoint=args.checkpoint,
        eval_every=args.eval_every,
    )
