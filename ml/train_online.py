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

import argparse, collections, json, math, os, queue, random, sys, threading, time, warnings
import ctypes

# Prevent Windows from going to sleep during multi-day runs
if sys.platform == "win32":
    try:
        # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as e:
        print(f"\033[93m[WARN]\033[0m Failed to set Windows keep-awake state: {e}")

# Silencing persistent PyTorch transformer warning regarding nested tensors
warnings.filterwarnings("ignore", ".*enable_nested_tensor is True.*")
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from model import MarjapussiNet
from env import MarjapussiEnv, obs_to_tensors
from self_play import compute_advantages

CKPT_DIR = Path(__file__).parent / "checkpoints"
RUNS_DIR  = Path(__file__).parent / "runs"
CKPT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

from train.utils import Log, Transition
from train.pool import BatchInferenceServer, EnvPool
from train.loss import train_step


def run_episode(env: MarjapussiEnv, model, device, mc_rollouts: int, stage: int,
                inference_server, start_trick: int = None, verbose_timing: bool = False, seed: int = None,
                force_heuristic_bidding: bool = False,
                force_heuristic_passing: bool = False) -> list[Transition]:
    """
    Play one complete game. Returns a list of Transitions.
    stage 0 = heuristic (bootstrap), 1 = model.
    """
    transitions = []
    t_step_total = 0.0
    t_adv_total = 0.0
    t_inf_total = 0.0
    n_adv_calls = 0
    try:
        obs = env.reset(start_trick=start_trick, seed=seed)
        steps = 0
        info = {}
        while not env.done and steps < 300:
            steps += 1
            legal = env.legal_actions
            if not legal:
                break

            t0_step = time.perf_counter()

            # Compute tensors ONCE per step (reused for inference & storage)
            t_before = obs_to_tensors(obs)

            active_player = obs.get("active_player", 0)

            # Choose action
            is_bid = bool(legal and legal[0].get("action_token") in (41, 42))
            is_pass = bool(legal and legal[0].get("action_token") == 43)
            
            # ALWAYS query the model for value prediction so GAE mathematically baselines correctly!
            model_action_pos, val_pred, model_log_prob = inference_server.infer(t_before)
            
            if stage == 0 or len(legal) == 1 or (is_bid and force_heuristic_bidding) or (is_pass and force_heuristic_passing):
                action_pos = env.get_heuristic_action() if (stage == 0 or is_bid or is_pass) else 0
                # Use a neutral log_prob for heuristic actions to cleanly guide behavioral cloning
                log_prob = -1.0 
            else:
                action_pos = min(model_action_pos, len(legal) - 1)
                log_prob = model_log_prob

            # Targeted Curriculum Branching: only branch at the specific learning trick
            # Targeted Curriculum Branching: only branch at the specific learning trick
            advantage = 0.0
            trick_no = obs.get('trick_number', 1)
            
            # Infer the true phase based on action tokens because trick_number defaults to 1 early on.
            # Bidding actions are typically ACT_BID (41) or ACT_PASS_STOP (42).
            # Passing actions are typically ACT_PASS_CARDS (43).
            is_bidding = False
            is_passing = False
            if len(legal) > 0:
                first_tok = legal[0].get("action_token", 0)
                if first_tok in [41, 42]:
                    is_bidding = True
                elif first_tok == 43:
                    is_passing = True
            
            if start_trick == -1:
                is_target = is_bidding
            elif start_trick == 0:
                is_target = is_passing
            else:
                is_target = (trick_no == start_trick) and not is_bidding and not is_passing

            if len(legal) > 1 and mc_rollouts > 0:
                # No stochastic branching — it's causing unpredictable overhead
                if is_target:
                    t0_adv = time.perf_counter()
                    try:
                        advs = env.get_advantages(policy="heuristic", num_rollouts=mc_rollouts)
                        if advs and action_pos < len(advs):
                            # The Rust environment calculates advantages from the physical perspective of POV (Seat 0).
                            # If the active player making this decision is on the opposing team, an action
                            # that is "good" for POV (+advantage) is actively BAD for the opponent! We MUST invert it.
                            raw_adv = advs[action_pos]
                            advantage = raw_adv if (active_player % 2 == env.pov % 2) else -raw_adv
                    except Exception:
                        pass
                    t_adv_total += time.perf_counter() - t0_adv
                    n_adv_calls += 1

            prev_my_pts = obs.get("points_my_team", 0)
            prev_opp_pts = obs.get("points_opp_team", 0)

            obs, done, info = env.step(legal[action_pos]["action_list_idx"])
            t_step_total += time.perf_counter() - t0_step

            # Calculate immediate tick reward based on trick point diffs
            my_diff = obs.get("points_my_team", 0) - prev_my_pts
            opp_diff = obs.get("points_opp_team", 0) - prev_opp_pts
            imm_r = (my_diff - opp_diff) / 420.0
            if active_player % 2 != env.pov % 2:
                imm_r = -imm_r

            transitions.append(Transition(
                obs=t_before,
                action_idx=action_pos,
                advantage=advantage,
                pts_my=0.0,
                pts_opp=0.0,
                value=val_pred,
                active_player=active_player,
                log_prob=log_prob,
                imm_r=imm_r,
            ))

        if verbose_timing and steps > 0:
            avg_step = 1000 * t_step_total / steps
            avg_inf  = 1000 * t_inf_total / steps
            # Percentage of time spent in Rust branching if it occurred
            branch_pct = (t_adv_total / t_step_total * 100.0) if t_step_total > 0 else 0
            
            Log.sim(
                f"Episode stats: {steps} steps | "
                f"StepAvg: {avg_step:.1f}ms (Inf: {avg_inf:.1f}ms) | "
                f"AdvCalls: {n_adv_calls} | "
                f"BranchingWork: {branch_pct:.1f}%"
            )

        # Fill in final outcome for all transitions
        if info and "team_points" in info:
            pts_team0 = info["team_points"][0]
            pts_team1 = info["team_points"][1]
            pts0 = pts_team0 / 420.0
            pts1 = pts_team1 / 420.0
            
            playing_party = info.get("playing_party")
            game_value = info.get("game_value", 0) / 420.0 # Normalize game value to [-0.5, 0.5] approximate range (e.g. 140/420 = 0.33)
            schwarz = info.get("schwarz", False)
            won_contract = info.get("won", False)
            
            # 1. Objective Alignment: RL Reward should be based on Game Value Swing, not raw tricks.
            if playing_party is not None:
                # Base contract points
                reward_magnitude = game_value
                if schwarz:
                    reward_magnitude *= 2.0 # Schwarz doubles game value swing

                # If the attackers won the contract, they get +reward, defenders get -reward.
                # If the attackers failed their contract, they get -reward, defenders get +reward.
                if won_contract:
                    diff0 = reward_magnitude if playing_party == 0 else -reward_magnitude
                else:
                    diff0 = -reward_magnitude if playing_party == 0 else reward_magnitude
            else:
                # Passgame (no one played). Winner is whoever has more trick points.
                base_passgame_reward = 115.0 / 420.0
                if pts_team0 > pts_team1:
                    diff0 = base_passgame_reward
                elif pts_team1 > pts_team0:
                    diff0 = -base_passgame_reward
                else:
                    diff0 = 0.0
            
            tricks_party_0 = sum(1 for t in info.get("tricks", []) if t["winner"] % 2 == 0)
            tricks_party_1 = sum(1 for t in info.get("tricks", []) if t["winner"] % 2 == 1)
        else:
            pts0 = pts1 = 0.5
            diff0 = 0.0
            playing_party = None
            tricks_party_0 = tricks_party_1 = 0

        # Subtract all the strictly intermediate rewards already distributed
        sum_imm_r_0 = sum(t.imm_r if t.active_player % 2 == 0 else -t.imm_r for t in transitions)
        remainder_0 = diff0 - sum_imm_r_0
        final_rewards = {0: remainder_0, 1: -remainder_0}
        
        # GAE calculation (gamma=0.99, lambda=0.95)
        gamma, lam = 0.99, 0.95
        gae_advs = [0.0] * len(transitions)
        returns  = [0.0] * len(transitions)
        last_gae = 0.0
        
        for t in reversed(range(len(transitions))):
            curr_team = transitions[t].active_player % 2
            v_t = transitions[t].value
            
            # Action's immediate reward points
            r_t = transitions[t].imm_r
            
            if t == len(transitions) - 1:
                next_v = 0.0
                r_t += final_rewards[curr_team] # give remaining unaccounted game points to last action
            else:
                next_team = transitions[t+1].active_player % 2
                next_v = transitions[t+1].value if next_team == curr_team else -transitions[t+1].value
                
            delta = r_t + gamma * next_v - v_t
            last_gae = delta + gamma * lam * last_gae
            gae_advs[t] = last_gae
            returns[t] = last_gae + v_t

        out_transitions = []
        for idx, t in enumerate(transitions):
            is_bid = bool(t.obs["action_feats"][0, t.action_idx, 1] > 0.5)
            curr_team = t.active_player % 2
            adv = t.advantage if abs(t.advantage) > 0.01 else gae_advs[idx]

            # Defender Bidding Mask: If the model didn't win the bid (defending party), 
            # only give it a bidding training signal if it played the attackers to 0 tricks.
            if is_bid and playing_party is not None and curr_team != playing_party:
                attacker_tricks = tricks_party_0 if playing_party == 0 else tricks_party_1
                if attacker_tricks > 0:
                    adv = 0.0

            processed_obs = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) 
                    else {sk: sv.detach().cpu() for sk, sv in v.items()} if isinstance(v, dict) 
                    else v) 
                for k, v in t.obs.items()
            }
            
            out_transitions.append(Transition(
                processed_obs, t.action_idx, adv,
                pts0 if t.active_player % 2 == 0 else pts1,
                pts1 if t.active_player % 2 == 0 else pts0,
                returns[idx], t.active_player, t.log_prob, t.imm_r
            ))

        return out_transitions
    except Exception as e:
        Log.error(f"Episode failed: {e}")
        return []


# ── Batch collation ───────────────────────────────────────────────────────────

def collate(transitions: list[Transition], device: str):
    if not transitions:
        return None

    B = int(len(transitions))
    max_seq = int(max(int(t.obs["token_ids"].shape[1]) for t in transitions))
    max_act = int(max(int(t.obs["action_feats"].shape[1]) for t in transitions))

    obs_a = {}
    for k in transitions[0].obs["obs_a"]:
        obs_a[k] = torch.cat([t.obs["obs_a"][k] for t in transitions], 0).to(device, non_blocking=True)

    tok = torch.zeros((B, max_seq), dtype=torch.long)
    tmask = torch.ones((B, max_seq), dtype=torch.bool)
    for i, t in enumerate(transitions):
        L = int(t.obs["token_ids"].shape[1])
        tok[i, :L] = t.obs["token_ids"][0]; tmask[i, :L] = t.obs["token_mask"][0]

    af = torch.zeros((B, max_act, 51))
    am = torch.ones((B, max_act), dtype=torch.bool)
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
        "value":       torch.tensor([t.value for t in transitions]).to(device, non_blocking=True),
        "log_prob_old":torch.tensor([t.log_prob for t in transitions]).to(device, non_blocking=True),
        "pts_target":  torch.tensor([[t.pts_my, t.pts_opp] for t in transitions]).to(device, non_blocking=True),
    }





# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_deterministic(model, device, n=100) -> dict:
    model.eval()
    server = BatchInferenceServer(model, device, max_batch=1, greedy=True) # Serial deterministic eval
    env = MarjapussiEnv()
    
    total_pts_diff = 0.0
    valid_games = 0
    
    try:
        for seed in range(1, n + 1):
            eps = run_episode(env, model, device, mc_rollouts=0, stage=1, inference_server=server, seed=seed)
            if eps:
                last_t = eps[-1]
                diff = (last_t.pts_my - last_t.pts_opp) * 420.0
                total_pts_diff += diff
                valid_games += 1
    finally:
        if env.proc:
            env.proc.kill()
        server.stop()
        
    avg_diff = total_pts_diff / max(1, valid_games)
    return {"avg_diff": avg_diff}


# ── Main loop ─────────────────────────────────────────────────────────────────

def train_online(
    rounds: int        = 100,
    start_round: int   = 0,
    games_per_round: int = 200,
    workers: int        = 8,
    mc_rollouts: int    = 4,
    train_batch: int    = 256,
    lr: float           = 3e-4,
    device: str         = "cpu",
    checkpoint: str | None = None,
    eval_every: int     = 10,
    ppo_epochs: int     = 3,
):
    use_amp = device.startswith("cuda")
    model   = MarjapussiNet().to(device)
    scaler  = GradScaler("cuda", enabled=use_amp)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rounds)

    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")

    Log.success(f"Online training | rounds={rounds} | games/round={games_per_round} "
                f"| workers={workers} | MC-rollouts={mc_rollouts} | ppo_epochs={ppo_epochs} | device={device}")
    Log.info(f"Params: {model.param_count():,}\n")

    run_dir  = RUNS_DIR / f"online_{int(time.time())}"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.jsonl"
    (RUNS_DIR / "latest").unlink(missing_ok=True)
    try: (RUNS_DIR / "latest").symlink_to(run_dir, target_is_directory=True)
    except: pass

    best_diff = -9999.0

    for rnd in range(start_round, rounds):
        t0 = time.time()
        stage = 0 if rnd < 3 else 1   # first 3 rounds: heuristic bootstrap

        # ── Collect games in parallel via threads ────────────────────────────
        Log.phase(f"Round {rnd+1}: Simulation")
        print(f"\033[94m[SIM]\033[0m Starting simulation with {workers} workers...", flush=True)
        sim_model = MarjapussiNet().to("cpu")
        sim_model.load_state_dict(model.state_dict())
        sim_model.eval()
        
        server = BatchInferenceServer(sim_model, "cpu", max_batch=min(workers * 4, 128))
        pool_envs = EnvPool(workers)
        all_transitions: list[Transition] = []

        # Curriculum logic
        progress = rnd / max(1, rounds)
        if progress < 0.30:
            train_phase = "trick"
            curriculum_trick = max(1, 8 - int(8 * (progress / 0.30)))
        else:
            if random.random() < 0.10:
                # 10% All Tricks Refresher
                train_phase = "trick"
                curriculum_trick = 8
            elif progress < 0.60:
                train_phase = "passing"
                curriculum_trick = 0
            elif progress < 0.80:
                train_phase = "bidding_value"
                curriculum_trick = -1
            else:
                train_phase = "bidding_prop"
                curriculum_trick = -1
        
        completed_games = 0
        progress_lock = threading.Lock()

        progress = rnd / max(1, rounds)
        force_bidding = progress < 0.90
        force_passing = progress < 0.30
        
        def collect_one(game_idx):
            env = pool_envs.get()
            try:
                res = run_episode(env, model, device, mc_rollouts, stage, server,
                                  start_trick=curriculum_trick,
                                  verbose_timing=(game_idx == 0),
                                  force_heuristic_bidding=force_bidding,
                                  force_heuristic_passing=force_passing)
            finally:
                pool_envs.put(env)
            
            nonlocal completed_games
            with progress_lock:
                completed_games += 1
                if completed_games % 1 == 0 or completed_games == games_per_round:
                    elapsed = time.time() - t0
                    gps = completed_games / max(elapsed, 0.1)
                    Log.sim(f"Round Progress: {completed_games}/{games_per_round} ({gps:.1f} games/s) | TargetTrick: {curriculum_trick}", end="")
            return res

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(collect_one, range(games_per_round)))
        print() # Newline after progress bar
        server.stop()
        pool_envs.close()

        for eps in results:
            all_transitions.extend(eps)

        random.shuffle(all_transitions)
        n_trans = len(all_transitions)
        gen_time = time.time() - t0

        if n_trans == 0:
            Log.warn(f"[Round {rnd+1}] No transitions collected — skipping")
            continue

        # ── Train on collected batch ─────────────────────────────────────────
        Log.phase(f"Round {rnd+1}: Optimization")
        model.train()
        t1 = time.time()
        loss_acc = collections.defaultdict(float)
        n_steps = 0

        for epoch in range(ppo_epochs):
            random.shuffle(all_transitions)
            for start in range(0, n_trans, train_batch):
                chunk = all_transitions[start:start + train_batch]
                batch = collate(chunk, device)
                if batch is None: continue
                losses = train_step(model, opt, scaler, batch, use_amp, train_phase=train_phase)
                for k, v in losses.items(): loss_acc[k] += v
                n_steps += 1
                if n_steps % 10 == 0:
                    prog = min(1.0, (start + train_batch) / n_trans)
                    Log.opt(f"Optimizing Round Batch (Epoch {epoch+1}/{ppo_epochs}): {prog:.1%}", end="")
            
            # Print newline after each epoch if we logged opt progress
            print()
        sched.step()

        avg_loss = {k: v / max(n_steps, 1) for k, v in loss_acc.items()}
        train_time = time.time() - t1

        # ── Summary ──────────────────────────────────────────────────────────
        policy_label = "heuristic" if stage == 0 else f"model-v{rnd+1}"
        rate = games_per_round / gen_time
        Log.success(f"Round {rnd+1:3d} Summary:")
        print(f"  » Policy:   {policy_label}")
        print(f"  » Games:    {games_per_round} ({rate:.1f} games/s)")
        print(f"  » Samples:  {n_trans} transitions")
        print(f"  » Losses:   Total: {avg_loss.get('total',0):.4f} | Pol: {avg_loss.get('policy',0):.4f} | Val: {avg_loss.get('value',0):.4f} | Ent: {avg_loss.get('entropy',0):.4f} | Pts: {avg_loss.get('pts',0):.4f}")
        print(f"  » Time:     Sim: {gen_time:.1f}s | Opt: {train_time:.1f}s")

        log_entry = {"round": rnd+1, "stage": stage, "n_games": games_per_round,
                     "n_transitions": n_trans, "losses": avg_loss,
                     "gen_secs": gen_time, "train_secs": train_time}

        # ── Evaluate + checkpoint ────────────────────────────────────────────
        if (rnd + 1) % eval_every == 0 and stage == 1:
            eval_metrics = eval_deterministic(model, device, n=100)
            avg_diff = eval_metrics["avg_diff"]
            print(f"           → Point diff (100 games): {avg_diff:+.1f}")
            log_entry["avg_diff"] = avg_diff
            torch.save(model.state_dict(), CKPT_DIR / f"online_round_{rnd+1}.pt")
            if avg_diff > best_diff:
                best_diff = avg_diff
                torch.save(model.state_dict(), CKPT_DIR / "best.pt")
                print(f"           → New best: {best_diff:+.1f}")
                
        # 50k games checkpoint (250 rounds * 200 games/round)
        if (rnd + 1) % 250 == 0:
            games_played = (rnd + 1) * games_per_round
            ckpt_name = f"model_{games_played // 1000}k.pt"
            torch.save(model.state_dict(), CKPT_DIR / ckpt_name)
            Log.info(f"Saved 50k milestone checkpoint: {ckpt_name}")

        torch.save(model.state_dict(), CKPT_DIR / "latest.pt")
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "update", **log_entry}) + "\n")

        # Free GPU memory and clear garbage
        del results, all_transitions
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    Log.success(f"Done. Best test point diff: {best_diff:+.1f}")
    Log.info(f"Checkpoint: {CKPT_DIR / 'latest.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Iterative online training for Marjapussi AI")
    p.add_argument("--rounds",           type=int,   default=100)
    p.add_argument("--games-per-round",  type=int,   default=200,
                   help="Games collected each round before training")
    p.add_argument("--workers",          type=int,   default=min(8, os.cpu_count() or 4),
                   help="Parallel game workers (each runs own ml_server)")
    p.add_argument("--mc-rollouts",      type=int,   default=4,
                   help="MC rollouts per action at decision points (0=disabled)")
    p.add_argument("--train-batch",      type=int,   default=192,
                   help="Batch size for training optimization step")
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint",       default=None)
    p.add_argument("--start-round",      type=int,   default=0,
                   help="Round to start/resume from")
    p.add_argument("--eval-every",       type=int,   default=10,
                   help="Evaluate win rate every N rounds")
    p.add_argument("--ppo-epochs",       type=int,   default=3,
                   help="Number of PPO optimization passes over collected data")
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
        start_round=args.start_round,
        eval_every=args.eval_every,
        ppo_epochs=args.ppo_epochs,
    )
