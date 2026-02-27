"""
Iterative Online Training for Marjapussi AI
============================================
Each round:
  1. Play N games with the CURRENT model (quality improves every round)
  2. At each multi-choice decision: ask Rust for MC advantages (try_all_actions)
  3. Train on the batch (advantage-weighted BC + points regression)
  4. Save checkpoint; next round uses the improved policy

This means training data quality grows automatically; no stale 1M-game dumps.

Usage:
  python ml/train_online.py --rounds 50 --games-per-round 500 --workers 8 --mc-rollouts 8 --device cuda

Recommended starting point:
  python ml/train_online.py --rounds 100 --games-per-round 200 --workers 8 --mc-rollouts 4 --device cuda
"""

from __future__ import annotations

import argparse, collections, json, math, os, random, sys, threading, time, warnings, uuid
import ctypes
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

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
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from model import ACTION_FEAT_DIM
from env import MarjapussiEnv, SUPPORTED_OBS_SCHEMA_VERSION, obs_to_tensors
from model_factory import (
    build_checkpoint_payload,
    create_model,
    load_state_compatible,
    parse_checkpoint,
)

DEFAULT_CKPT_DIR = Path(__file__).parent / "checkpoints"
DEFAULT_RUNS_DIR = Path(__file__).parent / "runs"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "train_online.toml"

from train.utils import Log, Transition
from train.pool import BatchInferenceServer, EnvPool
from train.loss import train_step
from train.reward import RewardConfig, contract_reward_from_pov, point_delta_reward, pov_team_points


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def configure_torch_runtime(device: str, workers: int) -> None:
    """Apply safe runtime settings for better training throughput."""
    torch.set_float32_matmul_precision("high")

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Reduce CPU over-subscription when many game workers are active.
    cpu_count = os.cpu_count() or 4
    if workers > 1:
        per_worker_threads = max(1, cpu_count // workers)
        torch.set_num_threads(max(1, min(4, per_worker_threads)))


def _load_config_file(path: str | Path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    cfg = _load_config_file(args.config)
    if not cfg:
        return args

    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest not in {"help"}
    }
    for key, value in cfg.items():
        if key not in defaults:
            continue
        if getattr(args, key) == defaults[key]:
            setattr(args, key, value)
    return args


def _is_bidding_action(legal: list[dict]) -> bool:
    return bool(legal and legal[0].get("action_token") in (41, 42))


def _is_passing_action(legal: list[dict]) -> bool:
    return bool(legal and legal[0].get("action_token") in (43, 52))

def _is_info_action(legal: list[dict]) -> bool:
    if not legal:
        return False
    # Trump/Q&A family (ask/answer/announce)
    return any(la.get("action_token") in (44, 45, 46, 47, 48, 49, 50) for la in legal)


def atomic_torch_save(payload: Any, path: Path, retries: int = 5, delay_s: float = 0.05) -> None:
    """
    Atomically publish checkpoints so readers never see partially-written files.
    Writes to a temp file in the same directory and then replaces target.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")

    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(max(0.0, delay_s))

    raise RuntimeError(f"Failed to atomically save checkpoint to {path}: {last_err}")


def run_episode(
    env: MarjapussiEnv,
    model,
    device,
    mc_rollouts: int,
    stage: int,
    inference_server,
    start_trick: int = None,
    verbose_timing: bool = False,
    seed: int = None,
    force_heuristic_bidding: bool = False,
    force_heuristic_passing: bool = False,
    pov: int | None = None,
    adv_query_mode: str = "target",
    adv_non_target_prob: float = 0.0,
    max_adv_calls_per_episode: int = 1,
    max_pass_adv_calls_per_episode: int = 4,
    max_info_adv_calls_per_episode: int = 2,
    reward_cfg: RewardConfig = RewardConfig(),
) -> list[Transition]:
    """
    Play one complete game and return training transitions for POV-controlled decisions only.
    stage 0: heuristic actions for POV (bootstrap data collection)
    stage 1: model actions for POV
    Non-POV seats always use heuristic policy to avoid hidden-information leakage.
    """
    transitions: list[Transition] = []
    pending: Transition | None = None
    t_step_total = 0.0
    t_adv_total = 0.0
    n_adv_calls = 0
    n_pass_adv_calls = 0
    n_info_adv_calls = 0

    try:
        episode_pov = pov if pov is not None else (random.randrange(4) if stage >= 1 else 0)
        obs = env.reset(pov=episode_pov, start_trick=start_trick, seed=seed)
        steps = 0
        info: dict = {}

        while not env.done and steps < 300:
            steps += 1
            legal = env.legal_actions
            if not legal:
                break

            t0_step = time.perf_counter()
            active_player = int(obs.get("active_player", 0))
            controls_turn = (active_player == 0)  # relative to POV

            # Transition reward is accumulated over all environment steps until POV acts again.
            if controls_turn and pending is not None:
                transitions.append(pending)
                pending = None

            is_bid = _is_bidding_action(legal)
            is_pass = _is_passing_action(legal)
            is_info = _is_info_action(legal)
            action_pos = 0

            if controls_turn:
                t_before = obs_to_tensors(obs, env.labels)
                model_action_pos, val_pred, model_log_prob = inference_server.infer(t_before)
                force_heuristic_turn = (
                    stage == 0
                    or len(legal) == 1
                    or (is_bid and force_heuristic_bidding)
                    or (is_pass and force_heuristic_passing)
                )

                if force_heuristic_turn:
                    action_pos = env.get_heuristic_action() if len(legal) > 1 else 0
                    log_prob = -1.0
                else:
                    action_pos = min(model_action_pos, len(legal) - 1)
                    log_prob = model_log_prob

                advantage = 0.0
                if len(legal) > 1 and mc_rollouts > 0:
                    trick_no = obs.get("trick_number", 1)
                    if start_trick == -1:
                        is_target = is_bid
                    elif start_trick == 0:
                        is_target = is_pass
                    else:
                        is_target = ((trick_no == start_trick) and not is_bid and not is_pass) or is_info

                    should_query = False
                    if is_pass:
                        can_query_adv = n_pass_adv_calls < max_pass_adv_calls_per_episode
                    elif is_info:
                        can_query_adv = n_info_adv_calls < max_info_adv_calls_per_episode
                    else:
                        can_query_adv = n_adv_calls < max_adv_calls_per_episode
                    if can_query_adv:
                        if adv_query_mode == "all":
                            should_query = True
                        elif adv_query_mode == "stochastic":
                            should_query = random.random() < adv_non_target_prob
                        elif adv_query_mode == "target_plus_stochastic":
                            should_query = is_target or (random.random() < adv_non_target_prob)
                        else:
                            # default: "target"
                            should_query = is_target

                    if should_query:
                        t0_adv = time.perf_counter()
                        try:
                            advs = env.get_advantages(policy="heuristic", num_rollouts=mc_rollouts)
                            if advs and action_pos < len(advs):
                                advantage = advs[action_pos]
                        except Exception:
                            pass
                        t_adv_total += time.perf_counter() - t0_adv
                        n_adv_calls += 1
                        if is_pass:
                            n_pass_adv_calls += 1
                        if is_info:
                            n_info_adv_calls += 1

                pending = Transition(
                    obs=t_before,
                    action_idx=action_pos,
                    advantage=advantage,
                    pts_my=0.0,
                    pts_opp=0.0,
                    value=val_pred,
                    active_player=active_player,
                    log_prob=log_prob,
                    is_forced=force_heuristic_turn,
                    imm_r=0.0,
                )
            else:
                action_pos = env.get_heuristic_action() if len(legal) > 1 else 0

            prev_obs = obs
            obs, _, info = env.step(legal[action_pos]["action_list_idx"])
            t_step_total += time.perf_counter() - t0_step

            # Reward from POV perspective; accumulate until next POV decision.
            step_reward = point_delta_reward(prev_obs, obs, reward_cfg)
            if pending is not None:
                pending = pending._replace(imm_r=pending.imm_r + step_reward)

        if pending is not None:
            transitions.append(pending)

        if (not info or "team_points" not in info) and not env.done:
            info = env.run_to_end("heuristic")

        if verbose_timing and steps > 0:
            avg_step = 1000 * t_step_total / steps
            branch_pct = (t_adv_total / t_step_total * 100.0) if t_step_total > 0 else 0.0
            Log.sim(
                f"Episode stats: {steps} steps | "
                f"StepAvg: {avg_step:.1f}ms | "
                f"AdvCalls: {n_adv_calls} (pass={n_pass_adv_calls},info={n_info_adv_calls}) | "
                f"BranchingWork: {branch_pct:.1f}%"
            )

        pov_party = env.pov % 2
        if info and "team_points" in info:
            pts_my, pts_opp = pov_team_points(info, pov_party)
            pts_my_norm = pts_my / reward_cfg.points_normalizer
            pts_opp_norm = pts_opp / reward_cfg.points_normalizer
            terminal_diff, tricks_party_0, tricks_party_1, playing_party = contract_reward_from_pov(
                info, pov_party, reward_cfg
            )
        else:
            pts_my_norm = 0.5
            pts_opp_norm = 0.5
            terminal_diff = 0.0
            tricks_party_0 = 0
            tricks_party_1 = 0
            playing_party = None

        if not transitions:
            return []

        sum_imm = sum(t.imm_r for t in transitions)
        remainder = terminal_diff - sum_imm

        gamma, lam = 0.99, 0.95
        gae_advs = [0.0] * len(transitions)
        returns = [0.0] * len(transitions)
        last_gae = 0.0

        for t in reversed(range(len(transitions))):
            v_t = transitions[t].value
            r_t = transitions[t].imm_r + (remainder if t == len(transitions) - 1 else 0.0)
            next_v = transitions[t + 1].value if t < len(transitions) - 1 else 0.0
            delta = r_t + gamma * next_v - v_t
            last_gae = delta + gamma * lam * last_gae
            gae_advs[t] = last_gae
            returns[t] = last_gae + v_t

        out_transitions = []
        for idx, t in enumerate(transitions):
            is_bid = bool(t.obs["action_feats"][0, t.action_idx, 1] > 0.5)
            adv = t.advantage if abs(t.advantage) > 0.01 else gae_advs[idx]

            # Defender bidding mask:
            # only keep defender bid training signal when attackers are held to 0 tricks.
            if is_bid and playing_party is not None and pov_party != playing_party:
                attacker_tricks = tricks_party_0 if playing_party == 0 else tricks_party_1
                if attacker_tricks > 0:
                    adv = 0.0

            processed_obs = {
                k: (
                    v.detach().cpu() if isinstance(v, torch.Tensor)
                    else {sk: sv.detach().cpu() for sk, sv in v.items()} if isinstance(v, dict)
                    else v
                )
                for k, v in t.obs.items()
            }

            out_transitions.append(Transition(
                processed_obs,
                t.action_idx,
                adv,
                pts_my_norm,
                pts_opp_norm,
                returns[idx],
                t.active_player,
                t.log_prob,
                t.is_forced,
                t.imm_r,
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

    af = torch.zeros((B, max_act, ACTION_FEAT_DIM))
    am = torch.ones((B, max_act), dtype=torch.bool)
    for i, t in enumerate(transitions):
        A = t.obs["action_feats"].shape[1]
        af[i, :A] = t.obs["action_feats"][0]; am[i, :A] = t.obs["action_mask"][0]

    ai = torch.tensor(
        [min(t.action_idx, transitions[i].obs["action_feats"].shape[1]-1)
         for i, t in enumerate(transitions)], dtype=torch.long)

    hidden_target = torch.cat(
        [t.obs["hidden_target"] for t in transitions], 0
    ).to(device, non_blocking=True)
    hidden_possible = torch.cat(
        [t.obs["hidden_possible"] for t in transitions], 0
    ).to(device, non_blocking=True)
    hidden_known = torch.cat(
        [t.obs["hidden_known"] for t in transitions], 0
    ).to(device, non_blocking=True)

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
        "is_forced":   torch.tensor([float(t.is_forced) for t in transitions]).to(device, non_blocking=True),
        "pts_target":  torch.tensor([[t.pts_my, t.pts_opp] for t in transitions]).to(device, non_blocking=True),
        "hidden_target": hidden_target,
        "hidden_possible": hidden_possible,
        "hidden_known": hidden_known,
    }





# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_deterministic(model, device, n=100, reward_cfg: RewardConfig = RewardConfig()) -> dict:
    model.eval()
    server = BatchInferenceServer(model, device, max_batch=1, greedy=True) # Serial deterministic eval
    env = MarjapussiEnv(include_labels=False)
    
    total_pts_diff = 0.0
    valid_games = 0
    
    try:
        for seed in range(1, n + 1):
            eps = run_episode(
                env,
                model,
                device,
                mc_rollouts=0,
                stage=1,
                inference_server=server,
                seed=seed,
                pov=(seed - 1) % 4,
                reward_cfg=reward_cfg,
            )
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
    model_family: str = "legacy",
    model_config: str | None = None,
    strict_param_budget: int | None = None,
    eval_every: int     = 10,
    save_latest_every: int = 16,
    ppo_epochs: int     = 3,
    min_ppo_epochs: int = 2,
    balance_opt_time: bool = False,
    target_opt_sim_ratio: float = 1.0,
    max_ppo_epochs: int = 5,
    target_kl: float = 0.03,
    max_clipfrac: float = 0.40,
    min_policy_improve: float = 0.001,
    opt_early_stop_patience: int = 1,
    adv_query_mode: str = "target",
    adv_non_target_prob: float = 0.0,
    max_adv_calls_per_episode: int = 1,
    max_pass_adv_calls_per_episode: int = 4,
    max_info_adv_calls_per_episode: int = 2,
    force_passing_until_progress: float = 0.55,
    force_bidding_until_progress: float = 0.90,
    hidden_loss_weight: float = 0.3,
    impossible_penalty_weight: float = 2.0,
    hidden_count_weight: float = 0.1,
    hidden_known_weight: float = 0.5,
    hidden_exclusive_weight: float = 0.5,
    forced_imitation_weight: float = 0.5,
    forced_imitation_bid_mult: float = 1.5,
    forced_imitation_pass_mult: float = 2.5,
    forced_imitation_decay_rounds: int = 128,
    named_checkpoint: str | None = None,
    points_normalizer: float = 420.0,
    passgame_base_reward: float = 115.0 / 420.0,
    step_delta_scale: float = 1.0 / 420.0,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
):
    configure_torch_runtime(device=device, workers=workers)
    use_amp = device.startswith("cuda")
    model, model_meta = create_model(
        model_family=model_family,
        model_config_path=model_config,
        strict_param_budget=strict_param_budget,
    )
    model = model.to(device)
    scaler  = GradScaler("cuda", enabled=use_amp)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rounds)
    reward_cfg = RewardConfig(
        points_normalizer=points_normalizer,
        passgame_base_reward=passgame_base_reward,
        step_delta_scale=step_delta_scale,
    )
    ckpt_dir = Path(checkpoints_dir)
    runs_dir = Path(runs_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint and Path(checkpoint).exists():
        state_dict, ckpt_meta, _ = parse_checkpoint(checkpoint, map_location=device)
        loaded, skipped = load_state_compatible(model, state_dict)
        ckpt_family = ckpt_meta.get("model_family")
        if ckpt_family and ckpt_family != model_meta.get("model_family"):
            Log.warn(
                f"Checkpoint family mismatch: ckpt={ckpt_family}, requested={model_meta.get('model_family')}. "
                f"Loaded compatible tensors only ({loaded} loaded, {skipped} skipped)."
            )
        else:
            print(f"Loaded checkpoint: {checkpoint} ({loaded} tensors, {skipped} skipped)")

    named_ckpt_path: Path | None = None
    if named_checkpoint:
        ncp = Path(named_checkpoint)
        named_ckpt_path = ncp if ncp.is_absolute() else (ckpt_dir / ncp)
        named_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        Log.info(f"Named checkpoint target: {named_ckpt_path}")

    Log.success(f"Online training | rounds={rounds} | games/round={games_per_round} "
                f"| workers={workers} | MC-rollouts={mc_rollouts} | ppo_epochs={ppo_epochs} | device={device}")
    Log.info(
        f"Model family: {model_meta.get('model_family')} | "
        f"config_hash={model_meta.get('model_config_hash')} | "
        f"params={model.param_count():,}"
    )
    Log.info(
        "Optimization control: "
        f"min_epochs={min_ppo_epochs}, base_epochs={ppo_epochs}, max_epochs={max_ppo_epochs}, "
        f"target_kl={target_kl:.3f}, max_clipfrac={max_clipfrac:.2f}, "
        f"min_policy_improve={min_policy_improve:.4f}, patience={opt_early_stop_patience}"
    )
    Log.info(
        f"Checkpoint cadence: eval_every={max(1, eval_every)} rounds, "
        f"save_latest_every={max(1, save_latest_every)} rounds"
    )
    if balance_opt_time:
        Log.info(f"Time balancing enabled: target opt/sim ratio={target_opt_sim_ratio:.2f}")
    Log.info(
        f"Adv branching: mode={adv_query_mode}, non_target_prob={adv_non_target_prob:.2f}, "
        f"max_calls_per_episode={max_adv_calls_per_episode}, "
        f"max_pass_calls_per_episode={max_pass_adv_calls_per_episode}, "
        f"max_info_calls_per_episode={max_info_adv_calls_per_episode}"
    )
    Log.info(
        f"Hidden-state aux loss: weight={hidden_loss_weight:.2f}, "
        f"impossible_penalty={impossible_penalty_weight:.2f}, "
        f"count_weight={hidden_count_weight:.2f}, "
        f"known_weight={hidden_known_weight:.2f}, "
        f"exclusive_weight={hidden_exclusive_weight:.2f}"
    )
    Log.info(
        f"Forced-action imitation: base_weight={forced_imitation_weight:.2f}, "
        f"bid_mult={forced_imitation_bid_mult:.2f}, pass_mult={forced_imitation_pass_mult:.2f}, "
        f"decay_rounds={forced_imitation_decay_rounds}"
    )
    Log.info(
        f"Reward knobs: points_norm={reward_cfg.points_normalizer:.1f}, "
        f"passgame_base={reward_cfg.passgame_base_reward:.4f}, "
        f"step_delta_scale={reward_cfg.step_delta_scale:.6f}"
    )
    Log.info(f"Params: {model.param_count():,}\n")

    run_dir  = runs_dir / f"online_{int(time.time())}"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.jsonl"
    (runs_dir / "latest").unlink(missing_ok=True)
    try: (runs_dir / "latest").symlink_to(run_dir, target_is_directory=True)
    except: pass

    best_diff = -9999.0
    train_start = time.time()
    total_rounds = max(1, rounds - start_round)

    for rnd in range(start_round, rounds):
        t0 = time.time()
        local_round = (rnd - start_round) + 1
        stage = 0 if rnd < 3 else 1   # first 3 rounds: heuristic bootstrap

        # ── Collect games in parallel via threads ────────────────────────────
        Log.phase(f"Round {rnd+1}/{rounds}: Simulation")
        print(f"\033[94m[SIM]\033[0m Starting simulation with {workers} workers...", flush=True)
        sim_model, _ = create_model(
            model_family=model_meta.get("model_family", "legacy"),
            model_config_path=model_meta.get("model_config_path"),
            strict_param_budget=strict_param_budget,
        )
        sim_model = sim_model.to("cpu")
        sim_model.load_state_dict(model.state_dict())
        sim_model.eval()
        
        server = BatchInferenceServer(sim_model, "cpu", max_batch=min(workers * 4, 128))
        pool_envs = EnvPool(workers, include_labels=True)
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

        force_passing = progress < force_passing_until_progress
        force_bidding = progress < force_bidding_until_progress
        
        def collect_one(game_idx):
            env = pool_envs.get()
            replacement_env = env
            try:
                res = run_episode(env, model, device, mc_rollouts, stage, server,
                                  start_trick=curriculum_trick,
                                  verbose_timing=(game_idx == 0),
                                  force_heuristic_bidding=force_bidding,
                                  force_heuristic_passing=force_passing,
                                  adv_query_mode=adv_query_mode,
                                  adv_non_target_prob=adv_non_target_prob,
                                  max_adv_calls_per_episode=max_adv_calls_per_episode,
                                  max_pass_adv_calls_per_episode=max_pass_adv_calls_per_episode,
                                  max_info_adv_calls_per_episode=max_info_adv_calls_per_episode,
                                  reward_cfg=reward_cfg)
            finally:
                if not env.is_alive():
                    rc = env.return_code()
                    rc_hex = f"0x{(rc & 0xFFFFFFFF):08X}" if rc is not None else "n/a"
                    Log.warn(
                        f"Worker env died (rc={rc}, {rc_hex}); creating replacement ml_server instance."
                    )
                    try:
                        env.close()
                    except Exception:
                        pass
                    replacement_env = MarjapussiEnv(include_labels=True)
                pool_envs.put(replacement_env)
            
            nonlocal completed_games
            with progress_lock:
                completed_games += 1
                if completed_games % 1 == 0 or completed_games == games_per_round:
                    elapsed = time.time() - t0
                    gps = completed_games / max(elapsed, 0.1)
                    games_left = max(0, games_per_round - completed_games)
                    eta_games = games_left / max(gps, 1e-6)
                    Log.sim(
                        f"Round {rnd+1}/{rounds} | "
                        f"Progress: {completed_games}/{games_per_round} ({gps:.1f} games/s) | "
                        f"ETA: {_format_eta(eta_games)} | "
                        f"TargetTrick: {curriculum_trick}",
                        end=""
                    )
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
            Log.warn(f"[Round {rnd+1}] No transitions collected - skipping")
            continue

        # ── Train on collected batch ─────────────────────────────────────────
        Log.phase(f"Round {rnd+1}/{rounds}: Optimization")
        model.train()
        t1 = time.time()
        loss_acc = collections.defaultdict(float)
        n_steps = 0
        epoch = 0
        target_opt_secs = gen_time * max(0.0, target_opt_sim_ratio) if balance_opt_time else None
        required_epochs = max(1, ppo_epochs, min_ppo_epochs)
        max_epochs = max(required_epochs, max_ppo_epochs)
        no_improve_epochs = 0
        prev_epoch_policy: float | None = None
        invalid_batch_detected = False
        while epoch < max_epochs:
            epoch_acc = collections.defaultdict(float)
            epoch_steps = 0

            random.shuffle(all_transitions)
            for start in range(0, n_trans, train_batch):
                chunk = all_transitions[start:start + train_batch]
                batch = collate(chunk, device)
                if batch is None: continue
                losses = train_step(
                    model,
                    opt,
                    scaler,
                    batch,
                    use_amp,
                    train_phase=train_phase,
                    hidden_loss_weight=hidden_loss_weight,
                    impossible_penalty_weight=impossible_penalty_weight,
                    hidden_count_weight=hidden_count_weight,
                    hidden_known_weight=hidden_known_weight,
                    hidden_exclusive_weight=hidden_exclusive_weight,
                    forced_imitation_weight=max(
                        0.05,
                        forced_imitation_weight
                        * max(0.0, 1.0 - (rnd / max(1, forced_imitation_decay_rounds))),
                    ),
                    forced_imitation_bid_mult=forced_imitation_bid_mult,
                    forced_imitation_pass_mult=forced_imitation_pass_mult,
                )
                if not math.isfinite(float(losses.get("total", float("nan")))):
                    invalid_batch_detected = True
                    Log.warn("Stopping optimization early due to non-finite batch loss.")
                    break
                for k, v in losses.items(): loss_acc[k] += v
                for k, v in losses.items(): epoch_acc[k] += v
                n_steps += 1
                epoch_steps += 1
                if n_steps % 10 == 0:
                    prog = min(1.0, (start + train_batch) / n_trans)
                    epoch_label = f"{epoch+1}/{max_epochs}"
                    Log.opt(f"Optimizing Round Batch (Epoch {epoch_label}): {prog:.1%}", end="")
            if invalid_batch_detected:
                print()
                break
            
            # Print newline after each epoch if we logged opt progress
            print()
            epoch += 1

            if epoch < required_epochs:
                continue

            epoch_avg = {k: (epoch_acc[k] / max(epoch_steps, 1)) for k in epoch_acc}
            cur_policy = float(epoch_avg.get("policy", 0.0))
            cur_kl = float(epoch_avg.get("approx_kl", 0.0))
            cur_clipfrac = float(epoch_avg.get("clipfrac", 0.0))

            if prev_epoch_policy is not None and (prev_epoch_policy - cur_policy) < min_policy_improve:
                no_improve_epochs += 1
            else:
                no_improve_epochs = 0
            prev_epoch_policy = cur_policy

            stop_reason = None
            if target_kl > 0 and cur_kl >= target_kl:
                stop_reason = f"KL {cur_kl:.4f} >= {target_kl:.4f}"
            elif max_clipfrac > 0 and cur_clipfrac >= max_clipfrac:
                stop_reason = f"clipfrac {cur_clipfrac:.3f} >= {max_clipfrac:.3f}"
            elif no_improve_epochs > opt_early_stop_patience:
                stop_reason = (
                    f"policy improvement below {min_policy_improve:.4f} for "
                    f"{no_improve_epochs} epochs"
                )
            elif balance_opt_time and target_opt_secs is not None and (time.time() - t1) >= target_opt_secs:
                stop_reason = f"reached time target ({target_opt_secs:.1f}s)"

            if stop_reason is not None:
                Log.opt(f"Stopping optimization at epoch {epoch}: {stop_reason}")
                break
        if n_steps > 0:
            sched.step()
        else:
            Log.warn("Skipping LR scheduler step because no optimizer updates were applied this round.")

        avg_loss = {k: v / max(n_steps, 1) for k, v in loss_acc.items()}
        train_time = time.time() - t1

        # ── Summary ──────────────────────────────────────────────────────────
        policy_label = "heuristic" if stage == 0 else f"model-v{rnd+1}"
        rate = games_per_round / gen_time
        Log.success(f"Round {rnd+1:3d} Summary:")
        elapsed_run = time.time() - train_start
        avg_round_time = elapsed_run / max(local_round, 1)
        rounds_left = max(0, total_rounds - local_round)
        eta_run = avg_round_time * rounds_left
        print(f"  - Policy:   {policy_label}")
        print(f"  - Round:    {rnd+1}/{rounds} (local {local_round}/{total_rounds})")
        print(f"  - RunETA:   {_format_eta(eta_run)} remaining (elapsed {_format_eta(elapsed_run)})")
        print(f"  - Games:    {games_per_round} ({rate:.1f} games/s)")
        print(f"  - Samples:  {n_trans} transitions")
        print(f"  - OptEpochs:{epoch}")
        print(f"  - Losses:   Total: {avg_loss.get('total',0):.4f} | Pol: {avg_loss.get('policy',0):.4f} | Imit: {avg_loss.get('forced_imitation',0):.4f} | Val: {avg_loss.get('value',0):.4f} | Ent: {avg_loss.get('entropy',0):.4f} | Pts: {avg_loss.get('pts',0):.4f} | Hidden: {avg_loss.get('hidden',0):.4f} | KL: {avg_loss.get('approx_kl',0):.4f} | Clip: {avg_loss.get('clipfrac',0):.3f}")
        print(f"  - HiddenAux: PosBCE: {avg_loss.get('hidden_pos_loss',0):.4f} | KnownBCE: {avg_loss.get('hidden_known_loss',0):.4f} | ImpBCE: {avg_loss.get('hidden_impossible_loss',0):.4f} | Count: {avg_loss.get('hidden_count_loss',0):.4f} | Exclusive: {avg_loss.get('hidden_exclusive_loss',0):.4f}")
        print(f"  - HiddenQ:   PosAcc: {avg_loss.get('hidden_pos_acc',0):.3f} | KnownAcc: {avg_loss.get('hidden_known_acc',0):.3f} | ExclusiveAcc: {avg_loss.get('hidden_exclusive_acc',0):.3f} | ImpossibleMass: {avg_loss.get('impossible_mass',0):.3f}")
        print(f"  - Time:     Sim: {gen_time:.1f}s | Opt: {train_time:.1f}s")

        log_entry = {"round": rnd+1, "stage": stage, "n_games": games_per_round,
                     "n_transitions": n_trans, "losses": avg_loss, "opt_epochs": epoch,
                     "gen_secs": gen_time, "train_secs": train_time}

        # ── Evaluate + checkpoint ────────────────────────────────────────────
        if (rnd + 1) % eval_every == 0 and stage == 1:
            eval_metrics = eval_deterministic(model, device, n=100, reward_cfg=reward_cfg)
            avg_diff = eval_metrics["avg_diff"]
            print(f"           -> Point diff (100 games): {avg_diff:+.1f}")
            log_entry["avg_diff"] = avg_diff
            atomic_torch_save(
                build_checkpoint_payload(
                    model,
                    metadata={
                        **model_meta,
                        "schema_version": SUPPORTED_OBS_SCHEMA_VERSION,
                        "action_encoding_version": 1,
                        "action_feat_dim": ACTION_FEAT_DIM,
                    },
                    extra_metadata={
                        "checkpoint_kind": "online_round",
                        "round": rnd + 1,
                        "avg_diff": avg_diff,
                    },
                ),
                ckpt_dir / f"online_round_{rnd+1}.pt",
            )
            if avg_diff > best_diff:
                best_diff = avg_diff
                atomic_torch_save(
                    build_checkpoint_payload(
                        model,
                        metadata={
                            **model_meta,
                            "schema_version": SUPPORTED_OBS_SCHEMA_VERSION,
                            "action_encoding_version": 1,
                            "action_feat_dim": ACTION_FEAT_DIM,
                        },
                        extra_metadata={
                            "checkpoint_kind": "best",
                            "round": rnd + 1,
                            "best_diff": best_diff,
                        },
                    ),
                    ckpt_dir / "best.pt",
                )
                print(f"           -> New best: {best_diff:+.1f}")
                
        # 50k games checkpoint (250 rounds * 200 games/round)
        if (rnd + 1) % 250 == 0:
            games_played = (rnd + 1) * games_per_round
            ckpt_name = f"model_{games_played // 1000}k.pt"
            atomic_torch_save(
                build_checkpoint_payload(
                    model,
                    metadata={
                        **model_meta,
                        "schema_version": SUPPORTED_OBS_SCHEMA_VERSION,
                        "action_encoding_version": 1,
                        "action_feat_dim": ACTION_FEAT_DIM,
                    },
                    extra_metadata={
                        "checkpoint_kind": "milestone",
                        "round": rnd + 1,
                        "games_played": games_played,
                    },
                ),
                ckpt_dir / ckpt_name,
            )
            Log.info(f"Saved 50k milestone checkpoint: {ckpt_name}")

        should_save_latest = (
            ((rnd + 1) % max(1, int(save_latest_every)) == 0) or ((rnd + 1) == rounds)
        )
        if should_save_latest:
            latest_payload = build_checkpoint_payload(
                model,
                metadata={
                    **model_meta,
                    "schema_version": SUPPORTED_OBS_SCHEMA_VERSION,
                    "action_encoding_version": 1,
                    "action_feat_dim": ACTION_FEAT_DIM,
                },
                extra_metadata={
                    "checkpoint_kind": "latest",
                    "round": rnd + 1,
                    "stage": stage,
                },
            )
            atomic_torch_save(latest_payload, ckpt_dir / "latest.pt")
            if named_ckpt_path is not None:
                atomic_torch_save(latest_payload, named_ckpt_path)
        with open(log_path, "a") as f:
            f.write(json.dumps({"event": "update", **log_entry}) + "\n")

        # Free GPU memory and clear garbage
        del results, all_transitions
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    Log.success(f"Done. Best test point diff: {best_diff:+.1f}")
    Log.info(f"Checkpoint: {ckpt_dir / 'latest.pt'}")
    if named_ckpt_path is not None:
        Log.info(f"Named checkpoint: {named_ckpt_path}")


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
    p.add_argument("--model-family",     choices=["legacy", "parallel_v2"], default="legacy",
                   help="Model family for training/inference")
    p.add_argument("--model-config",     default=None,
                   help="Optional model config path (used by parallel_v2)")
    p.add_argument("--strict-param-budget", type=int, default=None,
                   help="Optional strict max-parameter gate")
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint",       default=None)
    p.add_argument("--start-round",      type=int,   default=0,
                   help="Round to start/resume from")
    p.add_argument("--eval-every",       type=int,   default=10,
                   help="Evaluate win rate every N rounds")
    p.add_argument("--save-latest-every", type=int, default=16,
                   help="Save latest/named rolling checkpoints every N rounds (final round always saves)")
    p.add_argument("--ppo-epochs",       type=int,   default=3,
                   help="Number of PPO optimization passes over collected data")
    p.add_argument("--min-ppo-epochs",   type=int,   default=2,
                   help="Minimum PPO epochs per round before any early-stop condition can trigger")
    p.add_argument("--balance-opt-time", action="store_true",
                   help="Use simulation/optimization time ratio as an additional stop condition")
    p.add_argument("--target-opt-sim-ratio", type=float, default=1.0,
                   help="Optimization/simulation time ratio target when --balance-opt-time is enabled")
    p.add_argument("--max-ppo-epochs",   type=int,   default=5,
                   help="Hard cap for PPO epochs per round")
    p.add_argument("--target-kl",        type=float, default=0.03,
                   help="Early-stop PPO when per-epoch approx KL exceeds this")
    p.add_argument("--max-clipfrac",     type=float, default=0.40,
                   help="Early-stop PPO when per-epoch clip fraction exceeds this")
    p.add_argument("--min-policy-improve", type=float, default=0.001,
                   help="Minimum per-epoch policy-loss improvement to keep training the same round data")
    p.add_argument("--opt-early-stop-patience", type=int, default=1,
                   help="Allowed consecutive low-improvement epochs before early-stop")
    p.add_argument("--adv-query-mode",   type=str, default="target",
                   choices=["target", "target_plus_stochastic", "stochastic", "all"],
                   help="Where to run counterfactual advantage queries")
    p.add_argument("--adv-non-target-prob", type=float, default=0.0,
                   help="Probability to branch on non-target decisions (used by stochastic modes)")
    p.add_argument("--max-adv-calls-per-episode", type=int, default=1,
                   help="Hard cap of advantage-query calls per episode")
    p.add_argument("--max-pass-adv-calls-per-episode", type=int, default=4,
                   help="Hard cap of advantage-query calls per episode for passing decisions")
    p.add_argument("--max-info-adv-calls-per-episode", type=int, default=2,
                   help="Hard cap of advantage-query calls per episode for trump/Q&A decisions")
    p.add_argument("--force-passing-until-progress", type=float, default=0.55,
                   help="Force heuristic passing for progress ratio < this value")
    p.add_argument("--force-bidding-until-progress", type=float, default=0.90,
                   help="Force heuristic bidding for progress ratio < this value")
    p.add_argument("--hidden-loss-weight", type=float, default=0.3,
                   help="Weight of hidden-hand auxiliary loss in total loss")
    p.add_argument("--impossible-penalty-weight", type=float, default=2.0,
                   help="Penalty multiplier for predicting cards that are impossible by symbolic constraints")
    p.add_argument("--hidden-count-weight", type=float, default=0.1,
                   help="Weight for hidden-card count consistency loss on possible cards")
    p.add_argument("--hidden-known-weight", type=float, default=0.5,
                   help="Weight for known-card (symbolically confirmed) positive supervision loss")
    p.add_argument("--hidden-exclusive-weight", type=float, default=0.5,
                   help="Weight for per-card unique-seat assignment (set-theoretic exclusivity) loss")
    p.add_argument("--forced-imitation-weight", type=float, default=0.5,
                   help="Base weight for imitation loss on heuristic-forced actions")
    p.add_argument("--forced-imitation-bid-mult", type=float, default=1.5,
                   help="Extra multiplier for forced imitation on bidding actions")
    p.add_argument("--forced-imitation-pass-mult", type=float, default=2.5,
                   help="Extra multiplier for forced imitation on passing actions")
    p.add_argument("--forced-imitation-decay-rounds", type=int, default=128,
                   help="Rounds over which forced-action imitation weight decays")
    p.add_argument("--named-checkpoint", default=None,
                   help="Optional checkpoint filename/path to update every round")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                   help="TOML file for human-readable training defaults")
    p.add_argument("--points-normalizer", type=float, default=420.0,
                   help="Point normalization constant used by reward calculations")
    p.add_argument("--passgame-base-reward", type=float, default=(115.0 / 420.0),
                   help="Terminal reward magnitude for pass games")
    p.add_argument("--step-delta-scale", type=float, default=(1.0 / 420.0),
                   help="Scale for dense per-step point-delta reward")
    p.add_argument("--checkpoints-dir", default=str(DEFAULT_CKPT_DIR),
                   help="Directory for saving checkpoints (latest/best/round)")
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR),
                   help="Directory for run logs and latest symlink")
    args = p.parse_args()
    args = _apply_config_defaults(args, p)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_online(
        rounds=args.rounds,
        games_per_round=args.games_per_round,
        workers=args.workers,
        mc_rollouts=args.mc_rollouts,
        train_batch=args.train_batch,
        lr=args.lr,
        model_family=args.model_family,
        model_config=args.model_config,
        strict_param_budget=args.strict_param_budget,
        device=args.device,
        checkpoint=args.checkpoint,
        start_round=args.start_round,
        eval_every=args.eval_every,
        save_latest_every=args.save_latest_every,
        ppo_epochs=args.ppo_epochs,
        min_ppo_epochs=args.min_ppo_epochs,
        balance_opt_time=args.balance_opt_time,
        target_opt_sim_ratio=args.target_opt_sim_ratio,
        max_ppo_epochs=args.max_ppo_epochs,
        target_kl=args.target_kl,
        max_clipfrac=args.max_clipfrac,
        min_policy_improve=args.min_policy_improve,
        opt_early_stop_patience=args.opt_early_stop_patience,
        adv_query_mode=args.adv_query_mode,
        adv_non_target_prob=args.adv_non_target_prob,
        max_adv_calls_per_episode=args.max_adv_calls_per_episode,
        max_pass_adv_calls_per_episode=args.max_pass_adv_calls_per_episode,
        max_info_adv_calls_per_episode=args.max_info_adv_calls_per_episode,
        force_passing_until_progress=args.force_passing_until_progress,
        force_bidding_until_progress=args.force_bidding_until_progress,
        hidden_loss_weight=args.hidden_loss_weight,
        impossible_penalty_weight=args.impossible_penalty_weight,
        hidden_count_weight=args.hidden_count_weight,
        hidden_known_weight=args.hidden_known_weight,
        hidden_exclusive_weight=args.hidden_exclusive_weight,
        forced_imitation_weight=args.forced_imitation_weight,
        forced_imitation_bid_mult=args.forced_imitation_bid_mult,
        forced_imitation_pass_mult=args.forced_imitation_pass_mult,
        forced_imitation_decay_rounds=args.forced_imitation_decay_rounds,
        named_checkpoint=args.named_checkpoint,
        points_normalizer=args.points_normalizer,
        passgame_base_reward=args.passgame_base_reward,
        step_delta_scale=args.step_delta_scale,
        checkpoints_dir=args.checkpoints_dir,
        runs_dir=args.runs_dir,
    )
