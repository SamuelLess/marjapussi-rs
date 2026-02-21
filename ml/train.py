"""
Training script for Marjapussi ML agent.

Usage:
    python ml/train.py [--phase 1] [--games 1000] [--checkpoint path] [--smoke]

Bootstrapping curriculum (no human data needed):
    Stage 0 (games 0-199):    Random play — explore game mechanics
    Stage 1 (games 200-699):  Heuristic opponents — stable training signal
    Stage 2 (games 700+):     Self-play with counterfactual advantage
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Allow running ml/ scripts directly
sys.path.insert(0, str(Path(__file__).parent))

from model import MarjapussiNet
from self_play import run_episode, Transition

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
RUNS_DIR = Path(__file__).parent / "runs"


def collate_transitions(transitions: list[Transition]) -> Optional[dict]:
    """Collate a list of transitions into a batched model input."""
    if not transitions:
        return None

    # Stack all tensors - we need to handle variable sequence lengths
    batch = []
    max_seq = max(t.obs_tensors['token_ids'].shape[1] for t in transitions)
    max_actions = max(t.obs_tensors['action_feats'].shape[1] for t in transitions)

    obs_a_keys = list(transitions[0].obs_tensors['obs_a'].keys())

    stacked_obs_a = {}
    for key in obs_a_keys:
        stacked_obs_a[key] = torch.cat([t.obs_tensors['obs_a'][key] for t in transitions], dim=0)

    # Pad token sequences
    padded_tokens = torch.zeros(len(transitions), max_seq, dtype=torch.long)
    padded_mask = torch.ones(len(transitions), max_seq, dtype=torch.bool)
    for i, t in enumerate(transitions):
        L = t.obs_tensors['token_ids'].shape[1]
        padded_tokens[i, :L] = t.obs_tensors['token_ids'][0]
        padded_mask[i, :L] = t.obs_tensors['token_mask'][0]

    # Pad action features
    padded_actions = torch.zeros(len(transitions), max_actions, 51)
    padded_action_mask = torch.ones(len(transitions), max_actions, dtype=torch.bool)
    for i, t in enumerate(transitions):
        A = t.obs_tensors['action_feats'].shape[1]
        padded_actions[i, :A] = t.obs_tensors['action_feats'][0]
        padded_action_mask[i, :A] = t.obs_tensors['action_mask'][0]

    advantages = torch.tensor([t.advantage for t in transitions], dtype=torch.float32)
    action_indices = torch.tensor([t.action_idx for t in transitions], dtype=torch.long)

    # Points targets
    have_pts = [t for t in transitions if t.points_my_team_target is not None]
    if have_pts:
        pts_my = torch.tensor([t.points_my_team_target for t in have_pts], dtype=torch.float32)
        pts_opp = torch.tensor([t.points_opp_team_target for t in have_pts], dtype=torch.float32)
    else:
        pts_my = pts_opp = None

    return {
        'obs_a': stacked_obs_a,
        'token_ids': padded_tokens,
        'token_mask': padded_mask,
        'action_feats': padded_actions,
        'action_mask': padded_action_mask,
        'advantages': advantages,
        'action_indices': action_indices,
        'pts_my': pts_my,
        'pts_opp': pts_opp,
    }


def compute_loss(model: MarjapussiNet, batch: dict, device: str) -> dict[str, torch.Tensor]:
    """Compute combined PPO-style + auxiliary loss."""
    # Move to device
    b = {k: (v.to(device) if isinstance(v, torch.Tensor) else
             {kk: vv.to(device) for kk, vv in v.items()})
         for k, v in batch.items()
         if k not in ('advantages', 'action_indices', 'pts_my', 'pts_opp')}
    b['advantages'] = batch['advantages'].to(device)
    b['action_indices'] = batch['action_indices'].to(device)

    model_input = {
        'obs_a': b['obs_a'],
        'token_ids': b['token_ids'],
        'token_mask': b['token_mask'],
        'action_feats': b['action_feats'],
        'action_mask': b['action_mask'],
    }

    logits, card_logits, point_preds = model(model_input)  # [B,A], [B,3,36], [B,2]

    # Policy loss: advantage-weighted log-prob of chosen action
    log_probs = F.log_softmax(logits, dim=-1)  # [B, A]
    chosen_log_probs = log_probs.gather(1, b['action_indices'].unsqueeze(1)).squeeze(1)  # [B]
    policy_loss = -(chosen_log_probs * b['advantages']).mean()

    losses = {'policy': policy_loss, 'total': policy_loss}

    # Auxiliary: points regression
    if batch['pts_my'] is not None:
        pts_target = torch.stack([batch['pts_my'], batch['pts_opp']], dim=-1).to(device)  # [N, 2]
        # Use points from first N samples that have targets
        N = pts_target.shape[0]
        pts_pred = point_preds[:N]
        pts_loss = F.mse_loss(pts_pred, pts_target) * 0.1
        losses['points'] = pts_loss
        losses['total'] = losses['total'] + pts_loss

    return losses


def eval_vs_heuristic(model: MarjapussiNet, n_games: int = 50, device: str = 'cpu') -> float:
    """Estimate win rate of model vs heuristic opponents."""
    wins = 0
    for _ in range(n_games):
        ep = run_episode(model, stage=2, device=device)
        outcome = ep.outcome
        if outcome and outcome.get('won'):
            wins += 1
    return wins / n_games if n_games > 0 else 0.0


def train(
    phase: int = 1,
    total_games: int = 1000,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
    smoke: bool = False,
):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    # Run directory for this training session
    run_dir = RUNS_DIR / f"run_{int(time.time())}"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.jsonl"
    # Also write to "latest" symlink for UI
    latest = RUNS_DIR / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)

    model = MarjapussiNet().to(device)
    print(f"Model parameters: {model.param_count():,}")

    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    phase_start_trick = max(1, 10 - phase)  # phase 1 = last 5 tricks = trick 5+

    buffer: list[Transition] = []
    UPDATE_EVERY = 50  # PPO update frequency (games)
    EVAL_EVERY = 200
    best_win_rate = 0.0

    def log(entry: dict):
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    log({'event': 'start', 'phase': phase, 'total_games': total_games,
         'params': model.param_count()})

    for game_num in range(total_games):
        # Curriculum stage
        if game_num < 200:
            stage = 0
        elif game_num < 700:
            stage = 1
        else:
            stage = 2

        try:
            ep = run_episode(
                model=model if stage == 2 else None,
                stage=stage,
                device=device,
                phase_start_trick=phase_start_trick,
            )
            buffer.extend(ep.transitions)
        except Exception as e:
            print(f"  [game {game_num}] Error: {e}")
            continue

        # PPO update
        if (game_num + 1) % UPDATE_EVERY == 0 and buffer:
            model.train()
            batch = collate_transitions(buffer)
            if batch:
                optimizer.zero_grad()
                losses = compute_loss(model, batch, device)
                losses['total'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_vals = {k: float(v) for k, v in losses.items()}
                print(f"[Game {game_num+1}] Stage {stage} | "
                      f"Loss: {loss_vals['total']:.4f} | "
                      f"Buffer: {len(buffer)} transitions")
                log({'event': 'update', 'game': game_num+1, 'stage': stage,
                     'losses': loss_vals, 'buffer_size': len(buffer)})
            buffer.clear()
            model.eval()

        # Periodic evaluation
        if (game_num + 1) % EVAL_EVERY == 0 and stage == 2:
            win_rate = eval_vs_heuristic(model, n_games=20 if not smoke else 5, device=device)
            print(f"[Game {game_num+1}] Win rate vs heuristic: {win_rate:.1%}")
            log({'event': 'eval', 'game': game_num+1, 'win_rate': win_rate})

            # Save checkpoint
            ckpt_path = CHECKPOINT_DIR / f"checkpoint_{game_num+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
                print(f"  → New best: {best_win_rate:.1%}")

        if smoke and game_num >= 10:
            print("Smoke run complete.")
            break

    # Save final checkpoint
    torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")
    log({'event': 'done', 'best_win_rate': best_win_rate})
    print(f"Training done. Best win rate: {best_win_rate:.1%}")
    print(f"Checkpoints in: {CHECKPOINT_DIR}")
    print(f"Logs in: {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Marjapussi ML agent')
    parser.add_argument('--phase', type=int, default=1, help='Training phase (1=endgame)')
    parser.add_argument('--games', type=int, default=1000, help='Total games to run')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load checkpoint from path')
    parser.add_argument('--device', type=str, default='cpu', help='torch device')
    parser.add_argument('--smoke', action='store_true', help='Quick smoke test (10 games)')
    args = parser.parse_args()

    train(
        phase=args.phase,
        total_games=args.games,
        checkpoint_path=args.checkpoint,
        device=args.device,
        smoke=args.smoke,
    )
