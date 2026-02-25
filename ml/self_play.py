"""
Self-play loop with bootstrapping curriculum and counterfactual move evaluation.

Curriculum stages:
    0 (random):    All 4 seats use random policy
    1 (heuristic): All 4 seats use heuristic (Rust-side)
    2 (self-play): All 4 seats use the trained model
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from env import MarjapussiEnv, obs_to_tensors


ENTROPY_THRESHOLD = 1.2   # nats — above this, run counterfactual eval
MAX_LEGAL_ACTIONS = 36    # max action space size


@dataclass
class Transition:
    """One (obs, action, advantage) tuple for training."""
    obs_tensors: dict
    action_idx: int        # index into model's legal action list
    advantage: float
    # Auxiliary targets
    points_my_team_target: Optional[float] = None
    points_opp_team_target: Optional[float] = None


@dataclass
class Episode:
    transitions: list[Transition] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)


def policy_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return float(-(probs * log_probs).sum())


def model_policy(model: torch.nn.Module, obs: dict, device: str = 'cpu') -> tuple[int, float, torch.Tensor]:
    """
    Run model inference on a single observation.
    Returns (action_idx_in_legal_list, entropy, logits).
    """
    tensors = obs_to_tensors(obs)
    tensors = {k: (v.to(device) if isinstance(v, torch.Tensor) else
                   {kk: vv.to(device) for kk, vv in v.items()})
               for k, v in tensors.items()}
    with torch.no_grad():
        logits, _, _, _ = model(tensors)  # [1, A]
    logits = logits[0]  # [A]
    ent = policy_entropy(logits)
    # Sample from policy
    probs = F.softmax(logits, dim=-1)
    action_pos = torch.multinomial(probs, 1).item()
    return action_pos, ent, logits


def run_episode(
    model: Optional[torch.nn.Module],
    stage: int,
    device: str = 'cpu',
    phase_start_trick: int = 1,
) -> Episode:
    """
    Run one full game episode.
    stage 0: all random (Rust-side)
    stage 1: all heuristic (Rust-side)
    stage 2: model plays all seats
    """
    env = MarjapussiEnv(pov=0)
    obs = env.reset()
    episode = Episode()

    cf_policy = "heuristic" if stage <= 1 else "heuristic"  # for counterfactual branches

    # Play through the game, collecting transitions at uncertain points
    steps = 0
    while not env.done and steps < 300:
        legal = env.legal_actions
        if not legal:
            break

        if stage == 0:
            # Random: pick random legal action
            action_pos = random.randrange(len(legal))
            _, done, info = env.step(legal[action_pos]['action_list_idx'])
        elif stage == 1:
            # Heuristic: use simple priority
            action_pos = heuristic_select(legal)
            _, done, info = env.step(legal[action_pos]['action_list_idx'])
        else:
            # Model: run inference
            action_pos, entropy, logits = model_policy(model, obs, device)
            action_pos = min(action_pos, len(legal) - 1)

            # Capture tensors before stepping
            tensors = obs_to_tensors(obs)

            # Targeted Curriculum Branching: only branch at the specific trick we are learning
            trick_no = obs.get('trick_number', 1)
            is_target = (trick_no == phase_start_trick)
            is_stochastic = (random.random() < 0.05)
            
            should_cf = (len(legal) > 1 and (is_target or is_stochastic))

            if should_cf:
                advs = env.get_advantages(policy=cf_policy, num_rollouts=4)
                if advs and action_pos < len(advs):
                    chosen_advantage = advs[action_pos]
                    episode.transitions.append(Transition(
                        obs_tensors=tensors,
                        action_idx=action_pos,
                        advantage=chosen_advantage,
                    ))

            obs, done, info = env.step(legal[action_pos]['action_list_idx'])

        obs = env.obs
        steps += 1

    episode.outcome = env.run_to_end("heuristic") if not env.done else {}
    env.close()

    # Fill in point targets for any captured transitions
    if episode.outcome:
        my_pts = episode.outcome.get('team_points', [0, 0])[0]
        opp_pts = episode.outcome.get('team_points', [0, 0])[1]
        for t in episode.transitions:
            t.points_my_team_target = my_pts / 420.0
            t.points_opp_team_target = opp_pts / 420.0

    return episode


def compute_advantages(branches: list[dict], pov_seat: int = 0) -> dict[int, float]:
    """
    Given counterfactual branch outcomes, compute relative advantage per action.
    Metric: trick-points won by pov's party.
    Returns dict: action_idx -> advantage (normalized, mean=0).
    """
    scores = {}
    for branch in branches:
        idx = branch['action_idx']
        # Handle both single 'outcome' and plural 'outcomes'
        outcomes = branch.get('outcomes', [branch.get('outcome')])
        
        vals = []
        for outcome in outcomes:
            if not outcome: continue
            tricks = outcome.get('tricks', [])
            my_pts = sum(t['points'] for t in tricks if t['winner'] % 2 == pov_seat % 2)
            vals.append(float(my_pts))
        
        if vals:
            scores[idx] = sum(vals) / len(vals)

    if not scores:
        return {}

    values = list(scores.values())
    mean_score = sum(values) / len(values)
    std_score = max(math.sqrt(sum((v - mean_score)**2 for v in values) / len(values)), 1.0)

    return {idx: (s - mean_score) / std_score for idx, s in scores.items()}


def heuristic_select(legal: list[dict]) -> int:
    """Python-side heuristic (mirrors Rust heuristic_policy)."""
    # Prefer trump announcements
    for i, la in enumerate(legal):
        if la['action_token'] == 44:  # ACT_TRUMP
            return i
    # Prefer Ace (value 8)
    for i, la in enumerate(legal):
        if la.get('card_idx') is not None and la['card_idx'] % 9 == 8:
            return i
    # Prefer Ten (value 7)
    for i, la in enumerate(legal):
        if la.get('card_idx') is not None and la['card_idx'] % 9 == 7:
            return i
    # Prefer StopBidding
    for i, la in enumerate(legal):
        if la['action_token'] == 42:  # ACT_PASS_STOP
            return i
    return 0
