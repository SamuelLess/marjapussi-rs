# Spec: Training Lifecycle (Configurable, Human-First Recommended)

## 1. Objective

Define the canonical training lifecycle so the project can:

1. Start from human gameplay.
2. Improve with self-play.
3. Continue post-training with mixed data without regressions.

## 2. Supported Strategies

The pipeline must support these strategy profiles:

1. `human_first` (recommended when high-quality data exists):
   human dataset -> supervised pretraining -> RL/self-play fine-tuning.
2. `self_play_curriculum`:
   heuristic/random curriculum -> RL/self-play, no human pretraining.
3. `hybrid`:
   configurable alternation or mixing of supervised and self-play updates.

All strategies use the same observation schema and evaluation harness.
Checkpoint metadata must store the strategy id and its parameters.

## 3. Stage 0: Dataset Conversion

Input:

1. Legacy full-game records (for example `ml/dataset/games.json`).

Output:

1. Decision-point NDJSON records aligned with current observation schema.

Each record must contain at least:

1. `obs` in current ML observation format.
2. `action_taken` as legal-action index at that decision.
3. Outcome targets (`outcome_pts_my_team`, `outcome_pts_opp`).
4. Optional `chosen_advantage` if counterfactual data is available.

Conversion requirements:

1. Replay actions through current Rust rules engine.
2. Reconstruct POV-relative observations per decision.
3. Validate action legality at each replay step.
4. Drop or quarantine corrupted games with strict logging.

## 4. Stage 1: Supervised Pretraining (Optional by strategy)

Goal:

1. Learn baseline human-like priors before RL.

Primary losses:

1. Policy imitation loss on `action_taken`.
2. Auxiliary point/value regression as configured.

Requirements:

1. No PPO ratio assumptions in this stage.
2. Dataset shuffling and deterministic seed support.
3. Checkpoint metadata includes data snapshot and schema version.
4. Support optional hard cap on supervised optimizer steps (`max_steps`) for budgeted warm-starts.

## 5. Stage 2: Offline Validation Gate

A checkpoint can move to the next stage only if it passes:

1. Rule compliance checks.
2. No-leak observation checks.
3. Basic sanity against heuristic baseline.
4. Numerical stability checks (NaN, inf, mask violations).

## 6. Stage 3: RL/Self-Play Fine-Tuning

Goal:

1. Improve beyond imitation.

Requirements:

1. Reward semantics from `SPEC_REWARD_AND_SCORING.md`.
2. Counterfactual advantage from `SPEC_COUNTERFACTUAL_ADVANTAGE.md`.
3. Separate treatment for heuristic-forced actions vs model-sampled actions.
4. Curriculum controls for bidding/passing/trick phases.

## 7. Stage 4: Continuous Post-Training

Allowed data sources:

1. New human game logs.
2. Fresh self-play samples.
3. Mixed replay buffers.

Policy:

1. Human data is always allowed as supervised refresh.
2. Mixing ratios are tracked in checkpoint metadata.
3. Every post-training run must keep regression reports.

## 8. Experiment Parameterization

The training loop must expose parameters needed to compare strategies:

1. Strategy profile (`human_first`, `self_play_curriculum`, `hybrid`).
2. Data mixture ratios and schedules.
3. Curriculum phase schedule for trick/passing/bidding.
4. Counterfactual rollout policy mix.
5. Loss weighting schedule (imitation vs RL).
6. Pretraining budget control (for example fixed `max_steps` or ratio-based policy).

The same fixed evaluation suite must be used across strategy runs to compare outcomes.

## 8.1 Reference Profiles

Project recipes should keep at least two directly comparable profiles:

1. `train-65k-scratch`: RL/self-play only baseline.
2. `train-65k-human`: legacy-human pretraining warm-start, then same RL schedule.

Default human-first warm-start budget is configured as a front-loaded 50% step budget (`max_steps=512`) for the 65k profile.

## 9. Checkpoint Governance

Each checkpoint must store:

1. Model version and observation schema version.
2. Training stage and objective mix.
3. Dataset sources and hashes.
4. Eval metrics and seed set.
5. Difficulty profile defaults (temperature, search budget).

## 10. Difficulty Profiles

The serving layer maps checkpoints to difficulty tiers:

1. Easy.
2. Medium.
3. Hard.

Each tier is fully specified by:

1. Checkpoint id.
2. Inference temperature.
3. Optional rollout/search budget.
