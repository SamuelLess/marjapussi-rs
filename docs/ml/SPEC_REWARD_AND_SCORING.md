# Spec: Reward and Scoring Semantics

## 1. Purpose

Define exact scoring semantics and reward mapping so training targets match Marjapussi rules and do not drift over time.

## 2. Two Layers of Scoring

This project uses two related but distinct layers:

1. Official game/series scoring (rule-faithful scoreboard update).
2. RL training utility (normalized scalar reward for optimization).

Layer 2 must be derived from layer 1 with explicit formulas.

## 3. Rule Variables (Per Game)

For one finished game:

1. `A`: announced value (Ansage) by the playing party (S-party).
2. `P_S`: points made by S-party in the game.
3. `P_N`: points made by N-party in the game.
4. `S_fulfilled`: boolean, `P_S >= A`.
5. `N_schwarz`: boolean, N-party won zero tricks.
6. `no_one_played`: boolean, no party took the game in bidding.

## 4. Official Scoreboard Update (Rules)

For non-pass games (`no_one_played = false`):

1. S-party scoreboard delta:
   - `+A` if `S_fulfilled`
   - `-A` if not `S_fulfilled`
2. N-party scoreboard delta baseline:
   - `round_to_5(P_N)` where standard rounding applies
3. Special schwarz rule:
   - If `N_schwarz` and `S_fulfilled`, then N-party delta is replaced by `-2*A`
   - If `N_schwarz` and S-party does not fulfill, use baseline rule above

For pass games (`no_one_played = true`):

1. Use the project-defined pass-game policy.
2. This policy must be explicit and versioned in config.

Rounding helper:

1. `round_to_5(x)` means nearest multiple of 5 (for example 137 -> 135, 138 -> 140).

## 5. RL Reward Mapping

### 5.1 Base utility

For a POV party:

1. Start from contract outcome sign:
   - Positive if POV side achieved contract objective for its role.
   - Negative otherwise.
2. Magnitude is based on normalized game value:
   - `abs_u = A / 420.0`

### 5.2 Schwarz handling

Schwarz adjustment must be case-based, not a blanket multiplier:

1. Only apply doubling in rule-valid cases per Section 4.
2. Optional extension modes must be explicit and disabled by default.

### 5.3 Pass-game handling

For `no_one_played`:

1. Use configured pass-game utility (for example normalized point differential).
2. Keep this path separate from contract logic.

### 5.4 Shaping terms

If intermediate shaping is used:

1. It must be additive.
2. It must not invert terminal preference ordering.
3. Weight and normalization must be documented in checkpoint metadata.

## 6. PPO and Forced-Action Constraint

Policy ratio terms require valid behavior-policy log probabilities.

Mandatory rule:

1. Heuristic-forced actions must not be used in PPO ratio with synthetic/fake old log-prob.
2. Forced actions use imitation/distillation losses.
3. PPO ratio is only for sampled actions from tracked policy.

## 7. Config Surface

Reward/scoring config must include:

1. `scoring_mode` (`strict_rules` by default).
2. `pass_game_mode`.
3. Shaping weights.
4. Normalization constants.

Implementation note:

1. Reward computation is centralized in `ml/train/reward.py`.
2. `train_online.py` exposes core reward knobs:
   `--points-normalizer`, `--passgame-base-reward`, `--step-delta-scale`.

All config values are checkpointed and logged in evaluation output.

## 8. Required Tests

Table-driven tests must cover:

1. S fulfilled, no schwarz.
2. S failed, no schwarz.
3. N schwarz with S fulfilled (special -2*A for N).
4. N schwarz with S failed (no special replacement).
5. Rounding examples for N-party `round_to_5`.
6. Pass-game scoring mode behavior.
7. PPO forced-action exclusion behavior.

