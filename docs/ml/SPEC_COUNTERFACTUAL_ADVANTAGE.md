# Spec: Counterfactual Advantage Estimation

## 1. Purpose

Define how to estimate per-action quality at a decision point under partial observability, with lower variance than pure terminal reward.

## 2. Target Quantity

At state `s` with legal actions `A(s)`, estimate:

1. `Q_hat(s, a)` for each legal `a`.
2. Relative advantage `A_hat(s, a) = Q_hat(s, a) - mean_{a' in A(s)} Q_hat(s, a')`.

This is the preferred training signal for action discrimination.

## 3. Branching Policy

For each candidate action:

1. Clone game state.
2. Apply candidate action.
3. Roll out to terminal under evaluation policy mix.
4. Compute scalar terminal utility from reward spec.

## 4. Rollout Policy Mix

Support configurable mix:

1. Heuristic rollout policy.
2. Model rollout policy.
3. Historical checkpoint ensemble policy.

Deterministic-only heuristic rollouts are allowed for bootstrap but should not be the sole estimator for mature training.

## 5. Partial Observability Handling

For robust estimates:

1. Optionally sample multiple determinizations consistent with POV-visible info.
2. Reuse same random seeds across candidate actions where possible.
3. Keep rollout budgets symmetric across actions.

## 6. Normalization

Normalize per-state action estimates, for example:

1. Zero-mean per decision.
2. Scale by stable denominator (`max(std, floor)`).

Avoid global normalization that mixes unrelated states.

## 7. Candidate-Set Policy

By default evaluate all legal actions.

For large spaces (especially passing):

1. Evaluate constrained candidate subset.
2. Ensure subset includes policy action and diverse alternatives.
3. Log subset generation strategy for auditability.

## 8. Computational Controls

Expose configurable controls:

1. `num_rollouts`.
2. Determinization count.
3. Phase-specific branching triggers.
4. Timeout/cutoff policy.

Cache branch results where safe.

## 9. Training Integration

1. Use `A_hat` as policy target weighting.
2. Combine with value-based GAE fallback when counterfactual unavailable.
3. Keep estimator provenance in dataset fields.

## 10. Validation Requirements

Tests and diagnostics:

1. Deterministic reproducibility under fixed seed.
2. Advantage sign sanity on known tactical positions.
3. Stability under rollout count increase.
4. Runtime budget accounting per phase.

