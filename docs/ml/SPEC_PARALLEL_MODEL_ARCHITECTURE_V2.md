# Spec: Parallel Model Architecture V2 (Budgeted to <=28M Params)

## 1. Purpose

Define a second model family (`parallel_v2`) that runs in parallel to the existing `legacy` model family.

The existing model must remain available and unchanged for baseline comparison, rollback safety, and checkpoint compatibility.

The new model must:

1. Stay within the hard parameter budget (`<=28,000,000` trainable params).
2. Use domain knowledge from the symbolic engine as first-class input and supervision.
3. Improve early-game competence (bidding + passing) without regressing trick-play quality.

## 2. Hard Constraints

1. Parameter budget: `<=28,000,000`.
2. Backward compatibility: legacy checkpoints and inference paths keep working.
3. No hidden-information leaks: privacy invariants from `SPEC_OBSERVATION_PRIVACY.md` remain mandatory.
4. Action legality invariants remain unchanged: masked illegal actions before softmax in all phases.
5. New architecture must be selectable via config/CLI; default behavior is `parallel_v2` while `legacy` remains available explicitly.

## 3. Non-Goals

1. Replacing or deleting `legacy` model code.
2. Introducing full online search/planning in this iteration.
3. Changing core game rules.

## 4. Architecture Summary

`parallel_v2` is a multi-branch architecture with explicit specialization:

1. Public-state tower: deterministic symbolic features.
2. Belief/inference tower: hidden-hand constraints and deductions.
3. Event-history tower: sequence model over event tokens.
4. Phase-router + phase-specific policy heads: bidding, passing, trick, info/QA.
5. Shared auxiliary heads: hidden-hand prediction, value, points.

Compared to legacy, this reduces interference between phases and allocates capacity where domain complexity differs.

## 5. Target Parameter Budget

Target (approximate) allocation:

1. Event-history tower: `~15.2M`.
2. Public-state tower: `~3.2M`.
3. Belief/inference tower: `~3.8M`.
4. Action encoder + 4 phase heads: `~4.4M`.
5. Auxiliary heads (hidden/value/points): `~1.1M`.

Total target: `~27.7M` with a hard gate at `28.0M`.

If the initial implementation exceeds budget, reduce in this order:

1. Event tower width (`d_model` 320 -> 304).
2. Event tower layers (12 -> 10).
3. Phase-head hidden size.

## 6. Module Specification

## 6.1 Public-State Tower

Input sources (already available or directly derivable):

1. Hand masks and card features.
2. Opponent possible/confirmed masks.
3. Role, trump, trick position, cards remaining, phase flags.
4. Existing scalar context (points, trick index, parity flags).

Output:

1. `state_vec: [B, D_state]`, target `D_state=640`.

## 6.2 Belief/Inference Tower

Input sources:

1. `hidden_possible`, `hidden_known`, `hidden_target` (train only for target).
2. New deterministic inference channels from Rust (if available):
   - `hidden_forced_owner` (`[B,3,36]`, cards that are logically forced to a seat).
   - `hidden_exclusive_candidates` (`[B,36]` or equivalent mask).
3. Existing cards-remaining constraints.

Required behavior:

1. Encode set-theoretic constraints explicitly.
2. Preserve exclusivity bias: one hidden card belongs to at most one seat.
3. Keep uncertain-but-possible cards as low-penalty regions (do not over-punish uncertainty).

Output:

1. `belief_vec: [B, D_belief]`, target `D_belief=320`.

## 6.3 Event-History Tower

Model:

1. Transformer encoder over event tokens.
2. Target config: `d_model=320`, `n_heads=8`, `n_layers=12`, `ff=1280`.
3. Positional embedding with fixed max length matching existing truncation policy.

Output:

1. `history_vec: [B, D_hist]`, target `D_hist=320`.

## 6.4 Fusion and Phase Router

Fusion:

1. `fused = concat(state_vec, belief_vec, history_vec)`.
2. Optional fusion MLP to `D_fused=768`.

Router:

1. Phase id is read from deterministic observation phase fields.
2. Router selects one policy head among: `bidding`, `passing`, `trick`, `info`.
3. Fallback mapping for unknown phase routes to `trick` head.

## 6.5 Policy and Auxiliary Heads

Policy:

1. Shared action encoder over `action_feats`.
2. Phase-specific scorer head receives `concat(action_emb, fused)`.
3. Output shape and masking semantics must match legacy: logits `[B, A]`.

Auxiliary:

1. Hidden-hand head: logits `[B,3,36]`.
2. Points head: `[B,2]` in `[0,1]`.
3. Value head: `[B]`.

## 7. Domain Knowledge Integration Rules

## 7.1 Mandatory Symbolic Priors

`parallel_v2` must consume and respect:

1. Possible-card masks as hard feasibility prior.
2. Confirmed/known cards as strong supervision signal.
3. Cards-remaining per seat as mass/consistency prior.
4. Phase-specific deterministic features from the game engine.

## 7.2 Mandatory Loss Composition

Keep current hidden-head losses and maintain strict weighting hierarchy:

1. Primary objective: policy/value for winning.
2. Auxiliary objective: hidden prediction as regularizer and representation shaper.
3. Auxiliary loss must not dominate policy updates.

Required safeguards:

1. Configurable aux coefficients with documented defaults.
2. Ability to reduce aux weight during late training if policy stalls.
3. Numeric stability checks for non-finite loss/gradients.

## 8. Interface Contract

## 8.1 New Python Model API

Add file:

1. `ml/model_parallel.py`

Required class:

1. `MarjapussiParallelNet(config: ParallelModelConfig)`
2. `forward(batch) -> (policy_logits, card_logits, point_preds, value_pred)`
3. `param_count() -> int`

Forward return signatures must exactly match legacy shape semantics.

## 8.2 Model Factory

Add file:

1. `ml/model_factory.py`

Required behavior:

1. `create_model(model_family: str, config_path: str | None)`.
2. Supports `legacy` and `parallel_v2`.
3. Enforces hard param budget when `parallel_v2` is selected.

## 8.3 Training CLI Extensions

Add to:

1. `ml/train_online.py`
2. `ml/train_from_dataset.py`

Required arguments:

1. `--model-family` (`legacy|parallel_v2`).
2. `--model-config` (path to model config file).
3. `--strict-param-budget` (default `28000000` for parallel_v2).

## 8.4 Config File

Add:

1. `ml/config/model_parallel_v2.toml`

Must include:

1. Dimensions and layer counts.
2. Param budget ceiling.
3. Head sizes per phase.
4. Optional ablation toggles (belief tower on/off, shared vs split heads).

## 8.5 Checkpoint Metadata Contract

Checkpoint payload must include metadata:

1. `model_family`.
2. `model_config_hash`.
3. `schema_version`.
4. `action_encoding_version`.
5. `param_count`.

Legacy pure-state_dict checkpoints remain loadable as `model_family=legacy`.

## 9. Training Plan for Next Iteration

## 9.1 Stage A: Supervised Warm-Start (Human Data)

Goal:

1. Learn high-quality bidding/passing priors before self-play.

Requirements:

1. Stronger phase weighting for bid/pass decisions.
2. Loss-driven LR schedule with warmup and floor.
3. Track per-phase metrics, not only global loss.

## 9.2 Stage B: Self-Play RL

Goal:

1. Convert imitation priors into stronger strategic play.

Requirements:

1. Keep forced-imitation decay schedule.
2. Monitor non-finite loss and gradient health.
3. Evaluate periodically vs heuristic and legacy checkpoints.

## 9.3 Stage C: Comparative Evaluation

Mandatory comparisons:

1. `legacy` vs `parallel_v2` at equal compute budget.
2. `parallel_v2` with and without belief tower (ablation).
3. `parallel_v2` with and without phase-split heads (ablation).

## 10. Acceptance Criteria

## 10.1 Architecture/Interface

1. Given `--model-family parallel_v2`, model initializes and reports param count `<=28M`.
2. Given `--model-family legacy`, behavior matches prior baseline.
3. Given any legal batch, both models return the same output tensor shapes and mask semantics.

## 10.2 Training Stability

1. A `train-1k-human` style smoke run completes without non-finite optimizer updates.
2. Pretraining LR does not decay below configured floor before plateau conditions are met.
3. Checkpoints contain required metadata fields.

## 10.3 Quality Gates

Measured against current legacy baseline on the same seed/control set:

1. Bidding top-1 accuracy: non-regression, target `+5pp`.
2. Passing top-1 accuracy: non-regression, target `+5pp`.
3. Hidden impossible mass: non-regression, target reduction.
4. Self-play evaluation point diff: non-regression at equal rounds, target improvement.

## 11. Required Tests

Rust:

1. Inference feature integrity tests for new deterministic fields.
2. Observation schema compatibility tests with version bumps.

Python:

1. Model factory tests for family selection and budget gate.
2. Forward-shape parity tests (`legacy` vs `parallel_v2`).
3. Checkpoint metadata save/load tests.
4. Training smoke tests for pretrain and RL with `parallel_v2`.
5. Numeric stability tests (detect and fail on non-finite loss/gradients).

UI/Serving:

1. `ui_server` can load either model family checkpoint and keep existing payload contract.

## 12. Implementation Work Packages (for Another Agent)

1. WP1: Add model config and factory (`model_parallel.py`, `model_factory.py`, TOML).
2. WP2: Integrate `model_family` into training scripts and checkpoint metadata.
3. WP3: Add belief/inference feature plumbing from Rust observation to Python tensors.
4. WP4: Implement phase-router and phase-specific policy heads.
5. WP5: Add tests for parity, budget, metadata, and smoke training.
6. WP6: Run controlled benchmark matrix and publish comparison table.
7. WP7: Update docs index and migration notes.

Completion rule:

1. A work package is complete only when its acceptance tests pass and docs are updated.

## 13. Risks and Mitigations

1. Risk: Aux losses overpower policy learning.
   Mitigation: enforce weighted hierarchy, schedule, and gradient-norm monitoring.
2. Risk: Budget creep above 28M.
   Mitigation: hard fail in factory and CI budget test.
3. Risk: Schema drift across Rust/Python.
   Mitigation: explicit schema version gates + compatibility tests.
4. Risk: Phase heads overfit and hurt transfer.
   Mitigation: shared action encoder + ablation comparisons.

## 14. Deliverables

1. New model family implementation with config-driven dimensions.
2. Training and serving support for selecting model family.
3. Comparative benchmark report (`legacy` vs `parallel_v2`).
4. Updated documentation and test coverage.
