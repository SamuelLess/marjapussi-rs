# V2 ML Implementation Plan And Task List

## Purpose

Execution plan for the full `parallel_v2` rollout while preserving current `legacy` training/inference and existing checkpoints.

This plan is implementation-facing and intended for continuous checkpoint commits after each meaningful unit of work.

## Scope Summary

1. Keep current model/checkpoints usable (`legacy` path remains explicitly selectable for compatibility).
2. Implement full `parallel_v2` architecture integration per `SPEC_PARALLEL_MODEL_ARCHITECTURE_V2.md`.
3. Add model-family routing across training, serving, and UI.
4. Add tests, smoke training routes, and comparative benchmarking.
5. Prepare long-run training command (`>=65k games`) for v2 and report shortcomings.

## Hard Constraints

1. `parallel_v2` param budget gate: `<=28,000,000`.
2. No hidden-information leakage.
3. Action-mask semantics unchanged and verified.
4. Legacy checkpoints must still load and run.
5. Continuous git checkpoints throughout execution.

## Workstreams

## WS1: Planning and Contracts

1. Add/maintain this plan file and keep status current.
2. Keep `SPEC_PARALLEL_MODEL_ARCHITECTURE_V2.md` as source-of-truth contract.
3. Add migration notes for model family + checkpoint metadata.

Acceptance:

1. Plan lists all deliverables and completion criteria.
2. Contract docs referenced in ML index and project overview.

## WS2: Model Architecture

1. Add `ml/model_parallel.py` implementing `MarjapussiParallelNet`.
2. Add phase-router and phase-specific heads (`bidding`, `passing`, `trick`, `info`).
3. Add belief/inference tower and integrate existing hidden constraints.
4. Add `param_count()` and strict budget check helper.

Acceptance:

1. Forward signature parity with legacy model.
2. Param count reported and budget-enforced.
3. Model initializes and runs in smoke forward pass.

## WS3: Model Factory and Configuration

1. Add `ml/model_factory.py`:
   - `legacy` -> current model
   - `parallel_v2` -> new model
2. Add `ml/config/model_parallel_v2.toml`.
3. Add model config parsing and config hash utility.

Acceptance:

1. Family selection works by CLI/config.
2. Invalid family/config fails with clear errors.
3. Strict budget failure is explicit.

## WS4: Training Integration

1. Update `ml/train_from_dataset.py`:
   - model family selection
   - model config input
   - checkpoint metadata save/load
2. Update `ml/train_online.py` similarly.
3. Ensure inference server uses selected model family.
4. Preserve forced-imitation and hidden-loss weighting logic.

Acceptance:

1. Both scripts run with `legacy` and `parallel_v2`.
2. Metadata contains family/config/schema info.
3. Legacy checkpoints still load.

## WS5: Serving and UI Integration

1. Update `ml/ui_server.py` to load model family from checkpoint metadata.
2. Show model family/version in UI controller/debug area.
3. Keep fallback behavior explicit for incompatible checkpoints.
4. Expand debug payload with v2-specific visibility (head family, param count, metadata).

Acceptance:

1. UI can load either model family.
2. Mismatch messages are actionable and specific.
3. No regressions in pass-selection interaction.

## WS6: Tests

Rust tests:

1. Observation/inference invariants for new fields if introduced.

Python tests:

1. Model factory selection and error-path tests.
2. Param-budget enforcement tests.
3. Forward-shape parity (`legacy` vs `parallel_v2`).
4. Checkpoint metadata save/load tests.
5. UI loading tests for both families.
6. Smoke training tests for both families.

Acceptance:

1. Test suite passes in local environment.
2. New tests cover failure modes (mismatch, budget overflow, wrong family).

## WS7: Commands and Run Profiles

1. Add v2 smoke command(s), at minimum:
   - `train-1k-human-v2`
   - `train-4k-human-v2`
2. Add long-run command:
   - `train-65k-v2` (>=65k games total)
3. Keep legacy commands unchanged.

Acceptance:

1. Commands create isolated run roots under `ml/runs/<run_name>/...`.
2. Named checkpoints include family-specific naming.

## WS8: Benchmarking

1. Add evaluation script/report workflow comparing:
   - `legacy` best checkpoint
   - `parallel_v2` best checkpoint
2. Evaluate on:
   - deterministic self-play eval set
   - converted human game dataset (same-card setups where available)
3. Report:
   - point diff
   - bidding/pass match rates
   - hidden prediction quality
   - notable failure patterns

Acceptance:

1. Report contains metrics table and conclusion.
2. Shortcomings and concrete next actions documented.

## Commit Cadence

Required git checkpoint after each milestone:

1. Planning docs complete.
2. Model + factory scaffold compiles.
3. Training integration complete.
4. UI integration complete.
5. Tests added and passing.
6. Smoke run complete and stable.
7. Benchmark report produced.

## Delivery Checklist

1. [ ] WS1 complete
2. [ ] WS2 complete
3. [ ] WS3 complete
4. [ ] WS4 complete
5. [ ] WS5 complete
6. [ ] WS6 complete
7. [ ] WS7 complete
8. [ ] WS8 complete
9. [ ] Final summary with shortcomings and recommendations

## Notes

1. If runtime or hardware blocks full long-run completion in-session, launch is still prepared with exact command and artifacts path, and report marks pending metrics explicitly.
2. Do not remove legacy behavior unless explicitly approved and separately migrated.
