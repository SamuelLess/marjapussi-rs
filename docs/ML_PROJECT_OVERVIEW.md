# Marjapussi AI Project Overview

## 1. Mission

Build a fair, strong, and controllable Marjapussi AI system that can:

1. Play AI vs AI for training and evaluation.
2. Play AI vs humans in real games without information leakage.
3. Expose multiple difficulty levels by checkpoint or policy profile.
4. Continue improving from either human game data or self-play, without breaking rules.

This project is not only about maximizing score. It is about producing an AI system that is:

1. Rule-correct.
2. Partnership-aware.
3. Explainable at the system level (neurosymbolic, not black box only).
4. Operationally safe for public use.

## 2. Product Intent

The target product behavior is:

1. A single engine and model family supports both competitive and casual play.
2. Difficulty is controlled by explicit model checkpoints and inference settings.
3. The same game rules apply to all seats and all game modes.
4. Human users can trust that AI never sees hidden cards it should not see.

## 3. Core Design Philosophy

The ML system follows a neurosymbolic split:

1. Symbolic layer handles deterministic game logic and constraints.
2. Neural layer learns strategic tradeoffs under uncertainty.

We do not ask the network to relearn pure bookkeeping that the engine can compute exactly.
We do ask the network to learn uncertainty management, partnership timing, and tactical choice.

## 4. Training Strategy Policy

The recommended production default is human-data first when a vetted dataset is available:

1. Convert human game logs into current decision-point training format.
2. Run supervised pretraining on human decisions.
3. Validate for rule compliance and behavioral sanity.
4. Run RL/self-play fine-tuning.
5. Continue optional post-training from new human games, self-play, or mixed batches.

This policy is not mandatory.
The system must support parameterized strategy selection so experiments can compare methods:

1. Human-first laddering (supervised pretrain, then RL).
2. Pure self-play curriculum (no human pretrain).
3. Hybrid schedules with configurable human/self-play mixing.

The project must track which strategy was used for each checkpoint so results are comparable.

## 5. Fairness and Safety Requirements

Fair play is a product requirement, not an optimization preference.

Mandatory requirements:

1. No hidden-information leakage in ML observations.
2. Identical game-rule enforcement for humans and AI.
3. No seat-specific shortcuts that break relative-seat symmetry unless explicitly modeled.
4. Training and evaluation must detect leak regressions automatically.

## 6. Difficulty Levels

Difficulty is controlled through checkpoints and policy temperature, for example:

1. Easy: earlier checkpoint + higher temperature + limited search/counterfactual budget.
2. Medium: mid checkpoint + moderate temperature + moderate budget.
3. Hard: strongest checkpoint + low temperature + full budget.

Difficulty must be reproducible and tied to metadata:

1. Checkpoint id.
2. Training data mixture.
3. Training stage.
4. Eval metrics snapshot.

## 7. Scope and Boundaries

In scope:

1. Card-play phase quality.
2. Bidding and passing strategy quality.
3. Rule-faithful scoring and contract modeling.
4. Safe data pipeline for human and self-play data.

Out of scope:

1. UI aesthetics and frontend polish details.
2. Non-Marjapussi rule variants unless explicitly configured.

## 8. Reference Specifications

This overview is the intent document. Implementation contracts are in:

1. `docs/ml/README.md`
2. `docs/ml/SPEC_NEUROSYMBOLIC_ARCHITECTURE.md`
3. `docs/ml/SPEC_TRAINING_LIFECYCLE.md`
4. `docs/ml/SPEC_REWARD_AND_SCORING.md`
5. `docs/ml/SPEC_OBSERVATION_PRIVACY.md`
6. `docs/ml/SPEC_COUNTERFACTUAL_ADVANTAGE.md`
7. `docs/ml/SPEC_PASSING_CURRICULUM.md`
8. `docs/ml/SPEC_PARALLEL_MODEL_ARCHITECTURE_V2.md`
9. `docs/ml/TEST_AND_VALIDATION_PLAN.md`

## 9. Definition of Success

The project is successful when:

1. AI can play complete games against humans without rule or leakage incidents.
2. Difficulty tiers are stable and meaningful.
3. Training can be resumed with new human data and improve policy quality.
4. Self-play training can continue without catastrophic regressions.
5. All high-risk scoring and privacy invariants are test-covered.
