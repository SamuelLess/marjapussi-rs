# Spec: Neurosymbolic Architecture

## 1. Purpose

Define the required architecture boundaries between symbolic game logic and neural policy/value learning.

## 2. Architectural Principle

The model is hybrid by design:

1. Symbolic code computes exact, deterministic game facts.
2. Neural code learns strategic decisions from those facts plus history.

This avoids wasting model capacity on exact bookkeeping and reduces hidden-state hallucination.

## 3. Symbolic Responsibilities (Rust Engine)

The symbolic layer must provide:

1. Legal action set for current state.
2. Rule-correct transition and scoring.
3. Deterministic observation features that do not require learning.
4. Counterfactual branch simulation support.

Required deterministic features include:

1. My hand bitmask.
2. Played/current trick card masks.
3. Opponent possible-card masks from suit-following logic.
4. Opponent confirmed-card masks from Q/A semantics.
5. Current trump, announced trump history.
6. Relative active seat and cards remaining.

## 4. Neural Responsibilities (Policy/Value)

The neural layer must learn:

1. Action selection among legal actions.
2. Partnership risk management under uncertainty.
3. Timing and tradeoffs for bidding, passing, trump calls, and trick play.
4. Long-horizon contract-aware value estimation.

## 5. Stream Structure

### 5.1 Stream A: Explicit state

Consumes deterministic symbolic tensors.

Requirements:

1. Permutation-invariant hand encoding.
2. Dedicated opponent-belief encoding.
3. Explicit context scalars and one-hot fields (role, trump state, trick position).

### 5.2 Stream B: Event history

Consumes event token sequence.

Requirements:

1. Preserve action order, especially Q/A and trump updates.
2. Use bounded sequence length with deterministic truncation policy.
3. Keep tokenizer and Rust token ids version-locked.

### 5.3 Action scoring head

Scores only legal actions.

Requirements:

1. Illegal actions are masked before softmax.
2. Action encoding is expressive enough for all action families.
3. Passing actions must be uniquely representable (see passing spec).
4. Passing representation must include exact card identity (not histogram-only aliases).

## 6. Mandatory Invariants

1. No model pathway may consume forbidden hidden-info fields.
2. Action mask semantics must be consistent across train/eval/inference.
3. Token id mapping must stay synchronized across Rust and Python.
4. Policy head outputs are only interpreted within legal-action set.

## 7. Known Failure Modes To Guard Against

1. Hidden-information leaks through debug fields.
2. Feature drift between observation builder and model assumptions.
3. Incorrect symbolic helper features that encode wrong game semantics.
4. Action aliasing, especially in passing representation.

## 8. Versioning Rules

Any change to observation schema, token vocabulary, or action encoding must:

1. Bump an observation/version id.
2. Include migration notes for datasets/checkpoints.
3. Add compatibility tests in `TEST_AND_VALIDATION_PLAN.md`.

Versioning implementation policy:

1. `schema_version` is carried in observation payload metadata.
2. Loader/runtime must enforce compatibility before tensorization.
3. `schema_version` is not part of neural input features.

