# Spec: Passing-Phase Strategy and Curriculum

## 1. Purpose

Define how passing strategy is learned while preserving core tactical structure and allowing later optimization.

## 2. Strategic Intent

Passing behavior should initially reflect strong heuristics:

1. Forth pass role (`PassingForth`):
   MH passes 4 cards to VH.
2. Back pass role (`PassingBack`):
   VH passes 4 cards back to MH.

Role-specific intent:

1. Forth passer (MH) tends to simplify suit distribution and preserve high-value structures for the partnership.
2. Back passer (VH) tends to preserve trump/pair flexibility and support partner coordination.

Later training should allow policy improvements beyond heuristic defaults.

## 3. Action Representation Requirement

Passing actions must be encoded without destructive aliasing.

Required:

1. Exact pass-set representation (for example 36-bit mask for selected 4 cards).
2. Optional engineered summary features (suit histogram, value histogram) as auxiliary.
3. Current implementation target: action feature layout reserves `[33..68]` for the 36-bit pass-card mask.

Not sufficient alone:

1. Histogram-only representation, because many distinct pass sets collapse to same features.

## 4. Curriculum Stages

### Stage A: Heuristic imitation (strict)

1. Policy trained to imitate heuristic pass choices.
2. Candidate actions restricted to heuristic-valid family.

### Stage B: Guided exploration (semi-strict)

1. Keep heuristic constraints soft but active.
2. Evaluate limited alternative passes that still satisfy baseline structural rules.
3. Learn from counterfactual relative gains.

### Stage C: Full optimization (soft constraints)

1. Full legal pass space available.
2. Heuristic structure becomes regularizer, not hard gate.
3. Counterfactual ranking drives final policy shape.

## 5. Constraint Features

Passing evaluation features should include:

1. Number of suits retained.
2. Pair/half preservation status.
3. Card-point retention and transfer.
4. Partner-support proxies.

These are used for curriculum filters and diagnostics, not as hard-coded final policy.

## 6. Counterfactual Candidate Generation

For passing turns:

1. Always include chosen action.
2. Include top-K heuristic alternatives.
3. Include diverse structurally valid alternatives.
4. Optionally include a small number of unconstrained probes.

## 7. Optimization Objectives

Mix objectives per stage:

1. Imitation loss for stabilization.
2. Advantage-weighted policy loss for improvement.
3. Optional regularizer for extreme anti-heuristic behavior early.

## 8. Metrics

Track passing-specific metrics:

1. Agreement with heuristic in early stages.
2. Average post-passing contract success uplift.
3. Trump/pair preservation rates.
4. Downstream point differential impact.

## 9. Exit Criteria

Promote stage only if:

1. No rule or leakage regressions.
2. Stable optimization without collapse.
3. Passing metrics improve or remain neutral while overall performance improves.
