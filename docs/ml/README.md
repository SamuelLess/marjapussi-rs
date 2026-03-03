# Marjapussi ML Documentation Index

This folder defines the implementation contracts for the Marjapussi ML system.

Project-level intent is defined in:

1. `docs/ML_PROJECT_OVERVIEW.md`

Technical specs in this folder:

1. `SPEC_NEUROSYMBOLIC_ARCHITECTURE.md`
2. `SPEC_TRAINING_LIFECYCLE.md`
3. `SPEC_REWARD_AND_SCORING.md`
4. `SPEC_OBSERVATION_PRIVACY.md`
5. `SPEC_COUNTERFACTUAL_ADVANTAGE.md`
6. `SPEC_PASSING_CURRICULUM.md`
7. `SPEC_PARALLEL_MODEL_ARCHITECTURE_V2.md`
8. `TEST_AND_VALIDATION_PLAN.md`

Execution plans:

1. `PLAN_V2_IMPLEMENTATION_TASKLIST.md`

Operational tooling:

1. Fixed-deal evaluator: `ml/eval_fixed_deals.py`
2. Editable suites: `ml/eval/fixed_deals_100.json`, `ml/eval/fixed_deals_custom_template.json`

Order of implementation should follow:

1. Observation privacy and reward correctness first.
2. Dataset conversion plus training strategy setup (human-first, self-play, or hybrid).
3. Counterfactual and passing-policy curriculum refinement.
4. Continuous validation and checkpoint governance.
