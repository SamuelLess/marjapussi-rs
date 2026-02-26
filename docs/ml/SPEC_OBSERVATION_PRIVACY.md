# Spec: Observation Privacy and Data Boundaries

## 1. Purpose

Guarantee fair play by preventing hidden-information leakage from engine/debug data into ML training or inference.

## 2. Separation Model

Use two explicit observation surfaces:

1. `ObservationMl`: minimal, leak-safe, training/inference payload.
2. `ObservationDebug`: full introspection payload for UI/debug tooling.

These are separate types and separate serialization endpoints.

## 3. ML-Allowed Fields

ML payload may include only:

1. POV hand information.
2. Public trick/history information.
3. Deterministic symbolic deductions valid from POV (possible/confirmed masks).
4. Relative-seat context fields, coarse phase indicator, and legal actions.

## 4. ML-Forbidden Fields

Forbidden in ML payload:

1. Full other-player hands.
2. Exact hidden pass-card contents not visible to POV.
3. Any debug-only omniscient state.
4. Any future-information label not available at decision time.

## 5. UI/Debug Payload Policy

Debug payload may contain full state, but:

1. Must never be reused by ML code paths.
2. Must be clearly namespace-separated.
3. Must be guarded by explicit command/API route.

## 6. Legal Action Leak Controls

When it is not POV turn:

1. Legal action export must avoid exposing exact hidden card identities.
2. Encodings must not allow trivial reconstruction of hidden hand.

## 7. Dataflow Contracts

Requirements:

1. `env.py` and training loaders consume only `ObservationMl`.
2. UI server may consume both, but ML path remains strict.
3. Schema version id included in payload metadata (`schema_version`).
4. Schema version metadata is a compatibility gate only and must not be fed as a model feature.

## 8. Test Requirements

Automated tests must assert:

1. Forbidden keys are absent in ML JSON.
2. Hidden opponent fields cannot be inferred directly from action features.
3. Roundtrip serialization preserves allowed fields exactly.
4. UI debug route contains expected extended fields.

## 9. Operational Safeguards

1. Add static denylist check in CI for ML observation serializer.
2. Add runtime assertion in Python loader to reject schema-version mismatches.
3. Log and hard-fail if schema mismatch occurs.

