# SPEC: Debug UI for Play-vs-AI

## Purpose
Provide a practical in-game UI for:
- Human-vs-AI play across any seat.
- Inspection of AI policy and hidden-hand predictions.
- Fast debugging of neurosymbolic reasoning errors.

## Scope
- Frontend: `ml/ui/app.js` + related CSS/HTML.
- Backend payloads: `ml/ui_server.py` debug and game state messages.
- No changes to core game rules in Rust for this spec.

## User Stories
- As a developer, I can take over seat P1/P2/P3 during a live game.
- As a developer, I can see what cards the model predicts for each hidden hand.
- As a developer, I can compare predictions with symbolic constraints (`possible_bitmasks`, `confirmed_bitmasks`).
- As a developer, I can quickly start/reset games and run AI autoplay for non-human seats.

## Functional Requirements
1. Seat Control
- Toggle human control per seat (`P0..P3`) at runtime.
- Clear visual indicator for human vs AI seats.
- Prevent accidental double-input by disabling AI action while a seat is human.

2. Core Table View
- Show current trick, trump, scores, active player, legal actions.
- Render card plays with clear player attribution and trick winner.
- Support full game progression without page reload.

3. Hidden-State Prediction Panel
- For each relative opponent (`left`, `partner`, `right`), show:
  - Top-K predicted cards (`K = cards_remaining[seat]`).
  - Per-card probability.
  - Impossible-mass indicator (aggregate probability on impossible cards).
- Color code:
  - Green: card is symbolically possible.
  - Red: card is symbolically impossible.

4. Symbolic Belief Overlay
- Display `possible_bitmasks` and `confirmed_bitmasks` per hidden seat.
- Allow side-by-side visual comparison:
  - Ground-truth hand (debug mode only).
  - Model-predicted hand.
  - Symbolic belief mask.

5. Policy Debug View
- Show action distribution for active AI seat:
  - Top actions with probabilities.
  - Entropy indicator.
  - Chosen action highlight.

6. Debug Actions
- New game/reset.
- Step AI one move / autoplay toggle.
- Optional: force pass/debug pass cards for reproducible scenarios.

## Non-Functional Requirements
- UI updates at interactive speed (target <= 100ms render path per state update).
- Handle missing debug fields gracefully (no hard crash if prediction fields absent).
- Keep behavior deterministic for same websocket payload sequence.

## Data Contract (Frontend Expectations)
From `debug_state`:
- `hands`
- `tricks`
- `possible_bitmasks`
- `confirmed_bitmasks`
- `predicted_hands`
- `predicted_card_probs`
- `predicted_impossible_mass`

From `game_state`:
- Current observation, legal actions, active player, scoring fields.
- `ai_info` for policy/entropy display.

## Acceptance Criteria
1. Human can play full rounds against AI by taking over any non-P0 seat.
2. Hidden prediction section updates every debug refresh without JS errors.
3. Impossible predicted cards are visibly distinguishable from possible ones.
4. UI remains usable when model is unavailable (prediction section hidden or fallback text).
5. Developer can identify at least one reasoning mismatch using panel data in under 30 seconds.

## Suggested Implementation Order
1. Stabilize seat takeover + game controls.
2. Finalize prediction panel layout and color semantics.
3. Add side-by-side symbolic vs predicted comparison.
4. Polish policy view and tooltips.
5. Add small UX improvements (filters, compact mode, keyboard shortcuts).

