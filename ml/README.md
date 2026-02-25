# Marjapussi ML Experiment

A deep RL + Transformer approach to learning Marjapussi, trained from endgame outward.

---

## 1. Project Goals

- Train an AI to play the **card-playing phase** of Marjapussi, then extend to card-passing (Schieben) and finally bidding (Reizen).
- The model must handle **partial observability** (unknown opponent hands) by learning to count cards from the action history.
- Emit **high-quality training signals** by evaluating moves counterfactually (try all legal moves at a decision point, play out games, reward based on differential outcome — not just terminal win/loss).

---

## 2. Phased Training Plan

Training starts at the endgame (fully constrained) and expands backward. This keeps the action space and credit assignment tractable early on.

| Phase | Scope | Trigger to advance |
|---|---|---|
| P1 | Last 5 tricks of card phase (4 cards/player) | Stable policy; near-optimal on known positions |
| P2 | Last 7–8 tricks (6–7 cards/player) | P1 policy transfers well |
| P3 | Full 9-trick card phase from trick 1 | P2 converges |
| P4 | Add Schieben phase (card passing) | P3 converges |
| P5 | Add Reizen (bidding) | P4 converges |

Within each phase, the model is initialized from the previous checkpoint.

---

## 3. Tokenization Strategy

The model's input is a **single flat token sequence** combining:

```
[GAME_HEADER] [ACTION_HISTORY] [SEP] [CURRENT_STATE] [SEP] [LEGAL_ACTIONS]
```

### 3.1 Vocabulary

#### Special Tokens
| Token | Purpose |
|---|---|
| `[PAD]` | Sequence padding |
| `[UNKNOWN]` | Unknown/hidden information (opponent cards not yet deducible) |
| `[SEP]` | Section separator |
| `[MASK]` | Masked card for MLM pre-training objective |
| `[START_GAME]` | Start of game marker |
| `[START_TRICK_N]` (N=1..9) | Start of trick N — gives the model positional game context without computing it |
| `[TRUMP_NONE]` | No trump active |

#### Card Tokens (36)
All 36 Schafkopf cards: `r-A r-Z r-K r-O r-U r-9 r-8 r-7 r-6` × 4 suits (`r`, `s`, `e`, `g`).

#### Player/Seat Tokens (4)
`[P0]` `[P1]` `[P2]` `[P3]` — always **relative to the observing player**, so P0 = "me", P1 = "left", P2 = "partner", P3 = "right". This lets the model learn partnerships and trick-order relationships without needing 4 separate weight sets.

#### Absolute Role Tokens (5)
`[ROLE_VH]` `[ROLE_MH]` `[ROLE_LH]` `[ROLE_RH]` `[ROLE_NONE]` — emitted **once** at `[START_GAME]` to tell the model who *I* am in the absolute game structure. This is necessary because seat position carries irreducible strategic meaning:
- **VH**: won the bid, leads every trick start, subject to first-card constraints (must play Ace → Grün), received 4 cards from MH
- **MH**: partner of VH, passed/received cards, plays 3rd in first trick
- **LH/RH**: defending party, with different threat profiles and lead freedom

The two-level encoding separates *relationship structure* (relative/P0–P3) from *absolute role* (VH/MH/LH/RH).

> **Phase 5 note (bidding)**: When the bidding phase is added, a separate `[BID_POS_1]`…`[BID_POS_4]` token will be needed to encode the dealer-relative starting position. This is the primary asymmetry in the bidding phase — going first means committing with zero information, going last means seeing all prior bids. The `[ROLE_*]` tokens encode the *outcome* of bidding; the `[BID_POS_*]` tokens encode the *structural advantage* before bidding starts.

#### Action Event Tokens
| Token | Maps to Rust | Notes |
|---|---|---|
| `ACT_PLAY` | `ActionType::CardPlayed(card)` | Followed by card token |
| `ACT_BID` | `ActionType::NewBid(i32)` | Followed by bid value token |
| `ACT_PASS_STOP` | `ActionType::StopBidding` | |
| `ACT_PASS_CARDS` | `ActionType::Pass(cards)` | Followed by 4 card tokens (hidden for opponents: `[UNKNOWN]`×4) |
| `ACT_TRUMP` | `ActionType::AnnounceTrump(suit)` | Followed by suit token |
| `ACT_Q_PAIR` | `ActionType::Question(QuestionType::Yours)` | |
| `ACT_Q_HALF` | `ActionType::Question(QuestionType::YourHalf(suit))` | Followed by suit token |
| `ACT_A_YES_PAIR` | `ActionType::Answer(AnswerType::YesPair(suit))` | Followed by suit token |
| `ACT_A_NO_PAIR` | `ActionType::Answer(AnswerType::NoPair)` | |
| `ACT_A_YES_HALF` | `ActionType::Answer(AnswerType::YesHalf(suit))` | Followed by suit token |
| `ACT_A_NO_HALF` | `ActionType::Answer(AnswerType::NoHalf(suit))` | Followed by suit token |
| `ACT_TRICK_WON` | Implicit (end of 4-card trick) | Followed by winner player token (for explicit card counting help) |

#### Suit Tokens (4)
`SUIT_R` `SUIT_S` `SUIT_E` `SUIT_G` — reused for trump announcements, half-questions, etc.

#### Scalar Value Tokens (numeric bid values, ~61 tokens)
Bid values 120, 125, …, 420 → `BID_120` … `BID_420`. Fixed vocab size: 61.

#### State Section Tokens (structural)
`[MY_HAND]` `[TRUMP_IS]` `[TRICK_SO_FAR]` `[POINTS_MY_TEAM]` `[POINTS_OPP_TEAM]` `[CARDS_LEFT_P0]`…`[CARDS_LEFT_P3]` `[LEGAL]`

**Total estimated vocabulary size: ~140 tokens** — tiny, fast lookup.

---

### 3.2 Full Input Sequence Structure

```
[START_GAME] [ROLE_VH]              ← absolute role of the observing player
[P0] ACT_BID BID_140
[P1] ACT_BID BID_145
[P2] ACT_PASS_STOP
[P3] ACT_PASS_STOP
[P0] ACT_PASS_STOP

[P1] ACT_PASS_CARDS [UNKNOWN] [UNKNOWN] [UNKNOWN] [UNKNOWN]   ← hidden from observer
[P0] ACT_PASS_CARDS r-A s-K e-6 g-7                          ← only visible if we are P0 or P1

[START_TRICK_1]
[P0] ACT_PLAY e-A
[P1] ACT_PLAY e-7
[P2] ACT_PLAY e-K
[P3] ACT_PLAY e-9
ACT_TRICK_WON [P0]

[P0] ACT_Q_PAIR
[P1] ACT_A_YES_PAIR SUIT_R
ACT_TRUMP SUIT_R                                               ← callback: NewTrump(Red)

[START_TRICK_2]
[P0] ACT_PLAY r-A
...

[SEP]
[MY_HAND] r-K s-A s-O g-Z g-9
[TRUMP_IS] SUIT_R
[TRICK_SO_FAR] [P3] ACT_PLAY g-A
[POINTS_MY_TEAM] 45                                            ← encoded as scalar, not token
[POINTS_OPP_TEAM] 30
[CARDS_LEFT_P0] 4  [CARDS_LEFT_P1] 4  [CARDS_LEFT_P2] 5  [CARDS_LEFT_P3] 5

[SEP]
[LEGAL] ACT_PLAY g-Z
[LEGAL] ACT_PLAY g-9
```

The model output is a **probability distribution over the legal action tokens** only. Illegal moves are masked to -inf in the logits before softmax.

---

### 3.3 Design Decisions & Rationale

#### Two-level seat encoding
**Relative (P0–P3)**: encodes relationships — who is my partner (P2), who plays before/after me (P1/P3). The model learns "P2 is always my teammate" without needing 4 separate weight sets for the same structural relationship.

**Absolute role (`[ROLE_VH/MH/LH/RH]`)**: seat position carries genuine strategic information that relative encoding cannot capture — VH leads first, is bound by the first-card rules, and holds the card-exchange context. LH/RH have a fundamentally different defensive goal. These are not redundant; they are given once as a game-level context token. The model uses relative encoding for *who does what in the sequence* and the role token for *what my strategic constraints are*.

#### Explicit `ACT_TRICK_WON` callback token
The `GameCallback::NewTrump` and trick resolution are not fully implicit in the action sequence alone. Emitting them as additional tokens gives the model a clean supervision signal and mirrors the `GameEvent.callback` structure already in the codebase — no inference needed.

#### `[START_TRICK_N]` positional tokens
These remove the need for the model to *count* tricks. This small amount of symbolic scaffolding significantly helps early training without compromising generality.

#### Card hand in state section (not hidden from transformer)
The transformer sees `[MY_HAND]` explicitly even though it appeared in the `ACT_PASS_CARDS` earlier. This redundancy is intentional — it's the *current* hand after all passing/playing, not the starting hand. Eliminates the need to compute a running set-difference to know what you hold.

#### Opponent card counts exposed
Explicit `[CARDS_LEFT_P1]` etc. gives the model ground-truth partial information that it would otherwise have to *count* from history. We do NOT reveal which cards opponents hold — just how many. This makes card counting optional (the model can bypass it early, but learn it later).

#### Legal actions in input
Appending `[LEGAL] <action>` tokens at the end means the model is never even tempted to output illegal actions. The classification head only scores these tokens. This is cleaner than masking at inference time alone.

---

## 4. Model Architecture

### 4.0 Core Design Principle

A pure Transformer is not ideal here. In NLP, transformers excel because natural language is *sparse in structure* — a word relates strongly to a few others. In Marjapussi, the required computations are different:

| Required operation | Nature | Best encoding |
|---|---|---|
| Card counting ("what's still out there") | Set union over history | **Explicit bitmask** — zero learning needed |
| Hand evaluation | Permutation-invariant over a set | **Sum of card embeddings (DeepSets)** |
| Trick evaluation ("who wins this trick") | Max over 4 cards under ordering | **Explicit comparison, or 4-card MLP** |
| Trump/Q&A ordering ("was this pair already asked?") | Genuinely sequential | **Small Transformer over event history** |
| Points progress | Running sum | **Scalar, direct** |
| Suit-following deductions | Pure logic | **Rust engine computes, model receives result** |

**The design**: explicit domain-structured state for everything computable, a small Transformer only for the genuinely sequential event history. These two streams are fused and scored over legal actions.

---

### 4.1 What the Rust Engine Must Pre-Compute (Not the Model's Job)

Before feeding anything to the model, the Rust observation serializer computes these deterministically from `GameState`:

**Played cards bitmask** (36 bits)
: Union of all cards in `all_tricks` + `current_trick`. Absolute ground truth — no inference needed.

**Possible cards per opponent** (36 bits × 3 opponents)
: Starts as (all 36 cards) minus (cards in my hand) minus (cards played). Narrows by suit-following: if opponent P1 didn't follow Grün in trick 3, all remaining Grün cards are zeroed from their bitmask. This is pure logic from `apply_action.rs` — tracking when a player plays off-suit.

**Confirmed cards per opponent** (36 bits × 3 opponents)
: Cards an opponent is *known* to hold. Populated from Q&A answers: if P1 answered `YesHalf(Red)`, we know they hold `r-O` or `r-K` (or both). If `YesPair(Red)`, both.

This gives the model **provably correct partial information** about opponent hands without any inference — the hardest part of card-game AI is done symbolically.

---

### 4.2 Card Embedding (Shared)

Every card is encoded with explicit structure — not just a learned ID:

```
card_features(c) = [
    suit_onehot(c),          # 4-dim  (Green, Acorns, Bells, Red)
    value_onehot(c),         # 9-dim  (Six..Ace)
    point_value(c),          # 1-dim  float: 0,0,0,0,2,3,4,10,11  /  11
    is_trump(c, state),      # 1-dim  bit
    is_higher_than_lead(c),  # 1-dim  bit: would this win the current trick?
]
# total: 16-dim raw features

card_emb(c) = ReLU(Linear(16 → 32))    # ~500 params, shared weights
```

This embedding is reused everywhere a card needs to be represented.

---

### 4.3 Stream A — Explicit State Vector

All *bookkeeping* information that is deterministically computable:

```
A1. My hand encoding
    hand_emb = sum(card_emb(c) for c in my_hand)    # 32-dim, DeepSets style
    # permutation-invariant, exact, no ordering artifact

A2. Opponent belief encoding (per opponent × 3)
    for each opponent o in {P1, P2=partner, P3}:
        opp_emb(o) = ReLU(Linear(
            concat(possible_bitmask(o),    # 36-dim
                   confirmed_bitmask(o),   # 36-dim
                   card_count(o)           # 1-dim integer / 9
            ) → 32-dim
        ))
    partner_emb  = opp_emb(P2)   # 32-dim
    left_emb     = opp_emb(P1)   # 32-dim
    right_emb    = opp_emb(P3)   # 32-dim

A3. Current trick encoding
    trick_emb = sum(card_emb(c) for c in current_trick)   # 32-dim
    trick_pos_onehot                                       # 4-dim: playing 1st/2nd/3rd/4th

A4. Game context (scalars + onehots)
    trump_onehot          # 5-dim (4 suits + none)
    trump_called_mask     # 4-dim bits: which suits have been announced ever
    trump_possibilities   # 3-dim onehot: Own / Yours / Ours  (from PlayerTrumpPossibilities)
    my_role_onehot        # 4-dim: VH/MH/LH/RH
    trick_number          # 1-dim: 1..9 / 9
    points_my_team        # 1-dim: current trick-points / 120
    points_opp_team       # 1-dim: current trick-points / 120
    last_trick_bonus_live # 1-dim: 1 if last trick started, 0 otherwise

# Concatenate all:
state_raw = concat(hand_emb,           # 32
                   partner_emb,        # 32
                   left_emb,           # 32
                   right_emb,          # 32
                   trick_emb,          # 32
                   trick_pos_onehot,   # 4
                   trump_onehot,       # 5
                   trump_called_mask,  # 4
                   trump_possibilities,# 3
                   my_role_onehot,     # 4
                   trick_number,       # 1
                   points_my_team,     # 1
                   points_opp_team,    # 1
                   last_trick_bonus)   # 1
# state_raw: 184-dim

state_vec = ReLU(Linear(184 → 128))   # ~24K params
```

---

### 4.4 Stream B — Sequential Event History (Transformer)

Only the parts of the game that are **genuinely sequential** and not expressible as a static snapshot:

- Q&A exchanges and their order (who asked what, when, what was answered)
- Trump history (which suit, after which trick)
- Bidding sequence (when added in Phase 5)

The card-playing trick sequence is *partially* covered by the bitmask, but the **order and context** of Q&A matters (e.g., "partner was asked about Grün and said no, then later answered yes to having a Red half — can I infer the pair?").

```
Vocabulary: ~145 tokens (see §3.1), embedded to 64-dim

Transformer encoder:
    Layers:    12
    Heads:     8
    Hidden:    256-dim
    FF dim:    1024
    Max length: ~1024 tokens (full game history fits easily)

history_vec = mean_pool(encoder(event_sequence))    # 64-dim
# ~400K params total for this block
```

---

### 4.5 Action Scoring Head

Each legal action gets its own embedding, then is scored jointly with the fused context:

```
For each legal action a:
    action_features(a) = concat(
        action_type_onehot(a),     # ~15-dim (ACT_PLAY, ACT_TRUMP, etc.)
        card_emb(a.card) if a has card else zeros(32),  # 32-dim
        suit_onehot(a.suit) if a has suit else zeros(4)  # 4-dim
    )
    action_emb(a) = ReLU(Linear(51 → 32))

    score(a) = Linear(
        concat(action_emb(a),   # 32
               state_vec,       # 128
               history_vec)     # 64
        → 1
    )

policy = softmax(scores over legal actions)
```

Illegal actions are never even represented — the scoring set is exactly the legal action set from `legal_actions.rs`.

---

### 4.6 Auxiliary Heads (Training Signals)

```
# Opponent card prediction (multi-label per opponent)
for o in {P1, P2, P3}:
    card_logits(o) = Linear(concat(state_vec, history_vec) → 36)
    # Binary cross-entropy against ground truth (known from full game record)
    # Only trained, not used at inference

# Final points regression
points_pred = Linear(concat(state_vec, history_vec) → 2)
# MSE against actual end-game trick-points for each team
# Scaled to [0, 1], denormalized at eval time

# Trick winner prediction (optional, Phase 3+)
trick_winner = Linear(concat(trick_emb, state_vec) → 4)
# Cross-entropy against FinishedTrick.winner
```

These heads share the `state_vec` and `history_vec` representations, so they regularize the main trunk during training.

---

### 4.7 Full Architecture Diagram

```
GameState (Rust)
    │
    ├─[deterministic]→ played_bitmask (36-bit)
    ├─[deterministic]→ possible_cards[P1,P2,P3] (36-bit × 3)
    ├─[deterministic]→ confirmed_cards[P1,P2,P3] (36-bit × 3)
    └─[serialized]──→ event_sequence (token ids)

                              ┌─────────────────────┐
 card features (16-dim) ─────►│  Card Embedding MLP  │──► card_emb (32-dim)
                              └─────────────────────┘
                                         │ (shared weights, used in A1, A2, A3, A5)
                                         ▼
 ┌────────────────── Stream A: Explicit State ──────────────────────┐
 │  hand_emb (DeepSets sum)                          32-dim         │
 │  partner/left/right belief encodings              96-dim         │
 │  current trick sum                                32-dim         │
 │  trick position, trump, role, points, etc.        24-dim         │
 │                         Linear(184→128) + ReLU                   │
 │                              state_vec            128-dim        │
 └──────────────────────────────────────────────────────────────────┘
                                         │
 ┌────────────── Stream B: Event History ──────────────────────────┐
 │  token_ids → embed(256) → Transformer(12L,8H,256d)               │
 │                              history_vec (mean pool) 256-dim    │
 └──────────────────────────────────────────────────────────────────┘
                                         │
                         concat(state_vec, history_vec) → 192-dim
                                         │
                    ┌────────────────────┴────────────────────┐
                    │            Action Scoring Head           │
                    │  action_emb(a) + fused context → score  │
                    │          softmax over legal set          │
                    └──────────────────────────────────────────┘
                        │                      │
                  policy(action)          aux heads
                  (main output)      (card pred, points pred)
```

---

### 4.8 Parameter Count

| Component | Params |
|---|---|
| Card embedding MLP | ~500 |
| Stream A (state encoder) | ~24K |
| Stream B (12-layer Transformer, 256-dim) | ~9.5M |
| Action scoring head | ~7K |
| Auxiliary heads | ~15K |
| **Total** | **~10M** |

The game complexity is in the *data structure*, not the model size. A 10M-param model will train reasonably fast on modern CPUs and instantly on GPUs for Phase 1 (endgame positions).

---

## 5. What Domain Knowledge to Build In (and What to Leave Out)

| Feature | Include? | Where | Reasoning |
|---|---|---|---|
| Played cards bitmask | **Yes** | Rust pre-compute → Stream A | Pure bookkeeping, zero learning value |
| Possible/confirmed opponent cards | **Yes** | Rust pre-compute → Stream A | Deterministic from suit-following + Q&A; not probabilistic inference |
| Card count per opponent | **Yes** | Scalar in Stream A | Derivable from bitmask but saves model a summation |
| Trump suit | **Yes** | Onehot in Stream A | Direct card ordering effect, critical |
| Team trick-points so far | **Yes** | Scalar in Stream A | Direct optimization target |
| Legal actions explicitly | **Yes** | Scoring head input | Rules are complex; symbolic enforcement saves capacity |
| Hand as a set (not sequence) | **Yes** | DeepSets in Stream A | Hand is genuinely unordered; positional encoding would be noise |
| Q&A order and content | **Yes** | Stream B Transformer | Genuinely sequential; inference requires knowing *when* things were asked |
| Trick winner of current trick | **No** | — | Computable from `cards.rs::high_card`; benchmark first, add if needed |
| Which cards the partner holds exactly | **No** | — | Must be inferred; this is the core of good partnership play |
| "Safe suit to lead" | **No** | — | Pure strategy; the model earns this |
| Bid value context | **Yes (Phase 5)** | Scalar in Stream A | Target score affects risk tolerance significantly |


---

## 6. Training Pipeline

### 6.1 Bootstrapping Without Human Data (Default Path)

Human game data is **not required**. Training works via a curriculum:

| Stage | Opponent | Purpose |
|---|---|---|
| 0: Random (first ~200 games) | Random legal moves | Explore game space; learn basic legality |
| 1: Heuristic (~500 games) | Rule-based agent | Stable opponent; learn trick-taking basics |
| 2: Self-play (ongoing) | Trained model copies | Model improves against itself |

The same PPO + counterfactual evaluation loop runs throughout. If human game data becomes available later, a behavior-cloning loss can be added as an optional enhancement.

### 6.2 Pre-training on Human Data (Optional)

**Data source**: Recorded games from `marjapussi.de` (when available).

**Objectives**: next-move prediction, masked card prediction, team points regression.

### 6.2 Counterfactual Move Evaluation (the Core Training Signal)

This is the key mechanism for producing clean RL signals without noisy terminal rewards.

For every training game, at each decision point `t` where the current player is uncertain (policy entropy above threshold, NOT a forced/obvious transposition):

1. **Clone** the `GameState` (the Rust struct already derives `Clone`).
2. **Try all N legal actions** at step `t`, producing N parallel game branches.
3. **Simulate** each branch to terminal state using the current policy (self-play).
4. **Compute outcome vector** for each branch: `(trick_points_my_team, trump_points_my_team, last_trick_bonus, won_bid)`.
5. **Rank actions** by this outcome vector.
6. **Generate training targets**: actions ranked above median → positive advantage, below → negative advantage. Use this as the advantage estimate for PPO, bypassing the value function for this specific move.

> ⚠️ This is essentially **partial MCTS without a value function**, applied only at uncertain decision points. It avoids the classic RL problem of sparse terminal rewards and conflicting signals from individual tricks.

**Special case — last 4 cards per player (endgame)**: enumerate all possible hands consistent with observable information (remaining cards minus what's been played) and **exhaustively solve** the subtree. This gives a perfect oracle signal for Phase 1 training.

### 6.3 RL Fine-tuning (PPO)

- Self-play with 4 model instances.
- Policy gradient with PPO, advantage from counterfactual evaluation above.
- Reward: **outcome differential** — not raw terminal reward.
- KL penalty from previous-stage policy to prevent catastrophic forgetting.
- Training runs continuously; checkpoint saved after each evaluation window.

### 6.4 Large-Scale Training (1 Million Games)

To train the model on a massive scale (e.g., 1 million games) using 32 parallel threads for game simulation, first ensure your Python virtual environment is activated:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

Then, run the following command (which equates to 100 rounds of 10,000 games each):
```bash
python ml/train_online.py --rounds 100 --games-per-round 10000 --workers 32 --mc-rollouts 4 --device cuda
```

Alternatively, you can trigger this via the `justfile` preset:
```bash
just train-1m
```

---

## 7. Rust Engine Requirements

The following capabilities need to be added to the Rust library to support the training loop:

- [ ] **`GameState::clone` from mid-game** — already available (derives `Clone`) ✅
- [ ] **Serialization of full `GameState` to JSON** — `serde::Serialize` partially done; `GamePhase` and `Player` need `#[derive(Serialize)]` ⚠️
- [ ] **Detached simulation mode** — run a game from a cloned state headlessly to terminal, returning `GameFinishedInfo`. Needs a simple loop around `legal_actions + apply_action`.
- [ ] **`GameState::cards_played`** — derive the set of already-played cards from `all_tricks + current_trick` (for card count bookkeeping).
- [ ] **Observation serializer** — function that takes a `GameState` and a `PlaceAtTable` (POV) and emits the token sequence described in §3.2 as a `Vec<u32>`. This is the main interface to the Python training code.
- [ ] **Python bindings (pyo3) or JSON stdio interface** — the training script needs to call the Rust engine.

---

## 8. Interfaces to Python Training Code

### Option A: `pyo3` native bindings (recommended for performance)
Expose `Game`, `GameState`, `clone_from_state(state) -> Game`, `step(action) -> GameEvent`, `legal_actions() -> Vec<u32>`, `serialize_observation(pov) -> Vec<u32>` as Python-callable Rust functions.

### Option B: JSON stdio protocol (simpler to start)
Each step: Python sends `{"action": <token_id>}` on stdin, Rust responds with `{"obs": [...], "legal": [...], "done": bool, "reward": 0}` on stdout. Easy to debug, good enough for Phase 1.

---

## 9. Decisions Made

| Question | Decision |
|---|---|
| Human game data required? | **No** — bootstrapping curriculum (random → heuristic → self-play) |
| Python interface | **JSON stdio** (spawn Rust binary as subprocess) |
| Python framework | **Pure PyTorch** |
| Model size | **~10M params** (see §4.8); scale up only if underfitting |
| Heuristic baseline | **Yes, built first** — needed for stage 1 and evaluation |
| Evaluation metric | Win rate vs heuristic + avg score diff, every 200 games |

---

## 10. Visualization UI

A local web app (`ml/ui/`) for watching and interacting with AI self-play.

### 10.1 Layout

```
┌─────────────────────────────────────────────────────────────┐
│  MARJAPUSSI AI  │ New Game │ Phase: P1 Endgame │ [Training] │
├─────────────────────┬──────────────┬────────────────────────┤
│                     │  AI INFO     │  TRAINING MONITOR      │
│   GAME BOARD        │  (per seat)  │  Loss: 0.42 ↓          │
│                     │  Policy %    │  Win vs heuristic: 61% │
│   [card table]      │  Entropy     │  Games: 1,247          │
│   [current trick]   │  Belief grid │  Phase: P1 (trick 5+)  │
│   [trump/scores]    │  [Take over] │  [loss curve]          │
│                     │              │                        │
│  [Proceed move ▶]   │              │                        │
└─────────────────────┴──────────────┴────────────────────────┘
```

### 10.2 Features

**Game board**
- 4 seats as card fans (face-up for human seat, face-down for AI seats)
- Current trick in center, trump + team scores permanently visible
- **"Proceed move ▶"** — steps the game one AI move
- **"Auto-play"** toggle — advances at 1 move/second

**AI info panel** (updates every move)
- Per-seat table: each legal action → probability %
- Entropy bar: low = confident, high = uncertain (counterfactual eval fires here)
- Belief state: 4×9 card grid per opponent — ✓ confirmed / ? possible / ✗ impossible
- **"Take over Seat N"** — switches that seat to human control immediately

**Human seat interaction**
- Legal cards highlighted; click to play
- Q&A prompts as inline dialog

**Training monitor** (active when `train.py` is running in background)
- Live loss curve, win rate vs heuristic (rolling 200-game window)
- Training stage indicator
- Checkpoint selector: load any saved checkpoint mid-session

### 10.3 Server

`python ml/ui_server.py` — `aiohttp` + WebSocket on `localhost:8765`.
- Manages current game state (env + model)
- Broadcasts game state JSON after each move
- Accepts: `proceed`, `human_action`, `take_over_seat`, `load_checkpoint`
- Tails `ml/runs/latest/log.jsonl` → forwards training events to UI

---

*This document is the living plan for the ML experiment. Update as implementation proceeds.*
