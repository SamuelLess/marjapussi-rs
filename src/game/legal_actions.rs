#[cfg(feature = "public-api")]
use std::cell::RefCell;

use crate::bits::{self, CardSet};
use crate::game::cards::{Card, Suit};
use crate::game::gameevent::{ActionType, AnswerType, GameAction, QuestionType};
use crate::game::gamestate::GamePhase;
use crate::game::player::{PlayerTrumpPossibilities, HAND_SIZE};
use crate::game::points::Points;
use crate::game::Game;

impl GamePhase {
    #[cfg(feature = "public-api")]
    pub fn legal_actions(&self, game: &Game) -> Vec<GameAction> {
        let mut out = vec![];
        self.push_legal_actions(game, &mut out);
        out
    }

    /// Appends the legal actions of this phase to `out` (without undo requests).
    pub(crate) fn push_legal_actions(&self, game: &Game, out: &mut Vec<GameAction>) {
        match self {
            GamePhase::WaitingForStart => out.extend(
                game.state
                    .players
                    .iter()
                    .filter(|p| !game.state.players_started.contains(&p.place_at_table))
                    .map(|p| GameAction {
                        action_type: ActionType::Start,
                        player: p.place_at_table,
                    }),
            ),
            GamePhase::Bidding => push_bidding(game, out),
            GamePhase::PassingForth | GamePhase::PassingBack => push_passing(game, out),
            GamePhase::Raising => {
                // The bids from the highest down, without stopping (`legal_bidding` reversed,
                // its StopBidding removed).
                let start = out.len();
                push_bidding(game, out);
                out[start..].reverse();
                out.pop();
                push_cards(game, out);
            }
            GamePhase::StartTrick => {
                push_question(game, out);
                push_cards(game, out);
            }
            GamePhase::Trick => push_cards(game, out),
            GamePhase::AnsweringPair | GamePhase::AnsweringHalf(_) => push_answer(game, out),
            GamePhase::Ended => {}
            GamePhase::PendingUndo(_previous_phase) => {
                let next_player = game.last_state.as_ref().unwrap().player_at_turn.next();
                for player in [next_player, next_player.partner()] {
                    if !game.state.players_accept_undo.contains(&player) {
                        out.push(GameAction {
                            action_type: ActionType::UndoAccept,
                            player,
                        });
                        out.push(GameAction {
                            action_type: ActionType::UndoDecline,
                            player,
                        });
                    }
                }
            }
        }
    }
}

/// Membership without generating card, pass or bid lists; other actions use their short list.
#[cfg(not(feature = "public-api"))]
pub(crate) fn is_legal_in_phase(game: &Game, action: &GameAction) -> bool {
    let at_turn = action.player == game.state.player_at_turn;
    match (&game.state.phase, &action.action_type) {
        (GamePhase::WaitingForStart, ActionType::Start) => {
            action.player.0 < 4 && !game.state.players_started.contains(&action.player)
        }
        (GamePhase::Bidding | GamePhase::Raising, ActionType::NewBid(value)) => {
            at_turn
                && *value > game.state.value.0
                && *value <= MAX_BID
                && (*value - game.state.value.0) % BID_STEP == 0
        }
        (GamePhase::Bidding, ActionType::StopBidding) => at_turn,
        (GamePhase::PassingForth | GamePhase::PassingBack, ActionType::Pass(cards)) => {
            at_turn && is_legal_pass(game, cards)
        }
        (
            GamePhase::Raising | GamePhase::StartTrick | GamePhase::Trick,
            ActionType::CardPlayed(card),
        ) => at_turn && allowed_cards(game).contains(card.index()),
        (GamePhase::StartTrick, ActionType::Question(_) | ActionType::AnnounceTrump(_))
        | (GamePhase::AnsweringPair | GamePhase::AnsweringHalf(_), ActionType::Answer(_))
        | (GamePhase::PendingUndo(_), ActionType::UndoAccept | ActionType::UndoDecline) => {
            let mut legal = Vec::new();
            game.state.phase.push_legal_actions(game, &mut legal);
            legal.contains(action)
        }
        _ => false,
    }
}

#[cfg(feature = "public-api")]
pub fn legal_bidding(game: &Game) -> Vec<GameAction> {
    let mut out = Vec::with_capacity(62);
    push_bidding(game, &mut out);
    out
}

const BID_STEP: i32 = 5;
const MAX_BID: i32 = 420;

fn push_bidding(game: &Game, out: &mut Vec<GameAction>) {
    let start_value = game.state.value + Points(BID_STEP);
    let player = game.state.player_at_turn;
    out.push(GameAction {
        action_type: ActionType::StopBidding,
        player,
    });
    for allowed_value in (start_value.0..=MAX_BID).step_by(BID_STEP as usize) {
        out.push(GameAction {
            action_type: ActionType::NewBid(allowed_value),
            player,
        })
    }
}

/// Every 4-card subset of the hand, in the order of `itertools::combinations` over the hand,
/// each sorted from high to low.
#[cfg(feature = "public-api")]
pub fn legal_passing(game: &Game) -> Vec<GameAction> {
    let mut out = vec![];
    push_passing(game, &mut out);
    out
}

fn push_passing(game: &Game, out: &mut Vec<GameAction>) {
    let cards = &game.state.player_at_turn().cards;
    let player = game.state.player_at_turn;
    let n = cards.len();
    if n < 4 {
        return;
    }
    out.reserve(n * (n - 1) * (n - 2) * (n - 3) / 24);
    let mut generate = |#[cfg(feature = "public-api")] pool: &mut Vec<Vec<Card>>| {
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    for d in c + 1..n {
                        let pass = sorted_desc([
                            cards[a].index(),
                            cards[b].index(),
                            cards[c].index(),
                            cards[d].index(),
                        ])
                        .map(Card::from_index);
                        #[cfg(feature = "public-api")]
                        let pass = {
                            let mut buffer =
                                pool.pop().unwrap_or_else(|| Vec::with_capacity(pass.len()));
                            buffer.clear();
                            buffer.extend_from_slice(&pass);
                            buffer
                        };
                        out.push(GameAction {
                            action_type: ActionType::Pass(pass),
                            player,
                        })
                    }
                }
            }
        }
    };
    #[cfg(feature = "public-api")]
    PASS_POOL.with(|pool| generate(&mut pool.borrow_mut()));
    #[cfg(not(feature = "public-api"))]
    generate();
}

/// Four card indices from high to low (a sorting network).
fn sorted_desc(mut x: [u8; 4]) -> [u8; 4] {
    for (i, j) in [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)] {
        if x[i] < x[j] {
            x.swap(i, j);
        }
    }
    x
}

// A game generates 126 + 715 four-card passes, each a `Vec<Card>` by the public type. Their
// buffers are recycled per thread (at most `PASS_POOL_MAX`), so a simulation loop does not
// allocate them again; measured in the pull request that added it.
#[cfg(feature = "public-api")]
const PASS_POOL_MAX: usize = 1024;

#[cfg(feature = "public-api")]
thread_local! {
    static PASS_POOL: RefCell<Vec<Vec<Card>>> = const { RefCell::new(Vec::new()) };
}

/// Empties `actions`, keeping the buffers of passes for the next passing list.
#[cfg(feature = "public-api")]
pub(crate) fn recycle(actions: &mut Vec<GameAction>) {
    if !matches!(
        actions.first().map(|a| &a.action_type),
        Some(ActionType::Pass(_))
    ) {
        actions.clear();
        return;
    }
    PASS_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for action in actions.drain(..) {
            if let ActionType::Pass(v) = action.action_type {
                if pool.len() < PASS_POOL_MAX {
                    pool.push(v);
                }
            }
        }
    });
}

/// Whether `cards` is a pass the player at turn may make: four cards of their hand from high
/// to low, i.e. exactly the members of `legal_passing`.
#[cfg(not(feature = "public-api"))]
pub(crate) fn is_legal_pass(game: &Game, cards: &[Card]) -> bool {
    let hand = &game.state.player_at_turn().cards;
    cards.len() == 4
        && cards.windows(2).all(|w| w[0] > w[1])
        && cards.iter().all(|c| hand.contains(c))
}

/// The first trick is played while the player still holds a full hand.
pub fn is_first_trick(hand_len: usize) -> bool {
    hand_len == HAND_SIZE
}

/// The cards the player at turn may play (the rule: `bits::play_levels`), in hand order.
#[cfg(feature = "public-api")]
pub fn legal_cards(game: &Game) -> Vec<GameAction> {
    let mut out = Vec::with_capacity(9);
    push_cards(game, &mut out);
    out
}

fn allowed_cards(game: &Game) -> CardSet {
    let state = &game.state;
    let cards = &state.player_at_turn().cards;
    let trick = if state.current_trick.len() == 4 {
        &[][..]
    } else {
        &state.current_trick[..]
    };
    let mut idx = [0u8; 4];
    for (slot, c) in idx.iter_mut().zip(trick) {
        *slot = bits::index(c);
    }
    let hand = CardSet::from_cards(cards);
    let first_trick = is_first_trick(cards.len());
    bits::play_levels(&idx[..trick.len()], state.trump, first_trick).allowed(hand)
}

fn push_cards(game: &Game, out: &mut Vec<GameAction>) {
    let player = game.state.player_at_turn;
    let cards = &game.state.player_at_turn().cards;
    let allowed = allowed_cards(game);
    for &card in cards {
        if allowed.contains(bits::index(&card)) {
            out.push(GameAction {
                action_type: ActionType::CardPlayed(card),
                player,
            });
        }
    }
}

/// Suits of which `hand` holds both King and Ober, in enum order.
fn pair_suits(hand: CardSet) -> impl Iterator<Item = Suit> {
    bits::SUITS
        .into_iter()
        .filter(move |&s| CardSet::halves(s).is_subset(hand))
}

#[cfg(feature = "public-api")]
pub fn legal_question(game: &Game) -> Vec<GameAction> {
    let mut out = Vec::with_capacity(10);
    push_question(game, &mut out);
    out
}

fn push_question(game: &Game, actions: &mut Vec<GameAction>) {
    let player = game.state.player_at_turn();
    let place = player.place_at_table;
    let action = |action_type| GameAction {
        action_type,
        player: place,
    };
    if player.trump == PlayerTrumpPossibilities::Own {
        for suit in pair_suits(CardSet::from_cards(&player.cards)) {
            if !game.state.trump_called.contains(&suit) {
                actions.push(action(ActionType::AnnounceTrump(suit)));
            }
        }
    }
    if player.trump != PlayerTrumpPossibilities::Ours {
        actions.push(action(ActionType::Question(QuestionType::Yours)));
    }
    for suit in [Suit::Red, Suit::Bells, Suit::Acorns, Suit::Green] {
        actions.push(action(ActionType::Question(QuestionType::YourHalf(suit))));
    }
}

#[cfg(feature = "public-api")]
pub fn legal_answer(game: &Game) -> Vec<GameAction> {
    let mut out = Vec::with_capacity(4);
    push_answer(game, &mut out);
    out
}

fn push_answer(game: &Game, actions: &mut Vec<GameAction>) {
    let question_event = game
        .all_events
        .iter()
        .rev()
        .find(|e| matches!(e.last_action.action_type, ActionType::Question(..)))
        .expect("Trying to find answers without question asked!");
    let hand = CardSet::from_cards(&game.state.player_at_turn().cards);
    let player = game.state.player_at_turn;
    let start = actions.len();
    match question_event.last_action.action_type {
        ActionType::Question(QuestionType::Yours) => {
            for suit in pair_suits(hand) {
                //don't allow double calling
                if !game.state.trump_called.contains(&suit) {
                    actions.push(GameAction {
                        action_type: ActionType::Answer(AnswerType::YesPair(suit)),
                        player,
                    });
                }
            }
            if actions.len() == start {
                actions.push(GameAction {
                    action_type: ActionType::Answer(AnswerType::NoPair),
                    player,
                })
            }
        }
        ActionType::Question(QuestionType::YourHalf(suit)) => {
            let answer = if (hand & CardSet::halves(suit)).is_empty() {
                AnswerType::NoHalf(suit)
            } else {
                AnswerType::YesHalf(suit)
            };
            actions.push(GameAction {
                action_type: ActionType::Answer(answer),
                player,
            });
        }
        _ => {
            println!("{:?}", game);
            panic!("Trying to find answers without question asked!")
        }
    }
}
