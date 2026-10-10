use crate::bits::{self, CardSet};
use crate::game::cards::Card;
#[cfg(feature = "public-api")]
use crate::game::current_time_string;
use crate::game::gameevent::{ActionType, AnswerType, GameAction, GameCallback, QuestionType};
#[cfg(feature = "public-api")]
use crate::game::gameinfo::GameMetaInfo;
use crate::game::gamestate::{FinishedTrick, GamePhase, GameState};
use crate::game::player::PlayerTrumpPossibilities;
use crate::game::points::{Points, CARD_POINTS, LAST_TRICK_BONUS};
use crate::game::Game;

#[cfg(feature = "public-api")]
impl ActionType {
    /// Returns the next meta info, state, callback and undo snapshot without changing `game`.
    pub fn apply_action(
        self,
        action: &GameAction,
        game: &Game,
    ) -> (
        GameMetaInfo,
        GameState,
        Option<GameCallback>,
        Option<GameState>,
    ) {
        let action = GameAction {
            action_type: self,
            player: action.player,
        };
        let mut next = Game {
            info: game.info.clone(),
            state: game.state.clone(),
            legal_actions: vec![],
            last_state: game.last_state.clone(),
            all_events: vec![],
        };
        let callback = next.apply_in_place(&action);
        (next.info, next.state, callback, next.last_state)
    }
}

impl Game {
    /// Applies a legal action to `info`, `state` and `last_state` in place and returns the
    /// callback. `last_state` (the undo snapshot) is the state before a bid or a card, is kept
    /// by undo requests and accepts, and is cleared by every other action.
    pub(crate) fn apply_in_place(&mut self, action: &GameAction) -> Option<GameCallback> {
        let mut callback = None;
        let state = &mut self.state;
        match action.action_type {
            ActionType::Start => {
                if !state.players_started.contains(&action.player) {
                    state.players_started.push(action.player);
                }
                if state.players_started.len() == 4 {
                    state.started = true;
                    #[cfg(feature = "public-api")]
                    {
                        self.info.start_time = Some(current_time_string());
                    }
                    state.phase = GamePhase::Bidding;
                }
                self.last_state = None;
            }
            ActionType::NewBid(value) => {
                snapshot(&mut self.last_state, state);
                state
                    .bidding_history
                    .push((action.action_type.to_owned(), action.player));
                state.value = Points(value);
                if state.phase == GamePhase::Raising {
                    state.phase = GamePhase::Trick;
                } else {
                    let mut next_player = state.player_at_turn.next();
                    while !state.player_at_place(next_player).bidding {
                        next_player = next_player.next();
                    }
                    if next_player == state.player_at_turn {
                        //same player can't bid against himself
                        state.phase = GamePhase::PassingForth;
                        state.player_at_turn = state.player_at_turn.partner();
                    } else {
                        // continue bidding
                        state.player_at_turn = next_player;
                    }
                }
            }
            ActionType::StopBidding => {
                snapshot(&mut self.last_state, state);
                state
                    .bidding_history
                    .push((action.action_type.to_owned(), action.player));
                state.player_at_turn_mut().bidding = false;
                state.bidding_players -= 1;
                let mut next_player = state.player_at_turn.next();
                if state.bidding_players >= 1 {
                    while !state.player_at_place(next_player).bidding {
                        next_player = next_player.next();
                    }
                }
                state.player_at_turn = next_player;

                //bidding ends
                if state.bidding_players == 1 && state.value > Points(115) {
                    state.phase = GamePhase::PassingForth;
                    for player in &state.players {
                        if player.bidding {
                            state.player_at_turn = player.place_at_table.partner();
                        }
                    }
                }
                if state.bidding_players == 0 {
                    // nobody takes game
                    state.phase = GamePhase::Trick;
                }
            }
            ActionType::Pass(ref cards) => {
                state
                    .player_at_place_mut(action.player)
                    .cards
                    .retain(|x| !cards.contains(x));
                state
                    .player_at_place_mut(action.player.partner())
                    .cards
                    .extend_from_slice(cards);
                if state.phase == GamePhase::PassingBack {
                    state.phase = GamePhase::Raising;
                } else {
                    state.phase = GamePhase::PassingBack;
                    state.player_at_turn = action.player.partner();
                };
                self.last_state = None;
            }
            ActionType::CardPlayed(card) => {
                snapshot(&mut self.last_state, state);
                act_card(card, state);
                state.player_at_place_mut(action.player).play_card(card);
                if state.player_at_turn().cards.is_empty() {
                    state.phase = GamePhase::Ended;
                    #[cfg(feature = "public-api")]
                    {
                        self.info.end_time = Some(current_time_string());
                    }
                }
            }
            ActionType::AnnounceTrump(suit) => {
                //can only happen once per suit
                callback = Some(GameCallback::NewTrump(suit));
                state.trump_called.push(suit);
                state.trump = Some(suit);
                state.phase = GamePhase::Trick;
                self.last_state = None;
            }
            ActionType::Question(QuestionType::Yours) => {
                let asker = state.player_at_place_mut(action.player);
                if asker.trump == PlayerTrumpPossibilities::Own {
                    asker.trump = PlayerTrumpPossibilities::Yours;
                }
                state.phase = GamePhase::AnsweringPair;
                state.player_at_turn = state.player_at_turn.partner();
                self.last_state = None;
            }
            ActionType::Question(QuestionType::YourHalf(suit)) => {
                //can happen multiple times per suit
                state.player_at_place_mut(action.player).trump = PlayerTrumpPossibilities::Ours;
                state.phase = GamePhase::AnsweringHalf(suit);
                state.player_at_turn = state.player_at_turn.partner();
                self.last_state = None;
            }
            ActionType::Answer(AnswerType::NoPair) => {
                state.phase = GamePhase::Trick;
                state.player_at_turn = state.player_at_turn.partner();
                self.last_state = None;
            }
            ActionType::Answer(AnswerType::YesPair(suit)) => {
                //can happen only once per suit
                callback = Some(GameCallback::NewTrump(suit));
                state.trump_called.push(suit);
                state.trump = Some(suit);
                state.phase = GamePhase::Trick;
                state.player_at_turn = state.partner().place_at_table;
                self.last_state = None;
            }
            ActionType::Answer(AnswerType::NoHalf(suit)) => {
                callback = Some(GameCallback::NoHalf(suit));
                state.phase = GamePhase::Trick;
                state.player_at_turn = state.partner().place_at_table;
                self.last_state = None;
            }
            ActionType::Answer(AnswerType::YesHalf(suit)) => {
                //can happen multiple times per suit
                let asker_hand = CardSet::from_cards(&state.partner().cards);
                if !(asker_hand & CardSet::halves(suit)).is_empty() {
                    if state.trump_called.contains(&suit) {
                        callback = Some(GameCallback::StillTrump(suit));
                    } else {
                        callback = Some(GameCallback::NewTrump(suit));
                        state.trump_called.push(suit);
                        state.trump = Some(suit);
                    }
                } else {
                    callback = Some(GameCallback::OnlyHalf(suit));
                }
                state.player_at_turn = state.partner().place_at_table;
                state.phase = GamePhase::Trick;
                self.last_state = None;
            }
            ActionType::UndoAccept => {
                if !state.players_accept_undo.contains(&action.player) {
                    state.players_accept_undo.push(action.player);
                }
                if state.players_accept_undo.len() == 2 {
                    if let Some(mut previous) = self.last_state.take() {
                        previous.players_accept_undo.clear();
                        *state = previous;
                    }
                }
            }
            ActionType::UndoDecline => {
                if let GamePhase::PendingUndo(previous_phase) = &mut state.phase {
                    let previous = std::mem::replace(&mut **previous_phase, GamePhase::Ended);
                    state.phase = previous;
                    state.players_accept_undo.clear();
                }
                self.last_state = None;
            }
            ActionType::UndoRequest => {
                //last_state is always Some() because otherwise action not legal
                if self.last_state.is_some() {
                    let phase = std::mem::replace(&mut state.phase, GamePhase::Ended);
                    state.phase = GamePhase::PendingUndo(Box::new(phase));
                }
            }
        }
        callback
    }
}

/// Stores `state` as the undo snapshot, reusing the previous snapshot's buffers.
fn snapshot(last_state: &mut Option<GameState>, state: &GameState) {
    match last_state {
        Some(last) => last.clone_from(state),
        None => *last_state = Some(state.clone()),
    }
}

/// Puts `card` into the current trick and, when the trick is full, scores it and gives the
/// turn to its winner (rule owner: `bits::trick_winner`).
pub fn act_card(card: Card, next_game_state: &mut GameState) {
    let state = next_game_state;
    if state.current_trick.len() >= 4 {
        state.current_trick.clear();
    }
    state.current_trick.push(card);
    state.phase = GamePhase::Trick;
    state.player_at_turn = state.player_at_turn.next();
    if state.current_trick.len() == 4 {
        let cards: [Card; 4] = [
            state.current_trick[0],
            state.current_trick[1],
            state.current_trick[2],
            state.current_trick[3],
        ];
        let idx = cards.map(|c| bits::index(&c));
        // After four plays the turn is back at the leader; the winner sits `winner` seats on.
        let winner = bits::trick_winner(&idx, state.trump);
        for _ in 0..winner {
            state.player_at_turn = state.player_at_turn.next();
        }
        state.phase = GamePhase::StartTrick;
        let mut points: i32 = idx.iter().map(|&i| CARD_POINTS[i as usize] as i32).sum();
        if state.all_tricks.len() == 8 {
            points += LAST_TRICK_BONUS;
        }
        state.all_tricks.push(FinishedTrick {
            cards,
            winner: state.player_at_turn,
            points: Points(points),
        });
    }
}
