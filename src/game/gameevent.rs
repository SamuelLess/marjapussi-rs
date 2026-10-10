use crate::storage::{Pass, Time};
use std::fmt::Debug;

use serde::Serialize;

use crate::game::cards::{Card, Suit};
use crate::game::player::PlaceAtTable;

/// This is everything that happened since the last game state.
/// Meant to broadcast implicit information about the game that follows actions
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(not(feature = "public-api"), derive(Copy))]
pub struct GameEvent {
    pub last_action: GameAction,
    /// Inner change that can not be known from single last action
    pub callback: Option<GameCallback>,
    pub player_at_turn: PlaceAtTable,
    pub time: Time,
}

/// Meant for broadcasting, hides passing cards.
#[derive(Debug, Clone)]
#[cfg(feature = "public-api")]
pub enum GameEventPlayer {
    PublicEvent(GameEvent),
    HiddenEvent,
}

/// Internal information after each action, i.e. questions, answers and trump changes.
#[derive(Debug, Clone, Copy, Serialize)]
pub enum GameCallback {
    NewTrump(Suit),
    /// A repeated half question confirms a previously announced suit; current trump is unchanged.
    StillTrump(Suit),
    NoHalf(Suit),
    OnlyHalf(Suit),
}

/// This is what a player can create.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(not(feature = "public-api"), derive(Copy))]
pub struct GameAction {
    pub action_type: ActionType,
    pub player: PlaceAtTable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(not(feature = "public-api"), derive(Copy))]
pub enum ActionType {
    Start,
    NewBid(i32),
    StopBidding,
    Pass(Pass),
    CardPlayed(Card),
    Question(QuestionType),
    Answer(AnswerType),
    AnnounceTrump(Suit),
    UndoRequest,
    UndoDecline,
    UndoAccept,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QuestionType {
    Yours,
    YourHalf(Suit),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AnswerType {
    YesPair(Suit),
    NoPair,
    YesHalf(Suit),
    NoHalf(Suit),
}
