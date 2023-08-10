use crate::game::cards::{Card, Suit};
use crate::game::player::PlaceAtTable;
use serde::Serialize;
use std::fmt::Debug;

/*
This is everything that happened since the last game state.
Meant to broadcast implicit information about the game that follows actions.
 */
#[derive(Debug, Clone, Serialize)]
pub struct GameEvent {
    pub last_action: GameAction,
    pub callback: Option<GameCallback>, //inner change that can not be known from single last action
    pub next_player_at_turn: PlaceAtTable,
    pub time: String,
}

/*
Meant for broadcasting, hides passing cards.
 */
#[derive(Debug, Clone)]
pub enum GameEventPlayer {
    PublicEvent(GameEvent),
    HiddenEvent,
}

/*
Internal information after each action, i.e. questions, answers and trump changes.
 */
#[derive(Debug, Clone, Serialize)]
pub enum GameCallback {
    NewTrump(Suit),
    StillTrump(Suit), //when asked again for half but is already trump
    NoHalf(Suit),
    OnlyHalf(Suit),
}

/*
This is what a player can create.
 */
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GameAction {
    pub action_type: ActionType,
    pub player: PlaceAtTable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ActionType {
    Start,
    NewBid(i32),
    StopBidding,
    Pass(Vec<Card>),
    CardPlayed(Card),
    Question(QuestionType),
    Answer(AnswerType),
    AnnounceTrump(Suit),
    UndoRequest,
    UndoDecline,
    UndoAccept,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum QuestionType {
    Yours,
    YourHalf(Suit),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum AnswerType {
    YesPair(Suit),
    NoPair,
    YesHalf(Suit),
    NoHalf(Suit),
}
