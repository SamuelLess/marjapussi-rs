use crate::game::cards::{Card, Suit};
use crate::game::gameevent::ActionType;
use crate::game::player::{PlaceAtTable, Player};
use crate::game::points::Points;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    WaitingForStart,
    Bidding,
    PassingForth,
    PassingBack,
    Raising,
    Trick,
    StartTrick,
    AnsweringPair,
    AnsweringHalf(Suit),
    Ended,
    PendingUndo(Box<GamePhase>),
}

#[derive(Debug, Clone, Serialize)]
pub struct FinishedTrick {
    pub cards: [Card; 4],
    pub winner: PlaceAtTable,
    pub points: Points,
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub phase: GamePhase,
    pub started: bool,
    pub players_started: Vec<PlaceAtTable>,
    pub players_accept_undo: Vec<PlaceAtTable>,
    pub bidding_players: u8, //starts at 4
    pub bidding_history: Vec<(ActionType, PlaceAtTable)>,
    pub trump: Option<Suit>,
    pub trump_called: Vec<Suit>,
    pub player_at_turn: PlaceAtTable,
    pub players: [Player; 4],
    pub value: Points,
    pub all_tricks: Vec<FinishedTrick>,
    pub current_trick: Vec<Card>,
}

impl GameState {
    pub fn player_at_turn(&self) -> &Player {
        &self.players[self.player_at_turn.0 as usize]
    }
    pub fn player_at_turn_mut(&mut self) -> &mut Player {
        &mut self.players[self.player_at_turn.0 as usize]
    }
    pub fn partner(&self) -> &Player {
        &self.players[self.player_at_turn.partner().0 as usize]
    }
    pub fn prev_player(&self) -> &Player {
        &self.players[self.player_at_turn.prev().0 as usize]
    }
    pub fn player_at_place(&self, place: PlaceAtTable) -> &Player {
        &self.players[place.0 as usize]
    }
    pub fn player_at_place_mut(&mut self, place: PlaceAtTable) -> &mut Player {
        &mut self.players[place.0 as usize]
    }

    pub fn players_perspective(&self, place: PlaceAtTable) -> [String; 4] {
        [
            self.player_at_place(place.clone()).name.clone(),
            self.player_at_place(place.next()).name.clone(),
            self.player_at_place(place.partner()).name.clone(),
            self.player_at_place(place.prev()).name.clone(),
        ]
    }

    pub fn players_perspective_cards(&self, place: PlaceAtTable) -> [u8; 4] {
        [
            self.player_at_place(place.clone()).cards.len() as u8,
            self.player_at_place(place.next()).cards.len() as u8,
            self.player_at_place(place.partner()).cards.len() as u8,
            self.player_at_place(place.prev()).cards.len() as u8,
        ]
    }

    pub fn players_started(&self) -> Vec<String> {
        let mut started = vec![];
        for player in &self.players {
            if self.players_started.contains(&player.place_at_table) {
                started.push(player.name.clone());
            }
        }
        started
    }
}
