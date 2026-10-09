use serde::Serialize;

use crate::game::cards::{Card, Suit};
use crate::game::gameevent::ActionType;
use crate::game::player::{PlaceAtTable, Player};
use crate::game::points::Points;

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

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FinishedTrick {
    pub cards: [Card; 4],
    pub winner: PlaceAtTable,
    pub points: Points,
}

#[derive(Debug)]
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

impl Clone for GameState {
    fn clone(&self) -> Self {
        GameState {
            phase: self.phase.clone(),
            started: self.started,
            players_started: self.players_started.clone(),
            players_accept_undo: self.players_accept_undo.clone(),
            bidding_players: self.bidding_players,
            bidding_history: self.bidding_history.clone(),
            trump: self.trump,
            trump_called: self.trump_called.clone(),
            player_at_turn: self.player_at_turn,
            players: self.players.clone(),
            value: self.value,
            all_tricks: self.all_tricks.clone(),
            current_trick: self.current_trick.clone(),
        }
    }

    /// Reuses this state's buffers: the undo snapshot is refreshed on every bid and card.
    fn clone_from(&mut self, source: &Self) {
        self.phase.clone_from(&source.phase);
        self.started = source.started;
        self.players_started.clone_from(&source.players_started);
        self.players_accept_undo
            .clone_from(&source.players_accept_undo);
        self.bidding_players = source.bidding_players;
        self.bidding_history.clone_from(&source.bidding_history);
        self.trump = source.trump;
        self.trump_called.clone_from(&source.trump_called);
        self.player_at_turn = source.player_at_turn;
        for (p, q) in self.players.iter_mut().zip(&source.players) {
            p.clone_from(q);
        }
        self.value = source.value;
        self.all_tricks.clone_from(&source.all_tricks);
        self.current_trick.clone_from(&source.current_trick);
    }
}

impl GameState {
    pub fn create(players: [Player; 4]) -> Self {
        GameState {
            started: false,
            players_started: vec![],
            players_accept_undo: vec![],
            phase: GamePhase::WaitingForStart,
            trump: None,
            trump_called: vec![],
            player_at_turn: PlaceAtTable(0),
            value: Points(115),
            bidding_players: 4,
            bidding_history: vec![],
            players,
            all_tricks: vec![],
            current_trick: vec![],
        }
    }
    pub fn player_at_turn(&self) -> &Player {
        &self.players[self.player_at_turn.0 as usize]
    }

    pub fn player_at_turn_mut(&mut self) -> &mut Player {
        &mut self.players[self.player_at_turn.0 as usize]
    }

    pub fn partner(&self) -> &Player {
        &self.players[self.player_at_turn.partner().0 as usize]
    }

    pub fn partner_mut(&mut self) -> &mut Player {
        let partner_idx = self.player_at_turn.partner().0 as usize;
        &mut self.players[partner_idx]
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
            self.player_at_place(place).name.clone(),
            self.player_at_place(place.next()).name.clone(),
            self.player_at_place(place.partner()).name.clone(),
            self.player_at_place(place.prev()).name.clone(),
        ]
    }

    pub fn players_perspective_cards(&self, place: PlaceAtTable) -> [u8; 4] {
        [
            self.player_at_place(place).cards.len() as u8,
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
