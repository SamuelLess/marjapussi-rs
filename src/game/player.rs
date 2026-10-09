use std::fmt::Debug;

use rand::rng;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::game::cards::{get_all_cards, Card};

/// Cards dealt to each player.
pub const HAND_SIZE: usize = 9;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct PlaceAtTable(pub u8);

impl PlaceAtTable {
    pub fn partner(&self) -> Self {
        PlaceAtTable((self.0 + 2) % 4)
    }

    pub fn next(&self) -> Self {
        PlaceAtTable((self.0 + 1) % 4)
    }

    pub fn prev(&self) -> Self {
        PlaceAtTable((self.0 + 3) % 4)
    }
    pub(crate) fn party(&self) -> Self {
        PlaceAtTable(self.0 % 2)
    }
}

impl Debug for PlaceAtTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Seat({})", self.0)
    }
}

pub struct Player {
    pub name: String,
    pub partner: PlaceAtTable,
    pub next_player: PlaceAtTable,
    pub place_at_table: PlaceAtTable,
    pub cards: Vec<Card>,
    last_played: Option<Card>,
    pub tricks: Vec<Vec<Card>>,
    pub trump: PlayerTrumpPossibilities,
    pub bidding: bool,
}

impl Clone for Player {
    fn clone(&self) -> Self {
        Player {
            name: self.name.clone(),
            partner: self.partner,
            next_player: self.next_player,
            place_at_table: self.place_at_table,
            cards: self.cards.clone(),
            last_played: self.last_played,
            tricks: self.tricks.clone(),
            trump: self.trump,
            bidding: self.bidding,
        }
    }

    /// Reuses the name and card buffers (see `GameState::clone_from`).
    fn clone_from(&mut self, source: &Self) {
        self.name.clone_from(&source.name);
        self.partner = source.partner;
        self.next_player = source.next_player;
        self.place_at_table = source.place_at_table;
        self.cards.clone_from(&source.cards);
        self.last_played = source.last_played;
        self.tricks.clone_from(&source.tricks);
        self.trump = source.trump;
        self.bidding = source.bidding;
    }
}

impl PartialEq for Player {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Player {}

impl Debug for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Player")
            .field("name", &self.name)
            .field("cards", &self.cards.len())
            .finish()
    }
}

impl Player {
    pub fn partner_place(&self) -> PlaceAtTable {
        self.place_at_table.partner()
    }
    pub fn play_card(&mut self, card: Card) {
        self.cards.retain(|c| *c != card);
        self.last_played = Some(card);
    }
    pub fn undo_play_card(&mut self) {
        if let Some(last_card) = self.last_played {
            self.cards.push(last_card)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerTrumpPossibilities {
    Own,
    Yours,
    Ours,
}

/*
Creates a Single Player without partner or next_player
 */
fn create_player(name: String, cards: Vec<Card>, place: u8) -> Player {
    Player {
        name,
        bidding: true,
        next_player: PlaceAtTable(place).partner(),
        partner: PlaceAtTable(place).partner(),
        place_at_table: PlaceAtTable(place),
        cards,
        last_played: None,
        tricks: vec![],
        trump: PlayerTrumpPossibilities::Own,
    }
}

pub fn create_players(names: [String; 4], cards: Option<[Vec<Card>; 4]>) -> [Player; 4] {
    let players_cards = cards.unwrap_or_else(|| {
        //random shuffled cads
        let mut deck = get_all_cards();
        deck.shuffle(&mut rng());
        [0, 1, 2, 3].map(|i| deck[i * HAND_SIZE..(i + 1) * HAND_SIZE].to_vec())
    });

    let p0 = create_player(names[0].clone(), players_cards[0].clone(), 0);
    let p1 = create_player(names[1].clone(), players_cards[1].clone(), 1);
    let p2 = create_player(names[2].clone(), players_cards[2].clone(), 2);
    let p3 = create_player(names[3].clone(), players_cards[3].clone(), 3);

    [p0, p1, p2, p3]
}
