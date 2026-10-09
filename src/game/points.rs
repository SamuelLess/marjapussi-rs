use std::ops::{Add, AddAssign};

use serde::Serialize;

use crate::game::cards::{Card, Suit};
use crate::game::player::PlaceAtTable;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Points(pub i32);

impl Add for Points {
    type Output = Points;
    fn add(self, rhs: Self) -> Self::Output {
        Points(self.0 + rhs.0)
    }
}

impl AddAssign for Points {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0 + rhs.0;
    }
}

/// The point table (single owner): points of a card by `Value`, in enum order
/// (6, 7, 8, 9, Unter, Ober, King, Ten, Ace).
pub const VALUE_POINTS: [i32; 9] = [0, 0, 0, 0, 2, 3, 4, 10, 11];

/// Points of an announced pair by `Suit`, in enum order (Green, Acorns, Bells, Red).
pub const PAIR_POINTS: [i32; 4] = [40, 60, 80, 100];

/// Bonus for winning the last trick.
pub const LAST_TRICK_BONUS: i32 = 20;

/// Points of each card by bit index (`bits::index`), derived from `VALUE_POINTS`.
pub const CARD_POINTS: [u8; 36] = {
    let mut t = [0u8; 36];
    let mut i = 0;
    while i < 36 {
        t[i] = VALUE_POINTS[i % 9] as u8;
        i += 1;
    }
    t
};

pub fn points_pair(suit: Suit) -> Points {
    Points(PAIR_POINTS[suit as usize])
}

pub fn points_card(card: Card) -> Points {
    Points(VALUE_POINTS[card.value as usize])
}

pub fn points_trick(trick: Vec<Card>) -> Points {
    trick
        .into_iter()
        .fold(Points(0), |acc, c| acc + points_card(c))
}

pub fn points_players(tricks: Vec<(Vec<Card>, PlaceAtTable)>) -> [Points; 4] {
    let mut points = [Points(0); 4];
    for (trick, place) in tricks {
        points[place.0 as usize] += points_trick(trick);
    }
    points
}
