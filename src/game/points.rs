use std::ops::{Add, AddAssign};

use serde::Serialize;

use crate::game::cards::{Card, Suit, Value};
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

pub fn points_pair(suit: Suit) -> Points {
    match suit {
        Suit::Red => Points(100),
        Suit::Bells => Points(80),
        Suit::Acorns => Points(60),
        Suit::Green => Points(40),
    }
}

pub fn points_card(card: Card) -> Points {
    match card.value {
        Value::Ace => Points(11),
        Value::Ten => Points(10),
        Value::King => Points(4),
        Value::Ober => Points(3),
        Value::Unter => Points(2),
        _ => Points(0),
    }
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
