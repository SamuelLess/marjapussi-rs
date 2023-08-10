#![allow(unused)]
extern crate test;

use std::cmp::max;
use std::fmt;

use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use itertools::concat;

use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Card {
    /**
     * Only compare cards with same color.
     */
    pub suit: Suit,
    pub value: Value,
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.suit, self.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, EnumIter, Serialize)]
pub enum Suit {
    Green,
    Acorns,
    Bells,
    Red,
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, EnumIter, Serialize)]
pub enum Value {
    Six,
    Seven,
    Eight,
    Nine,
    Unter,
    Ober,
    King,
    Ten,
    Ace,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/**
 * Returns whether first card is higher than second card.
 */
pub fn is_higher_card(higher: &Card, lower: &Card, trump: Option<Suit>) -> bool {
    match trump {
        Some(suit) => {
            //println!("{:?} {:?} {:?}", higher, lower, trump);
            (higher.suit == suit && lower.suit != suit)
                || (higher.suit == lower.suit && higher.value > lower.value)
        }
        None => higher.suit == lower.suit && higher.value > lower.value,
    }
}

pub fn higher_cards(card: &Card, trump: Option<Suit>, pool: Option<Vec<Card>>) -> Vec<Card> {
    match pool {
        Some(cards) => cards,
        None => get_all_cards(),
    }
    .into_iter()
    .filter(|maybe| is_higher_card(maybe, card, trump))
    .collect()
}

/**
* Returns highest card if exists.
*/
pub fn high_card(trick: Vec<&Card>, trump: Option<Suit>) -> Option<&Card> {
    if trick.is_empty() {
        return None;
    }
    let trick_suit = trick[0].suit;
    Some(
        match trump {
            Some(trump_suit) => {
                //if any trump in trick only count trump
                let all_suits: Vec<Suit> = trick.iter().map(|c| c.suit).rev().collect();
                if all_suits.contains(&trump_suit) {
                    trick
                        .into_iter()
                        .filter(|x| x.suit == trump_suit)
                        .collect::<Vec<_>>()
                } else {
                    trick.into_iter().filter(|x| x.suit == trick_suit).collect()
                }
            }
            None => trick.into_iter().filter(|x| x.suit == trick_suit).collect(),
        }
        .iter()
        .reduce(max)
        .unwrap(),
    )
}

/**
* Only for the first played card in the game. Proper play in rest of first trick handled elsewhere.
*/
pub fn allowed_first(cards: Vec<&Card>) -> Vec<&Card> {
    let mut allowed = cards
        .clone()
        .into_iter()
        .filter(|c| c.value == Value::Ace)
        .collect::<Vec<_>>();
    if allowed.is_empty() {
        allowed = cards
            .clone()
            .into_iter()
            .filter(|c| c.suit == Suit::Green)
            .collect::<Vec<_>>();
    }
    if !allowed.is_empty() {
        allowed
    } else {
        cards
    }
}

/**
* Returns general high card.
*/
pub fn allowed_cards<'a>(
    trick: Vec<&'a Card>,
    cards: Vec<&'a Card>,
    trump: Option<Suit>,
    first_trick: bool,
) -> Vec<&'a Card> {
    let current_high_card = high_card(trick.clone(), trump);
    /*println!(
        "trick={:?} wiht cards={:?} and trump={:?}",
        trick, cards, trump
    );*/
    match current_high_card {
        Some(current_high_card) => {
            let trick_suit = trick[0].suit;
            if first_trick {
                let first_trick_ace: Vec<&Card> = cards
                    .clone()
                    .into_iter()
                    .filter(|c| c.suit == trick_suit && c.value == Value::Ace)
                    .collect();
                if first_trick_ace.len() == 1 {
                    return first_trick_ace;
                }
            }
            let higher_cards: Vec<&Card> = cards
                .clone()
                .into_iter()
                .filter(|c| {
                    high_card(concat([trick.clone(), vec![c]]), trump) > Some(current_high_card)
                })
                .collect();
            if !higher_cards.is_empty() {
                return higher_cards;
            }
            let same_color_cards: Vec<&Card> = cards
                .clone()
                .into_iter()
                .filter(|c: &&Card| c.suit == trick_suit)
                .collect();
            if !same_color_cards.is_empty() {
                return same_color_cards;
            }
            cards
        }
        None => {
            //the trick has to be empty
            if !first_trick {
                cards
            } else {
                allowed_first(cards.clone())
            }
        }
    }
}

pub fn halves(cards: Vec<Card>) -> Vec<Suit> {
    let mut halves: Vec<Suit> = vec![];
    for suit in [Suit::Red, Suit::Bells, Suit::Acorns, Suit::Green] {
        if cards.contains(&Card {
            suit,
            value: Value::Ober,
        }) || cards.contains(&Card {
            suit,
            value: Value::King,
        }) {
            halves.push(suit);
        }
    }
    halves
}

pub fn pairs(cards: Vec<Card>) -> Vec<Suit> {
    let mut halves: Vec<Suit> = vec![];
    for suit in [Suit::Red, Suit::Bells, Suit::Acorns, Suit::Green] {
        if cards.contains(&Card {
            suit,
            value: Value::Ober,
        }) && cards.contains(&Card {
            suit,
            value: Value::King,
        }) {
            halves.push(suit);
        }
    }
    halves
}

pub fn get_all_cards() -> Vec<Card> {
    let mut cards: Vec<Card> = vec![];
    for suit in Suit::iter() {
        for value in Value::iter() {
            cards.push(Card { suit, value });
        }
    }
    cards
}

pub fn print_cards(cards: &[Card]) {
    let card_strs: Vec<String> = cards.iter().map(|c| format!("{}", c)).collect();
    let s = card_strs.join(", ");
    println!("{}", s)
}
