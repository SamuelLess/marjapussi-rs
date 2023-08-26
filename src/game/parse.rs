use crate::game::cards::{Card, Suit, Value};
use crate::game::gameinfo::GameInfoDatabase;
use std::io::{Error, ErrorKind};
use std::iter::zip;
use strum::IntoEnumIterator;

pub fn parse_card(card: String) -> Result<Card, Error> {
    if card.len() != 3 {
        return Err(Error::new(ErrorKind::Other, "wrong card format"));
    }
    let suit_char = card.chars().next().unwrap();
    let value_char = card.chars().last().unwrap();
    let suits_str = "gesr".chars();
    let values_str = "6789UOKZA".chars();

    let mut suit: Option<Suit> = None;
    let mut value: Option<Value> = None;

    let suits = Suit::iter();
    let values = Value::iter();

    for (c, s) in zip(suits_str, suits) {
        if c == suit_char {
            suit = Some(s);
        }
    }

    for (c, v) in zip(values_str, values) {
        if c == value_char {
            value = Some(v);
        }
    }

    if value.is_none() || suit.is_none() {
        return Err(Error::new(ErrorKind::Other, "wrong card format"));
    }

    Ok(Card {
        suit: suit.unwrap(),
        value: value.unwrap(),
    })
}

pub fn parse_python_format(_data: String) -> GameInfoDatabase {
    todo!()
}
