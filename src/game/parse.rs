use crate::game::cards::{Card, Suit, Value};
use crate::game::gameinfo::GameInfoDatabase;
use crate::game::Game;
use serde::Deserialize;
use std::collections::HashMap;
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
        return Err(Error::new(ErrorKind::Other, "Wrong card format"));
    }

    Ok(Card {
        suit: suit.unwrap(),
        value: value.unwrap(),
    })
}

#[derive(Debug, Deserialize)]
struct LegacyGameFormat {
    name: String,
    created: String,
    started: String,
    finished: String,
    players: Vec<String>,
    cards: HashMap<String, Vec<String>>,
    game_value: i32,
    actions: Vec<String>,
    players_sup: HashMap<String, Vec<String>>,
    players_points: HashMap<String, i32>,
    tricks: Vec<Vec<String>>,
    schwarz_game: bool,
}

fn parse_cards_vec(cards: Vec<String>) -> Vec<Card> {
    cards.into_iter().map(|c| parse_card(c).unwrap()).collect()
}

pub fn parse_legacy_format(data: String) -> Result<GameInfoDatabase, Error> {
    let game_data: LegacyGameFormat = serde_json::from_str(data.as_str())?;
    println!("{:#?}", game_data);
    let names: [String; 4] = game_data.players.clone().try_into().unwrap();

    let cards = [
        parse_cards_vec(game_data.cards.get(&names[0]).unwrap().clone()),
        parse_cards_vec(game_data.cards.get(&names[1]).unwrap().clone()),
        parse_cards_vec(game_data.cards.get(&names[2]).unwrap().clone()),
        parse_cards_vec(game_data.cards.get(&names[3]).unwrap().clone()),
    ];

    let game_replay = Game::new(game_data.name, names, Some(cards));

    println!("{:#?}", game_replay);

    for _action in game_data.actions {}

    //let obj = game_data.ok().unwrap();
    println!("---got to the end of the function---");
    Err(Error::new(
        ErrorKind::Other,
        "The game could not be parsed.",
    ))
}
