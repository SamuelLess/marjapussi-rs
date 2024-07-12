use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use std::iter::zip;

use serde::Deserialize;
use strum::IntoEnumIterator;

use crate::game::cards::{Card, Suit, Value};
use crate::game::gameinfo::GameInfoDatabase;
use crate::game::Game;

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

mod tests {
    use super::*;

    #[test]
    pub fn test_parse_card() {
        assert_eq!(
            parse_card(String::from("r-A")).unwrap(),
            Card {
                suit: Suit::Red,
                value: Value::Ace,
            }
        );
        assert_eq!(
            parse_card(String::from("g-O")).unwrap(),
            Card {
                suit: Suit::Green,
                value: Value::Ober,
            }
        );
        assert_eq!(
            parse_card(String::from("s-6")).unwrap(),
            Card {
                suit: Suit::Bells,
                value: Value::Six,
            }
        );
    }

    #[test]
    pub fn test_parse_python_game() {
        let input: String = String::from(
            r#"
        {
           "_id":{
              "$oid":"643b0e6be1f3816d1dad798f"
           },
           "name":"thüringen",
           "created":"2023-04-15 22:45:13",
           "started":"2023-04-15 22:45:34",
           "series_id":"",
           "finished":"2023-04-15 22:51:55",
           "players":[
              "marjastephen",
              "Elinka",
              "Jonas",
              "Gast1"
           ],
           "cards":{
              "marjastephen":[
                 "r-6",
                 "e-Z",
                 "e-K",
                 "e-7",
                 "g-K",
                 "g-O",
                 "g-8",
                 "g-7",
                 "g-6"
              ],
              "Elinka":[
                 "r-A",
                 "r-O",
                 "r-U",
                 "r-9",
                 "s-6",
                 "e-A",
                 "e-O",
                 "g-A",
                 "g-9"
              ],
              "Jonas":[
                 "r-7",
                 "s-A",
                 "s-9",
                 "s-8",
                 "e-U",
                 "e-9",
                 "e-6",
                 "g-Z",
                 "g-U"
              ],
              "Gast1":[
                 "r-Z",
                 "r-K",
                 "r-8",
                 "s-Z",
                 "s-K",
                 "s-O",
                 "s-U",
                 "s-7",
                 "e-8"
              ]
           },
           "passed_cards":{
              "forth":[
                 "g-9",
                 "g-A",
                 "e-O",
                 "e-A"
              ],
              "back":[
                 "s-7",
                 "r-K",
                 "e-8",
                 "e-O"
              ]
           },
           "tricks":[
              [
                 "e-A",
                 "e-7",
                 "e-8",
                 "e-6"
              ],
              [
                 "g-A",
                 "g-6",
                 "s-7",
                 "g-U"
              ],
              [
                 "g-9",
                 "g-O",
                 "s-6",
                 "g-Z"
              ],
              [
                 "r-A",
                 "r-7",
                 "r-8",
                 "r-6"
              ],
              [
                 "e-O",
                 "e-9",
                 "r-Z",
                 "e-K"
              ],
              [
                 "s-U",
                 "g-7",
                 "r-9",
                 "s-8"
              ],
              [
                 "r-K",
                 "e-U",
                 "s-K",
                 "g-8"
              ],
              [
                 "r-O",
                 "s-9",
                 "s-O",
                 "g-K"
              ],
              [
                 "r-U",
                 "s-A",
                 "s-Z",
                 "e-Z"
              ]
           ],
           "actions":[
              "0,PROV,125",
              "1,PROV,130",
              "2,PROV,135",
              "3,PROV,155",
              "0,PROV,0",
              "1,PROV,0",
              "2,PROV,0",
              "1,PASS,g-9",
              "1,PASS,g-A",
              "1,PASS,e-O",
              "1,PASS,e-A",
              "3,PBCK,s-7",
              "3,PBCK,r-K",
              "3,PBCK,e-8",
              "3,PBCK,e-O",
              "3,PRMO,0",
              "3,TRCK,e-A",
              "0,TRCK,e-7",
              "1,TRCK,e-8",
              "2,TRCK,e-6",
              "3,TRCK,g-A",
              "0,TRCK,g-6",
              "1,TRCK,s-7",
              "2,TRCK,g-U",
              "3,QUES,mys",
              "3,TRCK,g-9",
              "0,TRCK,g-O",
              "1,TRCK,s-6",
              "2,TRCK,g-Z",
              "1,QUES,myr",
              "1,TRCK,r-A",
              "2,TRCK,r-7",
              "3,TRCK,r-8",
              "0,TRCK,r-6",
              "1,TRCK,e-O",
              "2,TRCK,e-9",
              "3,TRCK,r-Z",
              "0,TRCK,e-K",
              "3,TRCK,s-U",
              "0,TRCK,g-7",
              "1,TRCK,r-9",
              "2,TRCK,s-8",
              "1,TRCK,r-K",
              "2,TRCK,e-U",
              "3,TRCK,s-K",
              "0,TRCK,g-8",
              "1,QUES,you",
              "3,ANSW,nmy",
              "1,TRCK,r-O",
              "2,TRCK,s-9",
              "3,TRCK,s-O",
              "0,TRCK,g-K",
              "1,TRCK,r-U",
              "2,TRCK,s-A",
              "3,TRCK,s-Z",
              "0,TRCK,e-Z"
           ],
           "playing_player":"Gast1",
           "game_value":155,
           "players_points":{
              "marjastephen":0,
              "Elinka":199,
              "Jonas":0,
              "Gast1":121
           },
           "players_sup":{
              "marjastephen":[

              ],
              "Elinka":[
                 "r"
              ],
              "Jonas":[

              ],
              "Gast1":[
                 "s"
              ]
           },
           "schwarz_game":true
        }
      "#,
        );
        let _result = parse_legacy_format(input);
    }
}
