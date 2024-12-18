use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use std::iter::zip;

use serde::Deserialize;
use strum::IntoEnumIterator;

use crate::game::cards::{Card, Suit, Value};
use crate::game::gameevent::{ActionType, AnswerType, GameAction, QuestionType};
use crate::game::gameinfo::GameFinishedInfo;
use crate::game::player::PlaceAtTable;
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

pub fn parse_cards(cards: Vec<String>) -> Vec<Card> {
    cards.into_iter().map(|c| parse_card(c).unwrap()).collect()
}

#[derive(Debug, Deserialize, Clone)]
pub struct LegacyGameFormat {
    /// The id of the game found in '_id.$oid'
    #[serde(rename = "_id")]
    id: String,
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
    #[serde(default)]
    schwarz_game: bool,
}

fn parse_action(action: String) -> Result<GameAction, Error> {
    let err = Err(Error::new(
        ErrorKind::Other,
        format!("The action {} could not be parsed.", action),
    ));

    let parts: Vec<&str> = action.split(',').collect();
    if parts.len() != 3 {
        return err;
    }
    let player_seat = parts[0].parse::<u8>().unwrap();
    let action_type = parts[1];
    let action_value = parts[2];

    let action_type: ActionType = match action_type {
        "PROV" => {
            let val = action_value.parse::<i32>().unwrap();
            if val == 0 {
                ActionType::StopBidding
            } else {
                ActionType::NewBid(val)
            }
        }
        "PRMO" => ActionType::NewBid(action_value.parse::<i32>().unwrap()),
        "TRCK" => {
            let card = parse_card(action_value.to_string())?;
            ActionType::CardPlayed(card)
        }
        "QUES" => parse_ques(action.clone()),
        "ANSW" => parse_answ(action.clone()),
        _ => return err,
    };

    Ok(GameAction {
        action_type,
        player: PlaceAtTable(player_seat),
    })
}

fn parse_pass(actions: Vec<String>) -> GameAction {
    let player_seat = &actions[0][0..1].parse::<u8>().unwrap();
    let mut cards = vec![];
    for action in actions {
        let parts: Vec<&str> = action.split(',').collect();
        let card = parse_card(parts[2].to_string()).unwrap();
        cards.push(card);
    }
    cards.sort();
    cards.reverse();
    GameAction {
        action_type: ActionType::Pass(cards),
        player: PlaceAtTable(*player_seat),
    }
}

fn parse_suit(suit: &str) -> Suit {
    match suit {
        "g" => Suit::Green,
        "e" => Suit::Acorns,
        "s" => Suit::Bells,
        "r" => Suit::Red,
        _ => Suit::Red,
    }
}

fn parse_ques(action: String) -> ActionType {
    let mut action_type = ActionType::Question(QuestionType::Yours);
    if &action[7..9] == "my" {
        let col = parse_suit(&action[9..10]);
        action_type = ActionType::AnnounceTrump(col);
    }
    if &action[7..9] == "ou" {
        let col = parse_suit(&action[9..10]);
        action_type = ActionType::Question(QuestionType::YourHalf(col));
    }
    action_type
}

fn parse_answ(action: String) -> ActionType {
    let mut action_type = ActionType::Answer(AnswerType::NoPair);
    if &action[7..9] == "my" {
        let col = parse_suit(&action[9..10]);
        action_type = ActionType::Answer(AnswerType::YesPair(col));
    }
    if &action[7..9] == "no" {
        let col = parse_suit(&action[9..10]);
        action_type = ActionType::Answer(AnswerType::NoHalf(col));
    }
    if &action[7..9] == "ou" {
        let col = parse_suit(&action[9..10]);
        action_type = ActionType::Answer(AnswerType::YesHalf(col));
    }
    action_type
}

pub fn parse_legacy_format(game_data: LegacyGameFormat) -> Result<GameFinishedInfo, Error> {
    let names: [String; 4] = game_data.players.clone().try_into().unwrap();

    let cards = [
        parse_cards(game_data.cards.get(&names[0]).unwrap().clone()),
        parse_cards(game_data.cards.get(&names[1]).unwrap().clone()),
        parse_cards(game_data.cards.get(&names[2]).unwrap().clone()),
        parse_cards(game_data.cards.get(&names[3]).unwrap().clone()),
    ];

    let mut game_replay = Game::new(game_data.name, names, Some(cards));

    for _ in 0..4 {
        game_replay.apply_action_mut(game_replay.legal_actions[0].clone());
    }

    let mut pass_collect = vec![];
    for action in game_data.actions {
        //println!("{:?}", game_replay.legal_actions);
        //println!("{}", action.clone());
        if pass_collect.len() == 4 {
            let pass_action = parse_pass(pass_collect.clone());
            //println!("{:?}", pass_action);
            game_replay.apply_action_mut(pass_action);
            pass_collect = vec![];
        }
        if &action[2..6] == "PASS" || &action[2..6] == "PBCK" {
            pass_collect.push(action.clone());
            continue;
        }
        let new_action = parse_action(action.clone())?;
        //println!("{:?}", new_action);
        if new_action.action_type == ActionType::NewBid(0) {
            continue;
        }
        game_replay.apply_action_mut(new_action);
    }
    //print!("{:#?}", game_replay.state.phase);
    let mut game_db = GameFinishedInfo::from(game_replay);
    game_db.set_times(game_data.created, game_data.started, game_data.finished);
    Ok(game_db)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::points::Points;

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
        let input: LegacyGameFormat = serde_json::from_str(
            r#"
        {
           "_id": "ffffffffe1f3816d1dad798f",
           "name":"NameOfGame",
           "created":"2023-04-15 22:45:13",
           "started":"2023-04-15 22:45:34",
           "series_id":"",
           "finished":"2023-04-15 22:51:55",
           "players":[
              "Player A",
              "Player B",
              "Player C",
              "Player D"
           ],
           "cards":{
              "Player A":[
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
              "Player B":[
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
              "Player C":[
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
              "Player D":[
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
           "playing_player":"Player D",
           "game_value":155,
           "players_points":{
              "Player A":0,
              "Player B":199,
              "Player C":0,
              "Player D":121
           },
           "players_sup":{
              "Player A":[

              ],
              "Player B":[
                 "r"
              ],
              "Player C":[

              ],
              "Player D":[
                 "s"
              ]
           },
           "schwarz_game":true
        }
      "#,
        )
        .unwrap();
        let result = parse_legacy_format(input.clone()).unwrap();
        assert_eq!(result.game_value, Points(input.game_value));
    }
}
