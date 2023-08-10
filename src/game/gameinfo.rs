use crate::game::cards::Card;
use crate::game::gameevent::{ActionType, GameCallback, GameEvent, GameEventPlayer};
use crate::game::gamestate::{FinishedTrick, GamePhase};
use crate::game::player::PlaceAtTable;
use crate::game::points::{points_pair, Points};
use crate::game::Game;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GameMetaInfo {
    pub name: String,
    pub create_time: String,
    pub start_time: Option<String>,
    pub player_names: [String; 4],
    pub player_start_cards: [Vec<Card>; 4],
}

//everything one player in the frontend wants to know
#[derive(Debug, Clone)]
pub struct GameInfoPlayer {
    pub meta_info: GameMetaInfo,
    pub players_pressed_start: Vec<String>,
    pub players_from_perspective: [String; 4],
    pub player_at_turn: String,
    pub own_cards: Option<Vec<Card>>,
    pub players_cards_number_perspective: [u8; 4],
    pub game_phase: GamePhase,
    pub bidding_history: Vec<(ActionType, PlaceAtTable)>,
    pub current_trick: Vec<Card>,
    pub last_trick: Option<FinishedTrick>,
    pub last_event: Option<GameEventPlayer>,
}

impl GameInfoPlayer {
    pub fn from_game(game: Game, place: PlaceAtTable) -> Self {
        GameInfoPlayer {
            meta_info: game.info.clone(),
            players_pressed_start: game.state.players_started(),
            players_from_perspective: game.state.players_perspective(place.clone()),
            player_at_turn: game
                .state
                .player_at_place(game.state.player_at_turn.clone())
                .name
                .clone(),
            own_cards: match game.state.started {
                true => Some(game.state.player_at_place(place.clone()).cards.clone()),
                false => None,
            },
            players_cards_number_perspective: game.state.players_perspective_cards(place),
            game_phase: game.state.phase,
            bidding_history: game.state.bidding_history,
            current_trick: game.state.current_trick,
            last_trick: game.state.all_tricks.clone().last().cloned(),
            last_event: match game.all_events.last() {
                None => None,
                Some(event) => {
                    //hide last event if cards were passed
                    if let ActionType::Pass(_cards) = event.last_action.action_type.clone() {
                        Some(GameEventPlayer::HiddenEvent)
                    } else {
                        Some(GameEventPlayer::PublicEvent((*event).clone()))
                    }
                }
            },
        }
    }
}

//everything the database needs to know
#[derive(Debug, Clone, Serialize)]
pub struct GameInfoDatabase {
    info: GameMetaInfo,
    game_value: Points,
    won: Option<bool>, //None if no_one_played
    no_one_played: bool,
    schwarz_game: bool,
    playing_party: Option<PlaceAtTable>,
    after_passing: Option<[Vec<Card>; 4]>,
    passed_cards: Option<(Vec<Card>, Vec<Card>)>,
    bidding_history: Vec<(ActionType, PlaceAtTable)>,
    tricks: Vec<FinishedTrick>, //PlaceAtTable for who got the trick
    all_events: Vec<GameEvent>,
}

impl From<Game> for GameInfoDatabase {
    fn from(game: Game) -> Self {
        if game.state.phase != GamePhase::Ended {
            panic!("Cannot convert unfinished game!");
        }
        let no_one_played = game.state.value.0 == 115;

        let mut players_points = [Points(0); 4];
        let mut players_tricks: [Vec<FinishedTrick>; 4] = [vec![], vec![], vec![], vec![]];
        for trick in &game.state.all_tricks {
            players_tricks[trick.winner.0 as usize].push(trick.clone());
            players_points[trick.winner.0 as usize] += trick.points;
        }
        let tricks_party_zero = players_tricks[0].len() + players_tricks[2].len();
        let schwarz_game = tricks_party_zero == 0 || tricks_party_zero == 9;

        let mut playing_party: Option<PlaceAtTable> = None;
        let mut won: Option<bool> = None;
        let mut passed_cards: Option<(Vec<Card>, Vec<Card>)> = None;
        let mut after_passing: Option<[Vec<Card>; 4]> = None;
        if !no_one_played {
            let mut playing_player = PlaceAtTable(0);
            let mut passed_forth: Option<Vec<Card>> = None;
            let mut passed_back: Option<Vec<Card>> = None;
            for event in &game.all_events {
                if ActionType::NewBid(game.state.value.0) == event.last_action.action_type {
                    playing_player = event.last_action.player.clone();
                    playing_party = Some(event.last_action.player.party());
                }
                if let Some(GameCallback::NewTrump(suit)) = event.callback {
                    players_points[event.last_action.player.clone().0 as usize] +=
                        points_pair(suit);
                }
                if let ActionType::Pass(cards) = event.last_action.action_type.clone() {
                    if passed_forth.is_none() {
                        passed_forth = Some(cards);
                    } else {
                        passed_back = Some(cards);
                    }
                }
            }
            passed_cards = Some((passed_forth.clone().unwrap(), passed_back.clone().unwrap()));

            let mut cards_after_passing = game.info.player_start_cards.clone();
            //partner cards
            cards_after_passing[playing_player.partner().0 as usize] = cards_after_passing
                [playing_player.partner().0 as usize]
                .clone()
                .into_iter()
                .filter(|c| !passed_forth.clone().unwrap().contains(c))
                .collect();
            cards_after_passing[playing_player.partner().0 as usize]
                .append(&mut passed_back.clone().unwrap());
            //cards of playing player
            cards_after_passing[playing_player.0 as usize]
                .append(&mut passed_forth.clone().unwrap());
            cards_after_passing[playing_player.0 as usize] = cards_after_passing
                [playing_player.0 as usize]
                .clone()
                .into_iter()
                .filter(|c| !passed_back.clone().unwrap().contains(c))
                .collect();

            after_passing = Some(cards_after_passing);

            let points_party = players_points[playing_party.clone().unwrap().0 as usize]
                + players_points[playing_party.clone().unwrap().partner().0 as usize];
            won = Some(points_party >= game.state.value);
        }

        GameInfoDatabase {
            info: game.info.clone(),
            game_value: game.state.value,
            won,
            no_one_played,
            schwarz_game,
            playing_party,
            after_passing,
            passed_cards,
            bidding_history: game.state.bidding_history,
            tricks: game.state.all_tricks,
            all_events: game.all_events,
        }
    }
}

pub struct GameInfoSpectator {}
