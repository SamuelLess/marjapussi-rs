use serde::Serialize;

use crate::game::cards::Card;
#[cfg(feature = "public-api")]
use crate::game::current_time_string;
use crate::game::gameevent::{ActionType, GameCallback, GameEvent};
#[cfg(feature = "public-api")]
use crate::game::gameevent::{GameAction, GameEventPlayer};
use crate::game::gamestate::{FinishedTrick, GamePhase};
use crate::game::player::{PlaceAtTable, Player, HAND_SIZE};
use crate::game::points::{points_pair, Points};
use crate::game::Game;

#[derive(Debug, Clone, Serialize)]
pub struct GameMetaInfo {
    pub name: String,
    pub create_time: String,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub player_names: [String; 4],
    pub player_start_cards: [Vec<Card>; 4],
}

impl GameMetaInfo {
    pub fn create(name: String, player_names: [String; 4], players: [Player; 4]) -> Self {
        GameMetaInfo {
            name,
            #[cfg(feature = "public-api")]
            create_time: current_time_string(),
            #[cfg(not(feature = "public-api"))]
            create_time: String::new(),
            start_time: None,
            end_time: None,
            player_names,
            player_start_cards: [
                players[0].cards.to_vec(),
                players[1].cards.to_vec(),
                players[2].cards.to_vec(),
                players[3].cards.to_vec(),
            ],
        }
    }
}

/// What one seat may see: public facts, its own hand and its own legal actions. Nothing here
/// reveals another seat's hidden cards; their start hands are empty in `meta_info`.
#[cfg(feature = "public-api")]
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
    pub legal_actions: Vec<GameAction>,
}

#[cfg(feature = "public-api")]
impl GameInfoPlayer {
    pub fn from_game(game: Game, place: PlaceAtTable) -> Self {
        let mut meta_info = game.info.clone();
        for (seat, cards) in meta_info.player_start_cards.iter_mut().enumerate() {
            if !game.state.started || seat != place.0 as usize {
                cards.clear();
            }
        }
        GameInfoPlayer {
            meta_info,
            players_pressed_start: game.state.players_started(),
            players_from_perspective: game.state.players_perspective(place),
            player_at_turn: game
                .state
                .player_at_place(game.state.player_at_turn)
                .name
                .to_string(),
            own_cards: match game.state.started {
                true => Some(game.state.player_at_place(place).cards.to_vec()),
                false => None,
            },
            players_cards_number_perspective: game.state.players_perspective_cards(place),
            game_phase: game.state.phase,
            bidding_history: game.state.bidding_history,
            current_trick: game.state.current_trick.to_vec(),
            last_trick: game.state.all_tricks.last().cloned(),
            last_event: match game.all_events.last() {
                None => None,
                Some(event) => {
                    //hide last event if cards were passed
                    if matches!(event.last_action.action_type, ActionType::Pass(_)) {
                        Some(GameEventPlayer::HiddenEvent)
                    } else {
                        Some(GameEventPlayer::PublicEvent((*event).clone()))
                    }
                }
            },
            // Only this seat's actions: another seat's card plays or passes would show its hand.
            legal_actions: game
                .legal_actions
                .iter()
                .filter(|a| a.player == place)
                .cloned()
                .collect(),
        }
    }
}

/// Everything the database needs to know
#[derive(Debug, Clone, Serialize)]
pub struct GameFinishedInfo {
    pub info: GameMetaInfo,
    pub game_value: Points,
    /// None if no_one_played
    pub won: Option<bool>,
    pub no_one_played: bool,
    pub schwarz_game: bool,
    pub playing_party: Option<PlaceAtTable>,
    pub after_passing: Option<[Vec<Card>; 4]>,
    pub passed_cards: Option<(Vec<Card>, Vec<Card>)>,
    pub bidding_history: Vec<(ActionType, PlaceAtTable)>,
    /// PlaceAtTable for who got the trick
    pub tricks: Vec<FinishedTrick>,
    pub all_events: Vec<GameEvent>,
}

/// Finished card and pair points, retaining the upstream all-pass convention.
fn team_points(tricks: &[FinishedTrick], events: &[GameEvent], no_one_played: bool) -> [i32; 2] {
    let mut points = [0; 2];
    for trick in tricks {
        points[trick.winner.0 as usize % 2] += trick.points.0;
    }
    if !no_one_played {
        for event in events {
            if let Some(GameCallback::NewTrump(suit)) = event.callback {
                points[event.last_action.player.0 as usize % 2] += points_pair(suit).0;
            }
        }
    }
    points
}

impl From<Game> for GameFinishedInfo {
    fn from(game: Game) -> Self {
        if game.state.phase != GamePhase::Ended {
            panic!("Cannot convert unfinished game!");
        }
        let no_one_played = game.state.value.0 == 115;

        let team_points = team_points(&game.state.all_tricks, &game.all_events, no_one_played);
        let tricks_party_zero = game
            .state
            .all_tricks
            .iter()
            .filter(|t| t.winner.0 % 2 == 0)
            .count();
        let schwarz_game = tricks_party_zero == 0 || tricks_party_zero == HAND_SIZE;

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
                    playing_player = event.last_action.player;
                    playing_party = Some(event.last_action.player.party());
                }
                if let ActionType::Pass(cards) = &event.last_action.action_type {
                    if passed_forth.is_none() {
                        passed_forth = Some(cards.to_vec());
                    } else {
                        passed_back = Some(cards.to_vec());
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
            // cards of playing player
            cards_after_passing[playing_player.0 as usize].append(&mut passed_forth.unwrap());
            cards_after_passing[playing_player.0 as usize] = cards_after_passing
                [playing_player.0 as usize]
                .clone()
                .into_iter()
                .filter(|c| !passed_back.clone().unwrap().contains(c))
                .collect();

            after_passing = Some(cards_after_passing);

            won = Some(team_points[playing_party.unwrap().0 as usize] >= game.state.value.0);
        }

        GameFinishedInfo {
            #[cfg(feature = "public-api")]
            info: game.info,
            #[cfg(not(feature = "public-api"))]
            info: (*game.info).clone(),
            game_value: game.state.value,
            won,
            no_one_played,
            schwarz_game,
            playing_party,
            after_passing,
            passed_cards,
            bidding_history: game.state.bidding_history,
            tricks: game.state.all_tricks.to_vec(),
            all_events: game.all_events,
        }
    }
}

impl GameFinishedInfo {
    pub fn set_times(&mut self, created: String, started: String, ended: String) {
        self.info.create_time = created;
        self.info.start_time = Some(started);
        self.info.end_time = Some(ended);
    }
}
