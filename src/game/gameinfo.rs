use crate::game::cards::Card;
use crate::game::gameevent::{ActionType, GameEventPlayer};
use crate::game::gamestate::GamePhase;
use crate::game::player::PlaceAtTable;
use crate::game::points::Points;
use crate::game::Game;

#[derive(Debug, Clone)]
pub struct GameMetaInfo {
    pub name: String,
    pub create_time: String,
    pub start_time: Option<String>,
    pub started: bool,
    pub all_player_names: [String; 4],
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
    pub last_trick: Option<([Card; 4], PlaceAtTable)>,
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
            own_cards: match game.info.started {
                true => Some(game.state.player_at_place(place.clone()).cards.clone()),
                false => None,
            },
            players_cards_number_perspective: game.state.players_perspective_cards(place),
            game_phase: game.state.phase,
            bidding_history: game.state.bidding_history,
            current_trick: game.state.current_trick,
            last_trick: game.state.last_trick,
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
pub struct GameInfoDatabase {
    meta_info: GameMetaInfo,
    game_value: Points,
    won: Option<bool>,
    noone_played: bool,
    schwarz_game: bool,
    playing_party: Option<PlaceAtTable>,
    players_cards: [Vec<Card>; 4],
    before_passing: Option<[Vec<Card>; 4]>,
    passed_cards: Option<(Vec<Card>, Vec<Card>)>,
    bidding_history: Vec<(ActionType, PlaceAtTable)>,
}

pub struct GameInfoSpectator {}
