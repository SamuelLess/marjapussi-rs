use std::time::SystemTime;

use chrono::offset::Local;
use chrono::DateTime;

use crate::game::errors::GameError;
use crate::game::gameevent::{ActionType, GameAction, GameEvent};
use crate::game::gamestate::GamePhase;
use crate::game::player::create_players;
use crate::game::points::Points;

use self::{cards::Card, gameinfo::GameMetaInfo, gamestate::GameState};

mod apply_action;
pub mod cards;
pub mod errors;
pub mod gameevent;
pub mod gameinfo;
pub mod gamestate;
pub mod legal_actions;
pub mod parse;
pub mod player;
pub mod points;
pub mod series;

/// Wrapper for Game and all of its details.
#[derive(Debug, Clone)]
pub struct Game {
    pub info: GameMetaInfo,
    pub state: GameState,
    pub legal_actions: Vec<GameAction>,
    pub last_state: Option<GameState>,
    pub all_events: Vec<GameEvent>,
}

impl Game {
    pub fn new(name: String, player_names: [String; 4], cards: Option<[Vec<Card>; 4]>) -> Self {
        let players = create_players(player_names.clone(), cards);

        let mut game = Game {
            info: GameMetaInfo::create(name, player_names, players.clone()),
            state: GameState::create(players.clone()),
            legal_actions: vec![],
            last_state: None,
            all_events: vec![],
        };
        game.legal_actions = game.legal_actions();
        game
    }

    /// Creates list with all legal actions in the current state of the game.
    pub fn legal_actions(&self) -> Vec<GameAction> {
        let mut legal = self.state.phase.legal_actions(self);
        let disallow_undo: Vec<GamePhase> = vec![
            GamePhase::WaitingForStart,
            GamePhase::Ended,
            GamePhase::PassingBack,
            GamePhase::Raising,
        ];
        if disallow_undo.contains(&self.state.phase) {
            return legal;
        }
        // disallow undo special cases
        match self.state.phase.clone() {
            GamePhase::PendingUndo(_) => {
                return legal.clone();
            }
            GamePhase::Bidding => {
                if self.state.value == Points(115) {
                    return legal.clone();
                }
            }
            _ => {}
        }
        if self.last_state.is_some() {
            legal.push(GameAction {
                action_type: ActionType::UndoRequest,
                player: self.last_state.clone().unwrap().player_at_turn.clone(),
            });
        }
        legal
    }

    /// Tries creating a new Game with state after applying a given action.
    /// Can fail and does not mutate the existing Game.
    pub fn apply_action(&self, action: GameAction) -> Result<Game, GameError> {
        if !self.legal_actions.contains(&action) {
            return Err(GameError::IllegalAction);
        }
        let (next_game_meta, next_game_state, this_callback, last_state) =
            action.clone().action_type.apply_action(&action, self);

        // next game Object
        let this_event = GameEvent {
            last_action: action,
            callback: this_callback,
            player_at_turn: next_game_state.player_at_turn.clone(),
            time: current_time_string(),
        };
        let mut next_all_events: Vec<GameEvent> = self.all_events.clone();
        next_all_events.push(this_event);
        let mut next_game = Game {
            info: next_game_meta,
            state: next_game_state,
            legal_actions: vec![],
            last_state,
            all_events: next_all_events,
        };
        next_game.legal_actions = next_game.legal_actions();
        Ok(next_game)
    }

    pub fn ended(&self) -> bool {
        self.state.phase == GamePhase::Ended
    }

    pub fn apply_action_mut(&mut self, action: GameAction) {
        if let Ok(next) = self.apply_action(action.clone()) {
            *self = next;
        } else {
            eprintln!("Discarded illegal action: {:?}", action);
            eprintln!("Game state: {:#?}", self);
            panic!();
        }
    }
}

pub fn current_time_string() -> String {
    let system_time = SystemTime::now();
    let datetime: DateTime<Local> = system_time.into();
    format!("{}", datetime.format("%Y-%m-%d %T"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::cards::Suit;
    use crate::game::gameevent::QuestionType;
    use crate::game::player::{PlaceAtTable, PlayerTrumpPossibilities};
    use rand::prelude::IndexedMutRandom;

    #[test]
    fn test_time_creation() {
        let time = current_time_string();
        assert_eq!(time.len(), 19);
    }

    fn helper_create_game() -> Game {
        let names = [
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
        ];
        Game::new(String::from("Testgame"), names, None)
    }

    #[test]
    fn test_creating() {
        let names = [
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
        ];
        let game = Game::new(String::from("Game Name"), names, None);
        assert_eq!(game.info.name, String::from("Game Name"));
    }

    #[test]
    fn test_starting() {
        let mut game = helper_create_game();
        let mut actions = game.legal_actions.clone();
        assert_eq!(actions.len(), 4);
        for _ in 0..4 {
            let act = actions.pop().unwrap();
            let res = game.apply_action(act);
            game = res.ok().unwrap();
            actions = game.legal_actions.clone();
        }
        assert_eq!(game.state.phase, GamePhase::Bidding);
    }

    #[test]
    fn test_random_game_controlled() {
        let mut game = helper_create_game();
        let mut actions = game.legal_actions();
        assert_eq!(actions.len(), 4);
        for _ in 0..4 {
            game = game.apply_action(actions.pop().unwrap()).ok().unwrap();
            actions.clone_from(&game.legal_actions);
        }
        assert_eq!(game.state.phase, GamePhase::Bidding);
        actions.clone_from(&game.legal_actions);
        assert_eq!(actions.len(), 62);
        let bid140 = GameAction {
            action_type: ActionType::NewBid(140),
            player: game.state.player_at_turn.clone(),
        };
        game = game.apply_action(bid140).ok().unwrap();
        let forbidden_action = actions.pop().unwrap();
        let res = game.apply_action(forbidden_action);
        assert!(res.is_err());
        for _ in 0..4 {
            actions = game.legal_actions.clone();
            let res = game.apply_action(actions[3].clone());
            game = res.ok().unwrap();
        }
        for _ in 0..3 {
            actions = game.legal_actions.clone();
            assert_eq!(game.state.phase, GamePhase::Bidding);
            let res = game.apply_action(actions[0].clone());
            game = res.ok().unwrap();
        }
        assert_eq!(game.state.value, Points(200));

        //passing forth
        assert_eq!(game.state.player_at_turn().name, String::from("S3"));
        assert_eq!(game.state.phase, GamePhase::PassingForth);
        assert_eq!(game.state.player_at_turn().cards.len(), 9);
        assert_eq!(game.legal_actions.len(), 127); //nCr(9,4) + 1
        actions = game.legal_actions.clone();
        let res = game.apply_action(actions[2].clone());
        game = res.ok().unwrap();
        assert_eq!(game.state.player_at_turn().cards.len(), 13);
        assert_eq!(
            game.state
                .player_at_place(game.state.player_at_turn.partner())
                .cards
                .len(),
            5
        );
        //passing back
        assert_eq!(game.state.player_at_turn().name, String::from("S1"));
        assert_eq!(game.state.phase, GamePhase::PassingBack);
        actions.clone_from(&game.legal_actions);
        let res = game.apply_action(actions[9].clone());
        game = res.ok().unwrap();
        assert_eq!(game.state.player_at_turn().cards.len(), 9);
        assert_eq!(
            game.state
                .player_at_place(game.state.player_at_turn.partner())
                .cards
                .len(),
            9
        );
        assert_eq!(game.state.phase, GamePhase::Raising);
        actions.clone_from(&game.legal_actions);
        let res = game.apply_action(actions[2].clone());
        game = res.ok().unwrap();
        assert_eq!(game.state.phase, GamePhase::Trick);
        actions.clone_from(&game.legal_actions);
        let _res = game.apply_action(actions[0].clone());
    }

    #[test]
    fn test_random_game_multi() {
        for _ in 0..100 {
            test_random_game_random();
        }
    }

    #[test]
    fn test_trump_possibilities_only_move_upwards() {
        let names = [
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
        ];
        let cards = [
            vec![
                "r-O".parse().unwrap(),
                "r-K".parse().unwrap(),
                "g-A".parse().unwrap(),
                "g-Z".parse().unwrap(),
                "g-9".parse().unwrap(),
                "s-6".parse().unwrap(),
                "s-7".parse().unwrap(),
                "e-6".parse().unwrap(),
                "e-7".parse().unwrap(),
            ],
            vec![
                "r-A".parse().unwrap(),
                "r-Z".parse().unwrap(),
                "r-U".parse().unwrap(),
                "r-9".parse().unwrap(),
                "r-8".parse().unwrap(),
                "r-7".parse().unwrap(),
                "r-6".parse().unwrap(),
                "s-A".parse().unwrap(),
                "s-Z".parse().unwrap(),
            ],
            vec![
                "s-K".parse().unwrap(),
                "s-O".parse().unwrap(),
                "s-U".parse().unwrap(),
                "s-9".parse().unwrap(),
                "s-8".parse().unwrap(),
                "g-K".parse().unwrap(),
                "g-O".parse().unwrap(),
                "g-U".parse().unwrap(),
                "g-8".parse().unwrap(),
            ],
            vec![
                "e-A".parse().unwrap(),
                "e-Z".parse().unwrap(),
                "e-K".parse().unwrap(),
                "e-O".parse().unwrap(),
                "e-U".parse().unwrap(),
                "e-9".parse().unwrap(),
                "e-8".parse().unwrap(),
                "g-7".parse().unwrap(),
                "g-6".parse().unwrap(),
            ],
        ];
        let mut game = Game::new(String::from("TrumpUp"), names, Some(cards));
        let p0 = PlaceAtTable(0);

        game.state.started = true;
        game.state.phase = GamePhase::StartTrick;
        game.state.player_at_turn = p0.clone();
        game.legal_actions = game.legal_actions();

        assert!(game.legal_actions.contains(&GameAction {
            action_type: ActionType::AnnounceTrump(Suit::Red),
            player: p0.clone(),
        }));
        assert!(game.legal_actions.contains(&GameAction {
            action_type: ActionType::Question(QuestionType::Yours),
            player: p0.clone(),
        }));

        game = game
            .apply_action(GameAction {
                action_type: ActionType::Question(QuestionType::Yours),
                player: p0.clone(),
            })
            .unwrap();
        assert_eq!(
            game.state.player_at_place(p0.clone()).trump,
            PlayerTrumpPossibilities::Yours
        );

        game.state.phase = GamePhase::StartTrick;
        game.state.player_at_turn = p0.clone();
        game.legal_actions = game.legal_actions();
        assert!(game.legal_actions.contains(&GameAction {
            action_type: ActionType::Question(QuestionType::Yours),
            player: p0.clone(),
        }));
        assert!(!game.legal_actions.contains(&GameAction {
            action_type: ActionType::AnnounceTrump(Suit::Red),
            player: p0.clone(),
        }));

        game = game
            .apply_action(GameAction {
                action_type: ActionType::Question(QuestionType::YourHalf(Suit::Green)),
                player: p0.clone(),
            })
            .unwrap();
        assert_eq!(
            game.state.player_at_place(p0.clone()).trump,
            PlayerTrumpPossibilities::Ours
        );

        game.state.phase = GamePhase::StartTrick;
        game.state.player_at_turn = p0.clone();
        game.legal_actions = game.legal_actions();
        assert!(!game.legal_actions.contains(&GameAction {
            action_type: ActionType::Question(QuestionType::Yours),
            player: p0.clone(),
        }));
        assert!(game.legal_actions.contains(&GameAction {
            action_type: ActionType::Question(QuestionType::YourHalf(Suit::Red)),
            player: p0,
        }));
    }

    pub fn test_random_game_random() {
        let mut game = helper_create_game();
        let mut actions = game.legal_actions.clone();
        assert_eq!(actions.len(), 4);
        for _ in 0..4 {
            game = game.apply_action(actions.pop().unwrap()).ok().unwrap();
            actions.clone_from(&game.legal_actions);
        }
        let mut i = 0;
        while game.state.phase != GamePhase::Ended {
            i += 1;
            if i >= 200 {
                panic!("Too many moves! Game: {:#?}\n ", game,);
            }
            actions.clone_from(&game.legal_actions);
            let opt_select = actions.choose_mut(&mut rand::rng());
            if opt_select.is_none() {
                panic!("Game: {:#?}\n Action: {:?}", game, opt_select);
            }
            let select = opt_select.unwrap().clone();
            let res = game.apply_action(select.clone()).ok();
            if let Some(successful) = res {
                game = successful;
            } else {
                panic!("Game: {:?}\n Action: {:?}", game, select);
            }
        }
        let total_points: Points = game
            .state
            .all_tricks
            .iter()
            .fold(Points(0), |acc, trick| acc + trick.points);
        assert_eq!(total_points, Points(140));
    }
}
