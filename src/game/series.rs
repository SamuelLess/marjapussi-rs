use crate::game::gameevent::GameAction;
use crate::game::gameinfo::GameInfoPlayer;
use crate::game::player::PlaceAtTable;
use crate::game::{current_time_string, Game};

struct Series {
    name: String,
    created: String,
    finished: Option<String>,
    num_of_games: u32,
    games: Vec<Game>,
    players_names: [String; 4],
    settings: SeriesSettings,
}

impl Series {
    pub fn new(
        name: String,
        players_names: [String; 4],
        num_of_games: u32,
        settings: Option<SeriesSettings>,
    ) -> Self {
        Series {
            name: name.clone(),
            created: current_time_string(),
            finished: None,
            players_names: players_names.clone(),
            num_of_games,
            games: vec![Game::new(name, players_names, None)],
            settings: settings.unwrap_or_default(),
        }
    }

    pub fn active_game_info(&mut self, player: PlaceAtTable) -> Option<GameInfoPlayer> {
        if self.games.len() >= (self.num_of_games as usize) {
            return None;
        } else if self.games.last().unwrap().clone().ended() {
            self.games.push(Game::new(
                self.name.clone(),
                self.players_names.clone(),
                None,
            ))
        }
        self.games
            .last()
            .map(|game| GameInfoPlayer::from_game(game.clone(), player))
    }
    pub fn active_game_apply(&mut self, action: GameAction) {
        if let Some(game) = self.games.last_mut() {
            game.apply_action_mutate_or_discard(action);
        }
    }
}

struct SeriesSettings {
    games: i32,
    schwarzfactor_fifths: i32,
    shuffle_players: bool,
    bonus_at: i32,
    bonus_value: i32,
    diff_plus_minus: bool,
    diff_divisor: i32,
}

impl Default for SeriesSettings {
    fn default() -> Self {
        SeriesSettings {
            games: 4,
            schwarzfactor_fifths: 10,
            shuffle_players: true,
            bonus_at: 500,
            bonus_value: 300,
            diff_plus_minus: true,
            diff_divisor: 5,
        }
    }
}
