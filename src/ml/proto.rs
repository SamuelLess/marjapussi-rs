use serde::{Deserialize, Serialize};

use crate::game::gameinfo::GameFinishedInfo;
use crate::game::player::PlaceAtTable;
use crate::game::Game;
use crate::ml::observation::{build_observation, ObservationJson};
use crate::ml::sim::{heuristic_policy, random_policy, run_to_end, try_all_actions};

// ─── Request/Response message types ─────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    /// Start a fresh game with random cards.
    NewGame {
        #[serde(default)]
        seed: Option<u64>,
        /// POV seat index (0..3). Defaults to 0.
        #[serde(default)]
        pov: u8,
    },
    /// Advance the game with the given action index (into legal_actions).
    Step { action_idx: usize },
    /// Return current observation without advancing.
    Observe,
    /// Run current game to end using the specified policy.
    RunToEnd {
        #[serde(default)]
        policy: PolicyName,
    },
    /// Counterfactual: try all legal actions, run each branch to end.
    TryAllActions {
        #[serde(default)]
        policy: PolicyName,
    },
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PolicyName {
    #[default]
    Random,
    Heuristic,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Obs {
        obs: ObservationJson,
        done: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        outcome: Option<OutcomeJson>,
    },
    Done {
        outcome: OutcomeJson,
    },
    TryAllResult {
        branches: Vec<BranchResult>,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Serialize)]
pub struct OutcomeJson {
    pub won: Option<bool>,
    pub no_one_played: bool,
    pub schwarz: bool,
    pub game_value: i32,
    pub tricks: Vec<TrickOutcome>,
}

#[derive(Debug, Serialize)]
pub struct TrickOutcome {
    pub winner: u8,
    pub points: i32,
    pub cards: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct BranchResult {
    pub action_idx: usize,
    pub action_token: u32,
    pub card_idx: Option<usize>,
    pub outcome: OutcomeJson,
}

fn info_to_outcome(info: &GameFinishedInfo) -> OutcomeJson {
    OutcomeJson {
        won: info.won,
        no_one_played: info.no_one_played,
        schwarz: info.schwarz_game,
        game_value: info.game_value.0,
        tricks: info.tricks.iter().map(|t| TrickOutcome {
            winner: t.winner.0,
            points: t.points.0,
            cards: t.cards.iter().map(|c| format!("{}", c)).collect(),
        }).collect(),
    }
}

// ─── Server state ─────────────────────────────────────────────────────────────

pub struct Server {
    game: Option<Game>,
    pov: PlaceAtTable,
}

impl Server {
    pub fn new() -> Self {
        Server { game: None, pov: PlaceAtTable(0) }
    }

    pub fn handle(&mut self, req: Request) -> Response {
        match req {
            Request::NewGame { seed: _, pov } => {
                self.pov = PlaceAtTable(pov);
                let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
                let mut game = Game::new("ml".to_string(), names, None);
                // Auto-start: apply all 4 Start actions
                let mut actions = game.legal_actions.clone();
                for _ in 0..4 {
                    if let Some(a) = actions.pop() {
                        game = match game.apply_action(a) {
                            Ok(g) => g,
                            Err(e) => return Response::Error { message: format!("{:?}", e) },
                        };
                        actions = game.legal_actions.clone();
                    }
                }
                let done = game.ended();
                let obs = ObservationJson::from(build_observation(&game, self.pov.clone()));
                self.game = Some(game);
                Response::Obs { obs, done, outcome: None }
            }

            Request::Step { action_idx } => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };
                if action_idx >= game.legal_actions.len() {
                    return Response::Error {
                        message: format!("action_idx {} out of range ({})", action_idx, game.legal_actions.len()),
                    };
                }
                let action = game.legal_actions[action_idx].clone();
                let next = match game.apply_action(action) {
                    Ok(g) => g,
                    Err(e) => return Response::Error { message: format!("{:?}", e) },
                };
                let done = next.ended();
                let obs = ObservationJson::from(build_observation(&next, self.pov.clone()));
                let outcome = if done {
                    Some(info_to_outcome(&GameFinishedInfo::from(next.clone())))
                } else {
                    None
                };
                self.game = Some(next);
                Response::Obs { obs, done, outcome }
            }

            Request::Observe => {
                let game = match &self.game {
                    Some(g) => g,
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let done = game.ended();
                let obs = ObservationJson::from(build_observation(game, self.pov.clone()));
                Response::Obs { obs, done, outcome: None }
            }

            Request::RunToEnd { policy } => {
                let game = match self.game.take() {
                    Some(g) => g,
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let pol = match policy {
                    PolicyName::Random => random_policy(),
                    PolicyName::Heuristic => heuristic_policy(),
                };
                let (_final_game, info) = run_to_end(game, &pol);
                self.game = None;
                Response::Done { outcome: info_to_outcome(&info) }
            }

            Request::TryAllActions { policy } => {
                let game = match &self.game {
                    Some(g) => g,
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let pol = match policy {
                    PolicyName::Random => random_policy(),
                    PolicyName::Heuristic => heuristic_policy(),
                };
                let legal = game.legal_actions.clone();
                let branches_raw = try_all_actions(game, &pol);
                let branches = branches_raw.into_iter().enumerate().map(|(idx, (action, info))| {
                    let la = legal.get(idx);
                    use crate::game::gameevent::ActionType;
                    use crate::ml::observation::{card_index, tokens};
                    let (action_token, card_idx) = match &action.action_type {
                        ActionType::CardPlayed(c) => (tokens::ACT_PLAY, Some(card_index(c))),
                        ActionType::NewBid(_) => (tokens::ACT_BID, None),
                        ActionType::StopBidding => (tokens::ACT_PASS_STOP, None),
                        ActionType::AnnounceTrump(_) => (tokens::ACT_TRUMP, None),
                        _ => (0, None),
                    };
                    BranchResult {
                        action_idx: idx,
                        action_token,
                        card_idx,
                        outcome: info_to_outcome(&info),
                    }
                }).collect();
                Response::TryAllResult { branches }
            }
        }
    }
}

// ─── Main stdio loop ─────────────────────────────────────────────────────────

pub fn run_server() {
    use std::io::{self, BufRead, Write};
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut server = Server::new();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::Error { message: format!("Parse error: {}", e) };
                let out = serde_json::to_string(&resp).unwrap_or_default();
                let mut out_lock = stdout.lock();
                writeln!(out_lock, "{}", out).ok();
                continue;
            }
        };
        let resp = server.handle(req);
        let out = serde_json::to_string(&resp).unwrap_or_default();
        let mut out_lock = stdout.lock();
        writeln!(out_lock, "{}", out).ok();
        out_lock.flush().ok();
    }
}
