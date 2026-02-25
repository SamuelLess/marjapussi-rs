use serde::{Deserialize, Serialize};

use crate::game::gameinfo::GameFinishedInfo;
use crate::game::gamestate::GamePhase;
use crate::game::player::PlaceAtTable;
use crate::game::Game;
use crate::game::gameevent::ActionType;
use crate::ml::observation::{build_observation, ObservationJson};
use crate::ml::sim::{heuristic_policy, random_policy, run_to_end, try_all_actions};
use crate::ml::search::TtEntry;
use std::collections::HashMap;

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
        /// Optional: fast forward to a specific trick number (1..9). 0 = Passing, -1 = Bidding.
        #[serde(default)]
        start_trick: Option<i32>,
    },
    /// Advance the game with the given action index (into legal_actions).
    Step { action_idx: usize },
    /// Allow debug passing of arbitrary cards, bypassing the legal_actions index requirement.
    DebugPass {
        card_indices: Vec<usize>,
    },
    /// Return current observation without advancing.
    Observe,
    /// Request the heuristic policy's preferred action.
    GetHeuristicAction,
    /// Run current game to end using the specified policy.
    RunToEnd {
        #[serde(default)]
        policy: PolicyName,
    },
    /// Counterfactual: try all legal actions, run each branch to end.
    TryAllActions {
        #[serde(default)]
        policy: PolicyName,
        #[serde(default = "default_rollouts")]
        num_rollouts: usize,
    },
    /// Optimized counterfactual: return only advantage scores (mean 0, std 1).
    GetAdvantages {
        #[serde(default)]
        policy: PolicyName,
        #[serde(default = "default_rollouts")]
        num_rollouts: usize,
    },
}

fn default_rollouts() -> usize { 1 }

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
    HeuristicAction {
        action_idx: usize,
    },
    TryAllResult {
        branches: Vec<BranchResult>,
    },
    Advantages {
        advantages: Vec<f32>,
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
    pub team_points: [i32; 2],
    pub playing_party: Option<u8>,
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
    pub outcomes: Vec<OutcomeJson>,
}

fn info_to_outcome(info: &GameFinishedInfo) -> OutcomeJson {
    OutcomeJson {
        won: info.won,
        no_one_played: info.no_one_played,
        schwarz: info.schwarz_game,
        game_value: info.game_value.0,
        team_points: info.team_points,
        playing_party: info.playing_party.clone().map(|p| p.party().0),
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
    cache: HashMap<u128, TtEntry>,
}

impl Server {
    pub fn new() -> Self {
        Server { 
            game: None, 
            pov: PlaceAtTable(0),
            cache: HashMap::with_capacity(8192),
        }
    }

    pub fn handle(&mut self, req: Request) -> Response {
        match req {
            Request::NewGame { seed, pov, start_trick } => {
                self.pov = PlaceAtTable(pov);
                let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
                
                let cards = if let Some(s) = seed {
                    use rand::SeedableRng;
                    // Seed the RNG deterministically
                    let mut r = rand::rngs::StdRng::seed_from_u64(s);
                    let mut deck = crate::game::cards::get_all_cards();
                    use rand::seq::SliceRandom;
                    deck.shuffle(&mut r);
                    
                    let mut c: [Vec<crate::game::cards::Card>; 4] = [vec![], vec![], vec![], vec![]];
                    for i in 0..4 {
                        c[i] = deck[(i * 9)..((i + 1) * 9)].to_vec();
                    }
                    Some(c)
                } else {
                    None
                };

                let mut game = Game::new("ml".to_string(), names, cards);
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
                
                if let Some(target) = start_trick {
                    let target = target.min(9);
                    let policy = heuristic_policy();
                    while !game.ended() {
                        if target == -1 {
                            // Target -1 means Bidding phase
                            if matches!(game.state.phase, GamePhase::Bidding) {
                                break;
                            }
                        } else if target == 0 {
                            // Target 0 means Passing phase
                            if matches!(game.state.phase, GamePhase::PassingForth | GamePhase::PassingBack) {
                                break;
                            }
                        } else {
                            // Target >= 1 means Card Playing phase trick 1..=9
                            let is_playing = matches!(
                                game.state.phase,
                                GamePhase::StartTrick | GamePhase::Trick | GamePhase::AnsweringPair | GamePhase::AnsweringHalf(_)
                            );
                            let reached_trick = game.state.all_tricks.len() + 1 >= target as usize;
                            if is_playing && reached_trick {
                                break;
                            }
                        }
                        let idx = policy(&game, &mut self.cache);
                        let actions = &game.legal_actions;
                        if actions.is_empty() {
                            break;
                        }
                        let idx = idx.min(actions.len() - 1);
                        game.apply_action_mut(actions[idx].clone());
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

            Request::DebugPass { card_indices } => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };
                
                // Find the existing legal action that passes these specific cards
                let mut found_action = None;
                for a in &game.legal_actions {
                    if let ActionType::Pass(passed) = &a.action_type {
                        let mut sorted_passed = passed.clone();
                        sorted_passed.sort();
                        
                        let mut target_cards = vec![];
                        let all_cards = crate::game::cards::get_all_cards();
                        for &idx in &card_indices {
                            if idx < all_cards.len() {
                                target_cards.push(all_cards[idx].clone());
                            }
                        }
                        target_cards.sort();

                        if sorted_passed == target_cards {
                            found_action = Some(a.clone());
                            break;
                        }
                    }
                }

                if let Some(action) = found_action {
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
                } else {
                    Response::Error { message: "Pass combination not legal for current player".into() }
                }
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

            Request::GetHeuristicAction => {
                let game = match &self.game {
                    Some(g) => g,
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let mut policy_cache = HashMap::new();
                let idx = crate::ml::sim::heuristic_policy()(game, &mut policy_cache);
                Response::HeuristicAction { action_idx: idx }
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
                let (_final_game, info) = run_to_end(game, &pol, &mut self.cache);
                self.game = None;
                Response::Done { outcome: info_to_outcome(&info) }
            }

            Request::TryAllActions { policy, num_rollouts } => {
                let game = match &self.game {
                    Some(g) => g,
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let pol = match policy {
                    PolicyName::Random => random_policy(),
                    PolicyName::Heuristic => heuristic_policy(),
                };
                let _legal = game.legal_actions.clone();
                let branches_raw = try_all_actions(game, &pol, &mut self.cache, num_rollouts);
                let branches = branches_raw.into_iter().enumerate().map(|(idx, (action, infos))| {
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
                        outcomes: infos.into_iter().map(|info| info_to_outcome(&info)).collect(),
                    }
                }).collect();
                Response::TryAllResult { branches }
            }

            Request::GetAdvantages { policy, num_rollouts } => {
                let game = match &self.game {
                    Some(g) => g,
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let pol = match policy {
                    PolicyName::Random => random_policy(),
                    PolicyName::Heuristic => heuristic_policy(),
                };
                
                let branches_raw = try_all_actions(game, &pol, &mut self.cache, num_rollouts);
                if branches_raw.is_empty() {
                    return Response::Advantages { advantages: vec![] };
                }

                let mut scores = Vec::with_capacity(branches_raw.len());
                let acting_party = game.state.player_at_turn.party();

                for (_action, infos) in branches_raw {
                    let mut sum_val = 0.0;
                    for info in infos {
                        if info.no_one_played {
                            // If nobody bid, both teams get a small penalty.
                            sum_val -= 0.1;
                            continue;
                        }

                        // Determine if the acting team won the contract
                        let is_playing_team = info.playing_party.map_or(false, |p| p.0 == acting_party.0);
                        let won_by_playing = info.won.unwrap_or(false);
                        
                        let won = if is_playing_team { won_by_playing } else { !won_by_playing };
                        let win_loss = if won { 1.0 } else { -1.0 };
                        
                        // Scale value exactly as python does (MAX_GAME_POINTS = 420.0)
                        let val = info.game_value.0 as f32 / 420.0;
                        let schwarz_mult = if info.schwarz_game { 2.0 } else { 1.0 };
                        
                        sum_val += win_loss * val * schwarz_mult;
                    }
                    scores.push(sum_val / num_rollouts as f32);
                }

                // Normalize: (s - mean) / max(std, 1.0)
                let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
                let variance: f32 = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
                let std = (variance.sqrt()).max(1.0);

                let advantages = scores.into_iter().map(|s| (s - mean) / std).collect();
                Response::Advantages { advantages }
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
