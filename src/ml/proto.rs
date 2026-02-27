use serde::{Deserialize, Serialize};

use crate::game::gameinfo::GameFinishedInfo;
use crate::game::gamestate::GamePhase;
use crate::game::player::PlaceAtTable;
use crate::game::Game;
use crate::game::gameevent::ActionType;
use crate::ml::observation::{
    build_observation, ObservationDebugJson, ObservationJson, ObservationTrainLabelsJson,
};
use crate::ml::pass_selection::{
    build_pick_legal_actions, candidate_scores, collect_pass_options, decode_pick_action_idx,
    is_pov_pass_turn, resolve_selected_pass_action_idx, selection_is_completable,
    PassSelectionState, PASS_PICK_TARGET,
};
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
        /// Include supervised-only labels (hidden opponent hands) as a separate payload.
        #[serde(default)]
        include_labels: bool,
    },
    /// Advance the game with the given action index (into legal_actions).
    Step { action_idx: usize },
    /// Allow debug passing of arbitrary cards, bypassing the legal_actions index requirement.
    DebugPass {
        card_indices: Vec<usize>,
    },
    /// Return current observation without advancing.
    Observe,
    /// Return current observation for an arbitrary POV seat without advancing.
    ObservePov {
        /// POV seat index (0..3). Defaults to 0.
        #[serde(default)]
        pov: u8,
    },
    /// Return debug-only payload (omniscient state for tooling/UI).
    ObserveDebug,
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
        labels: Option<ObservationTrainLabelsJson>,
        #[serde(skip_serializing_if = "Option::is_none")]
        outcome: Option<OutcomeJson>,
    },
    DebugObs {
        debug: ObservationDebugJson,
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
    include_labels: bool,
    cache: HashMap<u128, TtEntry>,
    pass_selection: PassSelectionState,
    pass_selection_owner: Option<PlaceAtTable>,
}

impl Server {
    pub fn new() -> Self {
        Server { 
            game: None, 
            pov: PlaceAtTable(0),
            include_labels: false,
            cache: HashMap::with_capacity(8192),
            pass_selection: PassSelectionState::default(),
            pass_selection_owner: None,
        }
    }

    fn clear_pass_selection(&mut self) {
        self.pass_selection.clear();
        self.pass_selection_owner = None;
    }

    fn pass_options_for_turn(
        &mut self,
        game: &Game,
        pov: PlaceAtTable,
    ) -> Option<Vec<crate::ml::pass_selection::PassActionOption>> {
        if !is_pov_pass_turn(game, pov) {
            return None;
        }

        let owner = game.state.player_at_turn.clone();
        let owner_changed = self
            .pass_selection_owner
            .as_ref()
            .map(|o| o.0 != owner.0)
            .unwrap_or(true);
        if owner_changed {
            self.pass_selection.clear();
            self.pass_selection_owner = Some(owner);
        }

        let options = collect_pass_options(game);
        if options.is_empty() {
            self.clear_pass_selection();
            return None;
        }

        let mut selected = self.pass_selection.selected().to_vec();
        selected.retain(|card_idx| {
            options
                .iter()
                .any(|opt| opt.cards.contains(card_idx))
        });
        while !selected.is_empty() && !selection_is_completable(&options, &selected) {
            selected.pop();
        }
        self.pass_selection.set_selected(selected);

        Some(options)
    }

    fn build_obs_payload_for_pov(
        &mut self,
        game: &Game,
        pov: PlaceAtTable,
    ) -> (ObservationJson, Option<ObservationTrainLabelsJson>) {
        let pass_options = self.pass_options_for_turn(game, pov.clone());
        let mut obs_full = build_observation(game, pov.clone());
        if let Some(options) = pass_options {
            obs_full.legal_actions = build_pick_legal_actions(&options, self.pass_selection.selected());
            obs_full.pass_selection_indices = self.pass_selection.selected().to_vec();
            obs_full.pass_selection_target = PASS_PICK_TARGET;
        }
        let labels = if self.include_labels && pov.0 == self.pov.0 {
            Some(ObservationTrainLabelsJson::from(&obs_full))
        } else {
            None
        };
        (ObservationJson::from(&obs_full), labels)
    }

    fn build_obs_payload(&mut self, game: &Game) -> (ObservationJson, Option<ObservationTrainLabelsJson>) {
        self.build_obs_payload_for_pov(game, self.pov.clone())
    }

    fn branch_score_for_infos(&self, infos: &[GameFinishedInfo], acting_party: PlaceAtTable) -> f32 {
        let mut sum_val = 0.0f32;
        for info in infos {
            if info.no_one_played {
                sum_val -= 0.1;
                continue;
            }

            let is_playing_team = info
                .playing_party
                .as_ref()
                .map_or(false, |p| p.0 == acting_party.0);
            let won_by_playing = info.won.unwrap_or(false);
            let won = if is_playing_team { won_by_playing } else { !won_by_playing };
            let win_loss = if won { 1.0 } else { -1.0 };
            let val = info.game_value.0 as f32 / 420.0;
            let schwarz_mult = if info.schwarz_game { 2.0 } else { 1.0 };
            sum_val += win_loss * val * schwarz_mult;
        }
        sum_val / infos.len().max(1) as f32
    }

    fn normalize_scores(scores: Vec<f32>) -> Vec<f32> {
        if scores.is_empty() {
            return vec![];
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>()
            / scores.len() as f32;
        let std = variance.sqrt().max(1.0);
        scores.into_iter().map(|s| (s - mean) / std).collect()
    }

    pub fn handle(&mut self, req: Request) -> Response {
        match req {
            Request::NewGame { seed, pov, start_trick, include_labels } => {
                self.pov = PlaceAtTable(pov);
                self.include_labels = include_labels;
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

                self.clear_pass_selection();
                let done = game.ended();
                let (obs, labels) = self.build_obs_payload(&game);
                self.game = Some(game);
                Response::Obs { obs, done, labels, outcome: None }
            }

            Request::Step { action_idx } => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };

                let actor = game.state.player_at_turn.clone();
                if let Some(pass_options) = self.pass_options_for_turn(&game, actor) {
                    if let Some(card_idx) = decode_pick_action_idx(action_idx) {
                        if let Err(msg) = self.pass_selection.select_card(&pass_options, card_idx) {
                            return Response::Error { message: msg };
                        }

                        if self.pass_selection.selected().len() < PASS_PICK_TARGET {
                            let (obs, labels) = self.build_obs_payload(&game);
                            return Response::Obs {
                                obs,
                                done: false,
                                labels,
                                outcome: None,
                            };
                        }

                        let pass_action_idx = match resolve_selected_pass_action_idx(
                            &pass_options,
                            self.pass_selection.selected(),
                        ) {
                            Some(idx) => idx,
                            None => {
                                self.clear_pass_selection();
                                return Response::Error {
                                    message: "Could not resolve selected pass cards to a legal pass action".into(),
                                };
                            }
                        };

                        let action = game.legal_actions[pass_action_idx].clone();
                        let next = match game.apply_action(action) {
                            Ok(g) => g,
                            Err(e) => return Response::Error { message: format!("{:?}", e) },
                        };
                        self.clear_pass_selection();

                        let done = next.ended();
                        let (obs, labels) = self.build_obs_payload(&next);
                        let outcome = if done {
                            Some(info_to_outcome(&GameFinishedInfo::from(next.clone())))
                        } else {
                            None
                        };
                        self.game = Some(next);
                        return Response::Obs { obs, done, labels, outcome };
                    }
                    // Non-sequential fallback: allow direct legal action index during pass phase.
                    // This is used when stepping non-POV seats from observations that expose
                    // raw legal pass combinations instead of sequential pick actions.
                    if action_idx >= game.legal_actions.len() {
                        return Response::Error {
                            message: format!(
                                "expected sequential pass-pick action_list_idx while passing, got {}",
                                action_idx
                            ),
                        };
                    }
                }

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
                self.clear_pass_selection();
                let done = next.ended();
                let (obs, labels) = self.build_obs_payload(&next);
                let outcome = if done {
                    Some(info_to_outcome(&GameFinishedInfo::from(next.clone())))
                } else {
                    None
                };
                self.game = Some(next);
                Response::Obs { obs, done, labels, outcome }
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
                    self.clear_pass_selection();
                    let done = next.ended();
                    let (obs, labels) = self.build_obs_payload(&next);
                    let outcome = if done {
                        Some(info_to_outcome(&GameFinishedInfo::from(next.clone())))
                    } else {
                        None
                    };
                    self.game = Some(next);
                    Response::Obs { obs, done, labels, outcome }
                } else {
                    Response::Error { message: "Pass combination not legal for current player".into() }
                }
            }

            Request::Observe => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let done = game.ended();
                let (obs, labels) = self.build_obs_payload(&game);
                Response::Obs { obs, done, labels, outcome: None }
            }

            Request::ObservePov { pov } => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let done = game.ended();
                let (obs, _labels) = self.build_obs_payload_for_pov(&game, PlaceAtTable(pov));
                Response::Obs {
                    obs,
                    done,
                    labels: None,
                    outcome: None,
                }
            }

            Request::ObserveDebug => {
                let game = match &self.game {
                    Some(g) => g,
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let obs_full = build_observation(game, self.pov.clone());
                let debug = ObservationDebugJson::from(&obs_full);
                Response::DebugObs { debug }
            }

            Request::GetHeuristicAction => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };

                if let Some(pass_options) = self.pass_options_for_turn(&game, self.pov.clone()) {
                    let legal_pick_actions = build_pick_legal_actions(&pass_options, self.pass_selection.selected());
                    if legal_pick_actions.is_empty() {
                        return Response::HeuristicAction { action_idx: 0 };
                    }

                    let mut policy_cache = HashMap::new();
                    let combo_idx = crate::ml::sim::heuristic_policy()(&game, &mut policy_cache)
                        .min(game.legal_actions.len().saturating_sub(1));
                    let preferred = pass_options
                        .iter()
                        .find(|opt| opt.game_action_idx == combo_idx)
                        .unwrap_or(&pass_options[0]);

                    let selected = self.pass_selection.selected();
                    let preferred_card = preferred
                        .cards
                        .iter()
                        .copied()
                        .find(|c| !selected.contains(c));
                    let selected_action_list_idx = preferred_card
                        .map(crate::ml::pass_selection::encode_pick_action_idx)
                        .filter(|idx| legal_pick_actions.iter().any(|la| la.action_list_idx == *idx))
                        .unwrap_or(legal_pick_actions[0].action_list_idx);

                    let pos = legal_pick_actions
                        .iter()
                        .position(|la| la.action_list_idx == selected_action_list_idx)
                        .unwrap_or(0);
                    return Response::HeuristicAction { action_idx: pos };
                }

                let mut policy_cache = HashMap::new();
                let idx = crate::ml::sim::heuristic_policy()(&game, &mut policy_cache);
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
                self.clear_pass_selection();
                Response::Done { outcome: info_to_outcome(&info) }
            }

            Request::TryAllActions { policy, num_rollouts } => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let pol = match policy {
                    PolicyName::Random => random_policy(),
                    PolicyName::Heuristic => heuristic_policy(),
                };
                let branches_raw = try_all_actions(&game, &pol, &mut self.cache, num_rollouts);
                let branches = if let Some(pass_options) = self.pass_options_for_turn(&game, self.pov.clone()) {
                    let selected = self.pass_selection.selected().to_vec();
                    let pick_actions = build_pick_legal_actions(&pass_options, &selected);
                    pick_actions
                        .into_iter()
                        .enumerate()
                        .map(|(idx, pick)| {
                            let pick_card = pick.card_idx.unwrap_or(0);
                            let outcomes = pass_options
                                .iter()
                                .filter(|opt| {
                                    selected.iter().all(|c| opt.cards.contains(c))
                                        && opt.cards.contains(&pick_card)
                                })
                                .filter_map(|opt| branches_raw.get(opt.game_action_idx))
                                .flat_map(|(_, infos)| infos.iter())
                                .map(info_to_outcome)
                                .collect();
                            BranchResult {
                                action_idx: idx,
                                action_token: pick.action_token,
                                card_idx: pick.card_idx,
                                outcomes,
                            }
                        })
                        .collect()
                } else {
                    branches_raw
                        .into_iter()
                        .enumerate()
                        .map(|(idx, (action, infos))| {
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
                        })
                        .collect()
                };
                Response::TryAllResult { branches }
            }

            Request::GetAdvantages { policy, num_rollouts } => {
                let game = match &self.game {
                    Some(g) => g.clone(),
                    None => return Response::Error { message: "No game in progress".into() },
                };
                let pol = match policy {
                    PolicyName::Random => random_policy(),
                    PolicyName::Heuristic => heuristic_policy(),
                };
                
                let branches_raw = try_all_actions(&game, &pol, &mut self.cache, num_rollouts);
                if branches_raw.is_empty() {
                    return Response::Advantages { advantages: vec![] };
                }

                let acting_party = game.state.player_at_turn.party();
                let action_scores: Vec<f32> = branches_raw
                    .iter()
                    .map(|(_, infos)| self.branch_score_for_infos(infos, acting_party.clone()))
                    .collect();

                let advantages = if let Some(pass_options) = self.pass_options_for_turn(&game, self.pov.clone()) {
                    let selected = self.pass_selection.selected().to_vec();
                    let pick_scores: Vec<f32> = candidate_scores(&pass_options, &selected, &action_scores)
                        .into_iter()
                        .map(|(_, score)| score)
                        .collect();
                    Server::normalize_scores(pick_scores)
                } else {
                    Server::normalize_scores(action_scores)
                };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_server_in_passing_turn() -> Server {
        let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
        let mut game = Game::new("proto_test".to_string(), names, None);

        let mut actions = game.legal_actions.clone();
        for _ in 0..4 {
            let a = actions.pop().expect("start action exists");
            game = game.apply_action(a).expect("start action valid");
            actions = game.legal_actions.clone();
        }

        let policy = heuristic_policy();
        let mut cache = HashMap::new();
        while !matches!(game.state.phase, GamePhase::PassingForth | GamePhase::PassingBack) {
            let idx = policy(&game, &mut cache).min(game.legal_actions.len().saturating_sub(1));
            let a = game.legal_actions[idx].clone();
            game = game.apply_action(a).expect("pre-pass rollout action valid");
        }

        let pov = game.state.player_at_turn.clone();
        let mut server = Server::new();
        server.pov = pov;
        server.game = Some(game);
        server
    }

    fn setup_server_in_passing_turn_with_mismatched_root_pov() -> (Server, u8) {
        let mut server = setup_server_in_passing_turn();
        let active = server
            .game
            .as_ref()
            .expect("game exists")
            .state
            .player_at_turn
            .0;
        server.pov = PlaceAtTable((active + 1) % 4);
        (server, active)
    }

    fn obs_from_response(resp: Response) -> ObservationJson {
        match resp {
            Response::Obs { obs, .. } => obs,
            other => panic!("expected obs response, got {:?}", other),
        }
    }

    #[test]
    fn sequential_pass_actions_are_exposed_in_passing_turn() {
        let mut server = setup_server_in_passing_turn();
        let obs = obs_from_response(server.handle(Request::Observe));
        assert!(!obs.legal_actions.is_empty());
        assert!(obs
            .legal_actions
            .iter()
            .all(|la| la.action_token == crate::ml::observation::tokens::ACT_PASS_PICK_CARD));
        assert!(obs.pass_selection_indices.is_empty());
        assert_eq!(obs.pass_selection_target, PASS_PICK_TARGET);
    }

    #[test]
    fn sequential_pass_step_accumulates_then_commits() {
        let mut server = setup_server_in_passing_turn();
        let mut obs = obs_from_response(server.handle(Request::Observe));
        let passing_phase = obs.phase.clone();

        for expected_len in 1..PASS_PICK_TARGET {
            let action_list_idx = obs.legal_actions[0].action_list_idx;
            obs = obs_from_response(server.handle(Request::Step { action_idx: action_list_idx }));
            assert_eq!(obs.pass_selection_indices.len(), expected_len);
            assert_eq!(obs.phase, passing_phase);
        }

        let action_list_idx = obs.legal_actions[0].action_list_idx;
        obs = obs_from_response(server.handle(Request::Step { action_idx: action_list_idx }));
        assert!(obs.pass_selection_indices.is_empty());
    }

    #[test]
    fn pass_pick_advantages_match_current_legal_action_count() {
        let mut server = setup_server_in_passing_turn();
        let obs = obs_from_response(server.handle(Request::Observe));
        let adv = match server.handle(Request::GetAdvantages {
            policy: PolicyName::Heuristic,
            num_rollouts: 1,
        }) {
            Response::Advantages { advantages } => advantages,
            other => panic!("expected advantages response, got {:?}", other),
        };
        assert_eq!(adv.len(), obs.legal_actions.len());
    }

    #[test]
    fn observe_pov_and_step_support_sequential_pass_when_root_pov_differs() {
        let (mut server, active) = setup_server_in_passing_turn_with_mismatched_root_pov();
        let mut seat_obs = obs_from_response(server.handle(Request::ObservePov { pov: active }));
        assert!(!seat_obs.legal_actions.is_empty());
        assert!(seat_obs
            .legal_actions
            .iter()
            .all(|la| la.action_token == crate::ml::observation::tokens::ACT_PASS_PICK_CARD));

        let pick_idx = seat_obs.legal_actions[0].action_list_idx;
        let root_obs_after = obs_from_response(server.handle(Request::Step { action_idx: pick_idx }));
        assert!(!root_obs_after.legal_actions.is_empty());

        seat_obs = obs_from_response(server.handle(Request::ObservePov { pov: active }));
        assert!(!seat_obs.pass_selection_indices.is_empty());
    }
}
