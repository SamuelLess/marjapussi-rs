use rand::prelude::IndexedRandom;
use rand::seq::SliceRandom;

use crate::game::gameevent::GameAction;

use crate::game::gameinfo::GameFinishedInfo;
use crate::game::gamestate::GamePhase;
use crate::game::Game;

/// A simple policy function type: given legal actions, choose one index.
pub type PolicyFn = Box<dyn Fn(&[GameAction]) -> usize + Send + Sync>;

/// Run a game from its current state to completion, using the provided policy
/// to select actions at each step.
pub fn run_to_end(mut game: Game, policy: &PolicyFn) -> (Game, GameFinishedInfo) {
    while game.state.phase != GamePhase::Ended {
        let actions = game.legal_actions.clone();
        if actions.is_empty() {
            break;
        }
        let idx = policy(&actions);
        let idx = idx.min(actions.len() - 1);
        game.apply_action_mut(actions[idx].clone());
    }
    let info = GameFinishedInfo::from(game.clone());
    (game, info)
}

/// Try every legal action at the current decision point, run each branch to
/// completion with the given policy, and return outcomes for all branches.
pub fn try_all_actions(game: &Game, policy: &PolicyFn) -> Vec<(GameAction, GameFinishedInfo)> {
    let actions = game.legal_actions.clone();
    let mut results = vec![];
    for action in actions {
        let mut branch = game.clone();
        branch.apply_action_mut(action.clone());
        let (_final_game, info) = run_to_end(branch, policy);
        results.push((action, info));
    }
    results
}

/// Random policy: uniformly picks a random legal action.
pub fn random_policy() -> PolicyFn {
    Box::new(|actions: &[GameAction]| {
        let mut rng = rand::rng();
        actions.choose(&mut rng)
            .map(|_| (0..actions.len()).collect::<Vec<_>>().choose(&mut rng).copied().unwrap_or(0))
            .unwrap_or(0)
    })
}

/// Heuristic policy: simple rule-based agent.
/// - Prefer playing an Ace if we can (maximum point capture).
/// - Otherwise play the highest card we're forced to (to win tricks).
/// - When leading: play lowest card to avoid giving away points.
/// - Announce own trump immediately if legal.
pub fn heuristic_policy() -> PolicyFn {
    Box::new(|actions: &[GameAction]| {
        use crate::game::gameevent::ActionType;
        use crate::game::cards::Value;

        // Priority: AnnounceTrump > play Ace > play Ten > play lowest card
        // Answers are forced (only one legal), bids: stop bidding (conservative)

        // If there's a trump announcement, take it immediately
        for (i, a) in actions.iter().enumerate() {
            if matches!(a.action_type, ActionType::AnnounceTrump(_)) {
                return i;
            }
        }

        // Collect card-play actions
        let card_plays: Vec<(usize, &crate::game::cards::Card)> = actions.iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if let ActionType::CardPlayed(card) = &a.action_type {
                    Some((i, card))
                } else {
                    None
                }
            })
            .collect();

        if !card_plays.is_empty() {
            // Prefer Ace
            if let Some(&(i, _)) = card_plays.iter().find(|(_, c)| c.value == Value::Ace) {
                return i;
            }
            // Prefer Ten
            if let Some(&(i, _)) = card_plays.iter().find(|(_, c)| c.value == Value::Ten) {
                return i;
            }
            // Play highest card to win
            let best = card_plays.iter()
                .max_by_key(|(_, c)| c.value.clone());
            if let Some(&(i, _)) = best {
                return i;
            }
        }

        // For answers: only one option (forced), return 0
        // For bids: prefer StopBidding
        for (i, a) in actions.iter().enumerate() {
            if matches!(a.action_type, ActionType::StopBidding) {
                return i;
            }
        }

        // Default: first action
        0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_started_game() -> Game {
        let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
        let mut game = Game::new("sim_test".to_string(), names, None);
        let mut actions = game.legal_actions.clone();
        for _ in 0..4 {
            game = game.apply_action(actions.pop().unwrap()).unwrap();
            actions = game.legal_actions.clone();
        }
        game
    }

    #[test]
    fn test_run_to_end_random() {
        let game = make_started_game();
        let (_final_game, info) = run_to_end(game, &random_policy());
        // A game should have 9 tricks
        assert_eq!(info.tricks.len(), 9);
    }

    #[test]
    fn test_run_to_end_heuristic() {
        let game = make_started_game();
        let (_final_game, info) = run_to_end(game, &heuristic_policy());
        assert_eq!(info.tricks.len(), 9);
    }

    #[test]
    fn test_try_all_actions_first_decision() {
        let game = make_started_game();
        // In bidding phase, try all bids
        let results = try_all_actions(&game, &random_policy());
        assert!(!results.is_empty());
        for (_, info) in &results {
            assert_eq!(info.tricks.len(), 9);
        }
    }
}
