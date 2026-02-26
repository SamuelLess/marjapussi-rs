use rand::prelude::IndexedRandom;

use crate::game::gameevent::GameAction;

use crate::game::gameinfo::GameFinishedInfo;
use crate::game::gamestate::GamePhase;
use crate::game::Game;

use std::collections::HashMap;
use crate::ml::search::TtEntry;

/// A simple policy function type: given the current game state and a transposition table, choose one action index.
pub type PolicyFn = Box<dyn Fn(&Game, &mut HashMap<u128, TtEntry>) -> usize + Send + Sync>;

/// Run a game from its current state to completion, using the provided policy
/// to select actions at each step.
pub fn run_to_end(mut game: Game, policy: &PolicyFn, cache: &mut HashMap<u128, TtEntry>) -> (Game, GameFinishedInfo) {
    while game.state.phase != GamePhase::Ended {
        let idx = policy(&game, cache);
        let actions = &game.legal_actions;
        if actions.is_empty() {
            break;
        }
        let idx = idx.min(actions.len() - 1);
        game.apply_action_mut(actions[idx].clone());
    }
    let info = GameFinishedInfo::from(game.clone());
    (game, info)
}

/// Try every legal action at the current decision point, run each branch to
/// completion with the given policy, and return outcomes for all branches.
pub fn try_all_actions(game: &Game, policy: &PolicyFn, cache: &mut HashMap<u128, TtEntry>, num_rollouts: usize) -> Vec<(GameAction, Vec<GameFinishedInfo>)> {
    let actions = game.legal_actions.clone();
    let mut results = vec![];
    for action in actions {
        let mut group = vec![];
        for _ in 0..num_rollouts {
            let mut branch = game.clone();
            branch.apply_action_mut(action.clone());
            let (_final_game, info) = run_to_end(branch, policy, cache);
            group.push(info);
        }
        results.push((action, group));
    }
    results
}

/// Random policy: uniformly picks a random legal action.
pub fn random_policy() -> PolicyFn {
    Box::new(|game: &Game, _cache: &mut HashMap<u128, TtEntry>| {
        let actions = &game.legal_actions;
        if actions.is_empty() { return 0; }
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
    Box::new(|game: &Game, _cache: &mut HashMap<u128, TtEntry>| {
        use crate::game::gameevent::{ActionType, QuestionType};
        use crate::game::cards::Value;
        let actions = &game.legal_actions;
        if actions.is_empty() { return 0; }

        // Last 3 tricks (trick 7, 8, 9): use optimal search
        // DISABLED due to excessive CPU utilization / hanging
        // if game.state.all_tricks.len() + 1 >= 7 {
        //     let (best_idx, _val) = crate::ml::search::find_best_action(game, 12, _cache); // 12 plies max (3 tricks * 4)
        //     if let Some(idx) = best_idx {
        //         return idx;
        //     }
        // }

        // Priority: AnnounceTrump > play Ace > play Ten > play lowest card
        // Answers are forced (only one legal), bids: stop bidding (conservative)

        // If there's a trump announcement, take it immediately
        for (i, a) in actions.iter().enumerate() {
            if matches!(a.action_type, ActionType::AnnounceTrump(_)) {
                return i;
            }
        }

        // If we can ask before leading, prefer informative questions that help
        // resolve our own halves into trump before we throw those cards away.
        let hand_now = game.state.player_at_turn().cards.clone();
        let my_halves = crate::game::cards::halves(hand_now.clone());
        let my_pairs = crate::game::cards::pairs(hand_now);
        let suit_rank = |s: &crate::game::cards::Suit| -> i32 {
            use crate::game::cards::Suit;
            match s {
                Suit::Red => 4,
                Suit::Bells => 3,
                Suit::Acorns => 2,
                Suit::Green => 1,
            }
        };
        let mut best_question: Option<(usize, i32)> = None;
        for (i, a) in actions.iter().enumerate() {
            let q_score = match &a.action_type {
                ActionType::Question(QuestionType::YourHalf(suit)) => {
                    let mut score = 0;
                    if my_halves.contains(suit) { score += 40; }
                    if my_pairs.contains(suit) { score -= 15; }
                    if !game.state.trump_called.contains(suit) { score += 15; }
                    else { score -= 20; }
                    score += suit_rank(suit);
                    Some(score)
                }
                ActionType::Question(QuestionType::Yours) => {
                    let mut score = 5;
                    if my_pairs.is_empty() { score += 10; }
                    if my_halves.len() >= 2 { score += 10; }
                    Some(score)
                }
                _ => None,
            };
            if let Some(score) = q_score {
                match best_question {
                    Some((_, best)) if score <= best => {}
                    _ => best_question = Some((i, score)),
                }
            }
        }
        if let Some((idx, _)) = best_question {
            return idx;
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

        let mut is_bidding_action = false;
        for a in actions {
            if matches!(a.action_type, ActionType::NewBid(_) | ActionType::StopBidding) {
                is_bidding_action = true; break;
            }
        }

        if is_bidding_action {
            let mut my_bids = vec![];
            let mut partner_bids = vec![];
            let mut current_max = 115;
            let my_seat = game.state.player_at_turn.0;
            let partner_seat = (my_seat + 2) % 4;

            for (action, p) in &game.state.bidding_history {
                if let ActionType::NewBid(val) = action {
                    if p.0 == my_seat { my_bids.push(*val - current_max); }
                    if p.0 == partner_seat { partner_bids.push(*val - current_max); }
                    current_max = *val;
                }
            }

            let hand = &game.state.players[my_seat as usize].cards;
            use crate::game::cards::{Suit, Value};
            let has_ace = hand.iter().any(|c| c.value == Value::Ace);
            let mut pair_points = 0;
            let mut halves = 0;
            
            for &suit in &[Suit::Acorns, Suit::Green, Suit::Bells, Suit::Red] {
                let has_king = hand.iter().any(|c| c.suit == suit && c.value == Value::King);
                let has_ober = hand.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                if has_king && has_ober {
                    pair_points += match suit {
                        Suit::Red => 100,
                        Suit::Bells => 80,
                        Suit::Acorns => 60,
                        Suit::Green => 40,
                    };
                } else if has_king || has_ober {
                    halves += 1;
                }
            }

            let partner_first_jump = partner_bids.first().copied();
            let partner_second_jump = partner_bids.get(1).copied();

            let partner_has_ace = partner_first_jump == Some(5);
            let partner_indicated_pair = partner_first_jump == Some(10) || partner_first_jump == Some(15);
            let partner_indicated_big_pair = partner_first_jump == Some(15);
            let partner_indicated_two_halves = partner_second_jump == Some(5);
            let partner_indicated_three_halves = partner_first_jump == Some(10); 

            let mut deductive_pair_points = pair_points;

            if pair_points == 0 {
                // Deductive Inference: We have no pairs, but can we mathematically guarantee a shared pair?
                if partner_indicated_big_pair {
                    deductive_pair_points = 80;
                } else if partner_indicated_pair {
                    deductive_pair_points = 40; 
                } else if partner_indicated_two_halves && halves >= 3 {
                    // We hold 3 halves. Partner indicated 2 halves. 3 + 2 = 5 cards for 4 suits -> Guaranteed Marriage overlap!
                    deductive_pair_points = 40;
                } else if partner_indicated_three_halves && halves >= 2 {
                    // We hold 2 halves. Partner indicated 3 halves. 3 + 2 = 5 cards for 4 suits -> Guaranteed Marriage overlap!
                    deductive_pair_points = 40;
                }
            } else {
                // If we also hold a pair, dynamically combine our known pair points.
                if partner_indicated_big_pair {
                    deductive_pair_points += 80;
                } else if partner_indicated_pair {
                    deductive_pair_points += 40;
                }
            }

            let mut max_willing_bid = 140;

            if !has_ace && !partner_has_ace {
                // strict Ace-gate: If neither player holds an Ace, they must instantly pass.
                // It is extremely dangerous to play a game without holding at least one Ace.
                max_willing_bid = 115;
            } else {
                if deductive_pair_points > 0 {
                    // Extend comfort zone using the combined guaranteed team pair values
                    max_willing_bid += (deductive_pair_points as f32 * 0.4) as i32;
                }
                if has_ace || partner_has_ace {
                    max_willing_bid += 10;
                }
            }

            if max_willing_bid > 200 {
                max_willing_bid = 200;
            }

            let mut desired_step = None;
            let my_bid_count = my_bids.len();

            if my_bid_count == 0 {
                // 1st Step Info Signal
                if pair_points >= 80 { desired_step = Some(15); }
                else if pair_points > 0 || halves >= 3 { desired_step = Some(10); }
                else if has_ace || halves >= 2 { desired_step = Some(5); }
            } else if my_bid_count == 1 {
                // 2nd Step Info Signal (e.g. signaling our 2 halves via a second consecutive +5 bid)
                if my_bids[0] == 5 && halves >= 2 && !has_ace { desired_step = Some(5); }
                else { desired_step = Some(5); }
            } else {
                // All subsequent steps: Only +5 padding to secure the bid logic
                desired_step = Some(5);
            }

            let mut chosen_action = None;
            if let Some(step) = desired_step {
                let target_bid = current_max + step;
                if target_bid <= max_willing_bid {
                    if let Some(idx) = actions.iter().position(|a| match a.action_type { ActionType::NewBid(v) => v == target_bid, _ => false }) {
                        chosen_action = Some(idx);
                    } else if let Some(idx) = actions.iter().position(|a| match a.action_type { ActionType::NewBid(v) => v <= max_willing_bid, _ => false }) {
                        chosen_action = Some(idx);
                    }
                }
            }
            if chosen_action.is_none() {
                chosen_action = actions.iter().position(|a| matches!(a.action_type, ActionType::StopBidding));
            }
            if let Some(idx) = chosen_action {
                return idx;
            }
        }

        let passing_actions: Vec<(usize, &Vec<crate::game::cards::Card>)> = actions.iter()
            .enumerate()
            .filter_map(|(i, a)| if let ActionType::Pass(cards) = &a.action_type { Some((i, cards)) } else { None })
            .collect();

        if !passing_actions.is_empty() {
            let my_seat = game.state.player_at_turn.0;
            let hand = &game.state.players[my_seat as usize].cards;
            let is_forth = matches!(game.state.phase, GamePhase::PassingForth);
            
            let mut best_score = -10000;
            let mut best_idx = passing_actions[0].0;

            for &(idx, passed_cards) in &passing_actions {
                let mut score = 0;
                let mut kept = hand.clone();
                for c in passed_cards {
                    if let Some(pos) = kept.iter().position(|x| x == c) {
                        kept.remove(pos);
                    }
                }
                
                let mut suits_kept = [0; 4];
                for c in &kept {
                    suits_kept[c.suit.clone() as usize] += 1;
                }
                let num_suits = suits_kept.iter().filter(|&&c| c > 0).count();
                
                use crate::game::cards::{Suit, Value};
                
                if is_forth {
                    // 1. Pass to leave max 2 suits
                    if num_suits <= 2 { score += 100; }
                    else if num_suits == 3 { score += 50; }
                    
                    // 2. Pass unknown valuable cards or Ace
                    for c in passed_cards {
                        if c.value == Value::Ace || c.value == Value::Ten { score += 10; }
                    }
                    
                    // 3. Never pass full pairs
                    for &suit in &[Suit::Acorns, Suit::Green, Suit::Bells, Suit::Red] {
                        let has_king_kept = kept.iter().any(|c| c.suit == suit && c.value == Value::King);
                        let has_ober_kept = kept.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                        if has_king_kept && has_ober_kept {
                            score += 50; // reward keeping full pairs
                        }
                        
                        let had_king = hand.iter().any(|c| c.suit == suit && c.value == Value::King);
                        let had_ober = hand.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                        if had_king && had_ober {
                            // Penalty for breaking pair
                            let passed_king = passed_cards.iter().any(|c| c.suit == suit && c.value == Value::King);
                            let passed_ober = passed_cards.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                            if passed_king || passed_ober {
                                score -= 200;
                            }
                        }
                    }
                } else {
                    // Backpasser
                    // 1. Keep at least one suit that won't become trump
                    if num_suits >= 2 { score += 50; }
                    
                    // 2. Maintain ability to call trump: keep pairs
                    for &suit in &[Suit::Acorns, Suit::Green, Suit::Bells, Suit::Red] {
                        let has_king_kept = kept.iter().any(|c| c.suit == suit && c.value == Value::King);
                        let has_ober_kept = kept.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                        if has_king_kept && has_ober_kept {
                            score += 100; 
                        }
                    }
                    
                    // 3. Pass back a full suit (cards of same suit)
                    let mut passed_suits = [0; 4];
                    for c in passed_cards {
                        passed_suits[c.suit.clone() as usize] += 1;
                    }
                    if passed_suits.iter().any(|&c| c >= 3) {
                        score += 80;
                    } else if passed_suits.iter().any(|&c| c >= 2) {
                        score += 30;
                    }
                }
                
                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }
            return best_idx;
        }

        // Default: first legal action
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
        let mut cache = std::collections::HashMap::new();
        let (_final_game, info) = run_to_end(game, &random_policy(), &mut cache);
        // A game should have 9 tricks
        assert_eq!(info.tricks.len(), 9);
    }

    #[test]
    fn test_run_to_end_heuristic() {
        let game = make_started_game();
        let mut cache = std::collections::HashMap::new();
        let (_final_game, info) = run_to_end(game, &heuristic_policy(), &mut cache);
        assert_eq!(info.tricks.len(), 9);
    }

    #[test]
    fn test_try_all_actions_first_decision() {
        let game = make_started_game();
        let mut cache = std::collections::HashMap::new();
        // In bidding phase, try all bids
        let results = try_all_actions(&game, &random_policy(), &mut cache, 5);
        assert!(!results.is_empty());
        for (_, infos) in &results {
            for info in infos {
                assert_eq!(info.tricks.len(), 9);
            }
        }
    }
}
