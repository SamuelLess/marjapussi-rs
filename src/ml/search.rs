use crate::game::Game;
use crate::game::gameevent::{ActionType, GameCallback};
use crate::game::points::points_pair;
use std::collections::HashMap;

// ── Transposition Table ─────────────────────────────────────────────────────

/// Flag stored alongside each TT entry to distinguish exact scores from bounds.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TtFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy)]
pub struct TtEntry {
    pub score: i32,
    pub depth: u8,
    pub flag:  TtFlag,
}

// ── Action pruning ──────────────────────────────────────────────────────────

/// Build a u64 bitmask of all cards that have been played.
/// card_id = suit * 9 + value.
fn played_cards_mask(game: &Game) -> u64 {
    let mut mask = 0u64;
    for trick in &game.state.all_tricks {
        for card in &trick.cards {
            let id = (card.suit as u8 * 9 + card.value as u8) as u64;
            mask |= 1 << id;
        }
    }
    for card in &game.state.current_trick {
        let id = (card.suit as u8 * 9 + card.value as u8) as u64;
        mask |= 1 << id;
    }
    mask
}

/// Return true if both King and Ober for `suit` are NOT in the played mask.
fn pair_possible(suit: &crate::game::cards::Suit, played_mask: u64) -> bool {
    use crate::game::cards::Value;
    let suit_offset = *suit as u64 * 9;
    let king_id = suit_offset + Value::King as u64;
    let ober_id = suit_offset + Value::Ober as u64;
    
    // If either has been played, the pair is dead.
    let king_played = (played_mask & (1 << king_id)) != 0;
    let ober_played = (played_mask & (1 << ober_id)) != 0;
    !king_played && !ober_played
}

/// Filter actions for the endgame search.
fn prune_actions<'a>(
    actions: &'a [crate::game::gameevent::GameAction], 
    played_mask: u64
) -> Vec<&'a crate::game::gameevent::GameAction> {
    use crate::game::gameevent::{ActionType, QuestionType};
    actions.iter().filter(|a| {
        match &a.action_type {
            ActionType::Question(QuestionType::YourHalf(suit)) => pair_possible(suit, played_mask),
            ActionType::AnnounceTrump(suit) => pair_possible(suit, played_mask),
            _ => true,
        }
    }).collect()
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Compute the minimax score for every legal action at the current position.
/// Returns (action_index, score) pairs.
pub fn find_all_action_scores(
    game: &Game,
    max_depth: usize,
    cache: &mut HashMap<u128, TtEntry>,
) -> Vec<(usize, i32)> {
    let actions = &game.legal_actions;
    if actions.is_empty() {
        return vec![];
    }

    // If only one legal move, return immediately (no search needed).
    if actions.len() == 1 {
        return vec![(0, 0)];
    }

    let is_max = game.state.player_at_turn.0 % 2 == 0;
    let mut alpha = -500_i32;
    let mut beta  =  500_i32;

    // Prune impossible pair/half-question actions before sorting.
    let mask = played_cards_mask(game);
    let pruned = prune_actions(actions, mask);
    
    // Sort actions by expected value so we get early cutoffs.
    let mut indexed: Vec<(usize, i32)> = pruned
        .iter()
        .map(|a| {
            // Find original index
            let orig_idx = actions.iter().position(|x| x == *a).unwrap_or(0);
            (orig_idx, move_ordering_score(a, game))
        })
        .collect();
    // Maximiser wants high-value moves first; minimiser wants low-value moves first.
    if is_max {
        indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    } else {
        indexed.sort_unstable_by(|a, b| a.1.cmp(&b.1));
    }

    let mut results = vec![(0usize, 0i32); actions.len()];
    for (i, _prescore) in &indexed {
        let action = &actions[*i];
        if let Ok(next) = game.apply_action(action.clone()) {
            let val = minimax(&next, max_depth.saturating_sub(1), alpha, beta, cache);
            results[*i] = (*i, val);
            if is_max {
                alpha = alpha.max(val);
            } else {
                beta = beta.min(val);
            }
            // Even at the root we can cut the remaining branches once the window closes.
            if beta <= alpha {
                break;
            }
        }
    }
    results
}

/// Find the single best action index (for use by heuristic_policy).
pub fn find_best_action(
    game: &Game,
    max_depth: usize,
    cache: &mut HashMap<u128, TtEntry>,
) -> (Option<usize>, i32) {
    // Forced move — no search.
    if game.legal_actions.len() <= 1 {
        return (game.legal_actions.first().map(|_| 0), 0);
    }

    let scores = find_all_action_scores(game, max_depth, cache);
    if scores.is_empty() {
        return (None, 0);
    }

    let is_max = game.state.player_at_turn.0 % 2 == 0;
    if is_max {
        let best = scores.into_iter().max_by_key(|&(_, v)| v);
        best.map(|(i, v)| (Some(i), v)).unwrap_or((None, 0))
    } else {
        let best = scores.into_iter().min_by_key(|&(_, v)| v);
        best.map(|(i, v)| (Some(i), v)).unwrap_or((None, 0))
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Heuristic ordering score for a single action: we want high-value plays first.
fn move_ordering_score(action: &crate::game::gameevent::GameAction, _game: &Game) -> i32 {
    match &action.action_type {
        ActionType::CardPlayed(card) => card_point_value(card),
        ActionType::AnnounceTrump(_) => 40, // bump, often decisive
        _ => 0,
    }
}

/// Point value of a card for move ordering (not game scoring, just a proxy).
fn card_point_value(card: &crate::game::cards::Card) -> i32 {
    use crate::game::cards::Value;
    match card.value {
        Value::Ace   => 11,
        Value::Ten   => 10,
        Value::King  =>  4,
        Value::Ober  =>  3,
        Value::Unter =>  2,
        _            =>  0,
    }
}

fn calculate_scores(game: &Game) -> (i32, i32) {
    let mut t0 = 0;
    let mut t1 = 0;
    for trick in &game.state.all_tricks {
        if trick.winner.0 % 2 == 0 { t0 += trick.points.0; } else { t1 += trick.points.0; }
    }
    for event in &game.all_events {
        if let Some(GameCallback::NewTrump(suit)) = event.callback {
            let pts = points_pair(suit).0;
            if event.last_action.player.0 % 2 == 0 { t0 += pts; } else { t1 += pts; }
        }
    }
    // (Last-trick bonus is organically handled by the main Game Engine, so we do not double-count here)
    (t0, t1)
}

/// Unique compact hash for the current game state (restricted to endgame info).
fn get_game_key(game: &Game) -> u128 {
    let mut key: u128 = 0;

    // Trump (3 bits at position 120)
    key |= (game.state.trump.map(|s| s as u8).unwrap_or(4) as u128) << 120;
    // Player at turn (2 bits at 124)
    key |= (game.state.player_at_turn.0 as u128) << 124;

    // Card locations: 3 bits per card, 36 cards = 108 bits
    for (p_idx, player) in game.state.players.iter().enumerate() {
        for card in &player.cards {
            let card_id = (card.suit as u8 * 9 + card.value as u8) as u32;
            if card_id < 36 {
                key |= (p_idx as u128 + 1) << (card_id * 3);
            }
        }
    }
    for card in &game.state.current_trick {
        let card_id = (card.suit as u8 * 9 + card.value as u8) as u32;
        if card_id < 36 {
            let shift = card_id * 3;
            key &= !(7 << shift);
            key |= (5u128) << shift;
        }
    }

    // Trump announced (4 bits at 116)
    let mut called_mask = 0u128;
    for suit in &game.state.trump_called {
        called_mask |= 1 << (*suit as u8);
    }
    key |= called_mask << 116;

    // Score differential (10 bits at 106)
    let (s0, s1) = calculate_scores(game);
    let diff = (s0 - s1 + 512) as u128;
    key |= (diff & 0x3FF) << 106;

    key
}

/// Alpha-beta minimax with transposition table and move ordering.
fn minimax(
    game: &Game,
    depth: usize,
    mut alpha: i32,
    mut beta: i32,
    cache: &mut HashMap<u128, TtEntry>,
) -> i32 {
    // Terminal / horizon check
    if game.ended() || depth == 0 {
        let (t0, t1) = calculate_scores(game);
        return t0 - t1;
    }

    let actions = &game.legal_actions;
    if actions.is_empty() {
        let (t0, t1) = calculate_scores(game);
        return t0 - t1;
    }

    // Forced-move fast path: no branching, no TT needed.
    if actions.len() == 1 {
        if let Ok(next) = game.apply_action(actions[0].clone()) {
            return minimax(&next, depth - 1, alpha, beta, cache);
        }
    }

    // Transposition table lookup.
    let key = get_game_key(game);
    if let Some(entry) = cache.get(&key) {
        if entry.depth >= depth as u8 {
            match entry.flag {
                TtFlag::Exact      => return entry.score,
                TtFlag::LowerBound => alpha = alpha.max(entry.score),
                TtFlag::UpperBound => beta  = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    let is_max = game.state.player_at_turn.0 % 2 == 0;
    let orig_alpha = alpha;
    let orig_beta  = beta;

    // Prune impossible pair/half-question actions before sorting.
    let mask = played_cards_mask(game);
    let pruned = prune_actions(actions, mask);
    
    // Move ordering: sort by quick heuristic.
    let mut sorted_actions: Vec<_> = pruned
        .iter()
        .map(|a| (move_ordering_score(a, game), (*a).clone()))
        .collect();
    if is_max {
        sorted_actions.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    } else {
        sorted_actions.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }

    // Early exit if all actions were pruned (shouldn't happen, but be safe).
    if sorted_actions.is_empty() {
        let (t0, t1) = calculate_scores(game);
        return t0 - t1;
    }

    let mut best_eval = if is_max { -500 } else { 500 };

    for (_, action) in &sorted_actions {
        if let Ok(next) = game.apply_action(action.clone()) {
            let eval = minimax(&next, depth - 1, alpha, beta, cache);
            if is_max {
                best_eval = best_eval.max(eval);
                alpha = alpha.max(eval);
            } else {
                best_eval = best_eval.min(eval);
                beta = beta.min(eval);
            }
            if beta <= alpha {
                break; // Alpha-beta cutoff
            }
        }
    }

    // Store result in TT with appropriate flag.
    let flag = if best_eval <= orig_alpha {
        TtFlag::UpperBound
    } else if best_eval >= orig_beta {
        TtFlag::LowerBound
    } else {
        TtFlag::Exact
    };
    cache.insert(key, TtEntry { score: best_eval, depth: depth as u8, flag });

    best_eval
}
