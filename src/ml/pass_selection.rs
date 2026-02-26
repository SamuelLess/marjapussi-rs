use std::collections::BTreeSet;

use crate::game::gameevent::ActionType;
use crate::game::gamestate::GamePhase;
use crate::game::player::PlaceAtTable;
use crate::game::Game;
use crate::ml::observation::{card_index, tokens, LegalActionObs};

pub const PASS_PICK_TARGET: usize = 4;
pub const PASS_PICK_ACTION_BASE: usize = 10_000;

#[derive(Debug, Clone)]
pub struct PassActionOption {
    pub game_action_idx: usize,
    pub cards: [usize; PASS_PICK_TARGET],
}

#[derive(Debug, Clone, Default)]
pub struct PassSelectionState {
    selected: Vec<usize>,
}

impl PassSelectionState {
    pub fn clear(&mut self) {
        self.selected.clear();
    }

    pub fn selected(&self) -> &[usize] {
        &self.selected
    }

    pub fn set_selected(&mut self, mut selected: Vec<usize>) {
        selected.sort_unstable();
        selected.dedup();
        self.selected = selected;
    }

    pub fn select_card(&mut self, options: &[PassActionOption], card_idx: usize) -> Result<(), String> {
        if card_idx >= 36 {
            return Err(format!("card index {card_idx} out of range"));
        }
        if self.selected.contains(&card_idx) {
            return Err(format!("card {card_idx} already selected"));
        }
        if self.selected.len() >= PASS_PICK_TARGET {
            return Err("pass selection already complete".to_string());
        }
        let candidates = candidate_cards(options, &self.selected);
        if !candidates.contains(&card_idx) {
            return Err(format!("card {card_idx} is not a valid pass pick in current selection"));
        }
        self.selected.push(card_idx);
        self.selected.sort_unstable();
        Ok(())
    }
}

pub fn is_pov_pass_turn(game: &Game, pov: PlaceAtTable) -> bool {
    if game.state.player_at_turn != pov {
        return false;
    }
    matches!(game.state.phase, GamePhase::PassingForth | GamePhase::PassingBack)
}

pub fn collect_pass_options(game: &Game) -> Vec<PassActionOption> {
    let mut options = Vec::new();
    for (idx, action) in game.legal_actions.iter().enumerate() {
        if let ActionType::Pass(cards) = &action.action_type {
            if cards.len() != PASS_PICK_TARGET {
                continue;
            }
            let mut mapped: Vec<usize> = cards.iter().map(card_index).collect();
            mapped.sort_unstable();
            mapped.dedup();
            if mapped.len() != PASS_PICK_TARGET {
                continue;
            }
            let arr: [usize; PASS_PICK_TARGET] = mapped
                .try_into()
                .expect("mapped pass cards always have exactly 4 entries");
            options.push(PassActionOption {
                game_action_idx: idx,
                cards: arr,
            });
        }
    }
    options
}

pub fn selection_is_completable(options: &[PassActionOption], selected: &[usize]) -> bool {
    options
        .iter()
        .any(|opt| selected.iter().all(|c| opt.cards.contains(c)))
}

pub fn candidate_cards(options: &[PassActionOption], selected: &[usize]) -> Vec<usize> {
    if selected.len() >= PASS_PICK_TARGET {
        return vec![];
    }
    let mut out = BTreeSet::new();
    for opt in options {
        if !selected.iter().all(|c| opt.cards.contains(c)) {
            continue;
        }
        for card_idx in opt.cards {
            if !selected.contains(&card_idx) {
                out.insert(card_idx);
            }
        }
    }
    out.into_iter().collect()
}

pub fn resolve_selected_pass_action_idx(
    options: &[PassActionOption],
    selected: &[usize],
) -> Option<usize> {
    if selected.len() != PASS_PICK_TARGET {
        return None;
    }
    let mut normalized = selected.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    if normalized.len() != PASS_PICK_TARGET {
        return None;
    }
    options.iter().find_map(|opt| {
        let mut cards = opt.cards;
        cards.sort_unstable();
        if cards.as_slice() == normalized.as_slice() {
            Some(opt.game_action_idx)
        } else {
            None
        }
    })
}

pub fn encode_pick_action_idx(card_idx: usize) -> usize {
    PASS_PICK_ACTION_BASE + card_idx
}

pub fn decode_pick_action_idx(action_idx: usize) -> Option<usize> {
    if (PASS_PICK_ACTION_BASE..(PASS_PICK_ACTION_BASE + 36)).contains(&action_idx) {
        Some(action_idx - PASS_PICK_ACTION_BASE)
    } else {
        None
    }
}

pub fn build_pick_legal_actions(options: &[PassActionOption], selected: &[usize]) -> Vec<LegalActionObs> {
    candidate_cards(options, selected)
        .into_iter()
        .map(|card_idx| LegalActionObs {
            action_token: tokens::ACT_PASS_PICK_CARD,
            card_idx: Some(card_idx),
            suit_idx: None,
            bid_value: None,
            pass_cards: None,
            action_list_idx: encode_pick_action_idx(card_idx),
        })
        .collect()
}

pub fn candidate_scores(
    options: &[PassActionOption],
    selected: &[usize],
    action_scores: &[f32],
) -> Vec<(usize, f32)> {
    let candidates = candidate_cards(options, selected);
    let mut out = Vec::with_capacity(candidates.len());
    for card_idx in candidates {
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for opt in options {
            if !selected.iter().all(|c| opt.cards.contains(c)) {
                continue;
            }
            if !opt.cards.contains(&card_idx) {
                continue;
            }
            if let Some(score) = action_scores.get(opt.game_action_idx) {
                sum += *score;
                n += 1;
            }
        }
        out.push((card_idx, if n == 0 { 0.0 } else { sum / n as f32 }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn option(idx: usize, cards: [usize; 4]) -> PassActionOption {
        PassActionOption {
            game_action_idx: idx,
            cards,
        }
    }

    #[test]
    fn candidate_generation_respects_prefix() {
        let options = vec![
            option(0, [0, 1, 2, 3]),
            option(1, [0, 1, 4, 5]),
            option(2, [6, 7, 8, 9]),
        ];
        assert_eq!(candidate_cards(&options, &[]), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(candidate_cards(&options, &[0]), vec![1, 2, 3, 4, 5]);
        assert_eq!(candidate_cards(&options, &[0, 1]), vec![2, 3, 4, 5]);
        assert_eq!(candidate_cards(&options, &[0, 1, 2, 3]), Vec::<usize>::new());
    }

    #[test]
    fn resolve_action_from_final_selection() {
        let options = vec![option(7, [1, 3, 4, 9]), option(8, [0, 2, 4, 6])];
        assert_eq!(resolve_selected_pass_action_idx(&options, &[9, 4, 1, 3]), Some(7));
        assert_eq!(resolve_selected_pass_action_idx(&options, &[0, 2, 4, 6]), Some(8));
        assert_eq!(resolve_selected_pass_action_idx(&options, &[0, 2, 4]), None);
    }

    #[test]
    fn candidate_scoring_aggregates_compatible_combos() {
        let options = vec![
            option(0, [0, 1, 2, 3]),
            option(1, [0, 1, 4, 5]),
            option(2, [6, 7, 8, 9]),
        ];
        let scores = vec![0.5, 1.0, -0.25];
        let mut out = candidate_scores(&options, &[0, 1], &scores);
        out.sort_by_key(|(c, _)| *c);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].0, 2);
        assert!((out[0].1 - 0.5).abs() < 1e-6);
        assert_eq!(out[2].0, 4);
        assert!((out[2].1 - 1.0).abs() < 1e-6);
    }
}
