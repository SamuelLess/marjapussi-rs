use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use super::rules::{
    finalize_confirmed_projection, rule_confirmed_consistency, rule_half_constraints,
};
use super::terms::{
    HalfConstraint, HalfConstraintGrid, HiddenConfirmedMask, HiddenPossibleMask, CARD_COUNT,
    HIDDEN_SEATS, SUIT_COUNT,
};

type HiddenSetOutput = (HiddenPossibleMask, HiddenConfirmedMask);

fn hidden_set_cache() -> &'static Mutex<HashMap<u64, HiddenSetOutput>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, HiddenSetOutput>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

static ENABLE_HIDDEN_SET_CONSTRAINTS: AtomicBool = AtomicBool::new(true);

pub fn hidden_set_constraints_enabled() -> bool {
    ENABLE_HIDDEN_SET_CONSTRAINTS.load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn set_hidden_set_constraints_enabled(enabled: bool) -> bool {
    ENABLE_HIDDEN_SET_CONSTRAINTS.swap(enabled, Ordering::Relaxed)
}

fn hash_hidden_set_input(
    possible_bitmasks: &HiddenPossibleMask,
    confirmed_bitmasks: &HiddenConfirmedMask,
    half_constraints: &HalfConstraintGrid,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for seat in 0..HIDDEN_SEATS {
        for card in 0..CARD_COUNT {
            possible_bitmasks[seat][card].hash(&mut hasher);
            confirmed_bitmasks[seat][card].hash(&mut hasher);
        }
    }
    for seat in 0..HIDDEN_SEATS {
        for suit in 0..SUIT_COUNT {
            match half_constraints[seat][suit] {
                HalfConstraint::Unknown => 0u8,
                HalfConstraint::RequireAtLeastOne => 1u8,
                HalfConstraint::RequireBoth => 2u8,
            }
            .hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Hidden-hand set-theory inference engine.
///
/// Rule order (iterated to fixpoint):
/// 1. `rule_confirmed_consistency`: confirmed cards must be possible and unique.
/// 2. `rule_half_constraints`: enforce Q&A constraints for half/pair.
///
/// Final projection:
/// - re-assert confirmed uniqueness and confirmed -> possible consistency.
pub fn apply_hidden_set_constraints(
    possible_bitmasks: &mut HiddenPossibleMask,
    confirmed_bitmasks: &mut HiddenConfirmedMask,
    half_constraints: &HalfConstraintGrid,
) {
    let key = hash_hidden_set_input(possible_bitmasks, confirmed_bitmasks, half_constraints);
    if let Ok(cache) = hidden_set_cache().lock() {
        if let Some((cached_possible, cached_confirmed)) = cache.get(&key) {
            *possible_bitmasks = *cached_possible;
            *confirmed_bitmasks = *cached_confirmed;
            return;
        }
    }

    const MAX_ITERS: usize = 16;

    type RuleFn = fn(&mut HiddenPossibleMask, &mut HiddenConfirmedMask, &HalfConstraintGrid) -> bool;
    const RULES: [RuleFn; 2] = [rule_confirmed_consistency, rule_half_constraints];

    for _ in 0..MAX_ITERS {
        let mut changed = false;
        for rule in RULES {
            changed |= rule(possible_bitmasks, confirmed_bitmasks, half_constraints);
        }
        if !changed {
            break;
        }
    }

    finalize_confirmed_projection(possible_bitmasks, confirmed_bitmasks);

    if let Ok(mut cache) = hidden_set_cache().lock() {
        if cache.len() > 2048 {
            cache.clear();
        }
        cache.insert(key, (*possible_bitmasks, *confirmed_bitmasks));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::inference::terms::HalfConstraint;

    #[test]
    fn test_set_constraints_single_candidate_stays_unique_possible() {
        let mut possible = [[false; 36]; 3];
        let mut confirmed = [[false; 36]; 3];
        let half = [[HalfConstraint::Unknown; 4]; 3];

        possible[2][7] = true;
        apply_hidden_set_constraints(&mut possible, &mut confirmed, &half);

        assert!(!confirmed[2][7]);
        assert!(!possible[0][7]);
        assert!(!possible[1][7]);
        assert!(possible[2][7]);
    }

    #[test]
    fn test_set_constraints_yes_half_forces_other_card() {
        let mut possible = [[true; 36]; 3];
        let mut confirmed = [[false; 36]; 3];
        let mut half = [[HalfConstraint::Unknown; 4]; 3];

        half[0][0] = HalfConstraint::RequireAtLeastOne;
        let green_ober = 5; // suit 0 * 9 + Ober index 5
        let green_king = 6; // suit 0 * 9 + King index 6
        possible[0][green_ober] = false;
        possible[0][green_king] = true;

        apply_hidden_set_constraints(&mut possible, &mut confirmed, &half);
        assert!(confirmed[0][green_king]);
    }
}
