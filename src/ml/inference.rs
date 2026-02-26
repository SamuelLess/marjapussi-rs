use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfConstraint {
    Unknown,
    RequireAtLeastOne,
    RequireBoth,
}

type HiddenSetOutput = ([[bool; 36]; 3], [[bool; 36]; 3]);

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
    possible_bitmasks: &[[bool; 36]; 3],
    confirmed_bitmasks: &[[bool; 36]; 3],
    half_constraints: &[[HalfConstraint; 4]; 3],
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for seat in 0..3 {
        for card in 0..36 {
            possible_bitmasks[seat][card].hash(&mut hasher);
            confirmed_bitmasks[seat][card].hash(&mut hasher);
        }
    }
    for seat in 0..3 {
        for suit in 0..4 {
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

pub fn apply_hidden_set_constraints(
    possible_bitmasks: &mut [[bool; 36]; 3],
    confirmed_bitmasks: &mut [[bool; 36]; 3],
    half_constraints: &[[HalfConstraint; 4]; 3],
) {
    let key = hash_hidden_set_input(
        possible_bitmasks,
        confirmed_bitmasks,
        half_constraints,
    );
    if let Ok(cache) = hidden_set_cache().lock() {
        if let Some((cached_possible, cached_confirmed)) = cache.get(&key) {
            *possible_bitmasks = *cached_possible;
            *confirmed_bitmasks = *cached_confirmed;
            return;
        }
    }

    const MAX_ITERS: usize = 16;

    for _ in 0..MAX_ITERS {
        let mut changed = false;

        // Enforce uniqueness for already confirmed cards.
        for card in 0..36 {
            let mut owner: Option<usize> = None;
            for seat in 0..3 {
                if confirmed_bitmasks[seat][card] {
                    if !possible_bitmasks[seat][card] {
                        confirmed_bitmasks[seat][card] = false;
                        changed = true;
                        continue;
                    }
                    if owner.is_none() {
                        owner = Some(seat);
                    } else {
                        confirmed_bitmasks[seat][card] = false;
                        changed = true;
                    }
                }
            }
            if let Some(owner_seat) = owner {
                for seat in 0..3 {
                    if seat != owner_seat && confirmed_bitmasks[seat][card] {
                        confirmed_bitmasks[seat][card] = false;
                        changed = true;
                    }
                }
            }
        }

        // Apply Q&A half constraints (per seat and suit).
        for seat in 0..3 {
            for suit in 0..4 {
                let ober_idx = suit * 9 + 5;
                let king_idx = suit * 9 + 6;
                match half_constraints[seat][suit] {
                    HalfConstraint::RequireBoth => {
                        if !possible_bitmasks[seat][ober_idx] {
                            possible_bitmasks[seat][ober_idx] = true;
                            changed = true;
                        }
                        if !possible_bitmasks[seat][king_idx] {
                            possible_bitmasks[seat][king_idx] = true;
                            changed = true;
                        }
                        if !confirmed_bitmasks[seat][ober_idx] {
                            confirmed_bitmasks[seat][ober_idx] = true;
                            changed = true;
                        }
                        if !confirmed_bitmasks[seat][king_idx] {
                            confirmed_bitmasks[seat][king_idx] = true;
                            changed = true;
                        }
                    }
                    HalfConstraint::RequireAtLeastOne => {
                        let ober_possible = possible_bitmasks[seat][ober_idx];
                        let king_possible = possible_bitmasks[seat][king_idx];
                        if !ober_possible && king_possible && !confirmed_bitmasks[seat][king_idx] {
                            confirmed_bitmasks[seat][king_idx] = true;
                            changed = true;
                        }
                        if !king_possible && ober_possible && !confirmed_bitmasks[seat][ober_idx] {
                            confirmed_bitmasks[seat][ober_idx] = true;
                            changed = true;
                        }
                    }
                    HalfConstraint::Unknown => {}
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Final consistency pass: confirmed implies possible and uniqueness.
    for card in 0..36 {
        let mut owner: Option<usize> = None;
        for seat in 0..3 {
            if confirmed_bitmasks[seat][card] {
                if !possible_bitmasks[seat][card] {
                    confirmed_bitmasks[seat][card] = false;
                    continue;
                }
                if owner.is_none() {
                    owner = Some(seat);
                } else {
                    confirmed_bitmasks[seat][card] = false;
                }
            }
        }
        if let Some(owner_seat) = owner {
            for seat in 0..3 {
                if seat != owner_seat {
                    confirmed_bitmasks[seat][card] = false;
                }
            }
        }
    }

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
