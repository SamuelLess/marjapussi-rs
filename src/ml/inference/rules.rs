use super::terms::{
    pair_card_indices, HalfConstraint, HalfConstraintGrid, HiddenConfirmedMask, HiddenPossibleMask,
    CARD_COUNT, HIDDEN_SEATS, SUIT_COUNT,
};

/// Rule 1:
/// Confirmed cards must remain possible and unique across hidden seats.
pub(crate) fn rule_confirmed_consistency(
    possible_bitmasks: &mut HiddenPossibleMask,
    confirmed_bitmasks: &mut HiddenConfirmedMask,
    _half_constraints: &HalfConstraintGrid,
) -> bool {
    let mut changed = false;
    for card in 0..CARD_COUNT {
        let mut owner: Option<usize> = None;
        for seat in 0..HIDDEN_SEATS {
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
            for seat in 0..HIDDEN_SEATS {
                if seat != owner_seat && confirmed_bitmasks[seat][card] {
                    confirmed_bitmasks[seat][card] = false;
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Rule 2:
/// Apply half/pair constraints from Q&A to possible/confirmed masks.
pub(crate) fn rule_half_constraints(
    possible_bitmasks: &mut HiddenPossibleMask,
    confirmed_bitmasks: &mut HiddenConfirmedMask,
    half_constraints: &HalfConstraintGrid,
) -> bool {
    let mut changed = false;
    for seat in 0..HIDDEN_SEATS {
        for suit in 0..SUIT_COUNT {
            let (ober_idx, king_idx) = pair_card_indices(suit);
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
    changed
}

/// Final post-fixpoint projection:
/// ensure confirmed -> possible and per-card unique ownership.
pub(crate) fn finalize_confirmed_projection(
    possible_bitmasks: &mut HiddenPossibleMask,
    confirmed_bitmasks: &mut HiddenConfirmedMask,
) {
    for card in 0..CARD_COUNT {
        let mut owner: Option<usize> = None;
        for seat in 0..HIDDEN_SEATS {
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
            for seat in 0..HIDDEN_SEATS {
                if seat != owner_seat {
                    confirmed_bitmasks[seat][card] = false;
                }
            }
        }
    }
}
