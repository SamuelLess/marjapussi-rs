/// Hidden seats from POV perspective:
/// 0 = left, 1 = partner, 2 = right.
pub const HIDDEN_SEATS: usize = 3;
/// 36-card deck.
pub const CARD_COUNT: usize = 36;
/// Four suits.
pub const SUIT_COUNT: usize = 4;
/// 9 ranks per suit.
pub const CARDS_PER_SUIT: usize = 9;
/// Value index in per-suit ordering for Ober.
pub const OBER_VALUE_INDEX: usize = 5;
/// Value index in per-suit ordering for King.
pub const KING_VALUE_INDEX: usize = 6;

pub type HiddenPossibleMask = [[bool; CARD_COUNT]; HIDDEN_SEATS];
pub type HiddenConfirmedMask = [[bool; CARD_COUNT]; HIDDEN_SEATS];
pub type HalfConstraintGrid = [[HalfConstraint; SUIT_COUNT]; HIDDEN_SEATS];

/// Constraint type derived from question/answer game events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfConstraint {
    /// No additional Q&A constraint for this (seat, suit).
    Unknown,
    /// Seat must hold at least one of (Ober, King) of this suit.
    RequireAtLeastOne,
    /// Seat must hold both (Ober, King) of this suit.
    RequireBoth,
}

pub fn pair_card_indices(suit_idx: usize) -> (usize, usize) {
    (
        suit_idx * CARDS_PER_SUIT + OBER_VALUE_INDEX,
        suit_idx * CARDS_PER_SUIT + KING_VALUE_INDEX,
    )
}
