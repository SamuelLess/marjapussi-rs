//! Bitset view of cards and the owner of the card-play rule.
//!
//! `play_levels` (which cards may be played) and `takes` / `trick_winner` (who wins a trick)
//! are the single implementation of these rules: `cards::allowed_cards`, `Game` and
//! `engine::fast` all call them. The test `play_levels_mirror_allowed_cards` keeps the former
//! `Vec`-based implementation as a reference oracle on random tricks and hands.

use std::fmt;
use std::ops::{BitAnd, BitOr};

use crate::game::cards::{Card, Suit, Value};

pub const SUITS: [Suit; 4] = [Suit::Green, Suit::Acorns, Suit::Bells, Suit::Red];
pub const VALUES: [Value; 9] = [
    Value::Six,
    Value::Seven,
    Value::Eight,
    Value::Nine,
    Value::Unter,
    Value::Ober,
    Value::King,
    Value::Ten,
    Value::Ace,
];

/// Bit index of a card: `suit * 9 + value`, in the engine's enum order.
pub fn index(card: &Card) -> u8 {
    card.suit as u8 * 9 + card.value as u8
}

pub fn card(index: u8) -> Card {
    Card {
        suit: SUITS[(index / 9) as usize],
        value: VALUES[(index % 9) as usize],
    }
}

/// A set of cards as 36 bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CardSet(pub u64);

impl CardSet {
    pub const EMPTY: CardSet = CardSet(0);
    pub const ALL: CardSet = CardSet((1 << 36) - 1);

    pub const fn single(index: u8) -> CardSet {
        CardSet(1 << index)
    }

    pub fn of(card: &Card) -> CardSet {
        CardSet::single(index(card))
    }

    pub fn from_cards<'a>(cards: impl IntoIterator<Item = &'a Card>) -> CardSet {
        cards
            .into_iter()
            .fold(CardSet::EMPTY, |set, c| set | CardSet::of(c))
    }

    pub const fn suit(suit: Suit) -> CardSet {
        CardSet(0x1FF << (suit as u64 * 9))
    }

    /// King and Ober of a suit.
    pub const fn halves(suit: Suit) -> CardSet {
        let base = suit as u64 * 9;
        CardSet((1 << (base + Value::King as u64)) | (1 << (base + Value::Ober as u64)))
    }

    /// The four cards of one value.
    pub const fn value(value: Value) -> CardSet {
        let v = value as u64;
        CardSet((1 << v) | (1 << (v + 9)) | (1 << (v + 18)) | (1 << (v + 27)))
    }

    pub const fn contains(self, index: u8) -> bool {
        self.0 & (1 << index) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn is_subset(self, other: CardSet) -> bool {
        self.0 & !other.0 == 0
    }

    /// Bit indices in ascending order.
    pub fn iter(self) -> impl Iterator<Item = u8> {
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let i = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            Some(i)
        })
    }
}

impl BitOr for CardSet {
    type Output = CardSet;
    fn bitor(self, rhs: CardSet) -> CardSet {
        CardSet(self.0 | rhs.0)
    }
}

impl BitAnd for CardSet {
    type Output = CardSet;
    fn bitand(self, rhs: CardSet) -> CardSet {
        CardSet(self.0 & rhs.0)
    }
}

impl fmt::Debug for CardSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter().map(card)).finish()
    }
}

/// The play rule for one decision: a hand may play `hand ∩ levels[k]` for the first
/// level `k` that meets the hand.
#[derive(Clone, Copy, Debug)]
pub struct PlayLevels {
    levels: [CardSet; 6],
    len: usize,
}

impl PlayLevels {
    /// The cards `hand` may play (the engine's `allowed_cards`).
    pub fn allowed(&self, hand: CardSet) -> CardSet {
        self.levels[..self.len]
            .iter()
            .map(|&level| level & hand)
            .find(|allowed| !allowed.is_empty())
            .unwrap_or(hand)
    }
}

/// Whether `challenger` takes the trick from the current `winner`.
#[inline]
pub fn takes(challenger: u8, winner: u8, trump: Option<Suit>) -> bool {
    let (cs, ws) = (challenger / 9, winner / 9);
    if cs == ws {
        return challenger > winner;
    }
    trump.is_some_and(|t| t as u8 == cs)
}

/// Position (0–3) of the card that currently wins `trick` (bit indices in play order).
/// The first card leads; panics on an empty trick.
#[inline]
pub fn trick_winner(trick: &[u8], trump: Option<Suit>) -> usize {
    let mut best = 0;
    for (i, &c) in trick.iter().enumerate().skip(1) {
        if takes(c, trick[best], trump) {
            best = i;
        }
    }
    best
}

/// The play rule: which cards may be played into `trick` (bit indices of the cards already
/// in the current trick, 0–3), given the trump and whether this is the game's first trick.
#[inline]
pub fn play_levels(trick: &[u8], trump: Option<Suit>, first_trick: bool) -> PlayLevels {
    let aces = CardSet::value(Value::Ace);
    let green = CardSet::suit(Suit::Green);
    let Some((&first, rest)) = trick.split_first() else {
        return if first_trick {
            PlayLevels {
                levels: [
                    aces,
                    green,
                    CardSet::ALL,
                    CardSet::EMPTY,
                    CardSet::EMPTY,
                    CardSet::EMPTY,
                ],
                len: 3,
            }
        } else {
            PlayLevels {
                levels: [CardSet::ALL; 6],
                len: 1,
            }
        };
    };
    let winner = rest
        .iter()
        .fold(first, |w, &c| if takes(c, w, trump) { c } else { w });
    let led = first / 9;
    let led_suit = CardSet::suit(SUITS[led as usize]);
    let winner_suit = winner / 9;

    // Cards that take the trick: higher cards of the winner's suit, and every trump if the
    // winner is not a trump. (The former `Vec` rule compared the new high card with the old one
    // by `Card`'s derived order; for trumps over a non-trump that could exclude them there, but
    // the trump level below admits the same cards, so the allowed set is identical.)
    let above_winner =
        CardSet(CardSet::suit(SUITS[winner_suit as usize]).0 & !((2u64 << winner) - 1));
    let trumps_over = match trump {
        Some(t) if t as u8 != winner_suit => CardSet::suit(t),
        _ => CardSet::EMPTY,
    };
    let higher = above_winner | trumps_over;
    let led_ace = if first_trick {
        CardSet::single(led * 9 + Value::Ace as u8)
    } else {
        CardSet::EMPTY
    };
    let trump_suit = trump.map(CardSet::suit).unwrap_or(CardSet::EMPTY);
    PlayLevels {
        levels: [
            led_ace,
            higher & led_suit,
            led_suit,
            higher,
            trump_suit,
            CardSet::ALL,
        ],
        len: 6,
    }
}
