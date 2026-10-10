use std::fmt;

use serde::Serialize;
use serde_with::{DeserializeFromStr, SerializeDisplay};
#[cfg(feature = "public-api")]
use strum_macros::EnumIter;

#[cfg(feature = "public-api")]
use crate::bits::{self, CardSet};
#[cfg(not(feature = "public-api"))]
use crate::inline::Filler;

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
pub const CARDS: usize = SUITS.len() * VALUES.len();
const SUIT_CODES: [char; 4] = ['g', 'e', 's', 'r'];
const VALUE_CODES: [char; 9] = ['6', '7', '8', '9', 'U', 'O', 'K', 'Z', 'A'];

#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, DeserializeFromStr, SerializeDisplay,
)]
pub struct Card {
    /// Only compare cards with same color
    pub suit: Suit,
    pub value: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize)]
#[cfg_attr(feature = "public-api", derive(EnumIter))]
pub enum Suit {
    Green,
    Acorns,
    Bells,
    Red,
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize)]
#[cfg_attr(feature = "public-api", derive(EnumIter))]
pub enum Value {
    Six,
    Seven,
    Eight,
    Nine,
    Unter,
    Ober,
    King,
    Ten,
    Ace,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Suit {
    pub const fn code(self) -> char {
        SUIT_CODES[self as usize]
    }

    pub fn from_code(code: char) -> Option<Self> {
        SUIT_CODES.iter().position(|&c| c == code).map(|i| SUITS[i])
    }
}

impl Value {
    pub const fn code(self) -> char {
        VALUE_CODES[self as usize]
    }

    pub fn from_code(code: char) -> Option<Self> {
        VALUE_CODES
            .iter()
            .position(|&c| c == code)
            .map(|i| VALUES[i])
    }
}

impl Card {
    /// Bit index, in enum order (suit, then value).
    #[inline]
    pub const fn index(self) -> u8 {
        self.suit as u8 * VALUES.len() as u8 + self.value as u8
    }

    #[inline]
    pub const fn from_index(index: u8) -> Self {
        Card {
            suit: suit_of(index),
            value: value_of(index),
        }
    }
}

#[inline]
pub const fn suit_of(index: u8) -> Suit {
    SUITS[index as usize / VALUES.len()]
}

#[inline]
pub const fn value_of(index: u8) -> Value {
    VALUES[index as usize % VALUES.len()]
}

#[cfg(not(feature = "public-api"))]
impl Filler for Card {
    const FILL: Self = Card::from_index(0);
}

#[cfg(not(feature = "public-api"))]
impl Filler for Suit {
    const FILL: Self = Suit::Green;
}

/// Whether `higher` takes `lower` (rule owner: `bits::takes`).
#[cfg(feature = "public-api")]
pub fn is_higher_card(higher: &Card, lower: &Card, trump: Option<Suit>) -> bool {
    bits::takes(higher.index(), lower.index(), trump)
}

#[cfg(feature = "public-api")]
pub fn higher_cards(card: &Card, trump: Option<Suit>, pool: Option<Vec<Card>>) -> Vec<Card> {
    pool.unwrap_or_else(get_all_cards)
        .into_iter()
        .filter(|maybe| is_higher_card(maybe, card, trump))
        .collect()
}

/// The card that currently wins the trick, if any (rule owner: `bits::trick_winner`).
#[cfg(feature = "public-api")]
pub fn high_card(trick: Vec<&Card>, trump: Option<Suit>) -> Option<&Card> {
    trick.into_iter().reduce(|winner, card| {
        if card == winner || bits::takes(bits::index(card), bits::index(winner), trump) {
            card
        } else {
            winner
        }
    })
}

/// Keeps the cards of `cards` that are in `allowed`, in their original order.
#[cfg(feature = "public-api")]
fn keep(cards: Vec<&Card>, allowed: CardSet) -> Vec<&Card> {
    let mut cards = cards;
    cards.retain(|c| allowed.contains(c.index()));
    cards
}

/// Only for the first played card in the game. Proper play in rest of first trick handled elsewhere.
#[cfg(feature = "public-api")]
pub fn allowed_first(cards: Vec<&Card>) -> Vec<&Card> {
    allowed_cards(vec![], cards, None, true)
}

/// The cards of `cards` that may be played into `trick`, in hand order
/// (the rule: `bits::play_levels`).
#[cfg(feature = "public-api")]
pub fn allowed_cards<'a>(
    trick: Vec<&'a Card>,
    cards: Vec<&'a Card>,
    trump: Option<Suit>,
    first_trick: bool,
) -> Vec<&'a Card> {
    let idx: Vec<_> = trick.into_iter().map(bits::index).collect();
    let hand = CardSet::from_cards(cards.iter().copied());
    let allowed = bits::play_levels(&idx, trump, first_trick).allowed(hand);
    keep(cards, allowed)
}

/// Suits of which `cards` hold the King or the Ober.
#[cfg(feature = "public-api")]
pub fn halves(cards: Vec<Card>) -> Vec<Suit> {
    let hand = CardSet::from_cards(&cards);
    SUITS
        .into_iter()
        .filter(|&s| !(hand & CardSet::halves(s)).is_empty())
        .collect()
}

/// Suits of which `cards` hold both the King and the Ober.
#[cfg(feature = "public-api")]
pub fn pairs(cards: Vec<Card>) -> Vec<Suit> {
    let hand = CardSet::from_cards(&cards);
    SUITS
        .into_iter()
        .filter(|&s| CardSet::halves(s).is_subset(hand))
        .collect()
}

pub fn get_all_cards() -> Vec<Card> {
    (0..CARDS as u8).map(Card::from_index).collect()
}

#[cfg(feature = "public-api")]
pub fn print_cards(cards: &[Card]) {
    let card_strs: Vec<String> = cards.iter().map(|c| format!("{}", c)).collect();
    let s = card_strs.join(", ");
    println!("{}", s)
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.suit.code(), self.value.code())
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self) // call method from Display
    }
}

impl std::str::FromStr for Card {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 3 {
            return Err(std::io::Error::other("wrong card format"));
        }
        // Preserve the upstream parser's acceptance of any middle byte.
        let code = s.as_bytes();
        let invalid = || std::io::Error::other("Wrong card format");
        Ok(Card {
            suit: Suit::from_code(code[0] as char).ok_or_else(invalid)?,
            value: Value::from_code(code[2] as char).ok_or_else(invalid)?,
        })
    }
}

#[cfg(all(test, feature = "public-api"))]
mod tests {
    use crate::game::parse::parse_cards;

    use super::*;

    #[test]
    fn test_is_higher_card() {
        let ra: Card = "r-A".parse().unwrap();
        let sz: Card = "s-Z".parse().unwrap();
        let su: Card = "s-U".parse().unwrap();
        assert!(!is_higher_card(&su, &sz, None));
        assert!(!is_higher_card(&ra, &sz, Some(Suit::Bells)));
        assert!(is_higher_card(&ra, &sz, Some(Suit::Red)))
    }

    #[test]
    fn test_higher_cards() {
        let ra = Card {
            suit: Suit::Red,
            value: Value::Ace,
        };
        let sz = Card {
            suit: Suit::Bells,
            value: Value::Ten,
        };

        let higher: Vec<Card> = parse_cards(
            vec![
                "s-A", "r-6", "r-7", "r-8", "r-9", "r-U", "r-O", "r-K", "r-Z", "r-A",
            ]
            .into_iter()
            .map(|c| c.to_string())
            .collect(),
        );
        assert_eq!(higher_cards(&ra, None, None), vec![]);
        assert_eq!(higher_cards(&sz, None, None), vec!["s-A".parse().unwrap()]);
        assert_eq!(higher_cards(&sz, Some(Suit::Red), None), higher)
    }

    #[test]
    fn test_high_card() {
        let ra = "r-A".parse().unwrap();
        let sz = "s-Z".parse().unwrap();
        assert_eq!(high_card(vec![], None), None);
        assert_eq!(high_card(vec![&ra, &sz], None), Some(&ra));
        assert_eq!(high_card(vec![&ra, &sz], Some(Suit::Bells)), Some(&sz));
    }

    #[test]
    fn test_allowed_first() {
        let rz: Card = "r-7".parse().unwrap();
        let s7: Card = "s-7".parse().unwrap();
        let gu: Card = "g-U".parse().unwrap();
        let ra: Card = "r-A".parse().unwrap();
        let sa: Card = "s-A".parse().unwrap();
        let mut cards: Vec<&Card> = vec![&rz, &s7];
        assert_eq!(allowed_first(cards.clone()), cards);
        cards.push(&gu);
        assert_eq!(allowed_first(cards.clone()), vec![&gu]);
        cards.push(&ra);
        cards.push(&sa);
        assert_eq!(allowed_first(cards.clone()), vec![&ra, &sa]);
    }

    #[test]
    fn test_allowed_cards() {
        let ga: Card = "g-A".parse().unwrap();
        let gz: Card = "g-Z".parse().unwrap();
        let go: Card = "g-O".parse().unwrap();
        let s7: Card = "s-7".parse().unwrap();
        let e7: Card = "e-7".parse().unwrap();
        let so: Card = "s-O".parse().unwrap();
        let ro: Card = "r-O".parse().unwrap();
        let ru: Card = "r-U".parse().unwrap();
        let gu: Card = "g-U".parse().unwrap();
        let g9: Card = "g-9".parse().unwrap();
        let mut cards: Vec<&Card> = vec![];
        let trick: Vec<&Card> = vec![];
        //everything empty
        assert_eq!(
            allowed_cards(trick, cards.clone(), None, false),
            vec![] as Vec<&Card>
        );
        //first trick, ace wasn't played
        let trick = vec![&gu];
        cards.push(&ru);
        cards.push(&g9);
        cards.push(&gz);
        cards.push(&ga);
        assert_eq!(
            allowed_cards(trick.clone(), cards.clone(), None, true),
            vec![&ga]
        );
        //same color and higher
        assert_eq!(
            allowed_cards(trick, cards.clone(), None, false),
            vec![&gz, &ga]
        );
        //trump
        let trick = vec![&so];
        assert_eq!(
            allowed_cards(trick, cards.clone(), Some(Suit::Red), false),
            vec![&ru]
        );
        //same color but no higher if already trump
        let trick = vec![&go, &ro];
        assert_eq!(
            allowed_cards(trick, cards.clone(), Some(Suit::Red), false),
            vec![&g9, &gz, &ga]
        );
        // no led suit available, trump exists but cannot beat current trump: still must trump
        let trick = vec![&go, &ro];
        let no_led_with_low_trump = vec![&ru, &s7, &e7];
        assert_eq!(
            allowed_cards(trick, no_led_with_low_trump, Some(Suit::Red), false),
            vec![&ru]
        );
    }
}
