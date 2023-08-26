extern crate test;

use crate::game::cards::{Card, Suit, Value};
use crate::game::parse::parse_card;

#[test]
pub fn test_parse_card() {
    assert_eq!(
        parse_card(String::from("r-A")).unwrap(),
        Card {
            suit: Suit::Red,
            value: Value::Ace
        }
    );
    assert_eq!(
        parse_card(String::from("g-O")).unwrap(),
        Card {
            suit: Suit::Green,
            value: Value::Ober
        }
    );
    assert_eq!(
        parse_card(String::from("s-6")).unwrap(),
        Card {
            suit: Suit::Bells,
            value: Value::Six,
        }
    );
}
