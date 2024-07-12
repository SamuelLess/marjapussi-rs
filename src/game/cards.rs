#![allow(unused)]

use std::cmp::max;
use std::fmt;

use itertools::concat;
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::game::parse::parse_card;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, DeserializeFromStr, SerializeDisplay)]
pub struct Card {
    /**
     * Only compare cards with same color.
     */
    pub suit: Suit,
    pub value: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, EnumIter, Serialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, EnumIter, Serialize)]
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

/**
 * Returns whether first card is higher than second card.
 */
pub fn is_higher_card(higher: &Card, lower: &Card, trump: Option<Suit>) -> bool {
    match trump {
        Some(suit) => {
            //println!("{:?} {:?} {:?}", higher, lower, trump);
            (higher.suit == suit && lower.suit != suit)
                || (higher.suit == lower.suit && higher.value > lower.value)
        }
        None => higher.suit == lower.suit && higher.value > lower.value,
    }
}

pub fn higher_cards(card: &Card, trump: Option<Suit>, pool: Option<Vec<Card>>) -> Vec<Card> {
    pool.unwrap_or_else(|| get_all_cards())
        .into_iter()
        .filter(|maybe| is_higher_card(maybe, card, trump))
        .collect()
}

/**
 * Returns highest card if exists.
 */
pub fn high_card(trick: Vec<&Card>, trump: Option<Suit>) -> Option<&Card> {
    if trick.is_empty() {
        return None;
    }
    let trick_suit = trick[0].suit;
    Some(
        match trump {
            Some(trump_suit) => {
                //if any trump in trick only count trump
                let all_suits: Vec<Suit> = trick.iter().map(|c| c.suit).rev().collect();
                if all_suits.contains(&trump_suit) {
                    trick
                        .into_iter()
                        .filter(|x| x.suit == trump_suit)
                        .collect::<Vec<_>>()
                } else {
                    trick.into_iter().filter(|x| x.suit == trick_suit).collect()
                }
            }
            None => trick.into_iter().filter(|x| x.suit == trick_suit).collect(),
        }
        .iter()
        .reduce(max)
        .unwrap(),
    )
}

/**
 * Only for the first played card in the game. Proper play in rest of first trick handled elsewhere.
 */
pub fn allowed_first(cards: Vec<&Card>) -> Vec<&Card> {
    let mut allowed = cards
        .clone()
        .into_iter()
        .filter(|c| c.value == Value::Ace)
        .collect::<Vec<_>>();
    if allowed.is_empty() {
        allowed = cards
            .clone()
            .into_iter()
            .filter(|c| c.suit == Suit::Green)
            .collect::<Vec<_>>();
    }
    if !allowed.is_empty() {
        allowed
    } else {
        cards
    }
}

/**
 * Returns general high card.
 */
pub fn allowed_cards<'a>(
    trick: Vec<&'a Card>,
    cards: Vec<&'a Card>,
    trump: Option<Suit>,
    first_trick: bool,
) -> Vec<&'a Card> {
    let current_high_card = high_card(trick.clone(), trump);
    /*println!(
        "trick={:?} wiht cards={:?} and trump={:?}",
        trick, cards, trump
    );*/
    match current_high_card {
        Some(current_high_card) => {
            let trick_suit = trick[0].suit;
            if first_trick {
                let first_trick_ace: Vec<&Card> = cards
                    .clone()
                    .into_iter()
                    .filter(|c| c.suit == trick_suit && c.value == Value::Ace)
                    .collect();
                if first_trick_ace.len() == 1 {
                    return first_trick_ace;
                }
            }
            let higher_cards: Vec<&Card> = cards
                .clone()
                .into_iter()
                .filter(|c| {
                    high_card(concat([trick.clone(), vec![c]]), trump) > Some(current_high_card)
                })
                .collect();
            if !higher_cards.is_empty() {
                return higher_cards;
            }
            let same_color_cards: Vec<&Card> = cards
                .clone()
                .into_iter()
                .filter(|c: &&Card| c.suit == trick_suit)
                .collect();
            if !same_color_cards.is_empty() {
                return same_color_cards;
            }
            cards
        }
        None => {
            //the trick has to be empty
            if !first_trick {
                cards
            } else {
                allowed_first(cards.clone())
            }
        }
    }
}

pub fn halves(cards: Vec<Card>) -> Vec<Suit> {
    let mut halves: Vec<Suit> = vec![];
    for suit in [Suit::Red, Suit::Bells, Suit::Acorns, Suit::Green] {
        if cards.contains(&Card {
            suit,
            value: Value::Ober,
        }) || cards.contains(&Card {
            suit,
            value: Value::King,
        }) {
            halves.push(suit);
        }
    }
    halves
}

pub fn pairs(cards: Vec<Card>) -> Vec<Suit> {
    let mut halves: Vec<Suit> = vec![];
    for suit in [Suit::Red, Suit::Bells, Suit::Acorns, Suit::Green] {
        if cards.contains(&Card {
            suit,
            value: Value::Ober,
        }) && cards.contains(&Card {
            suit,
            value: Value::King,
        }) {
            halves.push(suit);
        }
    }
    halves
}

pub fn get_all_cards() -> Vec<Card> {
    let mut cards: Vec<Card> = vec![];
    for suit in Suit::iter() {
        for value in Value::iter() {
            cards.push(Card { suit, value });
        }
    }
    cards
}

pub fn print_cards(cards: &[Card]) {
    let card_strs: Vec<String> = cards.iter().map(|c| format!("{}", c)).collect();
    let s = card_strs.join(", ");
    println!("{}", s)
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let suit = match self.suit {
            Suit::Red => "r",
            Suit::Bells => "s",
            Suit::Acorns => "e",
            Suit::Green => "g",
        };
        let value = match self.value {
            Value::Ace => "A",
            Value::Ten => "Z",
            Value::King => "K",
            Value::Ober => "O",
            Value::Unter => "U",
            Value::Nine => "9",
            Value::Eight => "8",
            Value::Seven => "7",
            Value::Six => "6",
        };
        write!(f, "{}-{}", suit, value)
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
        parse_card(String::from(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_higher_card() {
        let ra = Card {
            suit: Suit::Red,
            value: Value::Ace,
        };
        let sz = Card {
            suit: Suit::Bells,
            value: Value::Ten,
        };
        let su = Card {
            suit: Suit::Bells,
            value: Value::Ten,
        };
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
        let higher = vec![
            Card {
                suit: Suit::Bells,
                value: Value::Ace,
            },
            Card {
                suit: Suit::Red,
                value: Value::Six,
            },
            Card {
                suit: Suit::Red,
                value: Value::Seven,
            },
            Card {
                suit: Suit::Red,
                value: Value::Eight,
            },
            Card {
                suit: Suit::Red,
                value: Value::Nine,
            },
            Card {
                suit: Suit::Red,
                value: Value::Unter,
            },
            Card {
                suit: Suit::Red,
                value: Value::Ober,
            },
            Card {
                suit: Suit::Red,
                value: Value::King,
            },
            Card {
                suit: Suit::Red,
                value: Value::Ten,
            },
            Card {
                suit: Suit::Red,
                value: Value::Ace,
            },
        ];
        assert_eq!(higher_cards(&ra, None, None), vec![]);
        assert_eq!(
            higher_cards(&sz, None, None),
            vec![Card {
                suit: Suit::Bells,
                value: Value::Ace,
            }]
        );
        assert_eq!(higher_cards(&sz, Some(Suit::Red), None), higher)
    }

    #[test]
    fn test_high_card() {
        let ra = Card {
            suit: Suit::Red,
            value: Value::Ace,
        };
        let sz = Card {
            suit: Suit::Bells,
            value: Value::Ten,
        };
        assert_eq!(high_card(vec![], None), None);
        assert_eq!(high_card(vec![&ra, &sz], None), Some(&ra));
        assert_eq!(high_card(vec![&ra, &sz], Some(Suit::Bells)), Some(&sz));
    }

    #[test]
    fn test_allowed_first() {
        let mut cards: Vec<&Card> = vec![
            &Card {
                suit: Suit::Red,
                value: Value::Seven,
            },
            &Card {
                suit: Suit::Bells,
                value: Value::Seven,
            },
        ];
        assert_eq!(allowed_first(cards.clone()), cards);
        cards.push(&Card {
            suit: Suit::Green,
            value: Value::Unter,
        });
        assert_eq!(
            allowed_first(cards.clone()),
            vec![&Card {
                suit: Suit::Green,
                value: Value::Unter,
            }]
        );
        cards.push(&Card {
            suit: Suit::Red,
            value: Value::Ace,
        });
        cards.push(&Card {
            suit: Suit::Bells,
            value: Value::Ace,
        });
        assert_eq!(
            allowed_first(cards.clone()),
            vec![
                &Card {
                    suit: Suit::Red,
                    value: Value::Ace,
                },
                &Card {
                    suit: Suit::Bells,
                    value: Value::Ace,
                },
            ]
        );
    }

    #[test]
    fn test_allowed_cards() {
        let mut cards: Vec<&Card> = vec![];
        let trick: Vec<&Card> = vec![];
        //everything empty
        assert_eq!(
            allowed_cards(trick, cards.clone(), None, false),
            vec![] as Vec<&Card>
        );
        //first trick, ace wasn't played
        let trick = vec![&Card {
            suit: Suit::Green,
            value: Value::Unter,
        }];
        cards.push(&Card {
            suit: Suit::Red,
            value: Value::Unter,
        });
        cards.push(&Card {
            suit: Suit::Green,
            value: Value::Nine,
        });
        cards.push(&Card {
            suit: Suit::Green,
            value: Value::Ten,
        });
        cards.push(&Card {
            suit: Suit::Green,
            value: Value::Ace,
        });
        assert_eq!(
            allowed_cards(trick.clone(), cards.clone(), None, true),
            vec![&Card {
                suit: Suit::Green,
                value: Value::Ace,
            },]
        );
        //same color and higher
        assert_eq!(
            allowed_cards(trick, cards.clone(), None, false),
            vec![
                &Card {
                    suit: Suit::Green,
                    value: Value::Ten,
                },
                &Card {
                    suit: Suit::Green,
                    value: Value::Ace,
                },
            ]
        );
        //trump
        let trick = vec![&Card {
            suit: Suit::Bells,
            value: Value::Ober,
        }];
        assert_eq!(
            allowed_cards(trick, cards.clone(), Some(Suit::Red), false),
            vec![&Card {
                suit: Suit::Red,
                value: Value::Unter,
            },]
        );
        //same color but no higher if already trump
        let trick = vec![
            &Card {
                suit: Suit::Green,
                value: Value::Ober,
            },
            &Card {
                suit: Suit::Red,
                value: Value::Ober,
            },
        ];
        assert_eq!(
            allowed_cards(trick, cards.clone(), Some(Suit::Red), false),
            vec![
                &Card {
                    suit: Suit::Green,
                    value: Value::Nine,
                },
                &Card {
                    suit: Suit::Green,
                    value: Value::Ten,
                },
                &Card {
                    suit: Suit::Green,
                    value: Value::Ace,
                },
            ]
        );
    }
}
