#![allow(unused_imports)]
extern crate test;

use crate::game::cards::{
    allowed_cards, allowed_first, high_card, higher_cards, is_higher_card, Card, Suit, Value,
};
#[cfg(test)]
// Note this useful idiom: importing names from outer (for mod tests) scope.
use test::Bencher;

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
            value: Value::Ace
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
            value: Value::Unter
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
                value: Value::Ace
            },
            &Card {
                suit: Suit::Bells,
                value: Value::Ace
            }
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
            value: Value::Ace
        },]
    );
    //same color and higher
    assert_eq!(
        allowed_cards(trick, cards.clone(), None, false),
        vec![
            &Card {
                suit: Suit::Green,
                value: Value::Ten
            },
            &Card {
                suit: Suit::Green,
                value: Value::Ace
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
            value: Value::Unter
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
                value: Value::Nine
            },
            &Card {
                suit: Suit::Green,
                value: Value::Ten
            },
            &Card {
                suit: Suit::Green,
                value: Value::Ace
            },
        ]
    );
}

#[bench]
fn bench_allowed_cards(b: &mut Bencher) {
    b.iter(|| {
        let mut cards: Vec<&Card> = vec![];
        let trick: Vec<&Card> = vec![];
        //everything empty
        allowed_cards(trick, cards.clone(), None, false);
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
        allowed_cards(trick.clone(), cards.clone(), None, true);
        //same color and higher
        //allowed_cards(trick, cards.clone(), None, false);
        //trump
        let trick = vec![&Card {
            suit: Suit::Bells,
            value: Value::Ober,
        }];
        allowed_cards(trick, cards.clone(), Some(Suit::Red), false);
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
        allowed_cards(trick, cards.clone(), Some(Suit::Red), false);
    });
}
