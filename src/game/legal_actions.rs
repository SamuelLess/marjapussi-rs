use itertools::Itertools;

use crate::game::cards::{allowed_cards, Card, Suit};
use crate::game::gameevent::{ActionType, AnswerType, GameAction, QuestionType};
use crate::game::gamestate::GamePhase;
use crate::game::player::{Player, PlayerTrumpPossibilities};
use crate::game::points::Points;
use crate::game::{cards, Game};

impl GamePhase {
    pub fn legal_actions(&self, game: &Game) -> Vec<GameAction> {
        match self {
            GamePhase::WaitingForStart => {
                let mut start: Vec<GameAction> = vec![];
                for player in &game.state.players {
                    let mut has_started = false;
                    for possibly_same_player in game.state.players_started.clone() {
                        if player.place_at_table == possibly_same_player {
                            has_started = true;
                        }
                    }
                    if !has_started {
                        start.push(GameAction {
                            action_type: ActionType::Start,
                            player: player.place_at_table.clone(),
                        })
                    }
                }
                start
            }
            GamePhase::Bidding => legal_bidding(game),
            GamePhase::PassingForth => legal_passing(game),
            GamePhase::PassingBack => legal_passing(game),
            GamePhase::Raising => {
                let mut allowed = legal_bidding(game);
                allowed.reverse();
                allowed.pop();
                allowed.extend(legal_cards(game));
                allowed
            }
            GamePhase::StartTrick => {
                let mut allowed = vec![];
                allowed.extend(legal_question(game));
                allowed.extend(legal_cards(game));
                allowed
            }
            GamePhase::Trick => legal_cards(game),
            GamePhase::AnsweringPair => legal_answer(game),
            GamePhase::AnsweringHalf(_suit) => legal_answer(game),
            GamePhase::Ended => {
                vec![]
            }
            GamePhase::PendingUndo(_previous_phase) => {
                let mut undo: Vec<GameAction> = vec![];
                let next_player = game.last_state.clone().unwrap().player_at_turn.next();
                let next_player_partner = next_player.partner();
                let players_to_ask = vec![next_player, next_player_partner];
                for player in players_to_ask {
                    let mut has_accepted = false;
                    for possibly_accepted in &game.state.players_accept_undo {
                        if player == *possibly_accepted {
                            has_accepted = true;
                        }
                    }
                    if !has_accepted {
                        undo.push(GameAction {
                            action_type: ActionType::UndoAccept,
                            player: player.clone(),
                        });
                        undo.push(GameAction {
                            action_type: ActionType::UndoDecline,
                            player,
                        });
                    }
                }
                undo
            }
        }
    }
}

pub fn legal_bidding(game: &Game) -> Vec<GameAction> {
    let start_value = game.state.value + Points(5);
    let mut allowed_actions = vec![GameAction {
        action_type: ActionType::StopBidding,
        player: game.state.player_at_turn.clone(),
    }];
    for allowed_value in (start_value.0..=420).step_by(5) {
        allowed_actions.push(GameAction {
            action_type: ActionType::NewBid(allowed_value),
            player: game.state.player_at_turn.clone(),
        })
    }
    allowed_actions
}

pub fn legal_passing(game: &Game) -> Vec<GameAction> {
    let mut actions: Vec<GameAction> = vec![];
    for comb in game
        .state
        .player_at_turn()
        .cards
        .clone()
        .into_iter()
        .combinations(4)
    {
        let sorted: Vec<Card> = comb.clone().into_iter().sorted().rev().collect();
        actions.push(GameAction {
            action_type: ActionType::Pass(sorted),
            player: game.state.player_at_turn.clone(),
        })
    }
    actions
}

pub fn legal_cards(game: &Game) -> Vec<GameAction> {
    let mut actions: Vec<GameAction> = vec![];

    let cards = game.state.player_at_turn().cards.clone();
    let mut players_cards: Vec<&Card> = vec![];
    for c in &cards {
        players_cards.push(c);
    }
    let mut trick = game.state.current_trick.clone();
    if trick.len() == 4 {
        trick = vec![];
    }
    let trump = game.state.trump;
    let first_trick = players_cards.len() == 9;

    for card in allowed_cards(trick.iter().collect(), players_cards, trump, first_trick) {
        actions.push(GameAction {
            action_type: ActionType::CardPlayed(card.clone()),
            player: game.state.player_at_turn.clone(),
        })
    }

    actions
}

pub fn legal_question(game: &Game) -> Vec<GameAction> {
    let player: &Player = game.state.player_at_turn();
    let mut actions: Vec<GameAction> = vec![];
    let cards = player.cards.clone();
    let trump = player.trump.clone();
    let mut own_actions = vec![];
    for suit in cards::pairs(cards) {
        if game.state.trump_called.contains(&suit) {
            continue;
        }
        own_actions.push(GameAction {
            action_type: ActionType::AnnounceTrump(suit),
            player: player.place_at_table.clone(),
        })
    }
    let yours_actions = vec![GameAction {
        action_type: ActionType::Question(QuestionType::Yours),
        player: player.place_at_table.clone(),
    }];
    let mut ours_actions = vec![];
    for suit in [Suit::Red, Suit::Bells, Suit::Acorns, Suit::Green] {
        ours_actions.push(GameAction {
            action_type: ActionType::Question(QuestionType::YourHalf(suit)),
            player: player.place_at_table.clone(),
        })
    }
    match trump {
        PlayerTrumpPossibilities::Own => {
            actions.extend(own_actions);
            actions.extend(yours_actions);
            actions.extend(ours_actions);
        }
        PlayerTrumpPossibilities::Yours => {
            actions.extend(yours_actions);
            actions.extend(ours_actions);
        }
        PlayerTrumpPossibilities::Ours => {
            actions.extend(ours_actions);
        }
    }
    actions
}

pub fn legal_answer(game: &Game) -> Vec<GameAction> {
    let last_event = game.all_events.last().unwrap();
    let cards = game.state.player_at_turn().cards.clone();
    let mut actions: Vec<GameAction> = vec![];
    match last_event.last_action.action_type {
        ActionType::Question(QuestionType::Yours) => {
            for suit in cards::pairs(cards) {
                //don't allow double calling
                if game.state.trump_called.contains(&suit) {
                    continue;
                }
                actions.push(GameAction {
                    action_type: ActionType::Answer(AnswerType::YesPair(suit)),
                    player: game.state.player_at_turn.clone(),
                });
            }
            if actions.is_empty() {
                actions.push(GameAction {
                    action_type: ActionType::Answer(AnswerType::NoPair),
                    player: game.state.player_at_turn.clone(),
                })
            }
        }
        ActionType::Question(QuestionType::YourHalf(suit)) => {
            if cards::halves(cards).contains(&suit) {
                actions.push(GameAction {
                    action_type: ActionType::Answer(AnswerType::YesHalf(suit)),
                    player: game.state.player_at_turn.clone(),
                });
            } else {
                actions.push(GameAction {
                    action_type: ActionType::Answer(AnswerType::NoHalf(suit)),
                    player: game.state.player_at_turn.clone(),
                })
            }
        }
        _ => {
            println!("{:?}", game);
            panic!("Trying to find answers without question asked!")
        }
    }
    actions
}
