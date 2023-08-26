use self::{cards::Card, gameinfo::GameMetaInfo, gamestate::GameState};
use crate::game::errors::GameError;
use crate::game::gameevent::{
    ActionType, AnswerType, GameAction, GameCallback, GameEvent, QuestionType,
};
use crate::game::gamestate::{FinishedTrick, GamePhase};

use crate::game::cards::{allowed_cards, high_card, Suit};
use crate::game::player::{create_players, PlaceAtTable, Player, PlayerTrumpPossibilities};
use crate::game::points::{points_trick, Points};
use chrono::offset::Local;
use chrono::DateTime;
use itertools::Itertools;
use std::time::SystemTime;

pub mod cards;
pub mod errors;
pub mod gameevent;
pub mod gameinfo;
pub mod gamestate;
pub mod parse;
pub mod player;
pub mod points;

/**
 * Wrapper for Game and all of its details.
 */
#[derive(Debug, Clone)]
pub struct Game {
    pub info: GameMetaInfo,
    pub state: GameState,
    pub legal_actions: Vec<GameAction>,
    pub last_state: Option<GameState>,
    pub all_events: Vec<GameEvent>,
}

impl Game {
    pub fn new(name: String, player_names: [String; 4], cards: Option<[Vec<Card>; 4]>) -> Self {
        let players = create_players(player_names.clone(), cards);

        let mut game = Game {
            info: GameMetaInfo {
                name,
                player_names,
                create_time: current_time_string(),
                start_time: None,
                player_start_cards: [
                    players[0].cards.clone(),
                    players[1].cards.clone(),
                    players[2].cards.clone(),
                    players[3].cards.clone(),
                ],
                end_time: None,
            },
            state: GameState {
                started: false,
                players_started: vec![],
                players_accept_undo: vec![],
                phase: GamePhase::WaitingForStart,
                trump: None,
                trump_called: vec![],
                player_at_turn: PlaceAtTable(0),
                value: Points(115),
                bidding_players: 4,
                bidding_history: vec![],
                players,
                all_tricks: vec![],
                current_trick: vec![],
            },
            legal_actions: vec![],
            last_state: None,
            all_events: vec![],
        };
        game.legal_actions = game.legal_actions();
        game
    }

    /**
     * Creates list with all legal actions in the current state of the game.
     */
    pub fn legal_actions(&self) -> Vec<GameAction> {
        let mut legal = match self.state.phase.clone() {
            GamePhase::WaitingForStart => {
                let mut start: Vec<GameAction> = vec![];
                for player in &self.state.players {
                    let mut has_started = false;
                    for possibly_same_player in self.state.players_started.clone() {
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
            GamePhase::Bidding => legal_bidding(self),
            GamePhase::PassingForth => legal_passing(self),
            GamePhase::PassingBack => legal_passing(self),
            GamePhase::Raising => {
                let mut allowed = legal_bidding(self);
                allowed.reverse();
                allowed.pop();
                allowed.extend(legal_cards(self));
                allowed
            }
            GamePhase::StartTrick => {
                let mut allowed = vec![];
                allowed.extend(legal_question(self));
                allowed.extend(legal_cards(self));
                allowed
            }
            GamePhase::Trick => legal_cards(self),
            GamePhase::AnsweringPair => legal_answer(self),
            GamePhase::AnsweringHalf(_suit) => legal_answer(self),
            GamePhase::Ended => {
                vec![]
            }
            GamePhase::PendingUndo(_previous_phase) => {
                let mut undo: Vec<GameAction> = vec![];
                let next_player = self.last_state.clone().unwrap().player_at_turn.next();
                let next_player_partner = next_player.partner();
                let players_to_ask = vec![next_player, next_player_partner];
                for player in players_to_ask {
                    let mut has_accepted = false;
                    for possibly_accepted in &self.state.players_accept_undo {
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
        };
        //disallow undo in some phases
        match self.state.phase.clone() {
            GamePhase::WaitingForStart => {
                return legal.clone();
            }
            GamePhase::PendingUndo(_prev) => {
                return legal.clone();
            }
            GamePhase::Ended => {
                return legal.clone();
            }
            GamePhase::Bidding => {
                if self.state.value == Points(115) {
                    return legal.clone();
                }
            }
            GamePhase::PassingBack => {
                return legal.clone();
            }
            GamePhase::Raising => {
                return legal.clone();
            }
            _ => {}
        }
        if self.last_state.is_some() {
            legal.push(GameAction {
                action_type: ActionType::UndoRequest,
                player: self.last_state.clone().unwrap().player_at_turn.clone(),
            });
        }
        legal
    }

    /**
     * Tries creating a new Game with state after applying a given action.
     * Can fail and does not mutate the existing Game.
     */
    pub fn apply_action(&self, action: GameAction) -> Result<Game, GameError> {
        if !self.legal_actions.contains(&action) {
            return Err(GameError::IllegalAction);
        }
        let mut next_all_events = self.all_events.clone();
        let mut next_game_meta = self.info.clone();
        let mut next_game_state = self.state.clone();
        let mut this_callback: Option<GameCallback> = None;
        let mut last_state: Option<GameState> = None;
        match action.action_type.clone() {
            ActionType::Start => {
                let mut has_started = false;
                for player in self.state.players_started.clone() {
                    if player == action.player.clone() {
                        has_started = true;
                    }
                }
                if !has_started {
                    next_game_state.players_started.push(action.player.clone());
                }
                if next_game_state.players_started.len() == 4 {
                    next_game_state.started = true;
                    next_game_meta.start_time = Some(current_time_string());
                    next_game_state.phase = GamePhase::Bidding;
                }
            }
            ActionType::NewBid(value) => 'newbid: {
                last_state = Some(next_game_state.clone());
                next_game_state
                    .bidding_history
                    .push((action.action_type.clone(), action.player.clone()));

                next_game_state.value = Points(value);
                if next_game_state.phase == GamePhase::Raising {
                    next_game_state.phase = GamePhase::Trick;
                    break 'newbid;
                }

                let mut next_player: PlaceAtTable = next_game_state.player_at_turn.next();

                //TODO: redo this when you actually understand what you are doing
                while !next_game_state.player_at_place(next_player.clone()).bidding {
                    next_player = next_player.next();
                }
                if !(next_game_state.phase == GamePhase::Raising) {
                    //still bidding
                    next_game_state.player_at_turn = next_player.clone();
                } else {
                    //game value was raised
                    next_game_state.phase = GamePhase::Trick;
                }
            }
            ActionType::StopBidding => {
                last_state = Some(next_game_state.clone());
                next_game_state
                    .bidding_history
                    .push((action.action_type.clone(), action.player.clone()));
                next_game_state.player_at_turn_mut().bidding = false;
                next_game_state.bidding_players -= 1;
                let mut next_player = next_game_state.player_at_turn.next();
                if next_game_state.bidding_players >= 1 {
                    while !next_game_state.player_at_place(next_player.clone()).bidding {
                        next_player = next_player.next();
                    }
                }
                next_game_state.player_at_turn = next_player.clone();

                //bidding ends
                if next_game_state.bidding_players == 1 && next_game_state.value > Points(115) {
                    next_game_state.phase = GamePhase::PassingForth;

                    for player in &next_game_state.players {
                        if player.bidding {
                            next_game_state.player_at_turn = player.place_at_table.partner();
                        }
                    }
                }
                if next_game_state.bidding_players == 0 {
                    //nobody takes game
                    next_game_state.phase = GamePhase::Trick;
                }
            }
            ActionType::Pass(cards) => {
                {
                    let giver = next_game_state.player_at_place_mut(action.player.clone());
                    giver.cards.retain(|x| !cards.contains(x));
                }
                {
                    let receiver = next_game_state.player_at_place_mut(action.player.partner());
                    receiver.cards.extend(cards);
                }

                if next_game_state.phase == GamePhase::PassingBack {
                    next_game_state.phase = GamePhase::Raising;
                } else {
                    next_game_state.phase = GamePhase::PassingBack;
                    next_game_state.player_at_turn = action.player.partner();
                };
            }
            ActionType::CardPlayed(card) => {
                last_state = Some(next_game_state.clone());
                act_card(card.clone(), &mut next_game_state);
                next_game_state
                    .player_at_place_mut(action.player.clone())
                    .play_card(card);
                if next_game_state.player_at_turn().cards.is_empty() {
                    next_game_state.phase = GamePhase::Ended;
                    next_game_meta.end_time = Some(current_time_string());
                }
            }
            ActionType::AnnounceTrump(suit) => {
                //can only happen once per suit
                this_callback = Some(GameCallback::NewTrump(suit));
                next_game_state.trump_called.push(suit);
                next_game_state.trump = Some(suit);
                next_game_state.phase = GamePhase::Trick;
            }
            ActionType::Question(QuestionType::Yours) => {
                next_game_state.phase = GamePhase::AnsweringPair;
                next_game_state.player_at_turn = next_game_state.player_at_turn.partner();
            }
            ActionType::Question(QuestionType::YourHalf(suit)) => {
                //can happen multiple times per suit
                next_game_state.phase = GamePhase::AnsweringHalf(suit);
                next_game_state.player_at_turn = next_game_state.player_at_turn.partner();
            }
            ActionType::Answer(AnswerType::NoPair) => {
                next_game_state.phase = GamePhase::Trick;
                next_game_state.player_at_turn = next_game_state.player_at_turn.partner();
            }
            ActionType::Answer(AnswerType::YesPair(suit)) => {
                //can happen only once per suit
                this_callback = Some(GameCallback::NewTrump(suit));
                next_game_state.trump_called.push(suit);
                next_game_state.trump = Some(suit);
                next_game_state.phase = GamePhase::Trick;
                next_game_state.player_at_turn = next_game_state.partner().place_at_table.clone();
            }
            ActionType::Answer(AnswerType::NoHalf(suit)) => {
                this_callback = Some(GameCallback::NoHalf(suit));
                next_game_state.phase = GamePhase::Trick;
                next_game_state.player_at_turn = next_game_state.partner().place_at_table.clone();
            }
            ActionType::Answer(AnswerType::YesHalf(suit)) => {
                //can happen multiple times per suit
                let partner_cards = next_game_state.partner().cards.clone();

                if cards::halves(partner_cards).contains(&suit) {
                    if next_game_state.trump_called.contains(&suit) {
                        //if called for second time, can't be excluded
                        this_callback = Some(GameCallback::StillTrump(suit));
                    } else {
                        this_callback = Some(GameCallback::NewTrump(suit));
                        next_game_state.trump_called.push(suit);
                    }
                    next_game_state.trump = Some(suit);
                    next_game_state.phase = GamePhase::Trick;
                } else {
                    this_callback = Some(GameCallback::OnlyHalf(suit));
                }
                next_game_state.player_at_turn = next_game_state.partner().place_at_table.clone();
                next_game_state.phase = GamePhase::Trick;
            }
            ActionType::UndoAccept => {
                last_state = self.last_state.clone();
                let mut has_already_accepted = false;
                for player in self.state.players_accept_undo.clone() {
                    if player == action.player.clone() {
                        has_already_accepted = true;
                    }
                }
                if !has_already_accepted {
                    next_game_state
                        .players_accept_undo
                        .push(action.player.clone());
                }
                if next_game_state.players_accept_undo.len() == 2 && self.last_state.is_some() {
                    last_state = None;
                    next_game_state = self.last_state.clone().unwrap();
                    next_game_state.players_accept_undo = vec![];
                }
            }
            ActionType::UndoDecline => {
                if let GamePhase::PendingUndo(previous_phase) = next_game_state.phase {
                    next_game_state.phase = *previous_phase;
                    next_game_state.players_accept_undo = vec![];
                }
            }
            ActionType::UndoRequest => {
                last_state = self.last_state.clone();
                //last_state is always Some() because otherwise action not legal
                match self.last_state.clone() {
                    None => return Err(GameError::CannotUndo),
                    Some(_last) => {
                        next_game_state.phase =
                            GamePhase::PendingUndo(Box::new(next_game_state.phase.clone()));
                    }
                }
            }
        }

        // next game Object
        let this_event = GameEvent {
            last_action: action,
            callback: this_callback,
            next_player_at_turn: next_game_state.player_at_turn.clone(),
            time: current_time_string(),
        };
        next_all_events.push(this_event);

        /*if last_state.is_none() {
            last_state = Some(self.state.clone());
        }*/
        let mut next_game = Game {
            info: next_game_meta,
            state: next_game_state,
            legal_actions: vec![],
            last_state,
            all_events: next_all_events,
        };
        next_game.legal_actions = next_game.legal_actions();
        Ok(next_game)
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
        actions.push(GameAction {
            action_type: ActionType::Pass(comb),
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
    let mut btrick: Vec<&Card> = vec![];
    for c in &trick {
        btrick.push(c);
    }
    let trump = game.state.trump;
    let first_trick = players_cards.len() == 9;

    for card in allowed_cards(btrick, players_cards, trump, first_trick) {
        actions.push(GameAction {
            action_type: ActionType::CardPlayed(card.clone()),
            player: game.state.player_at_turn.clone(),
        })
    }

    actions
}

pub fn act_card(card: Card, next_game_state: &mut GameState) {
    if next_game_state.current_trick.len() >= 4 {
        next_game_state.current_trick = vec![card];
    } else {
        next_game_state.current_trick.push(card);
    }
    next_game_state.phase = GamePhase::Trick;
    next_game_state.player_at_turn = next_game_state.player_at_turn.next();
    if next_game_state.current_trick.len() == 4 {
        //determine next player
        let mut trick: Vec<&Card> = vec![];
        for c in &next_game_state.current_trick {
            trick.push(c);
        }
        let high_card = high_card(trick.clone(), next_game_state.trump).unwrap();
        for card in trick {
            if card == high_card {
                break;
            }
            next_game_state.player_at_turn = next_game_state.player_at_turn.next();
        }
        next_game_state.phase = GamePhase::StartTrick;
        //save trick
        let cards_in_last_trick: [Card; 4] =
            next_game_state.current_trick.clone().try_into().unwrap();
        next_game_state.all_tricks.push(FinishedTrick {
            cards: cards_in_last_trick.clone(),
            winner: next_game_state.player_at_turn.clone(),
            points: points_trick(cards_in_last_trick.try_into().unwrap()),
        });
    }
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

pub fn current_time_string() -> String {
    let system_time = SystemTime::now();
    let datetime: DateTime<Local> = system_time.into();
    format!("{}", datetime.format("%Y-%m-%d %T"))
}
