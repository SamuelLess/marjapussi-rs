use crate::game::cards::{high_card, Card};
use crate::game::gameevent::{ActionType, AnswerType, GameAction, GameCallback, QuestionType};
use crate::game::gameinfo::GameMetaInfo;
use crate::game::gamestate::{FinishedTrick, GamePhase, GameState};
use crate::game::points::{points_trick, Points};
use crate::game::{cards, current_time_string, Game};

impl ActionType {
    pub fn apply_action(
        self,
        action: &GameAction,
        game: &Game,
    ) -> (
        GameMetaInfo,
        GameState,
        Option<GameCallback>,
        Option<GameState>,
    ) {
        let mut next_game_meta = game.info.clone();
        let mut next_game_state = game.state.clone();
        let mut this_callback: Option<GameCallback> = None;
        let mut last_state: Option<GameState> = None;
        match self {
            ActionType::Start => {
                let mut has_started = false;
                for player in game.state.players_started.clone() {
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

                let mut next_player = next_game_state.player_at_turn.next();

                while !next_game_state.player_at_place(next_player.clone()).bidding {
                    next_player = next_player.next();
                }
                if next_player == next_game_state.player_at_turn {
                    //same player can't bid against himself
                    next_game_state.phase = GamePhase::PassingForth;
                    next_game_state.player_at_turn = next_game_state.player_at_turn.partner();
                    break 'newbid;
                } else {
                    // continue bidding
                    next_game_state.player_at_turn = next_player.clone();
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
                    // nobody takes game
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
                last_state = game.last_state.clone();
                let mut has_already_accepted = false;
                for player in game.state.players_accept_undo.clone() {
                    if player == action.player.clone() {
                        has_already_accepted = true;
                    }
                }
                if !has_already_accepted {
                    next_game_state
                        .players_accept_undo
                        .push(action.player.clone());
                }
                if next_game_state.players_accept_undo.len() == 2 && game.last_state.is_some() {
                    last_state = None;
                    next_game_state = game.last_state.clone().unwrap();
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
                last_state.clone_from(&game.last_state);
                //last_state is always Some() because otherwise action not legal
                match game.last_state.clone() {
                    None => {}
                    Some(_last) => {
                        next_game_state.phase =
                            GamePhase::PendingUndo(Box::new(next_game_state.phase.clone()));
                    }
                }
            }
        };
        (next_game_meta, next_game_state, this_callback, last_state)
    }
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
        // save trick
        let cards_in_last_trick: [Card; 4] =
            next_game_state.current_trick.clone().try_into().unwrap();
        next_game_state.all_tricks.push(FinishedTrick {
            cards: cards_in_last_trick.clone(),
            winner: next_game_state.player_at_turn.clone(),
            points: points_trick(cards_in_last_trick.try_into().unwrap()),
        });
    }
}
