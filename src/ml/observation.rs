use crate::game::cards::{get_all_cards, Card, Suit, Value};
use crate::game::gameevent::{ActionType, AnswerType, GameCallback, QuestionType};
use crate::game::player::PlaceAtTable;
use crate::game::Game;
use crate::ml::inference::{
    apply_hidden_set_constraints, hidden_set_constraints_enabled, HalfConstraint,
};

// ─── Card index helpers ─────────────────────────────────────────────────────

/// Canonical card ordering: all 36 cards as a flat index 0..35.
/// Order: Green 6..A, Acorns 6..A, Bells 6..A, Red 6..A (matches Suit/Value enum order).
pub fn card_index(card: &Card) -> usize {
    let suit_idx = match card.suit {
        Suit::Green => 0,
        Suit::Acorns => 1,
        Suit::Bells => 2,
        Suit::Red => 3,
    };
    let value_idx = match card.value {
        Value::Six => 0,
        Value::Seven => 1,
        Value::Eight => 2,
        Value::Nine => 3,
        Value::Unter => 4,
        Value::Ober => 5,
        Value::King => 6,
        Value::Ten => 7,
        Value::Ace => 8,
    };
    suit_idx * 9 + value_idx
}

pub fn card_from_index(idx: usize) -> Card {
    get_all_cards()[idx].clone()
}

pub fn card_point_value(card: &Card) -> f32 {
    match card.value {
        Value::Ace => 11.0,
        Value::Ten => 10.0,
        Value::King => 4.0,
        Value::Ober => 3.0,
        Value::Unter => 2.0,
        _ => 0.0,
    }
}

// ─── Observation ─────────────────────────────────────────────────────────────

/// Complete observation from one player's perspective.
/// All bitmasks are [bool; 36] indexed by card_index().
#[derive(Debug, Clone)]
pub struct Observation {
    /// Cards already played (in finished tricks or the completed current trick).
    pub played_bitmask: [bool; 36],
    /// My current hand.
    pub my_hand_bitmask: [bool; 36],
    /// For each of the 3 opponents (relative: left=0, partner=1, right=2):
    /// cards they CAN possibly hold (eliminated by suit-following deductions).
    pub possible_bitmasks: [[bool; 36]; 3],
    /// Cards confirmed to be in an opponent's hand from Q&A answers.
    pub confirmed_bitmasks: [[bool; 36]; 3],
    /// Current hand as list of card indices (convenience for embeddings).
    pub my_hand_indices: Vec<usize>,
    /// All hands (0=VH, 1=MH, etc). Always populated for debug purposes.
    pub all_hands: Vec<Vec<usize>>,
    /// Cards in the current trick being played (in play order).
    pub current_trick_indices: Vec<usize>,
    /// Who played each card in the current trick (relative seat 0..3).
    pub current_trick_players: Vec<usize>,
    /// Trump suit index (0=Green,1=Acorns,2=Bells,3=Red) or None.
    pub trump: Option<usize>,
    /// Which suits have ever been announced as trump (4 bits).
    pub trump_announced: [bool; 4],
    /// My role: 0=VH, 1=MH, 2=LH, 3=RH, 4=no-one-played
    pub my_role: usize,
    /// Trick number (1-indexed, 1..=9).
    pub trick_number: usize,
    /// Position in current trick (0=leading, 1,2,3=following).
    pub trick_position: usize,
    /// Cumulative trick-points for my team so far.
    pub points_my_team: i32,
    /// Cumulative trick-points for opponent team so far.
    pub points_opp_team: i32,
    /// Cards remaining for each player (relative: me=0, left=1, partner=2, right=3).
    pub cards_remaining: [usize; 4],
    /// The relative seat of the player who currently has the turn
    pub active_player: usize,
    /// Player trump possibilities: 0=Own, 1=Yours, 2=Ours (from PlayerTrumpPossibilities).
    pub trump_possibilities: usize,
    /// Whether to last trick bonus is still available (last trick not yet started).
    pub last_trick_bonus_live: bool,
    /// Serialized event history as token ids (Stream B input).
    pub event_tokens: Vec<u32>,
    /// Legal actions as (action_type_id, card_idx_or_0, suit_idx_or_0).
    pub legal_actions: Vec<LegalActionObs>,
    /// Debug-friendly phase string (for UI/runtime controls).
    pub phase: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LegalActionObs {
    /// Matches ActionToken enum values defined in proto.rs
    pub action_token: u32,
    pub card_idx: Option<usize>,
    pub suit_idx: Option<usize>,
    pub bid_value: Option<i32>,
    pub pass_cards: Option<Vec<usize>>,
    /// Index into the game's legal_actions list (used for stepping).
    pub action_list_idx: usize,
}

// ─── Token vocabulary ─────────────────────────────────────────────────────────

/// Token ids for the event history sequence (Stream B).
pub mod tokens {
    // Special tokens
    pub const PAD: u32 = 0;
    pub const START_GAME: u32 = 1;
    pub const SEP: u32 = 2;

    // Trick start markers [10..18] for tricks 1..9
    pub const START_TRICK_BASE: u32 = 10; // START_TRICK_N = 10 + (N-1)

    // Player tokens [20..23]
    pub const PLAYER_BASE: u32 = 20; // PLAYER_N = 20 + N (relative seat)

    // Role tokens [30..34]
    pub const ROLE_VH: u32 = 30;
    pub const ROLE_MH: u32 = 31;
    pub const ROLE_LH: u32 = 32;
    pub const ROLE_RH: u32 = 33;
    pub const ROLE_NONE: u32 = 34;

    // Action tokens [40..]
    pub const ACT_PLAY: u32 = 40;
    pub const ACT_BID: u32 = 41;
    pub const ACT_PASS_STOP: u32 = 42;
    pub const ACT_PASS_CARDS: u32 = 43;
    pub const ACT_TRUMP: u32 = 44;
    pub const ACT_Q_PAIR: u32 = 45;
    pub const ACT_Q_HALF: u32 = 46;
    pub const ACT_A_YES_PAIR: u32 = 47;
    pub const ACT_A_NO_PAIR: u32 = 48;
    pub const ACT_A_YES_HALF: u32 = 49;
    pub const ACT_A_NO_HALF: u32 = 50;
    pub const ACT_TRICK_WON: u32 = 51;

    // Suit tokens [60..63]
    pub const SUIT_BASE: u32 = 60; // SUIT_N = 60 + suit_idx

    // Card tokens [70..105] (36 cards)
    pub const CARD_BASE: u32 = 70; // CARD_N = 70 + card_index(card)

    // Unknown card token (for hidden passes)
    pub const UNKNOWN_CARD: u32 = 110;

    // Bid value tokens [120..180] for bids 120..420 step 5
    pub const BID_BASE: u32 = 120; // BID_VAL = 120 + (value - 120)/5

    pub const VOCAB_SIZE: u32 = 250;
}

pub fn suit_index(suit: Suit) -> usize {
    match suit {
        Suit::Green => 0,
        Suit::Acorns => 1,
        Suit::Bells => 2,
        Suit::Red => 3,
    }
}

pub fn bid_token(value: i32) -> u32 {
    tokens::BID_BASE + ((value - 120) / 5) as u32
}

// ─── Main observation builder ─────────────────────────────────────────────────

pub fn build_observation(game: &Game, pov: PlaceAtTable) -> Observation {
    let state = &game.state;
    let all_cards = get_all_cards();

    // ── Played bitmask ────────────────────────────────────────────────────────
    let mut played_bitmask = [false; 36];
    for trick in &state.all_tricks {
        for card in &trick.cards {
            played_bitmask[card_index(card)] = true;
        }
    }
    // current trick cards are in-play, not yet "played" (won), but visible to all
    for card in &state.current_trick {
        played_bitmask[card_index(card)] = true;
    }

    // ── My hand ───────────────────────────────────────────────────────────────
    let my_player = state.player_at_place(pov.clone());
    let mut my_hand_bitmask = [false; 36];
    let mut my_hand_indices = vec![];
    for card in &my_player.cards {
        let idx = card_index(card);
        my_hand_bitmask[idx] = true;
        my_hand_indices.push(idx);
    }
    
    // ── All hands (for debugging) ──────────────────────────────────────────────
    let mut all_hands = vec![];
    for p_idx in 0..4 {
        let seat = state.player_at_place(PlaceAtTable((pov.0 + p_idx) % 4));
        let mut seat_indices = vec![];
        for card in &seat.cards {
            seat_indices.push(card_index(card));
        }
        all_hands.push(seat_indices);
    }

    // ── Opponent relative indices ────────────────────────────────────────────
    // relative: 0=left(next), 1=partner, 2=right(prev)
    let opp_places = [pov.next(), pov.partner(), pov.prev()];

    // ── Possible cards per opponent ───────────────────────────────────────────
    // Start: all unplayed, not in my hand
    let mut possible_bitmasks = [[false; 36]; 3];
    for (i, _opp_place) in opp_places.iter().enumerate() {
        for card in &all_cards {
            let idx = card_index(card);
            if !played_bitmask[idx] && !my_hand_bitmask[idx] {
                possible_bitmasks[i][idx] = true;
            }
        }
    }

    // Narrow by suit-following: for each finished trick, if an opponent played
    // off-suit when the lead suit was not trump, they have no more of lead suit.
    let mut opp_no_suit: [[bool; 4]; 3] = [[false; 4]; 3]; // opp × suit

    for trick in &state.all_tricks {
        if trick.cards.is_empty() {
            continue;
        }
        let lead_suit = trick.cards[0].suit;
        let lead_player = trick.winner.clone(); // winner = who leads NEXT, trick.cards[0] was played by winner of PREV trick
        // Reconstruct who played which card: the lead player played cards[0], then clockwise
        let _ = lead_suit;
            let _ = lead_player;

    }

    // Use all_events to track suit-following violations more accurately
    {
        let mut current_trick_lead_suit: Option<Suit> = None;
        let mut historical_trump: Option<Suit> = None;
        let mut cards_in_trick: Vec<(PlaceAtTable, Card)> = vec![];

        for event in &game.all_events {
            // Reconstruct the history of the trump suit as it was known AT THE TIME
            if let ActionType::AnnounceTrump(suit) = &event.last_action.action_type {
                historical_trump = Some(suit.clone());
            } else if let Some(GameCallback::NewTrump(suit)) = &event.callback {
                historical_trump = Some(suit.clone());
            }

            if let ActionType::CardPlayed(card) = &event.last_action.action_type {
                let player = &event.last_action.player;
                if cards_in_trick.is_empty() {
                    // This player is leading the trick
                    current_trick_lead_suit = Some(card.suit.clone());
                } else if let Some(lead_suit) = &current_trick_lead_suit {
                    // This player is following. Did they play off-suit?
                    if card.suit != *lead_suit {
                        // Check: is the lead suit trump?
                        let trump_is_lead = historical_trump.as_ref().map(|t| t == lead_suit).unwrap_or(false);
                        
                        // We only mark them as void in the lead suit if they:
                        // A: Played a different suit when Trumps weren't lead
                        // B: Played a non-trump when Trumps WERE lead
                        let played_trump = historical_trump.as_ref().map(|t| t == &card.suit).unwrap_or(false);

                        // If they didn't follow suit, they MUST be void in the lead suit.
                        // (Even if they ruffed with a Trump, it still proves they are void in the led suit).
                        if !trump_is_lead || !played_trump {
                            for (i, opp_place) in opp_places.iter().enumerate() {
                                if opp_place.0 == player.0 {
                                    opp_no_suit[i][suit_index(lead_suit.clone())] = true;
                                }
                            }
                        }
                    }
                }
                cards_in_trick.push((player.clone(), card.clone()));
                if cards_in_trick.len() == 4 {
                    cards_in_trick.clear();
                    current_trick_lead_suit = None;
                }
            }
        }
    }

    // Apply suit-following eliminations to possible bitmasks
    for (opp_i, no_suit) in opp_no_suit.iter().enumerate() {
        for (suit_i, &eliminated) in no_suit.iter().enumerate() {
            if eliminated {
                for card in &all_cards {
                    let sidx = suit_index(card.suit);
                    if sidx == suit_i {
                        let cidx = card_index(card);
                        possible_bitmasks[opp_i][cidx] = false;
                    }
                }
            }
        }
    }

    // ── Confirmed cards per opponent ──────────────────────────────────────────
    let mut confirmed_bitmasks = [[false; 36]; 3];
    let mut half_constraints = [[HalfConstraint::Unknown; 4]; 3];
    for event in &game.all_events {
        let answerer = &event.last_action.player;
        // Find which relative opponent slot this answerer corresponds to
        let opp_slot = opp_places.iter().position(|p| p.0 == answerer.0);
        if let Some(slot) = opp_slot {
            match &event.last_action.action_type {
                ActionType::Answer(AnswerType::YesPair(suit)) => {
                    // Confirmed: Ober AND King of this suit
                    let ober = Card { suit: *suit, value: Value::Ober };
                    let king = Card { suit: *suit, value: Value::King };
                    confirmed_bitmasks[slot][card_index(&ober)] = true;
                    confirmed_bitmasks[slot][card_index(&king)] = true;
                    half_constraints[slot][suit_index(*suit)] = HalfConstraint::RequireBoth;
                }
                ActionType::Answer(AnswerType::YesHalf(suit)) => {
                    // Phase 3 Fix: We do NOT natively "confirm" either the Ober or King here,
                    // because we mathematically do not know which one they hold.
                    // Doing so was feeding false certainty into the neurosymbolic stream.
                    // Instead, we leave both cards as `possible = true` and let the network deduce it.
                    let suit_i = suit_index(*suit);
                    match half_constraints[slot][suit_i] {
                        HalfConstraint::RequireBoth => {}
                        _ => {
                            half_constraints[slot][suit_i] = HalfConstraint::RequireAtLeastOne;
                        }
                    }
                }
                ActionType::Answer(AnswerType::NoHalf(_suit)) => {}
                ActionType::Answer(AnswerType::NoPair) => {}
                _ => {}
            }
        }
    }

    // ── Current trick ─────────────────────────────────────────────────────────
    let mut current_trick_indices = vec![];
    let mut current_trick_players = vec![];
    {
        // Find who started the current trick from events
        let cards_in_finished = state.all_tricks.len() * 4;
        let card_events: Vec<_> = game.all_events.iter()
            .filter(|e| matches!(e.last_action.action_type, ActionType::CardPlayed(_)))
            .collect();
        for (i, ev) in card_events.iter().enumerate().skip(cards_in_finished) {
            if let ActionType::CardPlayed(card) = &ev.last_action.action_type {
                let relative_seat = {
                    let abs_seat = ev.last_action.player.0;
                    let pov_seat = pov.0;
                    ((abs_seat + 4 - pov_seat) % 4) as usize
                };
                current_trick_indices.push(card_index(card));
                current_trick_players.push(relative_seat);
                let _ = i;
            }
        }
    }

    // ── Points ────────────────────────────────────────────────────────────────
    let my_party = pov.party();
    let partner_place = pov.partner();
    let mut points_my_team = 0i32;
    let mut points_opp_team = 0i32;
    
    // Add points from tricks
    for trick in &state.all_tricks {
        let winner_party = trick.winner.party();
        if winner_party.0 == my_party.0 {
            points_my_team += trick.points.0;
        } else {
            points_opp_team += trick.points.0;
        }
    }

    // Add points from announced pairs (trump calls)
    for event in &game.all_events {
        if let Some(crate::game::gameevent::GameCallback::NewTrump(suit)) = event.callback {
            let caller_party = event.last_action.player.party();
            let pair_points = crate::game::points::points_pair(suit).0;
            if caller_party.0 == my_party.0 {
                points_my_team += pair_points;
            } else {
                points_opp_team += pair_points;
            }
        }
    }

    // ── trump_announced ───────────────────────────────────────────────────────
    let mut trump_announced = [false; 4];
    for &suit in &state.trump_called {
        trump_announced[suit_index(suit)] = true;
    }

    // ── Role (VH=0,MH=1,LH=2,RH=3,None=4) ───────────────────────────────────
    // Determine playing party from bidding history
    let my_role = determine_role(game, &pov);

    // ── Cards remaining ───────────────────────────────────────────────────────
    let cards_remaining = [
        my_player.cards.len(),
        state.player_at_place(pov.next()).cards.len(),
        state.player_at_place(partner_place.clone()).cards.len(),
        state.player_at_place(pov.prev()).cards.len(),
    ];
    if hidden_set_constraints_enabled() {
        apply_hidden_set_constraints(
            &mut possible_bitmasks,
            &mut confirmed_bitmasks,
            &half_constraints,
        );
    }

    // ── Trick number & position ───────────────────────────────────────────────
    let trick_number = state.all_tricks.len() + 1;
    let trick_position = state.current_trick.len();

    // ── trump_possibilities ───────────────────────────────────────────────────
    let trump_possibilities = match my_player.trump {
        crate::game::player::PlayerTrumpPossibilities::Own => 0,
        crate::game::player::PlayerTrumpPossibilities::Yours => 1,
        crate::game::player::PlayerTrumpPossibilities::Ours => 2,
    };

    // ── Event token sequence (Stream B) ──────────────────────────────────────
    let event_tokens = build_event_tokens(game, &pov, my_role);

    // ── Legal actions ─────────────────────────────────────────────────────────
    let legal_actions = build_legal_actions(game, &pov);

    // ── Active Player ─────────────────────────────────────────────────────────
    let active_player = ((game.state.player_at_turn.0 + 4 - pov.0) as usize) % 4;

    Observation {
        played_bitmask,
        my_hand_bitmask,
        possible_bitmasks,
        confirmed_bitmasks,
        my_hand_indices,
        all_hands,
        current_trick_indices,
        current_trick_players,
        trump: state.trump.map(suit_index),
        trump_announced,
        my_role,
        active_player,
        trick_number,
        trick_position,
        points_my_team,
        points_opp_team,
        cards_remaining,
        trump_possibilities,
        last_trick_bonus_live: trick_number <= 9,
        event_tokens,
        legal_actions,
        phase: format!("{:?}", state.phase),
    }
}

fn determine_role(game: &Game, pov: &PlaceAtTable) -> usize {
    // Find who won the bid (last NewBid action before PassingForth)
    let mut winner: Option<PlaceAtTable> = None;
    for event in &game.all_events {
        if let ActionType::NewBid(_) = event.last_action.action_type {
            winner = Some(event.last_action.player.clone());
        }
    }
    match winner {
        None => 4, // no-one-played
        Some(vh_place) => {
            let mh_place = vh_place.partner();
            if pov.0 == vh_place.0 {
                0 // VH
            } else if pov.0 == mh_place.0 {
                1 // MH
            } else if pov.0 == vh_place.next().0 {
                2 // LH (left of VH)
            } else {
                3 // RH (right of VH)
            }
        }
    }
}

fn build_event_tokens(game: &Game, pov: &PlaceAtTable, my_role: usize) -> Vec<u32> {
    use tokens::*;
    let mut toks = vec![];

    // Role token first
    toks.push(START_GAME);
    let role_tok = match my_role {
        0 => ROLE_VH,
        1 => ROLE_MH,
        2 => ROLE_LH,
        3 => ROLE_RH,
        _ => ROLE_NONE,
    };
    toks.push(role_tok);

    let mut trick_idx = 0usize;
    let mut cards_in_trick = 0usize;

    for event in &game.all_events {
        let player_abs = event.last_action.player.0;
        let player_rel = ((player_abs + 4 - pov.0) % 4) as u32;
        let player_tok = PLAYER_BASE + player_rel;

        match &event.last_action.action_type {
            ActionType::NewBid(val) => {
                toks.push(player_tok);
                toks.push(ACT_BID);
                toks.push(bid_token(*val));
            }
            ActionType::StopBidding => {
                toks.push(player_tok);
                toks.push(ACT_PASS_STOP);
            }
            ActionType::Pass(cards) => {
                toks.push(player_tok);
                toks.push(ACT_PASS_CARDS);
                // Reveal cards if we are the passer or receiver (POV-relevant)
                let is_relevant = event.last_action.player.0 == pov.0
                    || event.last_action.player.partner().0 == pov.0;
                if is_relevant {
                    for card in cards {
                        toks.push(CARD_BASE + card_index(card) as u32);
                    }
                } else {
                    for _ in cards {
                        toks.push(UNKNOWN_CARD);
                    }
                }
            }
            ActionType::CardPlayed(card) => {
                if cards_in_trick == 0 {
                    toks.push(START_TRICK_BASE + trick_idx as u32);
                }
                toks.push(player_tok);
                toks.push(ACT_PLAY);
                toks.push(CARD_BASE + card_index(card) as u32);
                cards_in_trick += 1;
                if cards_in_trick == 4 {
                    // trick winner
                    let trick = &game.state.all_tricks.get(trick_idx);
                    if let Some(t) = trick {
                        let winner_rel = ((t.winner.0 + 4 - pov.0) % 4) as u32;
                        toks.push(ACT_TRICK_WON);
                        toks.push(PLAYER_BASE + winner_rel);
                    }
                    cards_in_trick = 0;
                    trick_idx += 1;
                }
            }
            ActionType::AnnounceTrump(suit) => {
                toks.push(player_tok);
                toks.push(ACT_TRUMP);
                toks.push(SUIT_BASE + suit_index(*suit) as u32);
            }
            ActionType::Question(QuestionType::Yours) => {
                toks.push(player_tok);
                toks.push(ACT_Q_PAIR);
            }
            ActionType::Question(QuestionType::YourHalf(suit)) => {
                toks.push(player_tok);
                toks.push(ACT_Q_HALF);
                toks.push(SUIT_BASE + suit_index(*suit) as u32);
            }
            ActionType::Answer(AnswerType::YesPair(suit)) => {
                toks.push(player_tok);
                toks.push(ACT_A_YES_PAIR);
                toks.push(SUIT_BASE + suit_index(*suit) as u32);
            }
            ActionType::Answer(AnswerType::NoPair) => {
                toks.push(player_tok);
                toks.push(ACT_A_NO_PAIR);
            }
            ActionType::Answer(AnswerType::YesHalf(suit)) => {
                toks.push(player_tok);
                toks.push(ACT_A_YES_HALF);
                toks.push(SUIT_BASE + suit_index(*suit) as u32);
            }
            ActionType::Answer(AnswerType::NoHalf(suit)) => {
                toks.push(player_tok);
                toks.push(ACT_A_NO_HALF);
                toks.push(SUIT_BASE + suit_index(*suit) as u32);
            }
            // Ignore undo/start actions in history
            _ => {}
        }

        // Emit callback token if present
        if let Some(cb) = &event.callback {
            match cb {
                GameCallback::NewTrump(suit) | GameCallback::StillTrump(suit) => {
                    toks.push(ACT_TRUMP);
                    toks.push(SUIT_BASE + suit_index(*suit) as u32);
                }
                _ => {}
            }
        }
    }

    toks
}

fn build_legal_actions(game: &Game, pov: &PlaceAtTable) -> Vec<LegalActionObs> {
    use tokens::*;
    let mut result = vec![];
    for (idx, action) in game.legal_actions.iter().enumerate() {
        let obs = match &action.action_type {
            ActionType::CardPlayed(card) => {
                // Phase 1.1 Fix: X-Ray Vision Leak Protection.
                // If it's NOT the perspective player's turn, we CANNOT leak the exact 
                // card ID they are legally allowed to play, otherwise the neural net 
                // can see the opponent's entire hand!
                let is_pov_turn = game.state.player_at_turn == *pov;
                LegalActionObs {
                    action_token: ACT_PLAY,
                    card_idx: if is_pov_turn { Some(card_index(card)) } else { None },
                    suit_idx: None,
                    bid_value: None,
                    pass_cards: None,
                    action_list_idx: idx,
                }
            },
            ActionType::NewBid(v) => {
                LegalActionObs {
                    action_token: ACT_BID,
                    card_idx: None,
                    suit_idx: None,
                    bid_value: Some(*v),
                    pass_cards: None,
                    action_list_idx: idx,
                }
            },
            ActionType::StopBidding => LegalActionObs {
                action_token: ACT_PASS_STOP,
                card_idx: None,
                suit_idx: None,
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::AnnounceTrump(suit) => LegalActionObs {
                action_token: ACT_TRUMP,
                card_idx: None,
                suit_idx: Some(suit_index(*suit)),
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::Question(QuestionType::Yours) => LegalActionObs {
                action_token: ACT_Q_PAIR,
                card_idx: None,
                suit_idx: None,
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::Question(QuestionType::YourHalf(suit)) => LegalActionObs {
                action_token: ACT_Q_HALF,
                card_idx: None,
                suit_idx: Some(suit_index(*suit)),
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::Answer(AnswerType::YesPair(suit)) => LegalActionObs {
                action_token: ACT_A_YES_PAIR,
                card_idx: None,
                suit_idx: Some(suit_index(*suit)),
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::Answer(AnswerType::NoPair) => LegalActionObs {
                action_token: ACT_A_NO_PAIR,
                card_idx: None,
                suit_idx: None,
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::Answer(AnswerType::YesHalf(suit)) => LegalActionObs {
                action_token: ACT_A_YES_HALF,
                card_idx: None,
                suit_idx: Some(suit_index(*suit)),
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::Answer(AnswerType::NoHalf(suit)) => LegalActionObs {
                action_token: ACT_A_NO_HALF,
                card_idx: None,
                suit_idx: Some(suit_index(*suit)),
                bid_value: None,
                pass_cards: None,
                action_list_idx: idx,
            },
            ActionType::Pass(cards) => LegalActionObs {
                action_token: ACT_PASS_CARDS,
                card_idx: None,
                suit_idx: None,
                bid_value: None,
                // Do not leak exact passed cards when it's not the POV's turn.
                pass_cards: if game.state.player_at_turn == *pov {
                    Some(cards.iter().map(card_index).collect())
                } else {
                    None
                },
                action_list_idx: idx,
            },
            // Skip undo/start from ML training
            _ => continue,
        };
        result.push(obs);
    }
    result
}

// ─── Serialization for Python interop ────────────────────────────────────────

use serde::{Deserialize, Serialize};

pub const OBS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationJson {
    pub schema_version: u32,
    pub played_bitmask: Vec<bool>,
    pub my_hand_bitmask: Vec<bool>,
    pub possible_bitmasks: Vec<Vec<bool>>,
    pub confirmed_bitmasks: Vec<Vec<bool>>,
    pub my_hand_indices: Vec<usize>,
    pub current_trick_indices: Vec<usize>,
    pub current_trick_players: Vec<usize>,
    pub trump: Option<usize>,
    pub trump_announced: Vec<bool>,
    pub my_role: usize,
    pub active_player: usize,
    pub trick_number: usize,
    pub trick_position: usize,
    pub points_my_team: i32,
    pub points_opp_team: i32,
    pub cards_remaining: Vec<usize>,
    pub trump_possibilities: usize,
    pub last_trick_bonus_live: bool,
    pub event_tokens: Vec<u32>,
    pub legal_actions: Vec<LegalActionObs>,
    pub phase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationDebugJson {
    pub schema_version: u32,
    pub all_hands: Vec<Vec<usize>>,
    pub phase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationTrainLabelsJson {
    /// Relative hidden seats only (left, partner, right).
    pub hidden_hands: Vec<Vec<usize>>,
}

impl From<&Observation> for ObservationJson {
    fn from(o: &Observation) -> Self {
        ObservationJson {
            schema_version: OBS_SCHEMA_VERSION,
            played_bitmask: o.played_bitmask.to_vec(),
            my_hand_bitmask: o.my_hand_bitmask.to_vec(),
            possible_bitmasks: o.possible_bitmasks.iter().map(|b| b.to_vec()).collect(),
            confirmed_bitmasks: o.confirmed_bitmasks.iter().map(|b| b.to_vec()).collect(),
            my_hand_indices: o.my_hand_indices.clone(),
            current_trick_indices: o.current_trick_indices.clone(),
            current_trick_players: o.current_trick_players.clone(),
            trump: o.trump,
            trump_announced: o.trump_announced.to_vec(),
            my_role: o.my_role,
            active_player: o.active_player,
            trick_number: o.trick_number,
            trick_position: o.trick_position,
            points_my_team: o.points_my_team,
            points_opp_team: o.points_opp_team,
            cards_remaining: o.cards_remaining.to_vec(),
            trump_possibilities: o.trump_possibilities,
            last_trick_bonus_live: o.last_trick_bonus_live,
            event_tokens: o.event_tokens.clone(),
            legal_actions: o.legal_actions.clone(),
            phase: o.phase.clone(),
        }
    }
}

impl From<Observation> for ObservationJson {
    fn from(o: Observation) -> Self {
        ObservationJson::from(&o)
    }
}

impl From<&Observation> for ObservationDebugJson {
    fn from(o: &Observation) -> Self {
        ObservationDebugJson {
            schema_version: OBS_SCHEMA_VERSION,
            all_hands: o.all_hands.clone(),
            phase: o.phase.clone(),
        }
    }
}

impl From<Observation> for ObservationDebugJson {
    fn from(o: Observation) -> Self {
        ObservationDebugJson::from(&o)
    }
}

impl From<&Observation> for ObservationTrainLabelsJson {
    fn from(o: &Observation) -> Self {
        ObservationTrainLabelsJson {
            hidden_hands: o.all_hands.iter().skip(1).cloned().collect(),
        }
    }
}

impl From<Observation> for ObservationTrainLabelsJson {
    fn from(o: Observation) -> Self {
        ObservationTrainLabelsJson::from(&o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::game::player::PlaceAtTable;
    use crate::ml::inference::set_hidden_set_constraints_enabled;
    use rand::prelude::IndexedRandom;
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn test_observation_builds() {
        let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
        let mut game = Game::new("test".to_string(), names, None);
        // Start game
        let mut actions = game.legal_actions.clone();
        for _ in 0..4 {
            game = game.apply_action(actions.pop().unwrap()).unwrap();
            actions = game.legal_actions.clone();
        }
        let obs = build_observation(&game, PlaceAtTable(0));
        assert_eq!(obs.played_bitmask.len(), 36);
        assert_eq!(obs.possible_bitmasks.len(), 3);
        assert!(!obs.legal_actions.is_empty());
    }

    #[test]
    fn test_card_index_roundtrip() {
        for (i, card) in get_all_cards().iter().enumerate() {
            assert_eq!(card_index(card), i);
        }
    }

    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn assert_hidden_inference_consistent(base: &Observation, inferred: &Observation) {
        for seat in 0..3 {
            for card in 0..36 {
                if inferred.confirmed_bitmasks[seat][card] {
                    assert!(
                        inferred.possible_bitmasks[seat][card],
                        "confirmed card must be possible (seat={seat}, card={card})"
                    );
                }
            }
        }

        for card in 0..36 {
            let mut conf_seats = 0usize;
            for seat in 0..3 {
                if inferred.confirmed_bitmasks[seat][card] {
                    conf_seats += 1;
                }
            }
            assert!(conf_seats <= 1, "card {card} confirmed in multiple seats");
        }

        for hidden_seat in 1..=3 {
            for &card in &inferred.all_hands[hidden_seat] {
                if base.possible_bitmasks[hidden_seat - 1][card] {
                    assert!(
                        inferred.possible_bitmasks[hidden_seat - 1][card],
                        "inference removed true card that baseline allowed (seat={hidden_seat}, card={card})"
                    );
                }
            }
        }

        for card in 0..36 {
            if inferred.played_bitmask[card] || inferred.my_hand_bitmask[card] {
                continue;
            }
            let mut base_candidates = 0usize;
            let mut inf_candidates = 0usize;
            for seat in 0..3 {
                if base.possible_bitmasks[seat][card] {
                    base_candidates += 1;
                }
                if inferred.possible_bitmasks[seat][card] {
                    inf_candidates += 1;
                }
            }
            if base_candidates > 0 {
                assert!(
                    inf_candidates > 0,
                    "inference removed all candidates for unseen card {card}"
                );
            }
        }
    }

    #[test]
    fn test_hidden_inference_soundness_random_games() {
        let _guard = test_lock().lock().expect("test mutex poisoned");
        let mut rng = rand::rng();
        let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());

        for g in 0..20 {
            let mut game = Game::new(format!("rand_{g}"), names.clone(), None);
            let mut guard_steps = 0usize;
            while !game.ended() && guard_steps < 400 {
                for pov in 0..4 {
                    let prev = set_hidden_set_constraints_enabled(false);
                    let base_obs = build_observation(&game, PlaceAtTable(pov));
                    set_hidden_set_constraints_enabled(true);
                    let inf_obs = build_observation(&game, PlaceAtTable(pov));
                    set_hidden_set_constraints_enabled(prev);
                    assert_hidden_inference_consistent(&base_obs, &inf_obs);
                }
                let action = game
                    .legal_actions
                    .choose(&mut rng)
                    .expect("game must always have at least one legal action")
                    .clone();
                game = game.apply_action(action).expect("legal action should apply");
                guard_steps += 1;
            }
            assert!(guard_steps < 400, "game did not terminate within guard limit");
        }
    }
}
