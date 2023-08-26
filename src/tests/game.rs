#![allow(unused_imports)]
extern crate test;
use rand::seq::SliceRandom;

use crate::game::gameevent::{ActionType, GameAction};
use crate::game::gameinfo::GameInfoPlayer;
use crate::game::gamestate::GamePhase;
use crate::game::player::PlaceAtTable;
use crate::game::points::Points;
use crate::game::{current_time_string, Game};
use test::Bencher;

#[test]
fn test_time_creation() {
    let time = current_time_string();
    assert_eq!(time.len(), 19);
}

fn helper_create_game() -> Game {
    let names = [
        "S1".to_string(),
        "S2".to_string(),
        "S3".to_string(),
        "S4".to_string(),
    ];
    Game::new(String::from("Testgame"), names, None)
}

#[test]
fn test_creating() {
    let names = [
        "S1".to_string(),
        "S2".to_string(),
        "S3".to_string(),
        "S4".to_string(),
    ];
    let game = Game::new(String::from("Game Name"), names, None);
    assert_eq!(game.info.name, String::from("Game Name"));
}

#[test]
fn test_starting() {
    let mut game = helper_create_game();
    let mut actions = game.legal_actions.clone();
    assert_eq!(actions.len(), 4);
    for _ in 0..4 {
        let act = actions.pop().unwrap();
        let res = game.apply_action(act);
        game = res.ok().unwrap();
        actions = game.legal_actions.clone();
    }
    assert_eq!(game.state.phase, GamePhase::Bidding);
}

#[test]
fn test_random_game_controlled() {
    let mut game = helper_create_game();
    let mut actions = game.legal_actions();
    assert_eq!(actions.len(), 4);
    for _ in 0..4 {
        game = game.apply_action(actions.pop().unwrap()).ok().unwrap();
        actions = game.legal_actions.clone();
    }
    assert_eq!(game.state.phase, GamePhase::Bidding);
    actions = game.legal_actions.clone();
    assert_eq!(actions.len(), 62);
    let bid140 = GameAction {
        action_type: ActionType::NewBid(140),
        player: game.state.player_at_turn.clone(),
    };
    game = game.apply_action(bid140).ok().unwrap();
    let forbidden_action = actions.pop().unwrap();
    let res = game.apply_action(forbidden_action);
    assert!(res.is_err());
    for _ in 0..4 {
        actions = game.legal_actions.clone();
        let res = game.apply_action(actions[3].clone());
        //println!("Result: {:?}", res);
        //println!("{:#?}", game.legal_actions());
        game = res.ok().unwrap();
    }
    for _ in 0..3 {
        actions = game.legal_actions.clone();
        assert_eq!(game.state.phase, GamePhase::Bidding);
        let res = game.apply_action(actions[0].clone());
        game = res.ok().unwrap();
    }
    assert_eq!(game.state.value, Points(200));

    //passing forth
    assert_eq!(game.state.player_at_turn().name, String::from("S3"));
    assert_eq!(game.state.phase, GamePhase::PassingForth);
    assert_eq!(game.state.player_at_turn().cards.len(), 9);
    assert_eq!(game.legal_actions.len(), 127); //nCr(9,4) + 1
    actions = game.legal_actions.clone();
    let res = game.apply_action(actions[2].clone());
    game = res.ok().unwrap();
    assert_eq!(game.state.player_at_turn().cards.len(), 13);
    assert_eq!(
        game.state
            .player_at_place(game.state.player_at_turn.partner())
            .cards
            .len(),
        5
    );
    //println!("{:#?}", game.state);

    //passing back
    assert_eq!(game.state.player_at_turn().name, String::from("S1"));
    assert_eq!(game.state.phase, GamePhase::PassingBack);
    actions = game.legal_actions.clone();
    let res = game.apply_action(actions[9].clone());
    game = res.ok().unwrap();
    assert_eq!(game.state.player_at_turn().cards.len(), 9);
    assert_eq!(
        game.state
            .player_at_place(game.state.player_at_turn.partner())
            .cards
            .len(),
        9
    );
    assert_eq!(game.state.phase, GamePhase::Raising);
    actions = game.legal_actions.clone();
    let res = game.apply_action(actions[2].clone());
    game = res.ok().unwrap();
    //println!("lastevent: {:#?}", game.all_events.last().unwrap());
    assert_eq!(game.state.phase, GamePhase::Trick);
    actions = game.legal_actions.clone();
    //println!("actions {:#?}", actions);
    let _res = game.apply_action(actions[0].clone());
    //game = res.ok().unwrap();
    //println!("{:#?}", GameInfoPlayer::from_game(game, PlaceAtTable(2)));
}

#[test]
fn test_random_game_multi() {
    for _ in 0..100 {
        test_random_game_random();
    }
}

pub fn test_random_game_random() {
    let mut game = helper_create_game();
    let mut actions = game.legal_actions.clone();
    assert_eq!(actions.len(), 4);
    for _ in 0..4 {
        game = game.apply_action(actions.pop().unwrap()).ok().unwrap();
        actions = game.legal_actions.clone();
    }
    let mut i = 0;
    while game.state.phase != GamePhase::Ended {
        i += 1;
        if i >= 200 {
            panic!("Too many moves! Game: {:#?}\n ", game,);
        }
        actions = game.legal_actions.clone();
        let opt_select = actions.choose(&mut rand::thread_rng());
        if opt_select.is_none() {
            panic!("Game: {:#?}\n Action: {:?}", game, opt_select);
        }
        let select = opt_select.unwrap().clone();
        let res = game.apply_action(select.clone()).ok();
        if res.is_some() {
            game = res.unwrap();
        } else {
            panic!("Game: {:?}\n Action: {:?}", game, select);
        }
    }
}

#[bench]
fn bench_starting(b: &mut Bencher) {
    b.iter(|| {
        let mut game = helper_create_game();
        let mut actions = game.legal_actions.clone();
        for _ in 0..4 {
            game = game.apply_action(actions.pop().unwrap()).ok().unwrap();
            actions = game.legal_actions.clone();
        }
    })
}

#[bench]
fn bench_game_test(b: &mut Bencher) {
    b.iter(|| {
        test_random_game_controlled();
    })
}

#[bench]
fn bench_create(b: &mut Bencher) {
    b.iter(|| {
        let _game = helper_create_game();
    });
}

#[bench]
fn bench_play_random(b: &mut Bencher) {
    b.iter(|| {
        test_random_game_random();
    });
}
