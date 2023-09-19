#![feature(test)]
#![allow(dead_code)]

use std::io;

use crate::game::gameinfo::GameInfoDatabase;
use crate::game::gamestate::GamePhase;
use crate::game::Game;

mod game;
mod tests;

fn main() {
    test_play();
}

fn test_play() {
    println!("starting");
    //tokio::try_join!(run_warp(&http_addr), run_socket(&socket_addr)).unwrap();
    let mut game = Game::new(
        String::from("Eine Runde"),
        [
            String::from("S 1"),
            String::from("S 2"),
            String::from("S 3"),
            String::from("S 4"),
        ],
        None,
    );
    let mut actions = game.legal_actions();
    while game.state.phase != GamePhase::Ended {
        print!("{}[2J", 27 as char);
        println!(
            "Phase: {:?}, Player: {:?}",
            game.state.phase,
            game.state.player_at_turn().name
        );
        println!("Cards 0: {:?}", game.state.players[0]);
        println!("Cards 1: {:?}", game.state.players[1]);
        println!("Cards 2: {:?}", game.state.players[2]);
        println!("Cards 3: {:?}", game.state.players[3]);
        println!("actions:");
        for (i, action) in actions.iter().enumerate() {
            println!("{}: {:?}", i, action);
            if i > 4 {
                println!("...");
                break;
            }
        }
        if actions.len() > 4 {
            for (i, action) in actions.iter().rev().enumerate() {
                println!("{}: {:?}", actions.len() - 1 - i, action);
                if i > 4 {
                    println!("...");
                    break;
                }
            }
        }
        let mut user_input = String::new();
        let stdin = io::stdin(); // We get `Stdin` here.
        let _ = stdin.read_line(&mut user_input);
        let action: i32 = user_input.trim().parse().unwrap_or(0);
        println!("chose: {}", action);
        let res = game.apply_action(actions[action as usize].clone());
        println!("results {:?}", res);
        game = res.ok().unwrap();
        actions = game.legal_actions();
    }
    //println!("{:#?}", GameInfoDatabase::from(game));
    println!(
        "{}",
        serde_json::to_string(&GameInfoDatabase::from(game))
            .ok()
            .unwrap()
    );
}
