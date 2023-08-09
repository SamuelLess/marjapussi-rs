//! A chat server that broadcasts a message to all connections.
//!
//! This is a simple line-based server which accepts WebSocket connections,
//! reads lines from those connections, and broadcasts the lines to all other
//! connected clients.
//!
//! You can test this out by running:
//!
//!     cargo run --example server 127.0.0.1:12345
//!
//! And then in another window run:
//!
//!     cargo run --example client ws://127.0.0.1:12345/
//!
//! You can run the second command in multiple windows and then chat between the
//! two, seeing the messages from the other client as they're received. For all
//! connected clients they'll all join the same room and see everyone else's
//! messages.

#![feature(test)]
#![allow(dead_code)]

mod game;
mod tests;

use std::io;
use std::{
    collections::HashMap,
    env,
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use futures_channel::mpsc::{unbounded, UnboundedSender};
use futures_util::{future, pin_mut, stream::TryStreamExt, StreamExt};

use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::Message;

use crate::game::Game;
use crate::tests::game::test_random_game_random;
use warp::Filter;

type Tx = UnboundedSender<Message>;
type PeerMap = Arc<Mutex<HashMap<SocketAddr, Tx>>>;

async fn handle_connection(peer_map: PeerMap, raw_stream: TcpStream, addr: SocketAddr) {
    println!("Incoming TCP connection from: {}", addr);

    let ws_stream = tokio_tungstenite::accept_async(raw_stream)
        .await
        .expect("Error during the websocket handshake occurred");
    println!("WebSocket connection established: {}", addr);

    // Insert the write part of this peer to the peer map.
    let (tx, rx) = unbounded();
    peer_map.lock().unwrap().insert(addr, tx);

    let (outgoing, incoming) = ws_stream.split();

    let broadcast_incoming = incoming.try_for_each(|msg| {
        println!(
            "Received a message from {}: {}",
            addr,
            msg.to_text().unwrap()
        );
        let peers = peer_map.lock().unwrap();

        // We want to broadcast the message to everyone except ourselves.
        //let broadcast_recipients =
        //    peers.iter().filter(|(peer_addr, _)| peer_addr != &&addr).map(|(_, ws_sink)| ws_sink);

        let broadcast_recipients = peers.iter().map(|(_, ws_sink)| ws_sink);

        for recp in broadcast_recipients {
            recp.unbounded_send(Message::from("pong")).unwrap();
        }

        future::ok(())
    });

    let receive_from_others = rx.map(Ok).forward(outgoing);

    pin_mut!(broadcast_incoming, receive_from_others);
    future::select(broadcast_incoming, receive_from_others).await; // everything runs here, waiting for either incoming or outcoming stream to fail

    println!("{} disconnected", &addr);
    peer_map.lock().unwrap().remove(&addr);
}

async fn run_socket(socket_addr: &String) -> anyhow::Result<()> {
    //Result<(), IoError> {

    let state = PeerMap::new(Mutex::new(HashMap::new()));

    // Create the event loop and TCP listener we'll accept connections on.
    let try_socket = TcpListener::bind(&socket_addr).await;
    let listener = try_socket.expect("Failed to bind");
    println!("Socket listening on: {}", socket_addr);

    // Let's spawn the handling of each connection in a separate task.
    while let Ok((stream, addr)) = listener.accept().await {
        tokio::spawn(handle_connection(state.clone(), stream, addr));
    }

    Ok(())
}

async fn run_warp(http_addr: &String) -> anyhow::Result<()> {
    let server: SocketAddr = http_addr.parse()?;
    //.expect("Could not parse http address.");
    //.unwrap_or_else(|err| SocketAddr::from(([0,0,0,0], 3030)));
    let routes = warp::any().map(|| "Hallo, API!");
    println!("API listening on {}", http_addr);
    warp::serve(routes).run(server).await;
    Ok(())
}

#[tokio::main]
async fn main() {
    //test_play().await;
    for _ in 0..1000 {
        test_random_game_random();
    }
}

async fn start_endpoints() {
    let _socket_addr = env::args()
        .nth(1)
        .unwrap_or_else(|| "0.0.0.0:3060".to_string());
    let _http_addr = env::args()
        .nth(2)
        .unwrap_or_else(|| "0.0.0.0:3030".to_string());
}

async fn test_play() {
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
    loop {
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
}
