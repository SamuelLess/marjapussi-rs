/// Bulk dataset generator — parallel, no per-step subprocess overhead.
/// Generates N complete games and writes one NDJSON line per decision point.
///
/// Usage:
///   ml_generate -n 10000000 --policy heuristic --output ml/data/dataset.ndjson --threads 8
///
/// Output line format:
///   {"obs":{...}, "action_taken":3, "outcome_pts_my_team":85, "outcome_pts_opp":35, "won":true}

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use marjapussi::game::gameinfo::GameFinishedInfo;
use marjapussi::game::gamestate::GamePhase;
use marjapussi::game::Game;
use marjapussi::ml::observation::{build_observation, ObservationJson};
use marjapussi::ml::sim::{heuristic_policy, random_policy, PolicyFn};

struct Cli {
    games: u64,
    policy: String,
    output: PathBuf,
    threads: usize,
    from_trick: usize,
    log_every: u64,
}

fn parse_args() -> Cli {
    let args: Vec<String> = env::args().collect();
    let mut cli = Cli {
        games: 100_000,
        policy: "heuristic".to_string(),
        output: PathBuf::from("ml/data/dataset.ndjson"),
        threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        from_trick: 1,
        log_every: 10_000,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-n" | "--games" => { i += 1; cli.games = args[i].parse().unwrap_or(cli.games); }
            "-p" | "--policy" => { i += 1; cli.policy = args[i].clone(); }
            "-o" | "--output" => { i += 1; cli.output = PathBuf::from(&args[i]); }
            "-t" | "--threads" => { i += 1; cli.threads = args[i].parse().unwrap_or(cli.threads); }
            "--from-trick" => { i += 1; cli.from_trick = args[i].parse().unwrap_or(1); }
            "--log-every" => { i += 1; cli.log_every = args[i].parse().unwrap_or(10_000); }
            "-h" | "--help" => {
                eprintln!("Usage: ml_generate [-n GAMES] [-p POLICY] [-o OUTPUT] [-t THREADS] [--from-trick N] [--log-every N]");
                eprintln!("  POLICY: random | heuristic   OUTPUT: path to .ndjson file");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    cli
}

fn make_policy(policy: &str) -> PolicyFn {
    match policy {
        "random" => random_policy(),
        _ => heuristic_policy(),
    }
}

fn run_game_and_collect(policy: &PolicyFn, from_trick: usize) -> Vec<serde_json::Value> {
    let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
    let mut game = Game::new("gen".to_string(), names, None);

    // Auto-start: apply 4 Start actions
    let mut actions = game.legal_actions.clone();
    for _ in 0..4 {
        if let Some(a) = actions.pop() {
            match game.apply_action(a) {
                Ok(g) => { game = g; actions = game.legal_actions.clone(); }
                Err(_) => return vec![],
            }
        }
    }

    let mut decision_points: Vec<(ObservationJson, usize)> = vec![];

    let mut steps = 0;
    while game.state.phase != GamePhase::Ended && steps < 300 {
        steps += 1;
        let legal = game.legal_actions.clone();
        if legal.is_empty() { break; }

        let trick_no = game.state.all_tricks.len() + 1;
        let record = trick_no >= from_trick;
        let action_idx = policy(&legal).min(legal.len() - 1);

        if record {
            let pov = game.state.player_at_turn.clone();
            let obs = build_observation(&game, pov);
            let obs_json = ObservationJson::from(obs);
            decision_points.push((obs_json, action_idx));
        }

        match game.apply_action(legal[action_idx].clone()) {
            Ok(g) => game = g,
            Err(_) => break,
        }
    }

    if decision_points.is_empty() { return vec![]; }

    let outcome = GameFinishedInfo::from(game);

    let mut records = Vec::with_capacity(decision_points.len());
    for (obs_json, action_idx) in decision_points {
        let my_pts: i32 = outcome.tricks.iter()
            .filter(|t| t.winner.0 % 2 == 0)
            .map(|t| t.points.0)
            .sum();
        let opp_pts: i32 = outcome.tricks.iter()
            .filter(|t| t.winner.0 % 2 != 0)
            .map(|t| t.points.0)
            .sum();
        let rec = serde_json::json!({
            "obs": &obs_json,
            "action_taken": action_idx,
            "outcome_pts_my_team": my_pts,
            "outcome_pts_opp": opp_pts,
            "won": outcome.won,
        });
        records.push(rec);
    }
    records
}

fn main() {
    let cli = parse_args();

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).expect("Cannot create output directory");
    }
    let file = File::create(&cli.output).expect("Cannot create output file");
    let writer = Arc::new(Mutex::new(BufWriter::new(file)));

    let games_done   = Arc::new(AtomicU64::new(0));
    let points_total = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let total = cli.games;
    let from_trick = cli.from_trick;
    let log_every = cli.log_every;
    let policy_str = cli.policy.clone();

    eprintln!("Generating {} games with policy '{}' on {} threads", total, policy_str, cli.threads);
    eprintln!("Output: {}", cli.output.display());

    let handles: Vec<_> = (0..cli.threads).map(|_| {
        let writer = Arc::clone(&writer);
        let games_done = Arc::clone(&games_done);
        let points_total = Arc::clone(&points_total);
        let policy_str = policy_str.clone();

        std::thread::spawn(move || {
            let policy = make_policy(&policy_str);
            let mut local_buf: Vec<u8> = Vec::with_capacity(256 * 1024);

            loop {
                let game_no = games_done.fetch_add(1, Ordering::Relaxed);
                if game_no >= total { break; }

                let records = run_game_and_collect(&policy, from_trick);
                let nrec = records.len() as u64;

                for rec in &records {
                    if let Ok(line) = serde_json::to_string(rec) {
                        local_buf.extend_from_slice(line.as_bytes());
                        local_buf.push(b'\n');
                    }
                }
                points_total.fetch_add(nrec, Ordering::Relaxed);

                // Flush local buffer when it grows large
                if local_buf.len() > 512 * 1024 {
                    let mut w = writer.lock().unwrap();
                    w.write_all(&local_buf).ok();
                    local_buf.clear();
                }

                if (game_no + 1) % log_every == 0 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let done = game_no + 1;
                    let rate = done as f64 / elapsed;
                    let pts = points_total.load(Ordering::Relaxed);
                    let eta = (total as f64 - done as f64) / rate.max(1.0) / 60.0;
                    eprintln!("[{done}/{total}] {rate:.0} games/sec | {pts} decision points | ETA {eta:.1}min");
                }
            }

            if !local_buf.is_empty() {
                let mut w = writer.lock().unwrap();
                w.write_all(&local_buf).ok();
            }
        })
    }).collect();

    for h in handles { h.join().ok(); }

    writer.lock().unwrap().flush().ok();

    let elapsed = start.elapsed().as_secs_f64();
    let pts = points_total.load(Ordering::Relaxed);
    let size_gb = std::fs::metadata(&cli.output)
        .map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);

    eprintln!("\nDone! {total} games in {elapsed:.1}s ({:.0} games/sec)", total as f64 / elapsed);
    eprintln!("Wrote {pts} decision points — {size_gb:.2} GB → {}", cli.output.display());
}
