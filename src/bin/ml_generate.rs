/// Bulk dataset generator with optional Monte Carlo advantage estimation.
/// Parallel threads — no subprocess overhead — ~3900 games/sec on 32 CPUs.
/// MC mode with --mc-rollouts 8: ~300 games/sec (much better training signal).
///
/// Usage:
///   ml_generate -n 1000000 --mc-rollouts 8 --from-trick 5 -o ml/data/dataset.ndjson
///
/// Output NDJSON (one line per decision point):
///   {"obs":{...},"action_taken":3,"advantages":[0.8,-0.3,...],"outcome_pts_my_team":85,"won":true}

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
use marjapussi::ml::sim::{heuristic_policy, random_policy, run_to_end, PolicyFn};

// ─── CLI parsing ──────────────────────────────────────────────────────────────

struct Cli {
    games: u64, policy: String, output: PathBuf,
    threads: usize, from_trick: usize, log_every: u64, mc_rollouts: u32,
}

fn parse_args() -> Cli {
    let args: Vec<String> = env::args().collect();
    let n_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut c = Cli { games: 100_000, policy: "heuristic".into(),
        output: PathBuf::from("ml/data/dataset.ndjson"),
        threads: n_cpus, from_trick: 1, log_every: 10_000, mc_rollouts: 0 };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-n"|"--games"       => { i+=1; c.games      = args[i].parse().unwrap_or(c.games); }
            "-p"|"--policy"      => { i+=1; c.policy     = args[i].clone(); }
            "-o"|"--output"      => { i+=1; c.output     = PathBuf::from(&args[i]); }
            "-t"|"--threads"     => { i+=1; c.threads    = args[i].parse().unwrap_or(c.threads); }
            "--from-trick"       => { i+=1; c.from_trick = args[i].parse().unwrap_or(1); }
            "--log-every"        => { i+=1; c.log_every  = args[i].parse().unwrap_or(10_000); }
            "--mc-rollouts"|"-m" => { i+=1; c.mc_rollouts= args[i].parse().unwrap_or(0); }
            "-h"|"--help" => {
                eprintln!("Usage: ml_generate [-n N] [-p random|heuristic] [-o FILE] [-t THREADS]");
                eprintln!("  --from-trick N   only record decisions from trick N (1=all, 5=endgame)");
                eprintln!("  --mc-rollouts N  Monte Carlo rollouts per action (0=use outcome only)");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    c
}

fn make_policy(p: &str) -> PolicyFn { match p { "random" => random_policy(), _ => heuristic_policy() } }

// ─── Monte Carlo advantage ────────────────────────────────────────────────────
/// Run n_rollouts per legal action, return normalised advantages (mean=0, std=1).
fn mc_advantages(game: &Game, policy: &PolicyFn, n: u32, pov_parity: u8) -> Vec<f32> {
    let legal = game.legal_actions.clone();
    let scores: Vec<f32> = legal.iter().map(|action| {
        let sum: f32 = (0..n).map(|_| {
            match game.apply_action(action.clone()) {
                Err(_) => 0.5,
                Ok(branch) => {
                    let (_, info) = run_to_end(branch, policy);
                    info.tricks.iter()
                        .filter(|t| t.winner.0 % 2 == pov_parity)
                        .map(|t| t.points.0 as f32)
                        .sum::<f32>() / 120.0
                }
            }
        }).sum();
        sum / n as f32
    }).collect();

    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let std  = (scores.iter().map(|&s| (s - mean).powi(2)).sum::<f32>()
                / scores.len() as f32).sqrt().max(0.01);
    scores.iter().map(|&s| (s - mean) / std).collect()
}

// ─── Game runner ──────────────────────────────────────────────────────────────
fn run_game_collect(policy: &PolicyFn, from_trick: usize, mc_rollouts: u32)
    -> Vec<serde_json::Value>
{
    let names = ["P0","P1","P2","P3"].map(|s| s.to_string());
    let mut game = Game::new("gen".to_string(), names, None);
    // Auto-start
    let mut actions = game.legal_actions.clone();
    for _ in 0..4 {
        if let Some(a) = actions.pop() {
            match game.apply_action(a) {
                Ok(g) => { game = g; actions = game.legal_actions.clone(); }
                Err(_) => return vec![],
            }
        }
    }

    let mut decision_points: Vec<(ObservationJson, usize, Vec<f32>)> = vec![];

    let mut steps = 0;
    while game.state.phase != GamePhase::Ended && steps < 300 {
        steps += 1;
        let legal = game.legal_actions.clone();
        if legal.is_empty() { break; }
        let action_idx = policy(&legal).min(legal.len() - 1);
        let trick_no = game.state.all_tricks.len() + 1;

        if trick_no >= from_trick {
            let pov = game.state.player_at_turn.clone();
            let pov_parity = pov.0 % 2;
            let obs_json = ObservationJson::from(build_observation(&game, pov));

            let advs = if mc_rollouts > 0 && legal.len() > 1 {
                mc_advantages(&game, policy, mc_rollouts, pov_parity)
            } else {
                vec![]
            };
            decision_points.push((obs_json, action_idx, advs));
        }

        match game.apply_action(legal[action_idx].clone()) {
            Ok(g) => game = g,
            Err(_) => break,
        }
    }

    if decision_points.is_empty() { return vec![]; }
    let outcome = GameFinishedInfo::from(game);

    decision_points.into_iter().map(|(obs_json, action_idx, advs)| {
        let my_pts: i32 = outcome.tricks.iter()
            .filter(|t| t.winner.0 % 2 == 0)
            .map(|t| t.points.0).sum();
        let opp_pts = 120 - my_pts;
        let mut r = serde_json::json!({
            "obs": &obs_json,
            "action_taken": action_idx,
            "outcome_pts_my_team": my_pts,
            "outcome_pts_opp": opp_pts,
            "won": outcome.won,
        });
        if !advs.is_empty() {
            r["advantages"] = serde_json::json!(advs);
            r["chosen_advantage"] = serde_json::json!(advs.get(action_idx).copied().unwrap_or(0.0));
        }
        r
    }).collect()
}

// ─── main ─────────────────────────────────────────────────────────────────────
fn main() {
    let cli = parse_args();
    if let Some(p) = cli.output.parent() { std::fs::create_dir_all(p).ok(); }
    let writer    = Arc::new(Mutex::new(BufWriter::new(File::create(&cli.output).expect("open output"))));
    let done_ctr  = Arc::new(AtomicU64::new(0));
    let pts_ctr   = Arc::new(AtomicU64::new(0));
    let start     = Instant::now();
    let total     = cli.games;
    let mc        = cli.mc_rollouts;
    let from      = cli.from_trick;
    let log_every = cli.log_every;
    let pol_str   = cli.policy.clone();

    let mc_note = if mc > 0 { format!(" + MC-{mc} rollouts/action") } else { String::new() };
    eprintln!("Generating {} games | policy: {} | threads: {}{}", total, pol_str, cli.threads, mc_note);
    eprintln!("Output: {}", cli.output.display());

    let handles: Vec<_> = (0..cli.threads).map(|_| {
        let writer = Arc::clone(&writer);
        let done_ctr = Arc::clone(&done_ctr);
        let pts_ctr  = Arc::clone(&pts_ctr);
        let pol_str  = pol_str.clone();
        std::thread::spawn(move || {
            let policy = make_policy(&pol_str);
            let mut buf: Vec<u8> = Vec::with_capacity(512 * 1024);
            loop {
                let n = done_ctr.fetch_add(1, Ordering::Relaxed);
                if n >= total { break; }
                let records = run_game_collect(&policy, from, mc);
                pts_ctr.fetch_add(records.len() as u64, Ordering::Relaxed);
                for r in &records {
                    if let Ok(line) = serde_json::to_string(r) {
                        buf.extend_from_slice(line.as_bytes());
                        buf.push(b'\n');
                    }
                }
                if buf.len() > 512 * 1024 {
                    writer.lock().unwrap().write_all(&buf).ok();
                    buf.clear();
                }
                let done = n + 1;
                if done % log_every == 0 {
                    let secs = start.elapsed().as_secs_f64();
                    let rate = done as f64 / secs;
                    let pts  = pts_ctr.load(Ordering::Relaxed);
                    let eta  = (total as f64 - done as f64) / rate.max(1.0) / 60.0;
                    eprintln!("[{done}/{total}] {rate:.0} games/sec | {pts} decision pts | ETA {eta:.1} min");
                }
            }
            if !buf.is_empty() { writer.lock().unwrap().write_all(&buf).ok(); }
        })
    }).collect();

    for h in handles { h.join().ok(); }
    writer.lock().unwrap().flush().ok();

    let secs = start.elapsed().as_secs_f64();
    let pts  = pts_ctr.load(Ordering::Relaxed);
    let sz   = std::fs::metadata(&cli.output).map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
    eprintln!("\nDone! {total} games in {secs:.1}s ({:.0} games/sec)", total as f64 / secs);
    eprintln!("{pts} decision points — {sz:.2} GB → {}", cli.output.display());
}
