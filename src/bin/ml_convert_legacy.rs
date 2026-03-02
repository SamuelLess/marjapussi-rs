use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use clap::{Arg, Command};
use serde::Deserialize;

use marjapussi::game::cards::{Card, Suit, Value};
use marjapussi::game::gameevent::{ActionType, AnswerType, GameAction, QuestionType};
use marjapussi::game::gameinfo::GameFinishedInfo;
use marjapussi::game::gamestate::GamePhase;
use marjapussi::game::player::PlaceAtTable;
use marjapussi::game::Game;
use marjapussi::ml::observation::{build_observation, card_index, ObservationJson};
use marjapussi::ml::pass_selection::{
    build_pick_legal_actions, collect_pass_options, PASS_PICK_TARGET,
};

#[derive(Debug, Deserialize)]
struct LegacyGameRecord {
    #[serde(rename = "_id")]
    id: serde_json::Value,
    name: String,
    players: Vec<String>,
    cards: HashMap<String, Vec<String>>,
    actions: Vec<String>,
    #[serde(default)]
    players_points: HashMap<String, i32>,
}

#[derive(Debug, Deserialize)]
struct LegacyGameDataset {
    games: Vec<LegacyGameRecord>,
}

#[derive(Debug)]
struct DecisionPoint {
    obs: ObservationJson,
    action_idx: usize,
    pov_parity: u8,
    pov_seat: u8,
}

fn parse_args() -> (PathBuf, PathBuf, Option<usize>, Option<f64>) {
    let matches = Command::new("ml_convert_legacy")
        .about("Convert legacy full-game logs to decision-point NDJSON for supervised pretraining.")
        .arg(
            Arg::new("input")
                .long("input")
                .default_value("ml/dataset/games.ndjson")
                .help("Path to legacy dataset (.ndjson, JSON array, or {\"games\": [...]} object)."),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .default_value("ml/data/human_dataset.ndjson")
                .help("Output NDJSON path."),
        )
        .arg(
            Arg::new("limit")
                .long("limit")
                .value_parser(clap::value_parser!(usize))
                .help("Optional max number of games to process."),
        )
        .arg(
            Arg::new("min-player-winrate")
                .long("min-player-winrate")
                .value_parser(clap::value_parser!(f64))
                .help(
                    "Optional strict player win-rate threshold; emit only rows for players with winrate > threshold.",
                ),
        )
        .get_matches();

    let input = PathBuf::from(
        matches
            .get_one::<String>("input")
            .expect("missing --input"),
    );
    let output = PathBuf::from(
        matches
            .get_one::<String>("output")
            .expect("missing --output"),
    );
    let limit = matches.get_one::<usize>("limit").copied();
    let min_player_winrate = matches.get_one::<f64>("min-player-winrate").copied();
    (input, output, limit, min_player_winrate)
}

fn team_points_from_record(record: &LegacyGameRecord) -> Option<(f64, f64)> {
    if record.players.len() != 4 || record.players_points.is_empty() {
        return None;
    }
    let p0 = f64::from(*record.players_points.get(&record.players[0])?);
    let p1 = f64::from(*record.players_points.get(&record.players[1])?);
    let p2 = f64::from(*record.players_points.get(&record.players[2])?);
    let p3 = f64::from(*record.players_points.get(&record.players[3])?);
    Some((p0 + p2, p1 + p3))
}

fn compute_player_winrates(records: &[LegacyGameRecord], total: usize) -> HashMap<String, f64> {
    let mut wins: HashMap<String, f64> = HashMap::new();
    let mut games: HashMap<String, f64> = HashMap::new();

    for record in records.iter().take(total) {
        let Some((team0, team1)) = team_points_from_record(record) else {
            continue;
        };
        let winner_parity = if team0 > team1 {
            Some(0u8)
        } else if team1 > team0 {
            Some(1u8)
        } else {
            None
        };

        for (seat_idx, player) in record.players.iter().enumerate() {
            *games.entry(player.clone()).or_insert(0.0) += 1.0;
            match winner_parity {
                Some(parity) => {
                    if (seat_idx as u8) % 2 == parity {
                        *wins.entry(player.clone()).or_insert(0.0) += 1.0;
                    }
                }
                None => {
                    *wins.entry(player.clone()).or_insert(0.0) += 0.5;
                }
            }
        }
    }

    let mut out = HashMap::new();
    for (player, n_games) in games {
        if n_games > 0.0 {
            let n_wins = wins.get(&player).copied().unwrap_or(0.0);
            out.insert(player, n_wins / n_games);
        }
    }
    out
}

fn parse_suit_char(c: char) -> Result<Suit, String> {
    match c {
        'g' => Ok(Suit::Green),
        'e' => Ok(Suit::Acorns),
        's' => Ok(Suit::Bells),
        'r' => Ok(Suit::Red),
        _ => Err(format!("unknown suit char: {c}")),
    }
}

fn parse_value_char(c: char) -> Result<Value, String> {
    match c {
        '6' => Ok(Value::Six),
        '7' => Ok(Value::Seven),
        '8' => Ok(Value::Eight),
        '9' => Ok(Value::Nine),
        'U' => Ok(Value::Unter),
        'O' => Ok(Value::Ober),
        'K' => Ok(Value::King),
        'Z' => Ok(Value::Ten),
        'A' => Ok(Value::Ace),
        _ => Err(format!("unknown value char: {c}")),
    }
}

fn parse_card_token(card: &str) -> Result<Card, String> {
    let mut chars = card.chars();
    let suit = chars.next().ok_or_else(|| format!("bad card token: {card}"))?;
    let sep = chars.next().ok_or_else(|| format!("bad card token: {card}"))?;
    let value = chars.next().ok_or_else(|| format!("bad card token: {card}"))?;
    if sep != '-' || chars.next().is_some() {
        return Err(format!("bad card token: {card}"));
    }
    Ok(Card {
        suit: parse_suit_char(suit)?,
        value: parse_value_char(value)?,
    })
}

fn parse_cards(tokens: &[String]) -> Result<Vec<Card>, String> {
    tokens.iter().map(|t| parse_card_token(t)).collect()
}

fn split_action(raw: &str) -> Result<(u8, &str, &str), String> {
    let mut it = raw.splitn(3, ',');
    let seat = it
        .next()
        .ok_or_else(|| format!("missing seat in action: {raw}"))?
        .parse::<u8>()
        .map_err(|e| format!("bad seat in action '{raw}': {e}"))?;
    let code = it
        .next()
        .ok_or_else(|| format!("missing code in action: {raw}"))?;
    let val = it
        .next()
        .ok_or_else(|| format!("missing value in action: {raw}"))?;
    Ok((seat, code, val))
}

fn parse_question(val: &str) -> Result<ActionType, String> {
    if let Some(suit_char) = val.strip_prefix("my") {
        let suit = parse_suit_char(
            suit_char
                .chars()
                .next()
                .ok_or_else(|| format!("invalid question value: {val}"))?,
        )?;
        return Ok(ActionType::AnnounceTrump(suit));
    }
    if let Some(suit_char) = val.strip_prefix("ou") {
        let suit = parse_suit_char(
            suit_char
                .chars()
                .next()
                .ok_or_else(|| format!("invalid question value: {val}"))?,
        )?;
        return Ok(ActionType::Question(QuestionType::YourHalf(suit)));
    }
    Ok(ActionType::Question(QuestionType::Yours))
}

fn parse_answer(val: &str) -> Result<ActionType, String> {
    if let Some(suit_char) = val.strip_prefix("my") {
        let suit = parse_suit_char(
            suit_char
                .chars()
                .next()
                .ok_or_else(|| format!("invalid answer value: {val}"))?,
        )?;
        return Ok(ActionType::Answer(AnswerType::YesPair(suit)));
    }
    if let Some(suit_char) = val.strip_prefix("no") {
        let suit = parse_suit_char(
            suit_char
                .chars()
                .next()
                .ok_or_else(|| format!("invalid answer value: {val}"))?,
        )?;
        return Ok(ActionType::Answer(AnswerType::NoHalf(suit)));
    }
    if let Some(suit_char) = val.strip_prefix("ou") {
        let suit = parse_suit_char(
            suit_char
                .chars()
                .next()
                .ok_or_else(|| format!("invalid answer value: {val}"))?,
        )?;
        return Ok(ActionType::Answer(AnswerType::YesHalf(suit)));
    }
    Ok(ActionType::Answer(AnswerType::NoPair))
}

fn answer_half_suit_from_raw(raw: &str) -> Option<Suit> {
    let (_, code, val) = split_action(raw).ok()?;
    if code != "ANSW" {
        return None;
    }
    let suit_char = val
        .strip_prefix("no")
        .or_else(|| val.strip_prefix("ou"))
        .and_then(|s| s.chars().next())?;
    parse_suit_char(suit_char).ok()
}

fn is_pair_answer_raw(raw: &str) -> bool {
    let (_, code, val) = match split_action(raw) {
        Ok(parts) => parts,
        Err(_) => return false,
    };
    if code != "ANSW" {
        return false;
    }
    val == "nmy" || val.starts_with("my")
}

fn parse_single_action(raw: &str) -> Result<GameAction, String> {
    let (seat, code, val) = split_action(raw)?;
    let action_type = match code {
        "PROV" => {
            let bid = val
                .parse::<i32>()
                .map_err(|e| format!("invalid PROV value '{val}': {e}"))?;
            if bid == 0 {
                ActionType::StopBidding
            } else {
                ActionType::NewBid(bid)
            }
        }
        "PRMO" => ActionType::NewBid(
            val.parse::<i32>()
                .map_err(|e| format!("invalid PRMO value '{val}': {e}"))?,
        ),
        "TRCK" => ActionType::CardPlayed(parse_card_token(val)?),
        "QUES" => parse_question(val)?,
        "ANSW" => parse_answer(val)?,
        _ => return Err(format!("unsupported action code: {code}")),
    };
    Ok(GameAction {
        action_type,
        player: PlaceAtTable(seat),
    })
}

fn parse_pass_action(lines: &[String]) -> Result<GameAction, String> {
    if lines.len() != 4 {
        return Err(format!("pass group length must be 4, got {}", lines.len()));
    }
    let mut cards: Vec<Card> = Vec::with_capacity(4);
    let mut seat: Option<u8> = None;
    for raw in lines {
        let (s, code, val) = split_action(raw)?;
        if code != "PASS" && code != "PBCK" {
            return Err(format!("expected PASS/PBCK action, got {code}"));
        }
        match seat {
            Some(prev) if prev != s => {
                return Err(format!("pass group has mixed seats: {prev} and {s}"));
            }
            None => seat = Some(s),
            _ => {}
        }
        cards.push(parse_card_token(val)?);
    }
    cards.sort();
    cards.reverse();
    Ok(GameAction {
        action_type: ActionType::Pass(cards),
        player: PlaceAtTable(seat.unwrap_or(0)),
    })
}

fn action_matches(legal: &GameAction, target: &GameAction) -> bool {
    if legal.player != target.player {
        return false;
    }
    match (&legal.action_type, &target.action_type) {
        (ActionType::Pass(a), ActionType::Pass(b)) => {
            let mut ca = a.clone();
            let mut cb = b.clone();
            ca.sort();
            cb.sort();
            ca == cb
        }
        (a, b) => a == b,
    }
}

fn find_action_index(game: &Game, target: &GameAction) -> Option<usize> {
    game.legal_actions
        .iter()
        .position(|legal| action_matches(legal, target))
}

fn id_to_string(v: &serde_json::Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    if let Some(oid) = v.get("$oid").and_then(|x| x.as_str()) {
        return oid.to_string();
    }
    v.to_string()
}

fn build_seeded_game(record: &LegacyGameRecord) -> Result<Game, String> {
    let names: [String; 4] = record
        .players
        .clone()
        .try_into()
        .map_err(|_| format!("expected exactly 4 players, got {}", record.players.len()))?;

    let cards = [
        parse_cards(
            record
                .cards
                .get(&names[0])
                .ok_or_else(|| format!("missing cards for {}", names[0]))?,
        )?,
        parse_cards(
            record
                .cards
                .get(&names[1])
                .ok_or_else(|| format!("missing cards for {}", names[1]))?,
        )?,
        parse_cards(
            record
                .cards
                .get(&names[2])
                .ok_or_else(|| format!("missing cards for {}", names[2]))?,
        )?,
        parse_cards(
            record
                .cards
                .get(&names[3])
                .ok_or_else(|| format!("missing cards for {}", names[3]))?,
        )?,
    ];

    let mut game = Game::new(record.name.clone(), names, Some(cards));
    for _ in 0..4 {
        let action = game
            .legal_actions
            .first()
            .cloned()
            .ok_or_else(|| "missing start action".to_string())?;
        game.apply_action_mut(action);
    }
    Ok(game)
}

fn record_decision(game: &Game, action: &GameAction) -> Result<DecisionPoint, String> {
    let action_idx = find_action_index(game, action).ok_or_else(|| {
        illegal_action_error(game, action)
    })?;
    let obs = ObservationJson::from(build_observation(game, action.player.clone()));
    Ok(DecisionPoint {
        obs,
        action_idx,
        pov_parity: action.player.0 % 2,
        pov_seat: action.player.0,
    })
}

fn record_pass_pick_decisions(game: &Game, pass_action: &GameAction) -> Result<Vec<DecisionPoint>, String> {
    let pass_cards = match &pass_action.action_type {
        ActionType::Pass(cards) => cards.clone(),
        _ => return Err("record_pass_pick_decisions expects ActionType::Pass".to_string()),
    };
    if game.state.player_at_turn != pass_action.player {
        return Err(format!(
            "pass action player {:?} does not match player_at_turn {:?}",
            pass_action.player, game.state.player_at_turn
        ));
    }

    let pass_options = collect_pass_options(game);
    if pass_options.is_empty() {
        return Err("no pass options available while trying to record pass picks".to_string());
    }

    // Proxy ordering as requested: pick highest -> lowest card from chosen 4-card set.
    // parse_pass_action already normalizes pass cards into descending sorted order.
    let ordered_pick_cards: Vec<usize> = pass_cards.iter().map(card_index).collect();
    let mut selected: Vec<usize> = vec![];
    let mut decisions = Vec::with_capacity(PASS_PICK_TARGET);

    for card_idx in ordered_pick_cards {
        let legal_pick_actions = build_pick_legal_actions(&pass_options, &selected);
        if legal_pick_actions.is_empty() {
            return Err("empty pass-pick legal actions during sequential pass reconstruction".to_string());
        }
        let action_idx = legal_pick_actions
            .iter()
            .position(|la| la.card_idx == Some(card_idx))
            .ok_or_else(|| {
                let cands: Vec<String> = legal_pick_actions
                    .iter()
                    .filter_map(|la| la.card_idx)
                    .map(|c| c.to_string())
                    .collect();
                format!(
                    "chosen pass card {card_idx} not in sequential pick candidates [{}]",
                    cands.join(",")
                )
            })?;

        let mut obs = ObservationJson::from(build_observation(game, pass_action.player.clone()));
        obs.legal_actions = legal_pick_actions;
        obs.pass_selection_indices = selected.clone();
        obs.pass_selection_target = PASS_PICK_TARGET;

        decisions.push(DecisionPoint {
            obs,
            action_idx,
            pov_parity: pass_action.player.0 % 2,
            pov_seat: pass_action.player.0,
        });

        selected.push(card_idx);
        selected.sort_unstable();
        selected.dedup();
    }

    Ok(decisions)
}

fn illegal_action_error(game: &Game, action: &GameAction) -> String {
    let legal_preview: Vec<String> = game
        .legal_actions
        .iter()
        .take(8)
        .map(|a| format!("{:?}", a.action_type))
        .collect();
    format!(
        "action not legal in phase {:?}: {:?}; legal preview: {}",
        game.state.phase,
        action,
        legal_preview.join(" | ")
    )
}

fn replay_legacy_game(
    record: &LegacyGameRecord,
    allowed_players: Option<&HashSet<String>>,
) -> Result<Vec<serde_json::Value>, String> {
    let mut game = build_seeded_game(record)?;
    let mut pass_collect: Vec<String> = Vec::new();
    let mut decisions: Vec<DecisionPoint> = Vec::new();
    let mut skip_next_answer_after_ignored_question = false;

    let flush_pass_group = |game: &mut Game, pass_collect: &mut Vec<String>, decisions: &mut Vec<DecisionPoint>| -> Result<(), String> {
        if pass_collect.is_empty() {
            return Ok(());
        }
        if pass_collect.len() != 4 {
            return Err(format!(
                "incomplete pass group with {} actions",
                pass_collect.len()
            ));
        }
        let pass_action = parse_pass_action(pass_collect)?;
        let pick_decisions = record_pass_pick_decisions(game, &pass_action)?;
        decisions.extend(pick_decisions);
        game.apply_action_mut(pass_action);
        pass_collect.clear();
        Ok(())
    };

    for (idx, raw) in record.actions.iter().enumerate() {
        if pass_collect.len() == 4 {
            flush_pass_group(&mut game, &mut pass_collect, &mut decisions)?;
        }
        let (_, code, _) = split_action(raw)?;
        if skip_next_answer_after_ignored_question && code == "ANSW" {
            skip_next_answer_after_ignored_question = false;
            continue;
        }
        if code == "PASS" || code == "PBCK" {
            pass_collect.push(raw.clone());
            continue;
        }

        let mut action = parse_single_action(raw)?;
        if find_action_index(&game, &action).is_none() {
            let (seat, code, val) = split_action(raw)?;
            if code == "QUES" {
                let player = PlaceAtTable(seat);
                let mut candidates: Vec<GameAction> = Vec::new();
                if val == "you" {
                    if let Some(next_raw) = record.actions.get(idx + 1) {
                        if let Some(suit) = answer_half_suit_from_raw(next_raw) {
                            candidates.push(GameAction {
                                action_type: ActionType::Question(QuestionType::YourHalf(suit)),
                                player: player.clone(),
                            });
                        }
                    }
                    candidates.push(GameAction {
                        action_type: ActionType::Question(QuestionType::Yours),
                        player: player.clone(),
                    });
                } else if let Some(suit_char) = val.strip_prefix("ou").and_then(|s| s.chars().next()) {
                    let suit = parse_suit_char(suit_char)?;
                    candidates.push(GameAction {
                        action_type: ActionType::Question(QuestionType::YourHalf(suit)),
                        player: player.clone(),
                    });
                    if let Some(next_raw) = record.actions.get(idx + 1) {
                        if is_pair_answer_raw(next_raw) {
                            candidates.push(GameAction {
                                action_type: ActionType::Question(QuestionType::Yours),
                                player: player.clone(),
                            });
                        }
                    }
                }
                for candidate in candidates {
                    if find_action_index(&game, &candidate).is_some() {
                        action = candidate;
                        break;
                    }
                }
            }
        }
        if action.action_type == ActionType::NewBid(0) {
            continue;
        }
        if find_action_index(&game, &action).is_none() {
            if code == "QUES" {
                if let Some(next_raw) = record.actions.get(idx + 1) {
                    let (_, next_code, _) = split_action(next_raw)?;
                    skip_next_answer_after_ignored_question = next_code == "ANSW";
                }
                continue;
            }
            if code == "ANSW" {
                continue;
            }
            if matches!(action.action_type, ActionType::StopBidding) {
                continue;
            }
            return Err(illegal_action_error(&game, &action));
        }

        decisions.push(record_decision(&game, &action)?);
        game.apply_action_mut(action);
    }
    flush_pass_group(&mut game, &mut pass_collect, &mut decisions)?;

    if game.state.phase != GamePhase::Ended {
        return Err("game replay did not reach ended phase".to_string());
    }
    let outcome = GameFinishedInfo::from(game);
    let team0 = outcome.team_points[0];
    let team1 = outcome.team_points[1];
    let game_id = id_to_string(&record.id);

    let mut rows = Vec::with_capacity(decisions.len());
    for dp in decisions {
        let pov_seat_idx = usize::from(dp.pov_seat);
        let pov_player = record
            .players
            .get(pov_seat_idx)
            .cloned()
            .unwrap_or_else(|| format!("seat_{pov_seat_idx}"));
        if let Some(allowed) = allowed_players {
            if !allowed.contains(&pov_player) {
                continue;
            }
        }
        let (my_pts, opp_pts) = if dp.pov_parity == 0 {
            (team0, team1)
        } else {
            (team1, team0)
        };
        rows.push(serde_json::json!({
            "obs": dp.obs,
            "action_taken": dp.action_idx,
            "outcome_pts_my_team": my_pts,
            "outcome_pts_opp": opp_pts,
            "source": "human_legacy",
            "game_id": game_id,
            "pov_player": pov_player,
            "pov_seat": dp.pov_seat,
        }));
    }
    Ok(rows)
}

fn main() {
    let (input, output, limit, min_player_winrate) = parse_args();

    let mut file = File::open(&input).unwrap_or_else(|e| {
        panic!("failed to open {}: {e}", input.display());
    });
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", input.display());
    });

    let records: Vec<LegacyGameRecord> = match serde_json::from_str::<serde_json::Value>(&content) {
        Ok(serde_json::Value::Array(_)) => serde_json::from_str(&content).unwrap_or_else(|e| {
            panic!("failed to parse {} as JSON array: {e}", input.display());
        }),
        Ok(serde_json::Value::Object(_)) => {
            if let Ok(dataset) = serde_json::from_str::<LegacyGameDataset>(&content) {
                dataset.games
            } else if let Ok(single) = serde_json::from_str::<LegacyGameRecord>(&content) {
                vec![single]
            } else {
                panic!(
                    "failed to parse {} as JSON object. Expected either {{\"games\":[...]}} or a single game object.",
                    input.display()
                );
            }
        }
        Ok(_) => {
            panic!(
                "unsupported JSON top-level in {}. Expected array/object or NDJSON lines.",
                input.display()
            );
        }
        Err(_) => {
            // Backward-compatible fallback: one JSON object per line (NDJSON-like).
            let mut parsed = Vec::new();
            for (line_no, line) in content.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let rec: LegacyGameRecord = serde_json::from_str(trimmed).unwrap_or_else(|e| {
                    panic!(
                        "failed to parse {} line {} as JSON object: {e}",
                        input.display(),
                        line_no + 1
                    );
                });
                parsed.push(rec);
            }
            parsed
        }
    };
    let total = limit.unwrap_or(records.len()).min(records.len());
    let allowed_players: Option<HashSet<String>> = min_player_winrate.map(|thr| {
        let winrates = compute_player_winrates(&records, total);
        let mut ranked: Vec<(String, f64)> = winrates
            .iter()
            .map(|(name, wr)| (name.clone(), *wr))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let filtered: HashSet<String> = ranked
            .iter()
            .filter(|(_, wr)| *wr > thr)
            .map(|(name, _)| name.clone())
            .collect();
        eprintln!(
            "Player winrate filter enabled: threshold > {:.1}% | qualified: {} / {} players",
            thr * 100.0,
            filtered.len(),
            ranked.len()
        );
        if !ranked.is_empty() {
            let preview = ranked
                .iter()
                .take(12)
                .map(|(name, wr)| {
                    let tag = if filtered.contains(name) { "*" } else { " " };
                    format!("{tag}{name}:{:.1}%", wr * 100.0)
                })
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("Top winrates: {preview}");
        }
        filtered
    });

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            panic!("failed to create {}: {e}", parent.display());
        });
    }
    let out_file = File::create(&output).unwrap_or_else(|e| {
        panic!("failed to create {}: {e}", output.display());
    });
    let mut writer = BufWriter::new(out_file);

    let mut ok_games = 0usize;
    let mut failed_games = 0usize;
    let mut rows_written = 0usize;

    for (idx, record) in records.into_iter().take(total).enumerate() {
        match replay_legacy_game(&record, allowed_players.as_ref()) {
            Ok(rows) => {
                ok_games += 1;
                rows_written += rows.len();
                for row in rows {
                    serde_json::to_writer(&mut writer, &row).expect("failed to write row");
                    writer.write_all(b"\n").expect("failed to write newline");
                }
            }
            Err(e) => {
                failed_games += 1;
                eprintln!(
                    "[WARN] Skipping game {} (id={}): {e}",
                    idx + 1,
                    id_to_string(&record.id)
                );
            }
        }
    }
    writer.flush().expect("failed to flush output");

    eprintln!(
        "Converted {} / {} games (failed: {}) -> {} rows at {}",
        ok_games,
        total,
        failed_games,
        rows_written,
        output.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use marjapussi::game::gameevent::GameAction;
    use marjapussi::game::points::Points;

    fn build_game_in_passing_forth() -> Game {
        let names = [
            "S1".to_string(),
            "S2".to_string(),
            "S3".to_string(),
            "S4".to_string(),
        ];
        let mut game = Game::new("legacy_test".to_string(), names, None);
        let mut actions = game.legal_actions.clone();
        for _ in 0..4 {
            game = game.apply_action(actions.pop().expect("start action")).expect("start apply");
            actions = game.legal_actions.clone();
        }
        let bid140 = GameAction {
            action_type: ActionType::NewBid(140),
            player: game.state.player_at_turn.clone(),
        };
        game = game.apply_action(bid140).expect("bid apply");
        for _ in 0..4 {
            actions = game.legal_actions.clone();
            game = game.apply_action(actions[3].clone()).expect("bidding continue");
        }
        for _ in 0..3 {
            actions = game.legal_actions.clone();
            game = game.apply_action(actions[0].clone()).expect("bidding stop");
        }
        assert_eq!(game.state.value, Points(200));
        assert_eq!(game.state.phase, GamePhase::PassingForth);
        game
    }

    #[test]
    fn test_parse_card_token() {
        let c = parse_card_token("r-A").expect("parse r-A");
        assert_eq!(c.suit, Suit::Red);
        assert_eq!(c.value, Value::Ace);
    }

    #[test]
    fn test_parse_single_action_question_my_maps_to_announce() {
        let a = parse_single_action("1,QUES,mys").expect("parse action");
        assert_eq!(a.player, PlaceAtTable(1));
        assert_eq!(a.action_type, ActionType::AnnounceTrump(Suit::Bells));
    }

    #[test]
    fn test_parse_pass_action_group() {
        let group = vec![
            "2,PASS,r-A".to_string(),
            "2,PASS,e-7".to_string(),
            "2,PASS,g-K".to_string(),
            "2,PASS,s-9".to_string(),
        ];
        let action = parse_pass_action(&group).expect("parse pass group");
        assert_eq!(action.player, PlaceAtTable(2));
        match action.action_type {
            ActionType::Pass(cards) => assert_eq!(cards.len(), 4),
            _ => panic!("expected pass action"),
        }
    }

    #[test]
    fn test_record_pass_pick_decisions_builds_four_sequential_picks() {
        let game = build_game_in_passing_forth();
        let pass_action = game.legal_actions[0].clone();
        let rows = record_pass_pick_decisions(&game, &pass_action).expect("record pass picks");
        assert_eq!(rows.len(), PASS_PICK_TARGET);
        for (idx, dp) in rows.iter().enumerate() {
            assert_eq!(dp.obs.pass_selection_indices.len(), idx);
            assert_eq!(dp.obs.pass_selection_target, PASS_PICK_TARGET);
            assert!(
                dp.obs.legal_actions.iter().all(|la| la.action_token == 52),
                "all legal actions should be sequential pass pick token"
            );
            assert!(
                dp.action_idx < dp.obs.legal_actions.len(),
                "action_idx should point into synthetic pick legal actions"
            );
        }
    }

    #[test]
    fn test_compute_player_winrates_basic() {
        let rec_a = LegacyGameRecord {
            id: serde_json::json!({"$oid":"a"}),
            name: "a".to_string(),
            players: vec![
                "P0".to_string(),
                "P1".to_string(),
                "P2".to_string(),
                "P3".to_string(),
            ],
            cards: HashMap::new(),
            actions: vec![],
            players_points: HashMap::from([
                ("P0".to_string(), 100),
                ("P1".to_string(), 40),
                ("P2".to_string(), 80),
                ("P3".to_string(), 20),
            ]),
        };
        let rec_b = LegacyGameRecord {
            id: serde_json::json!({"$oid":"b"}),
            name: "b".to_string(),
            players: vec![
                "P0".to_string(),
                "P1".to_string(),
                "P2".to_string(),
                "P3".to_string(),
            ],
            cards: HashMap::new(),
            actions: vec![],
            players_points: HashMap::from([
                ("P0".to_string(), 20),
                ("P1".to_string(), 110),
                ("P2".to_string(), 10),
                ("P3".to_string(), 90),
            ]),
        };
        let wr = compute_player_winrates(&[rec_a, rec_b], 2);
        assert_eq!(wr.get("P0").copied(), Some(0.5));
        assert_eq!(wr.get("P1").copied(), Some(0.5));
        assert_eq!(wr.get("P2").copied(), Some(0.5));
        assert_eq!(wr.get("P3").copied(), Some(0.5));
    }

    #[test]
    fn test_compute_player_winrates_tie_counts_as_half() {
        let rec = LegacyGameRecord {
            id: serde_json::json!({"$oid":"c"}),
            name: "c".to_string(),
            players: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            cards: HashMap::new(),
            actions: vec![],
            players_points: HashMap::from([
                ("A".to_string(), 50),
                ("B".to_string(), 70),
                ("C".to_string(), 50),
                ("D".to_string(), 30),
            ]),
        };
        let wr = compute_player_winrates(&[rec], 1);
        assert_eq!(wr.get("A").copied(), Some(0.5));
        assert_eq!(wr.get("B").copied(), Some(0.5));
        assert_eq!(wr.get("C").copied(), Some(0.5));
        assert_eq!(wr.get("D").copied(), Some(0.5));
    }
}
