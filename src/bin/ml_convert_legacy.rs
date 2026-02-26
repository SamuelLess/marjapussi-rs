use std::collections::HashMap;
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
use marjapussi::ml::observation::{build_observation, ObservationJson};

#[derive(Debug, Deserialize)]
struct LegacyGameRecord {
    #[serde(rename = "_id")]
    id: serde_json::Value,
    name: String,
    players: Vec<String>,
    cards: HashMap<String, Vec<String>>,
    actions: Vec<String>,
}

#[derive(Debug)]
struct DecisionPoint {
    obs: ObservationJson,
    action_idx: usize,
    pov_parity: u8,
}

fn parse_args() -> (PathBuf, PathBuf, Option<usize>) {
    let matches = Command::new("ml_convert_legacy")
        .about("Convert legacy full-game logs to decision-point NDJSON for supervised pretraining.")
        .arg(
            Arg::new("input")
                .long("input")
                .default_value("ml/dataset/games.json")
                .help("Path to legacy JSON file (top-level list)."),
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
    (input, output, limit)
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
    })?;
    let obs = ObservationJson::from(build_observation(game, action.player.clone()));
    Ok(DecisionPoint {
        obs,
        action_idx,
        pov_parity: action.player.0 % 2,
    })
}

fn replay_legacy_game(record: &LegacyGameRecord) -> Result<Vec<serde_json::Value>, String> {
    let mut game = build_seeded_game(record)?;
    let mut pass_collect: Vec<String> = Vec::new();
    let mut decisions: Vec<DecisionPoint> = Vec::new();

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
        decisions.push(record_decision(game, &pass_action)?);
        game.apply_action_mut(pass_action);
        pass_collect.clear();
        Ok(())
    };

    for raw in &record.actions {
        if pass_collect.len() == 4 {
            flush_pass_group(&mut game, &mut pass_collect, &mut decisions)?;
        }
        let (_, code, _) = split_action(raw)?;
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
                if let Some(suit_char) = val.strip_prefix("my").and_then(|s| s.chars().next()) {
                    let suit = parse_suit_char(suit_char)?;
                    candidates.push(GameAction {
                        action_type: ActionType::Question(QuestionType::YourHalf(suit)),
                        player: player.clone(),
                    });
                }
                candidates.push(GameAction {
                    action_type: ActionType::Question(QuestionType::Yours),
                    player,
                });
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
        }));
    }
    Ok(rows)
}

fn main() {
    let (input, output, limit) = parse_args();

    let mut file = File::open(&input).unwrap_or_else(|e| {
        panic!("failed to open {}: {e}", input.display());
    });
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", input.display());
    });

    let records: Vec<LegacyGameRecord> = if content.trim_start().starts_with('[') {
        serde_json::from_str(&content).unwrap_or_else(|e| {
            panic!("failed to parse {} as JSON array: {e}", input.display());
        })
    } else {
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
    };
    let total = limit.unwrap_or(records.len()).min(records.len());

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
        match replay_legacy_game(&record) {
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
}
