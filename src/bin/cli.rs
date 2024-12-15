use std::fs::File;
use std::io::{Read, Write};

use clap::{Arg, Command};
use indicatif::ProgressIterator;
use serde_json;

use marjapussi::game::parse::parse_legacy_format;
use marjapussi::game::parse::LegacyGameFormat;

fn main() {
    let matches = Command::new("Game Parser")
        .version("0.1")
        .author("Your Name")
        .about("Parses a file in legacy JSON format with top level list")
        .arg(
            Arg::new("filename")
                .help("The JSON file to parse")
                .required(true)
                .index(1),
        )
        .get_matches();

    if let Some(filename) = matches.get_one::<String>("filename") {
        let mut file = File::open(filename).expect("File not found");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Error reading the file");

        let new_format_json = to_new_format(contents);

        // save to file
        let new_filename = format!("new-{}", filename);
        let mut new_file = File::create(new_filename).expect("Error creating the new file");
        new_file
            .write_all(new_format_json.as_bytes())
            .expect("Error writing the new file");
    }
}

fn to_new_format(contents: String) -> String {
    let game_data: Vec<LegacyGameFormat> = serde_json::from_str(contents.as_str()).unwrap();
    let mut converted = vec![];
    for game in game_data.into_iter().progress() {
        converted.push(parse_legacy_format(game).unwrap());
    }
    serde_json::to_string_pretty(&converted).expect("Error serializing the new format")
}
