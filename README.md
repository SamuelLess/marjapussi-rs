# MarjaPussi

Rust implementation of MarjaPussi, mostly following the rules
from [Wurzel e. V.](http://wurzel.org/pussi/indexba7e.html?seite=regeln), exactly following the rules
on [marjapussi.de](https://marjapussi.de/rules).

This package is hopefully going to be used for the backend of [marjapussi.de](https://marjapussi.de).

The library manages game phases, legal actions, card exchanges, tricks and scoring.
It provides the rules, not a strategy for choosing moves.

## Usage

1. Create a `marjapussi::game::Game` with `Game::new`; provide hands or let it deal randomly.
2. Get the current choices with `game.legal_actions()` and choose an action.
3. Call `game.apply_action(action)` for a new game or an error. To update in place,
   use `game.apply_action_mut(action)`; it panics if the action is illegal. Repeat until `game.ended()`.

`Game` contains every player's hand and the full event history. With the default build,
`GameInfoPlayer::from_game(game, seat)` creates a private player view: it hides other
starting hands and passed-card events, and includes only that seat's legal actions.
The seat's own hand appears after the game starts.

### Build modes

The default `public-api` feature keeps the public `Vec`/`String` types, cached legal
actions, player views, parser/series modules and utility binaries.

Use `cargo build --no-default-features` for lean simulations. This changes source-level
types: bounded lists use `InlineVec`, names and metadata use shared `Arc` storage,
and passes use `[Card; 4]`. Legal actions are computed on demand instead of stored
in a public field. Player views, parser/series modules, some helpers and the binaries
are unavailable. No wall-clock timestamps are recorded: event times are `0` and
metadata times are empty or unset.

Both modes retain event and bidding histories and a snapshot for supported undo
actions. Lean mode still uses heap storage; it is not allocation-free.

### Utilities (default build)

- `interactive` for playing the game in the terminal with full information against yourself.
  Run `cargo run --bin interactive`; its source also shows the action loop.
- `parse` for converting a JSON array of games from the format used by the Python
  implementation [here](https://github.com/SamuelLess/marjapussi).
  Run `cargo run --bin parse -- games.json`; it writes `new-games.json`.

## License

This project is licensed under the GPL-3.0 License – see the [LICENSE](LICENSE) file for details.

## Contributing

Please do not hesitate to reach out to [me](mailto:samuel@lessmann.dev).
There are a lot of undocumented future plans I would love to discuss with you.
