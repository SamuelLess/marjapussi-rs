# MarjaPussi

Rust implementation of Marjapussi with a rules engine, ML tooling, and local UI support.

Rule reference:

1. [marjapussi.de rules](https://marjapussi.de/rules) (primary reference)
2. [Wurzel e. V. rules](http://wurzel.org/pussi/indexba7e.html?seite=regeln) (secondary reference)

## Project Vision

This project aims to provide a fair Marjapussi AI stack that can:

1. Play against humans with just a player observation interface.
2. Play self-play matches for training and evaluation.
3. Expose multiple difficulty levels through checkpointed models.
4. Continue training from human game logs, self-play, or mixed strategies.

## Repository Contents

1. Core Rust game engine under `src/game`.
2. ML interface and server code under `src/ml`.
3. Python training/inference code under `ml/`.
4. Local browser UI under `ml/ui/`.

## Documentation

1. Project overview: `docs/ML_PROJECT_OVERVIEW.md`
2. ML spec index: `docs/ml/README.md`
3. Existing ML planning notes: `ml/README.md`

If there is a conflict, `docs/ML_PROJECT_OVERVIEW.md` and `docs/ml/*` are the canonical implementation contracts.

## Usage

Current utility binaries include:

1. `interactive`: play in terminal with full information.
2. `parse_legacy`: parse legacy game export format.
3. `ml_server`: JSON interface for the ML pipeline.
4. `ml_generate`: generate training datasets from simulated games.

Optimized local ML setup (Git Bash):

1. `just setup-ml`
2. `just ui`
3. `just train-65k`

## License

This project is licensed under the GPL-3.0 License. See [LICENSE](LICENSE).

## Contributing

Please open issues or pull requests with clear reproduction steps and expected behavior.
