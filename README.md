# ♟ Chess Game Review Agent

A Chess.com-style post-game analysis tool built with Python, powered by Stockfish.

## Features
- **PGN Import** — Load games from `.pgn` files or paste PGN text directly
- **Move Classification** — Categorizes every move as Brilliant, Great, Best, Excellent, Good, Inaccuracy, Mistake, or Blunder
- **Accuracy Score** — CAPS-style accuracy percentage (0–100%) using per-move win-probability loss
- **Elo Estimation** — Predicts your approximate rating based on move quality
- **Key Moments** — Highlights your best and worst moves with CPL breakdown
- **Live Progress** — Real-time progress indicator during analysis

## Requirements
- Python 3.10+
- [python-chess](https://pypi.org/project/python-chess/) (`pip install chess`)
- [Stockfish](https://stockfishchess.org/download/) binary (update `STOCKFISH_PATH` in the script)

## Usage
```bash
python chess_review_agent.py game.pgn          # analyse from file
python chess_review_agent.py game.pgn white    # specify your color
python chess_review_agent.py                   # paste PGN interactively
```

## How It Works
The agent replays the game move-by-move, evaluating each position with Stockfish at depth 18. It computes the centipawn loss (CPL) for every move, classifies them using Chess.com's thresholds, and generates a full performance report with accuracy and estimated Elo.
