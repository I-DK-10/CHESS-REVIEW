# Chess Game Review Agent

A Chess.com-style post-game analysis tool built with Python, powered by Stockfish's UCI engine and WDL (Win/Draw/Loss) evaluation model.

---

## Features

- **Flexible PGN Ingestion** — Analyze games by passing a `.pgn` file path or pasting multi-line PGN notation straight into the terminal.
- **WDL Win-Probability Evaluation** — Moves are evaluated based on **Win Probability (WP) Loss**, accounting for positional context and game phase rather than raw centipawn loss.
- **Chess.com Move Classifications** — Automatically labels every move as **Brilliant (`!!`)**, **Great (`!`)**, **Best (`★`)**, **Excellent (`✓`)**, **Good (`+`)**, **Inaccuracy (`?!`)**, **Mistake (`?`)**, or **Blunder (`??`)**.
- **Sacrifice & Only-Move Detection** — Multi-PV analysis identifies true sacrifices and only-good-moves to reward Brilliant and Great moves.
- **CAPS2-Style Accuracy Scoring** — Computes a per-move accuracy percentage (0–100%) reflecting how closely your play preserved winning chances.
- **Elo Estimation** — Predicts performance rating via calibrated interpolation benchmarks against real player games.
- **Terminal Dashboard** — Color-coded per-move breakdown table, visual category distribution bar charts, and a key-moments highlight reel.
- **Optimized Analysis** — Caches board evaluations between plies, cutting required engine operations in half.

---

## Move Classification System

| Category | Symbol | Description | WP Loss Threshold |
| :--- | :---: | :--- | :---: |
| **Brilliant** | `!!` | Genuine piece sacrifice (Queen sac leading to mate/win, or sound Rook/Minor sac) | $\le 2\%$ |
| **Great Move** | `!` | Non-recapture move in contested position maintaining advantage (or sound sac) | $\le 2\%$ |
| **Best Move** | `★` | Top engine move or virtually equal continuation | $\le 2\%$ |
| **Excellent** | `✓` | Very strong move, preserves advantage | $\le 5\%$ |
| **Good** | `+` | Solid move, slight drop in winning chances | $\le 9\%$ |
| **Inaccuracy** | `?!` | Sub-optimal move that gives away some edge | $\le 18\%$ |
| **Mistake** | `?` | Clear error (~1–2 pawns lost, $\text{CP Loss} < 220$, or when position remains winning) | $\le 35\%$ |
| **Blunder** | `??` | Critical blunder losing substantial material ($\text{CP Loss} \ge 220$ AND $\text{WP}_{\text{after}} < 65\%$) | $> 35\%$ |

---

## Prerequisites & Setup

### 1. Requirements
- **Python 3.10+**
- **python-chess**:
  ```bash
  pip install chess
  ```
- **Stockfish Binary**:
  Download a modern Stockfish release from [stockfishchess.org](https://stockfishchess.org/download/).

### 2. Configuration
Open [`chess_review_agent.py`](chess_review_agent.py) and ensure `STOCKFISH_PATH` points to your Stockfish executable:

```python
STOCKFISH_PATH: str = r"C:\Users\Dhruv k\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe"
```

Adjust analysis settings if needed:
- `ANALYSIS_DEPTH = 18` (Default depth for engine search)
- `MULTIPV = 2` (Required for Brilliant & Great move detection)

---

## Usage

### 1. Analyze from a PGN File
```bash
python chess_review_agent.py game.pgn
```

### 2. Specify Your Color via CLI
```bash
python chess_review_agent.py game.pgn white
# or
python chess_review_agent.py game.pgn black
```

### 3. Interactive Paste Mode
Run the agent without arguments to paste raw PGN text directly:
```bash
python chess_review_agent.py
```
> Paste your PGN text into the terminal and type `END` on a new line (or press Enter twice) to begin analysis.

---

## Sample Report Output

```text
══════════════════════════════════════════════════════════════════
  CHESS GAME REVIEW AGENT
══════════════════════════════════════════════════════════════════
  Event  : Live Chess - chess.com
  Date   : 2026.09.04
  White  : Player1 (1520)  vs  Black: Player2 (1490)
  Result : 1-0
  You played as: White
──────────────────────────────────────────────────────────────────

     #  Move      Best       WP Loss    Win%  Category
  ──────────────────────────────────────────────────────────────────
    1W  e4        e4            0.0%     53%  ★   Best Move
    2W  Nf3       Nf3           0.0%     54%  ★   Best Move
    3W  Bc4       Bc4           0.5%     54%  ✓   Excellent
    4W  d3        c3            2.8%     51%  +   Good
    5W  Bxf7+     Bxf7+         0.0%     68%  !!  Brilliant
    6W  Ng5+      Ng5+          0.0%     68%  !   Great Move
    ...

──────────────────────────────────────────────────────────────────
  PERFORMANCE SUMMARY
──────────────────────────────────────────────────────────────────
  Accuracy     :  84.2%
  Estimated Elo:  ~1960

  Moves analyzed :  32
  Avg WP Loss    :  3.4%

  !! Brilliant                         1  ■
  !  Great Move                        2  ■■
  ★  Best Move                        18  ■■■■■■■■■■■■■■■■■■
  ✓  Excellent                         6  ■■■■■■
  +  Good                              3  ■■■
  ?! Inaccuracy                        1  ■
  ?  Mistake                           1  ■

══════════════════════════════════════════════════════════════════

  KEY MOMENTS
──────────────────────────────────────────────────────────────────
  Move 5W:  Bxf7+  [!! Brilliant]  WP Loss=0.0%
  Move 6W:   Ng5+  [! Great Move]  WP Loss=0.0%
  Move 14W:  Qd2   [? Mistake]     WP Loss=16.2%  (best: Re1)
```

---

## How It Works

1. **Move-by-Move Evaluation**: Replays game positions using `python-chess` and sends UCI analysis requests to Stockfish with `UCI_ShowWDL = True`.
2. **Win Expectation**: Converts engine evaluations to winning probability using:
   $$\text{Win Expectation} = \frac{\text{wins} + \frac{\text{draws}}{2}}{\text{wins} + \text{draws} + \text{losses}}$$
   This is phase-aware (middlegame vs. endgame), preventing the distortions common in centipawn evaluation.
3. **Exponential Accuracy Mapping (CAPS2)**: Per-move accuracy is scored using Chess.com's calibrated exponential expectation curve:
   $$\text{Accuracy}_{\text{move}} = \max\left(0,\, \min\left(100,\, 103.1668 \times e^{-3.8 \times \text{WP Loss}} - 3.1668\right)\right)$$
4. **Elo Calibration**: Maps the mean accuracy to estimated ratings using empirical anchor points from Chess.com game data.

For a detailed walkthrough of all internal functions, formulas, and codebase comments, see [CODE_EXPLANATION.md](CODE_EXPLANATION.md).
