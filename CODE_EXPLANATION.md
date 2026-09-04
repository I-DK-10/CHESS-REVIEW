# ♟ Chess Review Agent — Code & Function Guide

This document provides a comprehensive breakdown of the core functions, data structures, and algorithms implemented in [`chess_review_agent.py`](file:///c:/Users/Dhruv%20k/Downloads/CHESS-REVIEW/chess_review_agent.py), along with the rationale and detailed comments included in the codebase.

---

## Table of Contents
1. [Overview & Architectural Concept](#1-overview--architectural-concept)
2. [Data Structures & Enums](#2-data-structures--enums)
3. [Engine Evaluation & Win Probability (WDL)](#3-engine-evaluation--win-probability-wdl)
4. [Move Classification Logic](#4-move-classification-logic)
5. [Accuracy & Elo Calculation](#5-accuracy--elo-calculation)
6. [PGN Ingestion & Game Setup](#6-pgn-ingestion--game-setup)
7. [Reporting & Presentation](#7-reporting--presentation)
8. [CLI Entry Point](#8-cli-entry-point)

---

## 1. Overview & Architectural Concept

The agent performs post-game analysis on chess games using Python (`python-chess`) and Stockfish.

Instead of relying solely on raw Centipawn Loss (CPL), which is non-linear and context-blind (e.g. losing 50cp when at +8.0 is negligible, but losing 50cp at +0.2 completely alters the game), the engine uses Stockfish's **WDL (Win/Draw/Loss)** model:
- Converts position evaluations into **Win Probability (0.0 to 1.0)**.
- Accounts for the **game phase (ply count)** and position dynamics.
- Evaluates moves based on **Win Probability Loss (`wp_loss`)**.

---

## 2. Data Structures & Enums

### `MoveCategory(Enum)`
Defines the Chess.com-style classifications, symbols, labels, and WP loss threshold criteria:

| Category | Symbol | WP Loss Condition |
| :--- | :---: | :--- |
| **Brilliant** | `!!` | WP Loss ≤ 0.02, sacrifices material, best move, and 2nd-best move has > 0.10 WP gap |
| **Great Move** | `!` | WP Loss ≤ 0.02, sacrifice OR only viable move (> 0.10 WP gap to 2nd-best) |
| **Best Move** | `★` | WP Loss ≤ 0.02 (≤ 2% win chance loss) |
| **Excellent** | `✓` | 0.02 < WP Loss ≤ 0.05 (2% – 5%) |
| **Good** | `+` | 0.05 < WP Loss ≤ 0.08 (5% – 8%) |
| **Inaccuracy** | `?!` | 0.08 < WP Loss ≤ 0.15 (8% – 15%) |
| **Mistake** | `?` | 0.15 < WP Loss ≤ 0.25 (15% – 25%) |
| **Blunder** | `??` | WP Loss > 0.25 (> 25% drop in winning chances) |

---

### `MoveAnalysis` (Dataclass)
Stores the full engine evaluation and classification for an individual half-move (ply).

```python
@dataclass
class MoveAnalysis:
    move_number: int  # Full move number (e.g. 1, 2, 14)
    color: chess.Color  # chess.WHITE or chess.BLACK
    san: str  # Standard Algebraic Notation (e.g. "Nf3")
    eval_before: Optional[float]  # Centipawns before move (from White POV)
    eval_after: Optional[float]  # Centipawns after move (from White POV)
    wp_before: Optional[float]  # Win probability before (0.0 – 1.0 from mover's POV)
    wp_after: Optional[float]  # Win probability after (0.0 – 1.0 from mover's POV)
    wp_loss: float  # max(0.0, wp_before - wp_after)
    best_san: str  # Engine's recommended best move in SAN
    category: MoveCategory  # Assigned classification
    is_user_move: bool  # True if played by the user being reviewed
```

---

### `GameReport` (Dataclass)
Aggregates game-wide review statistics:

```python
@dataclass
class GameReport:
    pgn_headers: dict[str, str]  # PGN header metadata (Event, Date, Players, Result)
    user_color: chess.Color  # Color played by the user
    move_analyses: list[MoveAnalysis]  # List of all analyzed half-moves
    accuracy: float = 0.0  # Overall accuracy score (0.0 – 100.0)
    estimated_elo: int = 0  # Estimated rating based on move quality
    category_counts: dict[str, int] = field(default_factory=dict)  # Frequency per category
```

---

## 3. Engine Evaluation & Win Probability (WDL)

### `_score_to_cp(score, pov) -> Optional[float]`
**Purpose:** Normalizes a Stockfish `PovScore` into centipawns from the perspective of the specified player (`pov`).

- **Forced Mates:** If a forced checkmate is found, returns `±10,000` (positive if delivering mate, negative if getting mated).
- **Standard Positions:** Returns `float(relative.score())`.

---

### `_score_to_win_prob(score, pov, ply) -> float`
**Purpose:** Maps the engine score to a real win probability ($0.0$ to $1.0$) using Stockfish's WDL model.

```
expectation = (wins + draws / 2) / (wins + draws + losses)
```

**Why this matters (from comments in code):**
> *"The WDL model maps the engine's internal evaluation to Win/Draw/Loss probabilities based on large-scale engine self-play data. This is position-aware: the same centipawn value produces different win probabilities in the middlegame vs. endgame, which is exactly why WP loss is more accurate than raw CPL for move classification."*

- If the position has a forced mate: returns `1.0` if winning, `0.0` if losing.
- Calls `relative.wdl(ply=ply).expectation()`, using the current `ply` to adjust for game phase.

---

### `analyse_game(game, user_color, engine, depth=ANALYSIS_DEPTH) -> list[MoveAnalysis]`
**Purpose:** Core analysis loop that steps through the entire game move by move and runs Stockfish.

**Key Features & Optimizations:**
1. **Engine WDL Configuration:** Enables `UCI_ShowWDL` on the Stockfish process.
2. **Multi-PV (`multipv=2`):** Queries the top 2 engine lines per position to evaluate both the best move and the gap to the 2nd-best move (required for Brilliant and Great move detection).
3. **Evaluation Caching:**
   - The position *after* move $N$ is identical to the position *before* move $N+1$.
   - The loop reuses the cached analysis from the previous ply, cutting expensive engine evaluations almost in half.
4. **Movers Perspective:** Win probabilities before and after the move are evaluated from the perspective of the side who made the move (`side_to_move`).
5. **Progress Feedback:** Emits an in-place terminal progress indicator showing current ply, percentage, and move notation.

---

## 4. Move Classification Logic

### `_classify_by_wp_loss(wp_loss: float) -> MoveCategory`
**Purpose:** Assigns baseline categories using threshold lookups:
- `wp_loss <= 0.02` $\rightarrow$ **Best Move**
- `wp_loss <= 0.05` $\rightarrow$ **Excellent**
- `wp_loss <= 0.08` $\rightarrow$ **Good**
- `wp_loss <= 0.15` $\rightarrow$ **Inaccuracy**
- `wp_loss <= 0.25` $\rightarrow$ **Mistake**
- `wp_loss > 0.25` $\rightarrow$ **Blunder**

---

### `_is_sacrifice(board_before: chess.Board, move: chess.Move) -> bool`
**Purpose:** Detects whether a move immediately sacrifices material (a key prerequisite for a Brilliant move).

- Evaluates piece values:
  - Pawn = 100, Knight = 320, Bishop = 330, Rook = 500, Queen = 900, King = 0.
- Handles regular captures, quiet sacrifices (piece moved to an empty defended square), and en-passant captures.
- Returns `True` if `moving_value > captured_value`.

---

### `classify_move(wp_loss, board_before, move, best_move, second_best_wp_loss) -> MoveCategory`
**Purpose:** Applies contextual classification for exceptional moves:

- **Brilliant (`!!`)**:
  - Move must match the engine's best move (`move == best_move`).
  - Move must be a material sacrifice (`_is_sacrifice == True`).
  - Must be the only viable move: the 2nd-best move has a win probability drop greater than 10% (`second_best_wp_loss > 0.10`).
- **Great Move (`!`):**
  - Best move that is either a sacrifice OR the only good move (without meeting both criteria for Brilliant).
- **Fallback:**
  - If not Brilliant or Great, returns the category from `_classify_by_wp_loss(wp_loss)`.

---

## 5. Accuracy & Elo Calculation

### `compute_accuracy(user_moves: list[MoveAnalysis]) -> float`
**Purpose:** Computes overall game accuracy using a CAPS2-style win-probability loss formula.

**Formula per move:**
$$\text{Move Accuracy} = \max\left(0,\, 100 \times \left(1 - \frac{\text{wp\_loss}}{\text{WP\_LOSS\_CAP}}\right)\right)$$
- `WP_LOSS_CAP = 0.50` (losing 50% or more win probability on a single move results in 0% accuracy for that move).
- Overall accuracy is the arithmetic mean of all per-move accuracies for the user.

**Design Note (from comments in code):**
> *"Because WP loss is already non-linear (it accounts for game phase and evaluation magnitude), a simple linear mapping works well and produces scores that closely match Chess.com's accuracy numbers."*

---

### `estimate_elo(accuracy: float) -> int`
**Purpose:** Estimates player rating from game accuracy using piecewise linear interpolation between calibrated anchors based on Chess.com benchmarks:

| Accuracy Range | Target Elo Rating | Category |
| :---: | :---: | :--- |
| **98.0% – 100%** | 3000 – 3200 | Super-GM / Engine-like |
| **95.0% – 98.0%** | 2700 – 3000 | Grandmaster |
| **90.0% – 95.0%** | 2300 – 2700 | IM / FM |
| **85.0% – 90.0%** | 2000 – 2300 | Strong Club Player |
| **80.0% – 85.0%** | 1750 – 2000 | Club Player |
| **70.0% – 80.0%** | 1400 – 1750 | Intermediate |
| **60.0% – 70.0%** | 1100 – 1400 | Casual Player |
| **50.0% – 60.0%** | 850 – 1100 | Developing Beginner |
| **40.0% – 50.0%** | 600 – 850 | Novice |
| **< 40.0%** | 400 – 600 | Floor |

---

## 6. PGN Ingestion & Game Setup

### `load_game(source: str) -> chess.pgn.Game`
**Purpose:** Ingests PGN from either a filesystem path or raw pasted PGN text.
- Inspects whether `source` is a valid file path.
- Handles Windows OS error catches for strings containing non-path characters (e.g. brackets, newlines).
- Parses game via `chess.pgn.read_game()` and raises descriptive exceptions if parsing fails.

---

### `detect_user_color(game: chess.pgn.Game, hint: Optional[str]) -> chess.Color`
**Purpose:** Determines which player's moves are being analyzed.
- Accepts explicit CLI arguments (`"white"`, `"w"`, `"black"`, `"b"`).
- Displays player names found in PGN headers (`White` vs `Black`) and prompts the user interactively if no hint was provided.

---

## 7. Reporting & Presentation

### `build_report(game, analyses, user_color) -> GameReport`
**Purpose:** Compiles the raw analyses into a `GameReport` structure, computing accuracy, estimated rating, and counting moves across all categories.

---

### `print_report(report: GameReport) -> None`
**Purpose:** Formats and prints the terminal report dashboard.
- **Match Metadata Banner:** Event, Date, Players, Result, User Color.
- **Per-Move Analysis Table:** Move number, Played SAN, Best SAN, WP Loss %, Win% after move, and colored Move Category.
- **Performance Summary Dashboard:** Overall Accuracy %, Estimated Elo, Total Moves Analyzed, Average WP Loss %.
- **Move Breakdown Bar Chart:** Visual frequency bars for each category using Unicode block characters (`■`) with ANSI color coding.
- **Key Moments / Highlights:** Quick summary of all Brilliant, Great, Mistake, and Blunder moves.

---

## 8. CLI Entry Point

### `main() -> None`
**Purpose:** Coordinates the complete execution lifecycle.
1. Reconfigures `sys.stdout` to `utf-8` on Windows to cleanly output chess symbols (`★`, `✓`, `■`).
2. Collects PGN either via CLI arguments or interactive multi-line terminal paste (supporting `END` or double-Enter delimiters).
3. Verifies Stockfish executable existence at `STOCKFISH_PATH`.
4. Spawns Stockfish via UCI engine protocol (`chess.engine.SimpleEngine.popen_uci`).
5. Executes `analyse_game()`, generates `GameReport`, and prints output.
