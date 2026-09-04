"""
chess_review_agent.py
─────────────────────────────────────────────────────────────────────────────
A Chess.com-style post-game analysis agent built with python-chess + Stockfish.

Uses Stockfish's WDL (Win/Draw/Loss) model to compute Win Probability loss
for each move, providing more accurate classifications than raw centipawn loss.
"""

from __future__ import annotations

import io
import math
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn


STOCKFISH_PATH: str = r"C:\Users\Dhruv k\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe"


ANALYSIS_DEPTH: int = 18

# Number of alternative moves Stockfish returns per position (multipv).
# We need ≥2 to detect "Brilliant" sacrifices vs. second-best alternatives.
MULTIPV: int = 2



class MoveCategory(Enum):
    """
    Chess.com-style move categories, ordered from best to worst.

    Classification uses Win Probability (WP) loss derived from Stockfish's
    WDL model.  WP loss measures how much your winning chances dropped —
    this is position-aware, unlike raw centipawn loss.

    ┌─────────────────┬─────────────┬───────────────────────────────────────┐
    │ Category        │ Symbol      │ WP Loss condition                     │
    ├─────────────────┼─────────────┼───────────────────────────────────────┤
    │ Brilliant       │ !!          │ WP≤0.02 AND material sacrifice AND    │
    │                 │             │ best move AND 2nd-best >0.10 WP worse │
    │ Great Move      │ !           │ WP≤0.02 AND sacrifice or only move    │
    │ Best Move       │  ★          │ WP Loss ≤ 0.02                        │
    │ Excellent       │  ✓          │ 0.02 < WP Loss ≤ 0.05                │
    │ Good            │  +          │ 0.05 < WP Loss ≤ 0.08                │
    │ Inaccuracy      │ ?!          │ 0.08 < WP Loss ≤ 0.15                │
    │ Mistake         │  ?          │ 0.15 < WP Loss ≤ 0.25                │
    │ Blunder         │ ??          │ WP Loss > 0.25                        │
    └─────────────────┴─────────────┴───────────────────────────────────────┘

    """
    BRILLIANT   = ("!!", "Brilliant")
    GREAT       = ("!",  "Great Move")
    BEST        = ("★",  "Best Move")
    EXCELLENT   = ("✓",  "Excellent")
    GOOD        = ("+",  "Good")
    INACCURACY  = ("?!", "Inaccuracy")
    MISTAKE     = ("?",  "Mistake")
    BLUNDER     = ("??", "Blunder")

    def __init__(self, symbol: str, label: str) -> None:
        self.symbol = symbol
        self.label  = label


# Win Probability Loss upper-bounds for non-contextual categories.
# Brilliant and Great are determined separately via positional logic.
# Thresholds based on Chess.com's CAPS2 expected-points model.
WP_LOSS_THRESHOLDS: list[tuple[float, MoveCategory]] = [
    (0.02,  MoveCategory.BEST),
    (0.05,  MoveCategory.EXCELLENT),
    (0.08,  MoveCategory.GOOD),
    (0.15,  MoveCategory.INACCURACY),
    (0.25,  MoveCategory.MISTAKE),
]
# Anything above 0.25 WP loss = BLUNDER


def _classify_by_wp_loss(wp_loss: float) -> MoveCategory:
    """Return the base category purely from win-probability loss."""
    for threshold, category in WP_LOSS_THRESHOLDS:
        if wp_loss <= threshold:
            return category
    return MoveCategory.BLUNDER


def _is_sacrifice(board_before: chess.Board, move: chess.Move) -> bool:
    """
    Detect whether a move gives up material immediately (capture that loses
    a piece, or a quiet piece drop).  Used as part of Brilliant detection.

    We compare the value of the piece being moved to the value of any
    captured piece.  If the moved piece is worth more than what it takes,
    the move is a sacrifice.
    """
    PIECE_VALUES = {
        chess.PAWN:   100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK:   500,
        chess.QUEEN:  900,
        chess.KING:   0,    # King moves are never sacrifices
    }
    moving_piece = board_before.piece_at(move.from_square)
    if moving_piece is None:
        return False

    captured_piece = board_before.piece_at(move.to_square)
    moving_value   = PIECE_VALUES.get(moving_piece.piece_type, 0)

    # En-passant capture
    if board_before.is_en_passant(move):
        captured_value = PIECE_VALUES[chess.PAWN]
    elif captured_piece is not None:
        captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
    else:
        captured_value = 0

    # It's a sacrifice if we give up more than we take (or give up something
    # for nothing: e.g., queen sac to a square)
    return moving_value > captured_value


def classify_move(
    wp_loss: float,
    board_before: chess.Board,
    move: chess.Move,
    best_move: chess.Move,
    second_best_wp_loss: Optional[float],
) -> MoveCategory:
    """
    Full classification logic, including Brilliant and Great Move detection.

    Parameters
    ----------
    wp_loss              : win-probability loss of the played move (always ≥ 0)
    board_before         : position before the move was made
    move                 : the move that was actually played
    best_move            : engine's top choice for this position
    second_best_wp_loss  : WP loss for the second-best engine move (None if unavailable)
    """
    base = _classify_by_wp_loss(wp_loss)

    if base is MoveCategory.BEST:
        if move == best_move:
            is_sac = _is_sacrifice(board_before, move)
            # The second-best move must be significantly worse (>10% WP gap)
            # for the move to be considered the "only good move"
            is_only_good_move = (
                second_best_wp_loss is not None and second_best_wp_loss > 0.10
            )

            # In Chess.com, Brilliant moves are very rare.
            # A move is Brilliant if it's a sacrifice AND the only good continuation.
            if is_sac and is_only_good_move:
                return MoveCategory.BRILLIANT

            # If it's the only good move (but not a sac), OR a sacrifice
            # (but there are other okay moves), it is a Great move.
            if is_sac or is_only_good_move:
                return MoveCategory.GREAT

            return MoveCategory.BEST

    return base


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoveAnalysis:
    """Stores the analysis result for a single half-move (ply)."""
    move_number:    int
    color:          chess.Color          # chess.WHITE or chess.BLACK
    san:            str                  # Standard Algebraic Notation
    eval_before:    Optional[float]      # Eval (cp) before the move
    eval_after:     Optional[float]      # Eval (cp) after the move
    wp_before:      Optional[float]      # Win probability before (0.0–1.0)
    wp_after:       Optional[float]      # Win probability after (0.0–1.0)
    wp_loss:        float                # Win probability loss (always ≥ 0)
    best_san:       str                  # Engine's preferred move in SAN
    category:       MoveCategory
    is_user_move:   bool                 # Did the reviewed player play this?


@dataclass
class GameReport:
    """Aggregate report for the full game."""
    pgn_headers:     dict[str, str]
    user_color:      chess.Color
    move_analyses:   list[MoveAnalysis] = field(default_factory=list)
    accuracy:        float = 0.0         # 0-100
    estimated_elo:   int   = 0
    category_counts: dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# PGN LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_game(source: str) -> chess.pgn.Game:
    """
    Load a PGN game from either a file path or a raw PGN string.

    Raises
    ------
    FileNotFoundError   : if a file path is given but does not exist.
    ValueError          : if the PGN cannot be parsed.
    """
    try:
        path = Path(source)
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                pgn_text = fh.read()
        else:
            # Treat the input as a literal PGN string
            pgn_text = source
    except OSError:
        # Invalid path characters in raw PGN string will cause OSError on Windows
        pgn_text = source

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError(
            "Could not parse a valid chess game from the provided input.\n"
            "Make sure the PGN is well-formed (headers + move text)."
        )
    return game


def detect_user_color(
    game: chess.pgn.Game,
    hint: Optional[str] = None,
) -> chess.Color:
    """
    Determine which color the user played.

    Resolution order:
      1. Explicit hint passed by the caller ("white" / "black").
      2. PGN White/Black header matched against common usernames / prompts.
      3. Default to WHITE if no hint is given.
    """
    if hint is not None:
        hint = hint.strip().lower()
        if hint in ("white", "w"):
            return chess.WHITE
        if hint in ("black", "b"):
            return chess.BLACK
        raise ValueError(f"Unrecognised color hint '{hint}'. Use 'white' or 'black'.")

    # Try to extract from headers (e.g., the user is "You" or a known username)
    white_player = game.headers.get("White", "")
    black_player = game.headers.get("Black", "")
    print(f"\n  White: {white_player}  |  Black: {black_player}")
    choice = input("  Which color did YOU play? [white/black, default=white]: ").strip().lower()

    if choice in ("black", "b"):
        return chess.BLACK
    return chess.WHITE


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_cp(score: chess.engine.PovScore, pov: chess.Color) -> Optional[float]:
    """
    Convert a Stockfish PovScore to centipawns from the given player's POV.
    Returns ±10000 for forced-mate positions.
    """
    relative = score.pov(pov)
    if relative.is_mate():
        mate_in = relative.mate()
        return math.copysign(10_000, mate_in)
    return float(relative.score())


def _score_to_win_prob(
    score: chess.engine.PovScore,
    pov: chess.Color,
    ply: int,
) -> float:
    """
    Convert a Stockfish PovScore to a win probability (0.0–1.0) using the
    built-in WDL model.

    The WDL model maps the engine's internal evaluation to Win/Draw/Loss
    probabilities based on large-scale engine self-play data.  The "expectation"
    (expected game outcome) is:
        expectation = (wins + draws/2) / (wins + draws + losses)

    This is position-aware: the same centipawn value produces different
    win probabilities in the middlegame vs. endgame, which is exactly why
    WP loss is more accurate than raw CPL for move classification.

    Parameters
    ----------
    score : the PovScore from engine analysis
    pov   : compute probability from this player's perspective
    ply   : current half-move count (used by the WDL model for phase adjustment)
    """
    relative = score.pov(pov)
    if relative.is_mate():
        mate_in = relative.mate()
        # Mate coming → 1.0;  getting mated → 0.0
        return 1.0 if mate_in > 0 else 0.0

    wdl = relative.wdl(ply=ply)
    # expectation(): (wins + draws/2) / 1000  → a value in [0, 1]
    return wdl.expectation()


def analyse_game(
    game: chess.pgn.Game,
    user_color: chess.Color,
    engine: chess.engine.SimpleEngine,
    depth: int = ANALYSIS_DEPTH,
) -> list[MoveAnalysis]:
    """
    Walk through every move in the game and evaluate each position with
    Stockfish at the given depth, using the WDL model for win-probability
    based classification.

    For each ply we compute:
      - eval_before / eval_after  (centipawns, for display context)
      - wp_before / wp_after      (win probability from the mover's POV)
      - wp_loss = max(0, wp_before - wp_after)

    Optimization: the eval_after of ply N is the same board state as
    eval_before of ply N+1.  We cache the multipv analysis from the
    "before" call and reuse it, cutting engine calls roughly in half.

    Returns a list of MoveAnalysis objects (one per half-move / ply).
    """
    results: list[MoveAnalysis] = []
    board = game.board()
    engine.configure({"UCI_ShowWDL": True})

    # Count total plies for progress bar
    total_plies = sum(1 for _ in game.mainline_moves())
    ply_index = 0

    # Cache: the multipv analysis of the current board position.
    cached_info: Optional[list] = None

    for move_node in game.mainline():
        move = move_node.move
        board_before = board.copy()
        side_to_move = board_before.turn

        # Current ply count (for WDL model phase adjustment)
        current_ply = board_before.ply()

        # ── Evaluate BEFORE the move ────────────────────────────────────────
        if cached_info is not None:
            info_before = cached_info
        else:
            info_before = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=MULTIPV,
            )

        best_pv = info_before[0]
        best_move = best_pv["pv"][0] if best_pv.get("pv") else move
        eval_before_cp = _score_to_cp(best_pv["score"], chess.WHITE)
        wp_before = _score_to_win_prob(best_pv["score"], side_to_move, current_ply)

        # ── WP loss for the second-best candidate (for Brilliant detection) ──
        second_best_wp_loss: Optional[float] = None
        if len(info_before) >= 2:
            second_pv = info_before[1]
            second_wp = _score_to_win_prob(
                second_pv["score"], side_to_move, current_ply
            )
            second_best_wp_loss = max(0.0, wp_before - second_wp)

        # ── Apply the actual move ─────────────────────────────────────────
        board.push(move)
        after_ply = board.ply()

        # ── Evaluate AFTER the move (and cache for next iteration) ──────────
        cached_info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=MULTIPV,
        )
        eval_after_cp = _score_to_cp(cached_info[0]["score"], chess.WHITE)
        # WP after from the MOVER's perspective (not the side now to move)
        wp_after = _score_to_win_prob(
            cached_info[0]["score"], side_to_move, after_ply
        )

        # ── Compute Win Probability Loss ─────────────────────────────────────
        wp_loss = max(0.0, wp_before - wp_after)

        # ── SAN strings ─────────────────────────────────────────────────────
        san = board_before.san(move)
        best_san = board_before.san(best_move)

        # ── Classify ────────────────────────────────────────────────────────
        category = classify_move(
            wp_loss, board_before, move, best_move, second_best_wp_loss
        )

        move_number = board_before.fullmove_number
        color = board_before.turn

        results.append(MoveAnalysis(
            move_number  = move_number,
            color        = color,
            san          = san,
            eval_before  = eval_before_cp,
            eval_after   = eval_after_cp,
            wp_before    = wp_before,
            wp_after     = wp_after,
            wp_loss      = wp_loss,
            best_san     = best_san,
            category     = category,
            is_user_move = (color == user_color),
        ))

        # ── Progress indicator ──────────────────────────────────────────────
        ply_index += 1
        pct = ply_index / total_plies * 100
        color_prefix = "W" if color == chess.WHITE else "B"
        sys.stdout.write(
            f"\r  Analysing: {ply_index}/{total_plies} plies "
            f"({pct:.0f}%)  |  {move_number}{color_prefix}. {san}     "
        )
        sys.stdout.flush()

    # Clear progress line
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()
    print(f"  ✓ Analysis complete — {total_plies} plies evaluated.\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY & ELO ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(user_moves: list[MoveAnalysis]) -> float:
    """
    Compute accuracy using Win Probability loss (CAPS2-style).

    Each move's accuracy is derived from how much win probability was lost:

        per-move accuracy = max(0, 100 * (1 - wp_loss / WP_LOSS_CAP))

    where WP_LOSS_CAP = 0.50 (losing 50% of your winning chances on a single
    move floors that move's accuracy at 0%).

    This replaces the old exponential CPL approximation with a direct
    win-probability-based formula.  Because WP loss is already non-linear
    (it accounts for game phase and evaluation magnitude), a simple linear
    mapping works well and produces scores that closely match Chess.com's
    accuracy numbers.

    Overall accuracy = arithmetic mean of per-move accuracies.
    """
    if not user_moves:
        return 0.0

    # A single-move WP loss of 0.50 or more floors that move's accuracy at 0%.
    WP_LOSS_CAP = 0.50

    per_move_accuracies: list[float] = []
    for m in user_moves:
        wpl = min(m.wp_loss, WP_LOSS_CAP)
        move_acc = 100.0 * (1.0 - wpl / WP_LOSS_CAP)
        move_acc = max(0.0, min(100.0, move_acc))
        per_move_accuracies.append(move_acc)

    accuracy = sum(per_move_accuracies) / len(per_move_accuracies)
    return max(0.0, min(100.0, accuracy))


def estimate_elo(accuracy: float) -> int:
    """
    Map accuracy percentage to an approximate Elo rating.

    Calibration points sourced from Chess.com's public accuracy ↔ Elo
    research and community analysis.  With per-move accuracy averaging,
    scores tend to be lower than with avg-CPL-first, so the anchors
    are tuned accordingly:

        accuracy  98%+ → ~3200  (super-GM, near-perfect play)
        accuracy   95% → ~2700  (GM)
        accuracy   90% → ~2300  (IM/FM)
        accuracy   85% → ~2000  (strong club player)
        accuracy   80% → ~1750  (club player)
        accuracy   70% → ~1400  (intermediate)
        accuracy   60% → ~1100  (casual)
        accuracy   50% → ~850   (beginner)
        accuracy  <40% →  600   (floor)

    We interpolate linearly between these anchor points.
    """
    anchors: list[tuple[float, int]] = [
        (100.0, 3200),
        (98.0,  3000),
        (95.0,  2700),
        (90.0,  2300),
        (85.0,  2000),
        (80.0,  1750),
        (70.0,  1400),
        (60.0,  1100),
        (50.0,   850),
        (40.0,   600),
        (0.0,    400),
    ]

    # Clamp to table bounds
    accuracy = max(0.0, min(100.0, accuracy))

    for i in range(len(anchors) - 1):
        acc_high, elo_high = anchors[i]
        acc_low,  elo_low  = anchors[i + 1]
        if acc_low <= accuracy <= acc_high:
            t   = (accuracy - acc_low) / (acc_high - acc_low)
            elo = elo_low + t * (elo_high - elo_low)
            return int(round(elo, -1))   # round to nearest 10

    return 400   # fallback floor


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORY_ORDER = [
    MoveCategory.BRILLIANT,
    MoveCategory.GREAT,
    MoveCategory.BEST,
    MoveCategory.EXCELLENT,
    MoveCategory.GOOD,
    MoveCategory.INACCURACY,
    MoveCategory.MISTAKE,
    MoveCategory.BLUNDER,
]

_CATEGORY_COLORS = {
    # ANSI escape codes for terminal coloring (falls back gracefully)
    MoveCategory.BRILLIANT:  "\033[96m",   # Bright Cyan
    MoveCategory.GREAT:      "\033[94m",   # Blue
    MoveCategory.BEST:       "\033[92m",   # Green
    MoveCategory.EXCELLENT:  "\033[92m",   # Green
    MoveCategory.GOOD:       "\033[32m",   # Dark Green
    MoveCategory.INACCURACY: "\033[93m",   # Yellow
    MoveCategory.MISTAKE:    "\033[91m",   # Red
    MoveCategory.BLUNDER:    "\033[31m",   # Dark Red
}
_RESET = "\033[0m"


def _color(text: str, category: MoveCategory) -> str:
    return f"{_CATEGORY_COLORS[category]}{text}{_RESET}"


def build_report(
    game: chess.pgn.Game,
    analyses: list[MoveAnalysis],
    user_color: chess.Color,
) -> GameReport:
    """Aggregate all move analyses into a GameReport."""
    user_moves = [m for m in analyses if m.is_user_move]

    accuracy      = compute_accuracy(user_moves)
    estimated_elo = estimate_elo(accuracy)

    counts = {cat.label: 0 for cat in _CATEGORY_ORDER}
    for m in user_moves:
        counts[m.category.label] += 1

    return GameReport(
        pgn_headers    = dict(game.headers),
        user_color     = user_color,
        move_analyses  = analyses,
        accuracy       = accuracy,
        estimated_elo  = estimated_elo,
        category_counts= counts,
    )


def print_report(report: GameReport) -> None:
    """Render the full analysis dashboard to stdout."""
    h = report.pgn_headers
    color_name = "White" if report.user_color == chess.WHITE else "Black"
    sep = "─" * 66

    print(f"\n{'═'*66}")
    print(f"  ♟  CHESS GAME REVIEW AGENT")
    print(f"{'═'*66}")
    print(f"  Event  : {h.get('Event', 'Unknown')}")
    print(f"  Date   : {h.get('Date', 'Unknown')}")
    print(f"  White  : {h.get('White', '?')}  vs  Black: {h.get('Black', '?')}")
    print(f"  Result : {h.get('Result', '?')}")
    print(f"  You played as: {color_name}")
    print(sep)

    # ── Per-move table ────────────────────────────────────────────────────
    print(f"\n  {'#':>4}  {'Move':<8}  {'Best':<8}  {'WP Loss':>8}  {'Win%':>6}  {'Category'}")
    print(f"  {sep}")

    for m in report.move_analyses:
        if not m.is_user_move:
            continue

        color_prefix = "W" if m.color == chess.WHITE else "B"
        move_label   = f"{m.move_number}{color_prefix}"

        # Show win probability after the move
        wp_str = ""
        if m.wp_after is not None:
            wp_str = f"{m.wp_after*100:.0f}%"

        cat_str = _color(
            f"{m.category.symbol:<3} {m.category.label}",
            m.category
        )

        print(
            f"  {move_label:>4}  {m.san:<8}  {m.best_san:<8}  "
            f"{m.wp_loss*100:>7.1f}%  {wp_str:>6}  {cat_str}"
        )

    # ── Summary dashboard ────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  PERFORMANCE SUMMARY")
    print(sep)
    print(f"  Accuracy     :  {report.accuracy:.1f}%")
    print(f"  Estimated Elo:  ~{report.estimated_elo}")
    print()

    user_moves = [m for m in report.move_analyses if m.is_user_move]
    if user_moves:
        avg_wp_loss = sum(m.wp_loss for m in user_moves) / len(user_moves)
        print(f"  Moves analyzed :  {len(user_moves)}")
        print(f"  Avg WP Loss    :  {avg_wp_loss*100:.1f}%")
    print()

    for cat in _CATEGORY_ORDER:
        count = report.category_counts.get(cat.label, 0)
        if count == 0:
            continue
        bar   = "■" * count
        label = _color(f"{cat.symbol} {cat.label}", cat)
        print(f"  {label:<35}  {count:>3}  {bar}")

    print(f"\n{'═'*66}\n")

    # ── Notable moments ───────────────────────────────────────────────────
    highlights = [
        m for m in report.move_analyses
        if m.is_user_move and m.category in (
            MoveCategory.BRILLIANT, MoveCategory.GREAT,
            MoveCategory.BLUNDER,   MoveCategory.MISTAKE,
        )
    ]
    if highlights:
        print("  KEY MOMENTS")
        print(sep)
        for m in highlights:
            color_prefix = "W" if m.color == chess.WHITE else "B"
            tag = _color(
                f"{m.category.symbol} {m.category.label}",
                m.category
            )
            note = (
                f"  Move {m.move_number}{color_prefix}: {m.san:>6}  "
                f"[{tag}]  WP Loss={m.wp_loss*100:.1f}%"
            )
            if m.san != m.best_san:
                note += f"  (best: {m.best_san})"
            print(note)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Usage examples:
        python chess_review_agent.py game.pgn         # from file
        python chess_review_agent.py game.pgn white   # explicit color
    """
    # Fix stdout encoding for Windows to print special characters
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ── Resolve input source ──────────────────────────────────────────────
    color_hint: Optional[str] = None

    if len(sys.argv) >= 2:
        pgn_source = sys.argv[1]
    else:
        print("Enter PGN file path, or paste PGN string.")
        print("(If pasting a PGN string, type 'END' on a new line or press Enter twice to finish):")
        
        lines = []
        empty_count = 0
        while True:
            try:
                line = input()
            except EOFError:
                break
                
            if line.strip() == "END":
                break
                
            if not line.strip():
                empty_count += 1
                # If they only typed a file path and pressed Enter, that's 1 empty line.
                if len(lines) == 1 and not lines[0].startswith("[") and " " not in lines[0]:
                    break
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                
            lines.append(line)
            
        pgn_source = "\n".join(lines).strip()
        if not pgn_source:
            print("\n[ERROR] No PGN provided.")
            sys.exit(1)

    if len(sys.argv) >= 3:
        color_hint = sys.argv[2]

    # ── Load PGN ──────────────────────────────────────────────────────────
    print("\nLoading PGN …")
    try:
        game = load_game(pgn_source)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Detect user color ─────────────────────────────────────────────────
    try:
        user_color = detect_user_color(game, hint=color_hint)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    color_name = "White" if user_color == chess.WHITE else "Black"
    print(f"  Reviewing your moves as: {color_name}")

    # ── Validate Stockfish path ───────────────────────────────────────────
    sf_path = Path(STOCKFISH_PATH)
    if not sf_path.exists():
        print(
            f"\n[ERROR] Stockfish binary not found at: {STOCKFISH_PATH}\n"
            "  Please update the STOCKFISH_PATH variable at the top of this script.\n"
            "  Download Stockfish from: https://stockfishchess.org/download/"
        )
        sys.exit(1)

    # ── Run engine analysis ───────────────────────────────────────────────
    total_plies = sum(1 for _ in game.mainline_moves())
    user_plies  = total_plies // 2 + (
        1 if (user_color == chess.WHITE and total_plies % 2 == 1)
        else 0
    )
    print(
        f"\nAnalysing {total_plies} moves ({user_plies} yours) "
        f"at depth {ANALYSIS_DEPTH} …  This may take a moment.\n"
    )

    try:
        with chess.engine.SimpleEngine.popen_uci(str(sf_path)) as engine:
            analyses = analyse_game(game, user_color, engine, depth=ANALYSIS_DEPTH)
    except chess.engine.EngineError as exc:
        print(f"\n[ENGINE ERROR] {exc}")
        sys.exit(1)

    # ── Build and print report ────────────────────────────────────────────
    report = build_report(game, analyses, user_color)
    print_report(report)


if __name__ == "__main__":
    main()
