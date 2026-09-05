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
    (0.09,  MoveCategory.GOOD),
    (0.18,  MoveCategory.INACCURACY),
    (0.30,  MoveCategory.MISTAKE),
]
# Moves with WP loss > 0.30 are BLUNDERS unless a winning buffer applies.


def _classify_by_wp_loss(
    wp_loss: float,
    wp_after: float = 0.5,
    cp_loss: float = 0.0,
) -> MoveCategory:
    """
    Return the base category purely from win-probability loss, with
    centipawn loss and winning buffers to prevent false blunders.
    """
    for threshold, category in WP_LOSS_THRESHOLDS:
        if wp_loss <= threshold:
            return category

    # wp_loss > 0.30:
    # To be a BLUNDER (??):
    # 1. The move must lose significant material: cp_loss >= 220 (over 2 pawns / piece drop).
    #    If cp_loss < 220 (e.g. shifting by ~1 pawn like move 15W), it is a MISTAKE (?), not a Blunder!
    # 2. The player must NOT remain comfortably winning: wp_after < 0.65.
    if cp_loss < 220.0 or wp_after >= 0.65:
        return MoveCategory.MISTAKE
    return MoveCategory.BLUNDER


def _is_sacrifice(board_before: chess.Board, move: chess.Move) -> tuple[bool, int, Optional[int]]:
    """
    Detect whether a move gives up material deliberately (genuine sacrifice).
    Returns (is_sacrifice, net_material_sacrificed, piece_type_sacrificed).

    A move is a sacrifice if:
      1. A piece (Queen, Rook, Bishop, Knight) lands on a square where the opponent
         can legally capture it, or a piece (especially Queen) is left en prise.
      2. It is NOT an immediate recapture on the square the opponent just played to.
      3. The net material given up is strictly positive (at least 150 points).
    """
    PIECE_VALUES = {
        chess.PAWN:   100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK:   500,
        chess.QUEEN:  900,
        chess.KING:   0,
    }
    moving_piece = board_before.piece_at(move.from_square)
    if moving_piece is None or moving_piece.piece_type in (chess.PAWN, chess.KING):
        return False, 0, None

    # Immediate recaptures on the square the opponent just played to are NOT sacrifices
    if len(board_before.move_stack) > 0:
        last_move = board_before.peek()
        if move.to_square == last_move.to_square and board_before.is_capture(move):
            return False, 0, None

    captured_piece = board_before.piece_at(move.to_square)
    moving_value   = PIECE_VALUES.get(moving_piece.piece_type, 0)

    if board_before.is_en_passant(move):
        captured_value = PIECE_VALUES[chess.PAWN]
    elif captured_piece is not None:
        captured_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
    else:
        captured_value = 0

    net_sac = moving_value - captured_value

    board_after = board_before.copy()
    board_after.push(move)
    mover_color = board_before.turn
    opp_color = not mover_color

    # 1. Direct sacrifice: Can the opponent legally capture the moved piece on its destination?
    can_opp_take_moved_piece = any(
        m.to_square == move.to_square for m in board_after.legal_moves
    )
    if can_opp_take_moved_piece and net_sac >= 150:
        return True, net_sac, moving_piece.piece_type

    # 2. Check if destination square is attacked by opponent
    if board_after.is_attacked_by(opp_color, move.to_square) and net_sac >= 150:
        return True, net_sac, moving_piece.piece_type

    # 3. Discovered / quiet Queen sacrifice: another piece moved and left our Queen
    # to be captured by opponent, and we didn't just take an enemy Queen
    if moving_piece.piece_type != chess.QUEEN and (captured_piece is None or captured_piece.piece_type != chess.QUEEN):
        for m in board_after.legal_moves:
            target = board_after.piece_at(m.to_square)
            if target and target.piece_type == chess.QUEEN and target.color == mover_color:
                return True, 900 - captured_value, chess.QUEEN

    return False, 0, None


def classify_move(
    wp_loss: float,
    board_before: chess.Board,
    move: chess.Move,
    best_move: chess.Move,
    second_best_wp_loss: Optional[float],
    wp_before: float = 0.5,
    wp_after: float = 0.5,
    cp_loss: float = 0.0,
) -> MoveCategory:
    """
    Full classification logic, including calibrated Brilliant and Great Move detection.

    Parameters
    ----------
    wp_loss              : win-probability loss of the played move (always ≥ 0)
    board_before         : position before the move was made
    move                 : the move that was actually played
    best_move            : engine's top choice for this position
    second_best_wp_loss  : WP loss for the second-best engine move (None if unavailable)
    wp_before            : win probability before the move
    wp_after             : win probability after the move
    cp_loss              : centipawn loss from the mover's perspective
    """
    base = _classify_by_wp_loss(wp_loss, wp_after, cp_loss)

    if base is MoveCategory.BEST:
        if move == best_move:
            is_sac, net_sac, sac_piece_type = _is_sacrifice(board_before, move)

            # Check if this move is a routine recapture
            is_recapture = False
            if len(board_before.move_stack) > 0:
                last_move = board_before.peek()
                if move.to_square == last_move.to_square and board_before.is_capture(move):
                    is_recapture = True

            moving_piece = board_before.piece_at(move.from_square)

            # ── BRILLIANT (!!) vs GREAT MOVE (!) FOR SACRIFICES ─────────────
            if is_sac and wp_loss <= 0.02 and wp_after >= 0.50:
                # In Chess.com:
                # If you are ALREADY decisively winning before the sacrifice (wp_before > 0.85),
                # finding a sacrifice (like a Queen sac for mate or deflection) is classified as
                # a GREAT MOVE (!), NOT Brilliant (!!), because the game was already won.
                if wp_before > 0.85:
                    return MoveCategory.GREAT

                # In contested or turning positions (wp_before <= 0.85):
                # 1. Queen Sacrifice:
                if sac_piece_type == chess.QUEEN:
                    return MoveCategory.BRILLIANT

                # 2. Rook Sacrifice:
                # Clean rook or exchange sacrifice (net_sac >= 170)
                if sac_piece_type == chess.ROOK and net_sac >= 170:
                    return MoveCategory.BRILLIANT

                # 3. Minor Piece Sacrifice:
                if sac_piece_type in (chess.BISHOP, chess.KNIGHT) and net_sac >= 200:
                    gap = second_best_wp_loss if second_best_wp_loss is not None else 0.0
                    if gap >= 0.12 and wp_before <= 0.75:
                        return MoveCategory.BRILLIANT

                # Other sound sacrifices fall back to Great Move
                return MoveCategory.GREAT

            # ── GREAT MOVE (!) FOR NON-SACRIFICES ───────────────────────────
            # Great moves are rare (typically 1-3 per game).
            # Disqualifications:
            # - Routine recaptures
            # - Escaping check (forced/routine king/block moves)
            # - Early opening theory (first 4 full moves)
            # - Already overwhelmingly winning positions (wp_before > 0.88)
            # Qualifications:
            # - The ONLY move in a contested position that maintains the win (gap >= 0.15)
            if not is_recapture and not board_before.is_check() and board_before.fullmove_number > 4:
                if 0.25 <= wp_before <= 0.88 and wp_loss <= 0.02:
                    gap = second_best_wp_loss if second_best_wp_loss is not None else 0.0
                    if gap >= 0.15:
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

        # ── Compute Centipawn Loss from mover's POV ─────────────────────────
        eval_before_mover = eval_before_cp if side_to_move == chess.WHITE else -eval_before_cp
        eval_after_mover  = eval_after_cp if side_to_move == chess.WHITE else -eval_after_cp
        cp_loss = max(0.0, eval_before_mover - eval_after_mover)

        # ── Classify ────────────────────────────────────────────────────────
        category = classify_move(
            wp_loss, board_before, move, best_move, second_best_wp_loss,
            wp_before=wp_before, wp_after=wp_after, cp_loss=cp_loss,
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
    Compute accuracy using a CAPS2-calibrated Win Probability curve.

    Formula:
        move_acc = max(0, min(100, 103.1668 * exp(-3.8 * wp_loss) - 3.1668))

    Calibrated against Chess.com CAPS2 benchmarks:
      - 0.00 WP loss  -> 100.0%
      - 0.01 WP loss  ->  96.0%
      - 0.03 WP loss  ->  88.8%
      - 0.06 WP loss  ->  79.0%
      - 0.12 WP loss  ->  62.2%
      - 0.22 WP loss  ->  41.5%
      - 0.35+ WP loss ->  <= 24%

    Overall accuracy = arithmetic mean of per-move accuracies.
    """
    if not user_moves:
        return 0.0

    per_move_accuracies: list[float] = []
    for m in user_moves:
        wpl = m.wp_loss
        move_acc = 103.1668 * math.exp(-3.8 * wpl) - 3.1668
        move_acc = max(0.0, min(100.0, move_acc))
        per_move_accuracies.append(move_acc)

    accuracy = sum(per_move_accuracies) / len(per_move_accuracies)
    return max(0.0, min(100.0, accuracy))


def estimate_elo(accuracy: float) -> int:
    """
    Map accuracy percentage to an approximate Elo rating.

    Calibrated with Chess.com benchmark data:
        accuracy 98%+  -> ~2850+ (GM / engine level)
        accuracy 95%   -> ~2450  (IM/FM)
        accuracy 93%   -> ~2250  (Master / strong candidate)
        accuracy 91.5% -> ~2050  (Direct calibration anchor)
        accuracy 88%   -> ~1750  (Club player)
        accuracy 85%   -> ~1500  (Intermediate)
        accuracy 80%   -> ~1250  (Casual / club)
        accuracy 75%   -> ~1050  (Developing)
        accuracy 70%   -> ~850   (Beginner)
        accuracy 60%   -> ~650   (Novice)
        accuracy <50%  -> ~400-500

    We interpolate linearly between these anchor points.
    """
    anchors: list[tuple[float, int]] = [
        (100.0, 3100),
        (98.0,  2850),
        (95.0,  2450),
        (93.0,  2250),
        (91.5,  2050),
        (88.0,  1750),
        (85.0,  1500),
        (80.0,  1250),
        (75.0,  1050),
        (70.0,   850),
        (60.0,   650),
        (50.0,   500),
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
