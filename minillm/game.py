"""Game engine for Drop-Tac-Toe: Tic-Tac-Toe with gravity.

The board is a 3x3 grid. Columns are labelled A, B, C (x-axis) and rows
1, 2, 3 (y-axis, bottom to top). A move drops a piece into a column and
the piece falls onto the lowest free cell, Connect-Four style. Moves are
written as the cell the piece lands on: "A1" is bottom-left, "C3" is
top-right. "C3" is only legal once C1 and C2 are already occupied —
pieces never float.

X and O alternate, X moves first. Three own pieces in a line (row,
column or diagonal) win. If all nine cells fill without a line the game
is a draw.

This module is deliberately free of any ML code: it is the "physics" of
the tiny world our language model will learn purely from transcripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COLS = "ABC"
N = 3  # board is N x N, and each column holds at most N pieces
WIN = 3  # own pieces in a row needed to win
X = "X"
O = "O"

assert len(COLS) == N, "one column label per column"
ROWS = "".join(str(r) for r in range(1, N + 1))


def _win_segments(n: int = N, win: int = WIN) -> list[list[tuple[int, int]]]:
    """Every straight segment of `win` cells on an n x n board, as
    (column, row) coordinates, 0-based, row 0 = bottom.

    Scans the four line directions (horizontal, vertical, both
    diagonals) from every anchor cell that leaves the segment on the
    board. With win == n this reduces to the full-length lines — at the
    shipped 3x3 exactly the classic 8 (3 verticals + 3 horizontals +
    2 diagonals). With win < n (exercise 9's 4x4 Connect-3 world) it
    enumerates every 3-cell winning window, which full-length lines
    would miss entirely."""
    segments = []
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
        for c in range(n):
            for r in range(n):
                end_c, end_r = c + (win - 1) * dc, r + (win - 1) * dr
                if 0 <= end_c < n and 0 <= end_r < n:
                    segments.append([(c + i * dc, r + i * dr) for i in range(win)])
    return segments


# Every winning line: at the default 3x3, the classic 8.
LINES: list[list[tuple[int, int]]] = _win_segments()

RESULT_X = "#X"  # transcript token: X won
RESULT_O = "#O"  # transcript token: O won
RESULT_DRAW = "#="  # transcript token: draw


def other(player: str) -> str:
    """The opponent of `player`."""
    return O if player == X else X


class IllegalMoveError(ValueError):
    """Raised when a move violates the rules (bad cell, floating piece, ...)."""


@dataclass
class Game:
    """Mutable game state.

    The board is stored as one string per column ("stack"), bottom to
    top: stacks == ["XO", "", "X"] means A1=X, A2=O, C1=X.
    """

    stacks: list[str] = field(default_factory=lambda: ["" for _ in range(N)])
    history: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Reading the state
    # ------------------------------------------------------------------
    @property
    def to_move(self) -> str:
        """X moves on even ply counts (0, 2, ...), O on odd ones."""
        return X if sum(len(s) for s in self.stacks) % 2 == 0 else O

    def piece_at(self, col: int, row: int) -> str | None:
        """Piece at 0-based (col, row), or None if the cell is empty."""
        stack = self.stacks[col]
        return stack[row] if row < len(stack) else None

    def legal_moves(self) -> list[str]:
        """All legal moves in notation form, e.g. ["A2", "B1", "C1"].

        Exactly one cell per non-full column is reachable: the one right
        on top of the current stack. That is the gravity rule.
        """
        if self.is_over():
            return []
        return [
            f"{COLS[c]}{len(stack) + 1}"
            for c, stack in enumerate(self.stacks)
            if len(stack) < N
        ]

    def winner(self) -> str | None:
        """"X" or "O" if a line of three exists, else None."""
        for line in LINES:
            first = self.piece_at(*line[0])
            if first is not None and all(self.piece_at(c, r) == first for c, r in line):
                return first
        return None

    def is_full(self) -> bool:
        return all(len(stack) == N for stack in self.stacks)

    def is_draw(self) -> bool:
        return self.is_full() and self.winner() is None

    def is_over(self) -> bool:
        return self.winner() is not None or self.is_full()

    @property
    def result_token(self) -> str | None:
        """The transcript token for the final result, or None if ongoing."""
        w = self.winner()
        if w == X:
            return RESULT_X
        if w == O:
            return RESULT_O
        if self.is_full():
            return RESULT_DRAW
        return None

    # ------------------------------------------------------------------
    # Changing the state
    # ------------------------------------------------------------------
    def push(self, move: str) -> None:
        """Play `move` (e.g. "B2") for the player whose turn it is.

        Raises IllegalMoveError with a human-readable reason if the move
        is malformed, the column is full, the named cell floats, or the
        game is already over.
        """
        if self.is_over():
            raise IllegalMoveError("the game is already over")
        move = move.strip().upper()
        if len(move) != 2 or move[0] not in COLS or move[1] not in ROWS:
            raise IllegalMoveError(
                f"'{move}' is not a cell between {COLS[0]}1 and {COLS[-1]}{N}"
            )
        col = COLS.index(move[0])
        row = int(move[1]) - 1
        height = len(self.stacks[col])
        if height >= N:
            raise IllegalMoveError(f"column {move[0]} is full")
        if row != height:
            raise IllegalMoveError(
                f"{move} would float: the next free cell in column {move[0]} "
                f"is {COLS[col]}{height + 1}"
            )
        self.stacks[col] += self.to_move
        self.history.append(move)

    @classmethod
    def from_moves(cls, moves: list[str]) -> "Game":
        """Replay a list of moves from the empty board."""
        game = cls()
        for move in moves:
            game.push(move)
        return game

    def copy(self) -> "Game":
        return Game(stacks=list(self.stacks), history=list(self.history))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def render(self) -> str:
        """ASCII board, row 3 on top:

             3 | . . .
             2 | . X .
             1 | O X .
               +------
                 A B C
        """
        lines = []
        for r in reversed(range(N)):
            cells = " ".join(self.piece_at(c, r) or "." for c in range(N))
            lines.append(f" {r + 1} | {cells}")
        lines.append("   +" + "-" * (2 * N))
        lines.append("     " + " ".join(COLS))
        return "\n".join(lines)
