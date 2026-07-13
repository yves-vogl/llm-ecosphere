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
X = "X"
O = "O"

# Every winning line as a list of (column, row) coordinates, 0-based,
# row 0 = bottom. 3 columns + 3 rows + 2 diagonals = 8 lines.
LINES: list[list[tuple[int, int]]] = (
    [[(c, r) for r in range(N)] for c in range(N)]  # verticals
    + [[(c, r) for c in range(N)] for r in range(N)]  # horizontals
    + [[(i, i) for i in range(N)], [(i, N - 1 - i) for i in range(N)]]  # diagonals
)

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

    stacks: list[str] = field(default_factory=lambda: ["", "", ""])
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
        if len(move) != 2 or move[0] not in COLS or move[1] not in "123":
            raise IllegalMoveError(f"'{move}' is not a cell between A1 and C3")
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
        lines.append("   +------")
        lines.append("     A B C")
        return "\n".join(lines)
