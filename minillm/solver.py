"""Exact solver for Drop-Tac-Toe, and the game enumerators built on it.

The game is tiny (well under 10,000 reachable positions), so we can
solve it *exactly* with negamax + memoization. The solver plays three
roles in this project:

1. It tells us the game-theoretic value of any position (who wins with
   perfect play) — ground truth no neural network can argue with.
2. It generates the *pretraining corpus*: every complete game that can
   possibly be played (`enumerate_all_games`).
3. It generates the *finetuning corpus*: games in which one side (the
   "expert") always plays a perfect move while the opponent tries
   everything (`enumerate_expert_games`). This is our miniature
   supervised-finetuning dataset.

States are immutable tuples of column strings, e.g. ("XO", "", "X"),
bottom to top — the same representation `game.Game` uses internally,
just hashable so `functools.lru_cache` can memoize on it.
"""

from __future__ import annotations

from functools import lru_cache

from .game import COLS, LINES, N, O, RESULT_DRAW, RESULT_O, RESULT_X, X

State = tuple[str, str, str]
EMPTY: State = ("", "", "")


# ----------------------------------------------------------------------
# Pure state helpers (no Game object: these run millions of times)
# ----------------------------------------------------------------------
def to_move(stacks: State) -> str:
    """X moves on even piece counts, O on odd ones."""
    return X if sum(len(s) for s in stacks) % 2 == 0 else O


def piece_at(stacks: State, col: int, row: int) -> str | None:
    stack = stacks[col]
    return stack[row] if row < len(stack) else None


def winner(stacks: State) -> str | None:
    for line in LINES:
        first = piece_at(stacks, *line[0])
        if first is not None and all(piece_at(stacks, c, r) == first for c, r in line):
            return first
    return None


def is_full(stacks: State) -> bool:
    return all(len(s) == N for s in stacks)


def legal_columns(stacks: State) -> list[int]:
    return [c for c in range(N) if len(stacks[c]) < N]


def notation(stacks: State, col: int) -> str:
    """Where a piece dropped into `col` would land, e.g. "B2"."""
    return f"{COLS[col]}{len(stacks[col]) + 1}"


def apply_move(stacks: State, col: int) -> State:
    """Drop a piece for the side to move; returns a new state."""
    piece = to_move(stacks)
    new = list(stacks)
    new[col] += piece
    return tuple(new)  # type: ignore[return-value]


def result_token(stacks: State) -> str:
    """Transcript token for a terminal state."""
    w = winner(stacks)
    if w == X:
        return RESULT_X
    if w == O:
        return RESULT_O
    assert is_full(stacks), "result_token called on a non-terminal state"
    return RESULT_DRAW


# ----------------------------------------------------------------------
# Negamax
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def negamax(stacks: State) -> int:
    """Game value from the perspective of the player to move.

    +1 = the player to move wins with perfect play,
     0 = perfect play ends in a draw,
    -1 = the player to move loses with perfect play.

    The single recursion rule: my value is the best I can get among all
    moves, and after any move the position is worth the *negation* of
    what it is worth for my opponent (zero-sum game).
    """
    if winner(stacks) is not None:
        return -1  # the previous player just completed a line: I lost
    if is_full(stacks):
        return 0
    return max(-negamax(apply_move(stacks, c)) for c in legal_columns(stacks))


def best_moves(stacks: State) -> tuple[int, list[str]]:
    """The position's value and *all* moves that achieve it.

    Ties are common (e.g. symmetric openings), so this returns a list;
    picking any element preserves the game-theoretic value.
    """
    value = negamax(stacks)
    moves = [
        notation(stacks, c)
        for c in legal_columns(stacks)
        if -negamax(apply_move(stacks, c)) == value
    ]
    return value, moves


def describe_root_value() -> str:
    """Human sentence for the value of the empty board."""
    value = negamax(EMPTY)
    return {
        1: "X (the first player) wins with perfect play",
        0: "perfect play ends in a draw",
        -1: "O (the second player) wins with perfect play",
    }[value]


# ----------------------------------------------------------------------
# Corpus enumeration
# ----------------------------------------------------------------------
def enumerate_all_games() -> list[dict]:
    """Every complete game that can be played, in deterministic order.

    Each game is a dict: {"moves": ["B1", "A1", ...], "result": "#X"}.
    This is the entire "language" — a closed world small enough to
    enumerate, which is exactly what makes it a good study object.
    """
    games: list[dict] = []

    def dfs(stacks: State, moves: list[str]) -> None:
        if winner(stacks) is not None or is_full(stacks):
            games.append({"moves": list(moves), "result": result_token(stacks)})
            return
        for col in legal_columns(stacks):
            moves.append(notation(stacks, col))
            dfs(apply_move(stacks, col), moves)
            moves.pop()

    dfs(EMPTY, [])
    return games


def enumerate_expert_games(expert: str, ties: str = "all") -> list[dict]:
    """Games where `expert` ("X" or "O") always plays a perfect move.

    The *opponent* branches over every legal reply, so the expert's
    perfect answers are demonstrated against every situation the
    opponent can create — not just against perfect counter-play. With
    ties="all" the expert also branches over all equally-good moves
    (more diverse data); ties="first" keeps only the first one.

    Each game carries an "expert" field. During finetuning the loss is
    masked so the model only imitates the expert's moves, never the
    opponent's arbitrary ones — the same trick real SFT uses to learn
    the assistant's turns but not the user's.
    """
    assert expert in (X, O)
    assert ties in ("all", "first")
    games: list[dict] = []

    def dfs(stacks: State, moves: list[str]) -> None:
        if winner(stacks) is not None or is_full(stacks):
            games.append(
                {"moves": list(moves), "result": result_token(stacks), "expert": expert}
            )
            return
        if to_move(stacks) == expert:
            _, options = best_moves(stacks)
            if ties == "first":
                options = options[:1]
        else:
            options = [notation(stacks, c) for c in legal_columns(stacks)]
        for move in options:
            col = COLS.index(move[0])
            moves.append(move)
            dfs(apply_move(stacks, col), moves)
            moves.pop()

    dfs(EMPTY, [])
    return games
