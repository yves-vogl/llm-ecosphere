"""The solver is our ground truth — it had better be right."""

from minillm.game import COLS, Game
from minillm.solver import (
    EMPTY,
    best_moves,
    enumerate_all_games,
    enumerate_expert_games,
    negamax,
    to_move,
)


def test_root_value_is_a_valid_game_value():
    assert negamax(EMPTY) in (-1, 0, 1)


def test_immediate_win_is_seen():
    # X to move (2 pieces each); A3 completes X's column A.
    state = ("XX", "OO", "")
    assert to_move(state) == "X"
    value, moves = best_moves(state)
    assert value == 1
    assert "A3" in moves


def test_double_threat_is_lost_for_the_defender():
    # X owns two pieces in column A and two in column C: A3 and C3 are
    # both winning drops. O (to move) can only block one of them.
    state = ("XX", "O", "XX")
    assert to_move(state) == "O"
    assert negamax(state) == -1  # the player to move loses


def test_mirror_symmetry():
    """Swapping columns A and C cannot change a position's value."""
    for moves in (["B1"], ["A1", "B1"], ["A1", "C1", "A2"], ["B1", "B2", "A1"]):
        game = Game.from_moves(moves)
        mirrored = tuple(reversed(game.stacks))
        assert negamax(tuple(game.stacks)) == negamax(mirrored)


def test_enumerate_all_games_are_legal_and_complete():
    games = enumerate_all_games()
    # The corpus is a closed world: exactly 1310 complete games exist.
    # Pinning the exact count catches any silently dropped subtree.
    assert len(games) == 1310
    seen = set()
    for g in games[:: max(1, len(games) // 200)] + games[-5:]:  # spot-check ~200
        replayed = Game.from_moves(g["moves"])
        assert replayed.is_over()
        assert replayed.result_token == g["result"]
    for g in games:
        key = tuple(g["moves"])
        assert key not in seen, "duplicate game in enumeration"
        seen.add(key)


def test_expert_corpus_size_is_pinned():
    assert len(enumerate_expert_games("X")) + len(enumerate_expert_games("O")) == 334


def test_expert_games_expert_moves_are_optimal():
    games = enumerate_expert_games("X")
    assert all(g["expert"] == "X" for g in games)
    for g in games[:: max(1, len(games) // 50)]:  # stride across the whole tree
        game = Game()
        for move in g["moves"]:
            if game.to_move == "X":
                _, optimal = best_moves(tuple(game.stacks))
                assert move in optimal, f"expert played non-optimal {move}"
            game.push(move)


def test_expert_games_opponent_branches_everything():
    """The very first O reply after the expert's opening must cover every
    legal option somewhere in the corpus."""
    games = enumerate_expert_games("X")
    _, openings = best_moves(EMPTY)
    for opening in openings:
        replies = {g["moves"][1] for g in games if g["moves"][0] == opening}
        game = Game.from_moves([opening])
        assert replies == set(game.legal_moves())
