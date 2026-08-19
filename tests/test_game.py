"""Rules of the world: gravity, turn order, win/draw detection."""

import pytest

from minillm.game import Game, IllegalMoveError


def test_gravity_pieces_cannot_float():
    game = Game()
    with pytest.raises(IllegalMoveError, match="float"):
        game.push("C3")  # C1 and C2 are still empty
    with pytest.raises(IllegalMoveError, match="float"):
        game.push("A2")
    game.push("A1")  # fine: bottom cell


def test_gravity_stacking_order():
    game = Game()
    game.push("C1")
    game.push("C2")
    game.push("C3")  # now legal: C1 and C2 are occupied
    assert game.stacks[2] == "XOX"
    with pytest.raises(IllegalMoveError, match="full"):
        game.push("C1")


def test_turn_alternation():
    game = Game()
    assert game.to_move == "X"
    game.push("B1")
    assert game.to_move == "O"
    game.push("B2")
    assert game.to_move == "X"


def test_malformed_moves_rejected():
    game = Game()
    for bad in ("D1", "A4", "AA", "1A", "", "A"):
        with pytest.raises(IllegalMoveError):
            game.push(bad)


def test_legal_moves_lists_one_cell_per_open_column():
    game = Game.from_moves(["A1", "A2", "A3"])  # column A full
    assert game.legal_moves() == ["B1", "C1"]


def test_vertical_win():
    game = Game.from_moves(["A1", "B1", "A2", "B2", "A3"])
    assert game.winner() == "X"  # X owns all of column A
    assert game.result_token == "#X"
    assert game.is_over()


def test_horizontal_win():
    game = Game.from_moves(["A1", "A2", "B1", "B2", "C1"])
    assert game.winner() == "X"  # X owns row 1


def test_diagonal_win():
    game = Game.from_moves(["A1", "B1", "B2", "C1", "C2", "A2", "C3"])
    assert game.winner() == "X"  # A1-B2-C3


def test_anti_diagonal_win():
    game = Game.from_moves(["C1", "B1", "B2", "A1", "A2", "C2", "A3"])
    assert game.winner() == "X"  # C1-B2-A3


def test_o_can_win_too():
    game = Game.from_moves(["A1", "B1", "A2", "B2", "C1", "B3"])
    assert game.winner() == "O"  # O owns column B
    assert game.result_token == "#O"


def test_draw():
    game = Game.from_moves(["A1", "B1", "A2", "B2", "C1", "A3", "C2", "C3", "B3"])
    assert game.winner() is None
    assert game.is_draw()
    assert game.result_token == "#="


def test_no_moves_after_game_over():
    game = Game.from_moves(["A1", "B1", "A2", "B2", "A3"])
    assert game.legal_moves() == []
    with pytest.raises(IllegalMoveError, match="over"):
        game.push("C1")


def test_render_shows_bottom_row_last():
    game = Game.from_moves(["A1"])
    lines = game.render().splitlines()
    assert lines[0].startswith(" 3")
    assert "X" in lines[2]  # row 1 line contains the piece
    assert lines[-1].strip() == "A B C"


# ----------------------------------------------------------------------
# Win-segment derivation (exercise 9): LINES for any board and win length
# ----------------------------------------------------------------------
def test_win_segments_at_3x3_are_the_classic_eight_lines():
    from minillm.game import LINES, N, _win_segments

    classic = (
        [[(c, r) for r in range(N)] for c in range(N)]        # verticals
        + [[(c, r) for c in range(N)] for r in range(N)]      # horizontals
        + [[(i, i) for i in range(N)], [(i, N - 1 - i) for i in range(N)]]
    )
    as_sets = lambda lines: {frozenset(line) for line in lines}
    assert as_sets(LINES) == as_sets(classic)
    assert len(LINES) == 8
    assert _win_segments() == LINES


def test_win_segments_4x4_connect3_covers_every_three_cell_window():
    """The 4x4 world of exercise 9: win length 3 on a 4x4 board needs
    every 3-cell segment, not just full-length lines."""
    from minillm.game import _win_segments

    segments = _win_segments(n=4, win=3)
    # 2 horizontal windows per row x 4 rows, same vertically = 16, plus
    # 4 windows per diagonal direction = 24 segments in total.
    assert len(segments) == 24
    as_sets = {frozenset(s) for s in segments}
    assert frozenset([(0, 0), (1, 0), (2, 0)]) in as_sets   # horizontal window
    assert frozenset([(1, 0), (2, 0), (3, 0)]) in as_sets   # shifted window
    assert frozenset([(0, 1), (1, 2), (2, 3)]) in as_sets   # off-corner diagonal
    assert frozenset([(3, 1), (2, 2), (1, 3)]) in as_sets   # anti-diagonal window
    # Full-length lines alone would be 4 + 4 + 2 = 10 - the derivation
    # must not collapse back to that.
    assert all(len(s) == 3 for s in segments)
