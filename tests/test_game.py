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
