"""Tensorization: shifted targets, padding, and SFT-style loss masking."""

import pytest
import torch

from minillm.dataset import build_tensors, split_games, to_gambler_games
from minillm.tokenizer import CharTokenizer, Tokenizer

GAME = {"moves": ["A1", "B1", "A2", "B2", "A3"], "result": "#X"}
BLOCK = 16
CHAR_BLOCK = 24  # char-level games need up to 22 tokens


def test_split_is_disjoint_and_deterministic():
    games = [{"moves": [f"g{i}"], "result": "#="} for i in range(100)]
    train1, val1 = split_games(games, val_frac=0.1, seed=42)
    train2, val2 = split_games(games, val_frac=0.1, seed=42)
    assert train1 == train2 and val1 == val2
    assert len(val1) == 10
    ids = lambda gs: {g["moves"][0] for g in gs}
    assert ids(train1) & ids(val1) == set()


def test_targets_are_inputs_shifted_left():
    tok = Tokenizer()
    x, y = build_tensors([GAME], tok, BLOCK)
    seq = tok.encode_game(GAME["moves"], GAME["result"])  # 8 tokens
    assert x.shape == (1, BLOCK) and y.shape == (1, BLOCK)
    assert x[0, : len(seq) - 1].tolist() == seq[:-1]
    assert y[0, : len(seq) - 1].tolist() == seq[1:]


def test_padding_is_ignored_in_targets():
    tok = Tokenizer()
    x, y = build_tensors([GAME], tok, BLOCK)
    seq_len = len(tok.encode_game(GAME["moves"], GAME["result"]))
    assert (x[0, seq_len - 1 :] == tok.pad_id).all()  # inputs padded with <pad>
    assert (y[0, seq_len - 1 :] == -1).all()  # targets padded with ignore_index


def test_expert_masking_hides_opponent_moves():
    tok = Tokenizer()
    game_o = {**GAME, "expert": "O"}  # X's moves (1st, 3rd, 5th) must be masked
    _, y = build_tensors([game_o], tok, BLOCK, expert_only=True)
    # y[t] predicts seq[t+1]; moves sit at seq indices 1..5 (move #1..#5).
    assert y[0, 0] == -1  # predicts move 1 (X) -> masked
    assert y[0, 1] != -1  # predicts move 2 (O) -> trained
    assert y[0, 2] == -1  # move 3 (X)
    assert y[0, 3] != -1  # move 4 (O)
    assert y[0, 4] == -1  # move 5 (X)
    assert y[0, 5] != -1  # result token stays trained
    assert y[0, 6] != -1  # <eos> stays trained

    game_x = {**GAME, "expert": "X"}
    _, y = build_tensors([game_x], tok, BLOCK, expert_only=True)
    assert (y[0, torch.tensor([0, 2, 4])] != -1).all()  # X's moves trained
    assert (y[0, torch.tensor([1, 3])] == -1).all()  # O's moves masked


# ----------------------------------------------------------------------
# Char-level tensorization: the same invariants, two tokens per move
# ----------------------------------------------------------------------
def test_char_targets_are_inputs_shifted_left():
    tok = CharTokenizer()
    x, y = build_tensors([GAME], tok, CHAR_BLOCK)
    seq = tok.encode_game(GAME["moves"], GAME["result"])  # 14 tokens
    assert len(seq) == 1 + 5 * 2 + 2 + 1
    assert x[0, : len(seq) - 1].tolist() == seq[:-1]
    assert y[0, : len(seq) - 1].tolist() == seq[1:]
    assert (x[0, len(seq) - 1 :] == tok.pad_id).all()
    assert (y[0, len(seq) - 1 :] == -1).all()


def test_char_expert_masking_hides_both_characters_of_opponent_moves():
    """The exercise's advertised trap: "move number = token index" breaks
    at two tokens per move. Move k occupies seq indices 2k-1 and 2k, so
    y[t] (predicting seq[t+1]) belongs to move t // 2 + 1."""
    tok = CharTokenizer()
    # GAME: A1 B1 A2 B2 A3 — X plays moves 1, 3, 5 (the A column).
    game_x = {**GAME, "expert": "X"}
    _, y = build_tensors([game_x], tok, CHAR_BLOCK, expert_only=True)
    x_move_targets = [0, 1, 4, 5, 8, 9]  # both chars of moves 1, 3, 5
    o_move_targets = [2, 3, 6, 7]        # both chars of moves 2, 4
    assert (y[0, torch.tensor(x_move_targets)] != -1).all()
    assert (y[0, torch.tensor(o_move_targets)] == -1).all()
    assert (y[0, torch.tensor([10, 11, 12])] != -1).all()  # "#", "X", <eos> trained

    game_o = {**GAME, "expert": "O"}
    _, y = build_tensors([game_o], tok, CHAR_BLOCK, expert_only=True)
    assert (y[0, torch.tensor(x_move_targets)] == -1).all()
    assert (y[0, torch.tensor(o_move_targets)] != -1).all()
    assert (y[0, torch.tensor([10, 11, 12])] != -1).all()


# ----------------------------------------------------------------------
# Gambler objective: imitate the WINNING side of decisive games only,
# drop draws entirely (mirrors the expert masking above, but keyed off
# "winner" instead of "expert").
# ----------------------------------------------------------------------
def test_to_gambler_games_drops_draws_and_tags_winner():
    games = [
        {"moves": ["A1"], "result": "#X"},
        {"moves": ["A1"], "result": "#O"},
        {"moves": ["A1"], "result": "#="},
    ]
    gambler_games = to_gambler_games(games)
    assert len(gambler_games) == 2  # the draw is dropped entirely
    assert gambler_games[0]["winner"] == "X"
    assert gambler_games[1]["winner"] == "O"
    # original games are untouched (build_tensors input must not be mutated)
    assert "winner" not in games[0]
    assert "winner" not in games[1]


def test_gambler_masking_hides_losing_side_moves():
    tok = Tokenizer()
    # GAME: A1 B1 A2 B2 A3 — X plays moves 1, 3, 5; O plays moves 2, 4.
    game_x_wins = {**GAME, "result": "#X"}
    gambler_games = to_gambler_games([game_x_wins])
    _, y = build_tensors(gambler_games, tok, BLOCK, winner_only=True)
    assert (y[0, torch.tensor([0, 2, 4])] != -1).all()  # X's moves (winner) trained
    assert (y[0, torch.tensor([1, 3])] == -1).all()  # O's moves (loser) masked
    assert y[0, 5] != -1  # result token stays trained
    assert y[0, 6] != -1  # <eos> stays trained

    game_o_wins = {**GAME, "result": "#O"}
    gambler_games = to_gambler_games([game_o_wins])
    _, y = build_tensors(gambler_games, tok, BLOCK, winner_only=True)
    assert (y[0, torch.tensor([0, 2, 4])] == -1).all()  # X's moves (loser) masked
    assert (y[0, torch.tensor([1, 3])] != -1).all()  # O's moves (winner) trained
    assert y[0, 5] != -1
    assert y[0, 6] != -1


def test_gambler_excludes_draw_games_from_training_data():
    """A draw has no winner to imitate — to_gambler_games must drop it
    before build_tensors ever sees it (there is no "winner" key to fall
    back on, so passing a draw straight through would KeyError)."""
    draw_game = {**GAME, "result": "#="}
    assert to_gambler_games([draw_game]) == []


def test_char_gambler_masking_hides_both_characters_of_losing_moves():
    tok = CharTokenizer()
    game_x_wins = {**GAME, "result": "#X"}
    gambler_games = to_gambler_games([game_x_wins])
    _, y = build_tensors(gambler_games, tok, CHAR_BLOCK, winner_only=True)
    x_move_targets = [0, 1, 4, 5, 8, 9]  # both chars of X's moves 1, 3, 5 (winner)
    o_move_targets = [2, 3, 6, 7]        # both chars of O's moves 2, 4 (loser)
    assert (y[0, torch.tensor(x_move_targets)] != -1).all()
    assert (y[0, torch.tensor(o_move_targets)] == -1).all()
    assert (y[0, torch.tensor([10, 11, 12])] != -1).all()  # "#", "X", <eos> trained


def test_winner_only_and_expert_only_are_mutually_exclusive():
    tok = Tokenizer()
    game = {**GAME, "expert": "X", "winner": "X"}
    with pytest.raises(AssertionError):
        build_tensors([game], tok, BLOCK, expert_only=True, winner_only=True)


def test_expert_masking_regression_guard_after_gambler_addition():
    """Explicit regression guard: build_tensors' expert_only path must
    stay byte-for-byte the same behaviour it had before winner_only was
    added (this duplicates the intent of
    test_expert_masking_hides_opponent_moves on purpose)."""
    tok = Tokenizer()
    game_o = {**GAME, "expert": "O"}
    _, y = build_tensors([game_o], tok, BLOCK, expert_only=True)
    assert (y[0, torch.tensor([0, 2, 4])] == -1).all()  # X's moves masked
    assert (y[0, torch.tensor([1, 3])] != -1).all()  # O's moves trained
    assert y[0, 5] != -1 and y[0, 6] != -1  # result + <eos> trained

    game_x = {**GAME, "expert": "X"}
    _, y = build_tensors([game_x], tok, BLOCK, expert_only=True)
    assert (y[0, torch.tensor([0, 2, 4])] != -1).all()  # X's moves trained
    assert (y[0, torch.tensor([1, 3])] == -1).all()  # O's moves masked
