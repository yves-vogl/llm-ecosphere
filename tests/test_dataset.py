"""Tensorization: shifted targets, padding, and SFT-style loss masking."""

import torch

from minillm.dataset import build_tensors, split_games
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
