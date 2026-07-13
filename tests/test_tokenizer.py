"""Token <-> id mapping must be exact and stable."""

import pytest

from minillm.tokenizer import (CHAR_VOCAB, MAX_GAME_CHARS, MAX_GAME_TOKENS,
                               MOVE_TOKENS, VOCAB, CharTokenizer, Tokenizer,
                               get_tokenizer)


def test_vocab_is_15_tokens():
    assert len(VOCAB) == 15
    assert Tokenizer().vocab_size == 15


def test_roundtrip():
    tok = Tokenizer()
    tokens = ["<bos>", "B1", "A1", "B2", "#X", "<eos>"]
    assert tok.decode(tok.encode(tokens)) == tokens


def test_encode_game_layout():
    tok = Tokenizer()
    ids = tok.encode_game(["B1", "A1"], "#X")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == ["<bos>", "B1", "A1", "#X", "<eos>"]


def test_unknown_token_raises():
    with pytest.raises(ValueError, match="unknown token"):
        Tokenizer().encode(["D5"])


def test_move_and_result_id_helpers():
    tok = Tokenizer()
    assert len(tok.move_ids) == 9
    assert len(tok.result_ids) == 3
    assert all(tok.is_move_id(i) for i in tok.move_ids)
    assert not tok.is_move_id(tok.bos_id)
    assert set(tok.decode(tok.move_ids)) == set(MOVE_TOKENS)


def test_max_game_tokens():
    # <bos> + 9 moves + result + <eos>
    assert MAX_GAME_TOKENS == 12


def test_save_load(tmp_path):
    tok = Tokenizer()
    tok.save(tmp_path / "vocab.json")
    assert Tokenizer.load(tmp_path / "vocab.json").token_to_id == tok.token_to_id


# ----------------------------------------------------------------------
# Character-level tokenizer (docs/08-exercises.md, exercise 1)
# ----------------------------------------------------------------------
def test_char_vocab_is_13_tokens():
    assert len(CHAR_VOCAB) == 13
    assert CharTokenizer().vocab_size == 13


def test_char_encode_game_splits_moves_and_result():
    tok = CharTokenizer()
    ids = tok.encode_game(["B1", "A1"], "#X")
    assert tok.decode(ids) == ["<bos>", "B", "1", "A", "1", "#", "X", "<eos>"]


def test_char_max_game_tokens():
    # <bos> + 9 two-character moves + two-character result + <eos>
    assert MAX_GAME_CHARS == 22
    nine_moves = [f"{c}{r}" for c in "ABC" for r in (1, 2, 3)]
    assert len(CharTokenizer().encode_game(nine_moves, "#=")) == MAX_GAME_CHARS


def test_char_move_encoding_and_prompt():
    tok = CharTokenizer()
    assert tok.tokens_per_move == 2
    assert tok.decode(tok.encode_move("B2")) == ["B", "2"]
    assert tok.decode(tok.encode_prompt(["B1", "A1"])) == ["<bos>", "B", "1", "A", "1"]


def test_char_encode_move_rejects_malformed_input_like_move_level():
    # Both tokenizers must fail identically on bad input instead of
    # silently encoding character soup.
    for bad in ("B22", "2B", "AA", "A#", ""):
        with pytest.raises(ValueError):
            CharTokenizer().encode_move(bad)
        with pytest.raises(ValueError):
            Tokenizer().encode_move(bad)


def test_move_level_prompt_matches_old_encoding():
    tok = Tokenizer()
    assert tok.tokens_per_move == 1
    assert tok.decode(tok.encode_prompt(["B1", "A1"])) == ["<bos>", "B1", "A1"]


def test_char_move_and_result_ids_are_disjoint():
    tok = CharTokenizer()
    assert set(tok.move_ids) & set(tok.result_ids) == set()
    assert all(tok.is_move_id(i) for i in tok.move_ids)
    assert all(tok.is_result_id(i) for i in tok.result_ids)
    assert not tok.is_move_id(tok.bos_id)


def test_char_group_units_reassembles_pairs():
    tok = CharTokenizer()
    tokens = ["B", "1", "A", "1", "#", "X", "<eos>"]
    assert tok.group_units(tokens) == ["B1", "A1", "#X", "<eos>"]
    # a dangling half-move survives as-is for the verifier to reject
    assert tok.group_units(["B", "1", "A", "<eos>"]) == ["B1", "A", "<eos>"]
    # move level: identity
    assert Tokenizer().group_units(["B1", "#X", "<eos>"]) == ["B1", "#X", "<eos>"]


def test_get_tokenizer_factory():
    assert get_tokenizer("move").name == "move"
    assert get_tokenizer("char").name == "char"
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_tokenizer("bpe")
