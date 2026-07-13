"""Tokenizer: the bridge between game transcripts and tensor land.

A tokenizer maps text to a sequence of integer ids and back. Real LLMs
learn a subword vocabulary (BPE) with ~100k entries; our world is so
small that we can write the whole vocabulary down by hand. Two
vocabularies live here, selectable with ``get_tokenizer(name)``:

move-level (``Tokenizer``, the default) — 15 tokens:

    id  token   meaning
     0  <pad>   filler after the sequence ends (never trained on)
     1  <bos>   beginning of sequence
     2  <eos>   end of sequence
     3  A1      } the nine cells a piece can land on.
    ...          } One token per move — the equivalent of a
    11  C3      } word-level tokenizer.
    12  #X      result: X won
    13  #O      result: O won
    14  #=      result: draw

A full game becomes:  <bos> B1 A1 B2 C1 B3 #X <eos>
which is at most 1 + 9 + 1 + 1 = 12 tokens long.

character-level (``CharTokenizer``, exercise 1 in docs/08-exercises.md)
— 13 tokens: the same three specials plus the ten characters
A B C 1 2 3 # X O =. Every move is now TWO tokens ("B2" -> "B" "2"),
and so is every result ("#X" -> "#" "X"). The same game becomes

    <bos> B 1 A 1 B 2 C 1 B 3 # X <eos>

and grows to at most 1 + 9*2 + 2 + 1 = 22 tokens. The model must now
learn what the move tokenizer gave it for free: that "B" followed by
"2" names one cell, and that a lone "B" is only half a move. That is
the trade real BPE tokenizers make, in miniature — shorter sequences
bought at the price of the model knowing what is *inside* a token.

Both classes share one interface, so everything downstream asks the
tokenizer instead of assuming one token per move:

    tokens_per_move    1 (move) or 2 (char)
    encode_move(m)     the id sequence for one move, e.g. "B2"
    encode_prompt(ms)  <bos> + moves, ready for a forward pass
    encode_game(...)   <bos> moves result <eos>  (dataset building)
    max_game_tokens    longest possible complete game in tokens
"""

from __future__ import annotations

import json
from pathlib import Path

from .game import COLS, N, RESULT_DRAW, RESULT_O, RESULT_X

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]
MOVE_TOKENS = [f"{c}{r}" for c in COLS for r in range(1, N + 1)]  # A1..C3
RESULT_TOKENS = [RESULT_X, RESULT_O, RESULT_DRAW]
VOCAB = SPECIAL_TOKENS + MOVE_TOKENS + RESULT_TOKENS

# Longest possible sequence: <bos> + 9 moves + result + <eos>
MAX_GAME_TOKENS = 1 + N * N + 1 + 1

# The character-level alphabet, derived from the move-level tokens so
# the two vocabularies can never drift apart: A B C 1 2 3, then the
# result characters # X O = (no overlap — a move never contains one of
# the result characters, which keeps "is this a move token?" a purely
# local question).
MOVE_CHARS = list(COLS) + [str(r) for r in range(1, N + 1)]
RESULT_CHARS = ["#"] + [t[1] for t in RESULT_TOKENS]
CHAR_VOCAB = SPECIAL_TOKENS + MOVE_CHARS + RESULT_CHARS

# <bos> + 9 two-character moves + two-character result + <eos>
MAX_GAME_CHARS = 1 + N * N * 2 + 2 + 1


class Tokenizer:
    """Move-level token <-> id mapping over the fixed 15-token vocab.

    One token per move: "which cell" and "which token" are the same
    question. Subclasses swap the vocabulary and the move <-> token
    granularity; the id-mapping plumbing is shared.
    """

    name = "move"
    vocab: list[str] = VOCAB
    tokens_per_move = 1  # a move is one token; overridden by CharTokenizer
    max_game_tokens = MAX_GAME_TOKENS

    def __init__(self) -> None:
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

    # -- basic properties ------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def move_ids(self) -> list[int]:
        """Ids of the nine cell tokens A1..C3."""
        return [self.token_to_id[t] for t in MOVE_TOKENS]

    @property
    def result_ids(self) -> list[int]:
        """Ids of the three result tokens #X, #O, #=."""
        return [self.token_to_id[t] for t in RESULT_TOKENS]

    def is_move_id(self, token_id: int) -> bool:
        """Does this id belong to a move (or, char-level, to part of one)?"""
        return self.id_to_token[token_id] in MOVE_TOKENS

    def is_result_id(self, token_id: int) -> bool:
        """Does this id belong to a result (or part of one)?"""
        return self.id_to_token[token_id] in RESULT_TOKENS

    # -- encode / decode -------------------------------------------------
    def encode(self, tokens: list[str]) -> list[int]:
        """Token strings -> ids. Unknown tokens raise (no <unk> here:
        in a world this small an unknown token is always a bug)."""
        try:
            return [self.token_to_id[t] for t in tokens]
        except KeyError as err:
            raise ValueError(f"unknown token {err.args[0]!r}") from err

    def decode(self, ids: list[int]) -> list[str]:
        """Ids -> token strings."""
        return [self.id_to_token[int(i)] for i in ids]

    def encode_move(self, move: str) -> list[int]:
        """One move (e.g. "B2") -> its id sequence: one id here, two in
        the char-level subclass. The move-assembly counterpart used by
        evaluate.py and play.py."""
        return self.encode([move])

    def encode_prompt(self, moves: list[str]) -> list[int]:
        """<bos> + moves as ids — the standard inference prompt."""
        ids = [self.bos_id]
        for move in moves:
            ids.extend(self.encode_move(move))
        return ids

    def encode_game(self, moves: list[str], result: str) -> list[int]:
        """A complete game -> <bos> moves result <eos> as ids."""
        return self.encode_prompt(list(moves)) + self._result_ids(result) + [self.eos_id]

    def _result_ids(self, result: str) -> list[int]:
        return self.encode([result])

    def group_units(self, tokens: list[str]) -> list[str]:
        """Re-assemble decoded tokens into transcript units (moves,
        results, specials). Identity at move level; the char-level
        subclass glues character pairs back together. Malformed tails
        (a dangling half-move) pass through untouched so the verifier
        can name them."""
        return list(tokens)

    # -- persistence (mirrors how real tokenizers ship a vocab file) ------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        saved = json.loads(Path(path).read_text())
        tok = cls()
        assert saved == tok.token_to_id, "vocab file does not match this code version"
        return tok


class CharTokenizer(Tokenizer):
    """Character-level tokenizer: "B2" becomes the two tokens "B", "2".

    Same interface as `Tokenizer`; the splitting and re-assembly of
    moves is hidden here so that dataset building and inference code
    never special-case the granularity (docs/08-exercises.md, ex. 1).
    """

    name = "char"
    vocab = CHAR_VOCAB
    tokens_per_move = 2
    max_game_tokens = MAX_GAME_CHARS

    @property
    def move_ids(self) -> list[int]:
        """Ids of the six characters that can appear inside a move."""
        return [self.token_to_id[c] for c in MOVE_CHARS]

    @property
    def result_ids(self) -> list[int]:
        """Ids of the four characters that can appear inside a result."""
        return [self.token_to_id[c] for c in RESULT_CHARS]

    def is_move_id(self, token_id: int) -> bool:
        return self.id_to_token[token_id] in MOVE_CHARS

    def is_result_id(self, token_id: int) -> bool:
        return self.id_to_token[token_id] in RESULT_CHARS

    def encode_move(self, move: str) -> list[int]:
        # Validate against the cell names first: splitting blindly would
        # happily encode "B22" or "A#", which the move-level tokenizer
        # rejects — the two classes must fail identically on bad input.
        if move not in MOVE_TOKENS:
            raise ValueError(f"unknown move {move!r}")
        return self.encode(list(move))  # "B2" -> ids of "B", "2"

    def _result_ids(self, result: str) -> list[int]:
        return self.encode(list(result))  # "#X" -> ids of "#", "X"

    def group_units(self, tokens: list[str]) -> list[str]:
        """Glue characters back into two-character units: "B","2" -> "B2".

        Specials stay single. A malformed pair ("A","#") or a dangling
        final character is emitted as-is — downstream verification will
        reject it with a readable message instead of us guessing."""
        units: list[str] = []
        i = 0
        while i < len(tokens):
            if tokens[i] in SPECIAL_TOKENS:
                units.append(tokens[i])
                i += 1
            elif i + 1 < len(tokens) and tokens[i + 1] not in SPECIAL_TOKENS:
                units.append(tokens[i] + tokens[i + 1])
                i += 2
            else:  # dangling half-unit before a special or end of stream
                units.append(tokens[i])
                i += 1
        return units


TOKENIZERS = {cls.name: cls for cls in (Tokenizer, CharTokenizer)}


def get_tokenizer(name: str = "move") -> Tokenizer:
    """Factory used by train.py (via --tokenizer) and by everything
    that loads a checkpoint (the checkpoint records which tokenizer
    it was trained with — mixing them up would be silent nonsense,
    since both vocabularies share the low ids)."""
    try:
        return TOKENIZERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown tokenizer {name!r} (choose from {sorted(TOKENIZERS)})"
        ) from None
