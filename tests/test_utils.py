"""The move-assembly helpers must reproduce the old single-token
semantics exactly at move level (that is what makes the char-tokenizer
eval numbers comparable to the baseline), and follow the chain rule at
char level."""

import torch
import torch.nn.functional as F

from minillm.config import ModelConfig
from minillm.game import Game
from minillm.model import GPT
from minillm.tokenizer import CharTokenizer, Tokenizer
from minillm.utils import (greedy_unit, legal_move_logprobs, logits_for_ids,
                           next_token_logits, sample_unit,
                           tokenizer_for_checkpoint)

CPU = torch.device("cpu")


def tiny_model(vocab_size: int) -> GPT:
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=vocab_size, block_size=24,
                            n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    model.eval()
    return model


def test_tokenizer_for_checkpoint_defaults_to_move():
    assert tokenizer_for_checkpoint({}).name == "move"  # pre-exercise checkpoints
    assert tokenizer_for_checkpoint({"tokenizer": "char"}).name == "char"


def test_legal_move_logprobs_matches_single_token_math_at_move_level():
    tok = Tokenizer()
    model = tiny_model(tok.vocab_size)
    game = Game.from_moves(["B1", "A1"])
    legal = game.legal_moves()
    logits = next_token_logits(model, tok, game.history, CPU)
    expected = F.log_softmax(logits, dim=-1)[tok.encode(legal)]
    got = legal_move_logprobs(model, tok, game.history, legal, CPU)
    assert torch.allclose(got, expected, atol=1e-6)


def test_greedy_unit_is_plain_argmax_at_move_level():
    tok = Tokenizer()
    model = tiny_model(tok.vocab_size)
    game = Game.from_moves(["B1"])
    logits = next_token_logits(model, tok, game.history, CPU)
    assert greedy_unit(model, tok, game.history, CPU) == tok.id_to_token[int(logits.argmax())]


def test_char_legal_move_logprobs_follow_the_chain_rule():
    """p("B2" | history) must equal p("B" | history) * p("2" | history + "B")."""
    tok = CharTokenizer()
    model = tiny_model(tok.vocab_size)
    game = Game()
    legal = game.legal_moves()  # A1, B1, C1
    got = legal_move_logprobs(model, tok, game.history, legal, CPU)
    for move, score in zip(legal, got):
        prefix = list(tok.encode_prompt(game.history))
        expected = 0.0
        for token_id in tok.encode_move(move):
            log_probs = F.log_softmax(logits_for_ids(model, prefix, CPU), dim=-1)
            expected += log_probs[token_id].item()
            prefix.append(token_id)
        assert abs(score.item() - expected) < 1e-5


def test_char_units_assemble_tokens_per_move_tokens():
    tok = CharTokenizer()
    model = tiny_model(tok.vocab_size)
    generator = torch.Generator().manual_seed(0)
    greedy = greedy_unit(model, tok, [], CPU)
    sampled = sample_unit(model, tok, [], CPU, generator)
    # Two decoded tokens joined; an untrained model may emit any pair
    # (including specials), so only the token count is guaranteed.
    for unit in (greedy, sampled):
        rest = unit
        count = 0
        while rest:
            for token in tok.token_to_id:
                if rest.startswith(token):
                    rest = rest[len(token):]
                    count += 1
                    break
            else:
                raise AssertionError(f"cannot split unit {unit!r} into tokens")
        assert count == tok.tokens_per_move


def test_sample_unit_greedy_when_temperature_zero():
    tok = Tokenizer()
    model = tiny_model(tok.vocab_size)
    generator = torch.Generator().manual_seed(0)
    assert sample_unit(model, tok, [], CPU, generator, temperature=0.0) == \
        greedy_unit(model, tok, [], CPU)
