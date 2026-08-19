"""The lookup-table baseline: table building, fallbacks, model interface."""

import torch

from minillm.baseline_lookup import LookupModel, prefix_coverage
from minillm.tokenizer import Tokenizer

TOK = Tokenizer()
GAMES = [
    {"moves": ["A1", "B1", "A2", "B2", "A3"], "result": "#X"},
    {"moves": ["A1", "B1", "A2", "C1", "A3"], "result": "#X"},
]


def test_table_records_next_token_counts_per_prefix():
    table = LookupModel(GAMES, TOK)
    bos = (TOK.bos_id,)
    # After <bos> both games open with A1: one entry, count 2.
    assert table.table[bos] == {TOK.encode(["A1"])[0]: 2}
    # After A1 B1 A2, the two games diverge: B2 once, C1 once.
    prefix = tuple(TOK.encode_prompt(["A1", "B1", "A2"]))
    counts = table.table[prefix]
    assert counts[TOK.encode(["B2"])[0]] == 1
    assert counts[TOK.encode(["C1"])[0]] == 1


def test_seen_prefix_returns_observed_distribution():
    table = LookupModel(GAMES, TOK)
    prefix = tuple(TOK.encode_prompt(["A1", "B1", "A2"]))
    logits = table.logits_for_prefix(prefix)
    probs = logits.exp()
    assert torch.isclose(probs[TOK.encode(["B2"])[0]], torch.tensor(0.5))
    assert torch.isclose(probs[TOK.encode(["C1"])[0]], torch.tensor(0.5))
    # Everything never observed after this prefix has probability zero.
    assert probs.sum().item() == 1.0


def test_unseen_prefix_falls_back_to_uniform_over_legal_moves():
    table = LookupModel(GAMES, TOK)
    prefix = tuple(TOK.encode_prompt(["C1"]))  # never in the toy corpus
    assert not table.seen(prefix)
    probs = table.logits_for_prefix(prefix).exp()
    # After C1 every column still has room: A1, B1, C2 are the legal cells.
    legal_ids = [TOK.encode([m])[0] for m in ("A1", "B1", "C2")]
    for token_id in legal_ids:
        assert torch.isclose(probs[token_id], torch.tensor(1 / 3))
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_unseen_terminal_prefix_falls_back_to_uniform_over_results():
    table = LookupModel(GAMES, TOK)
    # X wins with three in the A column via a different move order than
    # the training games used - a finished game the table never saw.
    moves = ["A1", "C1", "A2", "B1", "A3"]
    prefix = tuple(TOK.encode_prompt(moves))
    assert not table.seen(prefix)
    probs = table.logits_for_prefix(prefix).exp()
    for token_id in TOK.result_ids:
        assert torch.isclose(probs[token_id], torch.tensor(1 / 3))
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_call_wears_the_gpt_inference_interface():
    table = LookupModel(GAMES, TOK)
    x = torch.tensor([TOK.encode_prompt(["A1", "B1"])], dtype=torch.long)
    logits, loss = table(x)
    assert logits.shape == (1, 1, TOK.vocab_size)
    assert loss is None
    # Greedy continuation of game 1/2's shared prefix is A2 (count 2).
    assert int(logits[0, -1].argmax()) == TOK.encode(["A2"])[0]


def test_prefix_coverage_counts_seen_positions():
    table = LookupModel(GAMES, TOK)
    val = [
        {"moves": ["A1", "B1", "A2", "B2", "A3"], "result": "#X"},  # fully seen
        {"moves": ["B1", "A1", "B2"], "result": "#X"},  # only <bos> seen
    ]
    cov = prefix_coverage(table, TOK, val)
    # Game 1 contributes 5 seen prefixes; game 2 contributes 1 (empty
    # history) + 2 unseen.
    assert cov["positions"] == 8
    assert cov["seen"] == 6
    assert abs(cov["coverage"] - 6 / 8) < 1e-9
