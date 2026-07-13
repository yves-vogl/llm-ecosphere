"""arena.py tests: pure game-loop mechanics and opponent policies only.

Deliberately hermetic — no checkpoint under runs/ or corpus under data/
is ever loaded (both are gitignored and absent in CI). Where a "model"
is needed, tests build a tiny fresh GPT in-memory (same trick as
test_utils.py's tiny_model) and save it to a pytest tmp_path checkpoint,
never touching runs/.
"""

import random

import pytest
import torch

from minillm.arena import (make_model_policy, play_game, play_match,
                           print_summary, resolve_opponent)
from minillm.config import ModelConfig
from minillm.evaluate import optimal_opponent, random_opponent
from minillm.game import Game, IllegalMoveError, other
from minillm.model import GPT
from minillm.solver import best_moves
from minillm.tokenizer import Tokenizer

CPU = torch.device("cpu")


def tiny_model(vocab_size: int) -> GPT:
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=vocab_size, block_size=24,
                            n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    model.eval()
    return model


def first_move_policy(game: Game) -> str:
    """Deterministic stub policy: always the first legal move."""
    return game.legal_moves()[0]


def last_move_policy(game: Game) -> str:
    """Deterministic stub policy: always the last legal move."""
    return game.legal_moves()[-1]


def illegal_policy(game: Game) -> str:
    """Stub policy that always cheats."""
    return "Z9"


# ----------------------------------------------------------------------
# Opponent policies: random and solver
# ----------------------------------------------------------------------
def test_random_opponent_returns_only_legal_moves():
    rng = random.Random(0)
    policy = random_opponent(rng)
    game = Game()
    while not game.is_over():
        move = policy(game)
        assert move in game.legal_moves()
        game.push(move)


def test_solver_opponent_is_always_negamax_optimal():
    rng = random.Random(0)
    policy = optimal_opponent(rng)
    game = Game()
    while not game.is_over():
        move = policy(game)
        _, optimal = best_moves(tuple(game.stacks))
        assert move in optimal
        game.push(move)


def test_resolve_opponent_random_and_solver_labels():
    rng = random.Random(0)
    generator = torch.Generator().manual_seed(0)
    policy, label = resolve_opponent("random", CPU, 0.0, rng, generator)
    assert label == "random"
    assert callable(policy)

    policy, label = resolve_opponent("solver", CPU, 0.0, rng, generator)
    assert "solver" in label
    assert callable(policy)


def test_resolve_opponent_rejects_missing_checkpoint_path():
    rng = random.Random(0)
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(SystemExit):
        resolve_opponent("no/such/checkpoint.pt", CPU, 0.0, rng, generator)


# ----------------------------------------------------------------------
# play_game: the core loop, with cheap deterministic stub policies
# ----------------------------------------------------------------------
def test_play_game_terminates_with_a_valid_result():
    game = play_game({"X": first_move_policy, "O": last_move_policy})
    assert game.is_over()
    assert game.winner() in ("X", "O", None)
    assert game.result_token in ("#X", "#O", "#=")
    assert 5 <= len(game.history) <= 9  # shortest possible win is 5 plies


def test_play_game_between_solver_and_random_never_favours_random():
    """A sanity check on the loop + policies together: perfect play
    should never lose to a random mover, whichever side it's on."""
    rng = random.Random(0)
    solver = optimal_opponent(rng)
    rnd = random_opponent(rng)
    for x_policy, o_policy, solver_side in ((solver, rnd, "X"), (rnd, solver, "O")):
        for _ in range(20):
            game = play_game({"X": x_policy, "O": o_policy})
            assert game.winner() != other(solver_side)


def test_play_game_rejects_illegal_move_from_a_policy():
    with pytest.raises(IllegalMoveError):
        play_game({"X": illegal_policy, "O": last_move_policy})


# ----------------------------------------------------------------------
# play_match: W/D/L accounting
# ----------------------------------------------------------------------
def test_play_match_wdl_counts_sum_to_games_played():
    rng = random.Random(1)
    solver = optimal_opponent(rng)
    rnd = random_opponent(rng)
    outcomes = play_match(solver, rnd, n_games=30)
    assert outcomes["win"] + outcomes["draw"] + outcomes["loss"] == 30


def test_play_match_perfect_player_never_loses_to_random():
    rng = random.Random(2)
    solver = optimal_opponent(rng)
    rnd = random_opponent(rng)
    outcomes = play_match(solver, rnd, n_games=40)
    assert outcomes["loss"] == 0
    assert outcomes["win"] > 0  # random play must occasionally blunder


def test_play_match_requires_positive_games():
    with pytest.raises(AssertionError):
        play_match(first_move_policy, last_move_policy, n_games=0)


def test_print_summary_reports_consistent_percentages(capsys):
    outcomes = {"win": 3, "draw": 1, "loss": 1}
    from collections import Counter
    print_summary("model-a", "model-b", 5, Counter(outcomes), mode="strict argmax")
    out = capsys.readouterr().out
    assert "model-a" in out and "model-b" in out
    assert "60.0%" in out  # 3/5 wins
    assert "5 games" in out


# ----------------------------------------------------------------------
# make_model_policy: a tiny fresh model, no checkpoint file needed
# ----------------------------------------------------------------------
def test_make_model_policy_strict_always_returns_legal_moves():
    tok = Tokenizer()
    model = tiny_model(tok.vocab_size)
    generator = torch.Generator().manual_seed(0)
    policy = make_model_policy(model, tok, CPU, temperature=0.0, generator=generator)
    game = play_game({"X": policy, "O": policy})
    assert game.is_over()  # would have raised IllegalMoveError otherwise


def test_make_model_policy_sampling_stays_legal_and_is_seed_reproducible():
    tok = Tokenizer()
    model = tiny_model(tok.vocab_size)

    def run():
        generator = torch.Generator().manual_seed(42)
        policy = make_model_policy(model, tok, CPU, temperature=1.0, generator=generator)
        game = play_game({"X": policy, "O": policy})
        return game.history

    history_a, history_b = run(), run()
    assert history_a == history_b  # same seed -> same rollout


# ----------------------------------------------------------------------
# End-to-end CLI: a tiny checkpoint written to tmp_path, never runs/
# ----------------------------------------------------------------------
def write_tiny_checkpoint(path) -> None:
    tok = Tokenizer()
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=tok.vocab_size, block_size=16,
                            n_layer=1, n_head=2, n_embd=16, dropout=0.0))
    torch.save(model.checkpoint_dict(stage="test", tokenizer=tok.name), path)


def test_cli_end_to_end_vs_random_and_vs_second_checkpoint(tmp_path, monkeypatch, capsys):
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    write_tiny_checkpoint(ckpt_a)
    write_tiny_checkpoint(ckpt_b)

    from minillm import arena

    monkeypatch.setattr(
        "sys.argv",
        ["arena", "--model", str(ckpt_a), "--vs", "random", "--games", "6", "--seed", "0"],
    )
    arena.main()
    out = capsys.readouterr().out
    assert "6 games" in out
    assert "win" in out and "draw" in out and "loss" in out

    monkeypatch.setattr(
        "sys.argv",
        ["arena", "--model", str(ckpt_a), "--vs", str(ckpt_b), "--games", "4", "--seed", "0"],
    )
    arena.main()
    out = capsys.readouterr().out
    assert str(ckpt_b) in out
    assert "4 games" in out
