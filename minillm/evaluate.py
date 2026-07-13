"""Behavioural evaluation: what did the model actually learn?

Loss numbers say "the model fits the data"; these metrics say what that
means in the world the data describes:

  legality     Does the model respect the rules? Measured three ways:
               probability mass it puts on legal moves, how often its
               top choice is legal (teacher-forced on held-out games),
               and how often free-running self-play stays clean.
  refereeing   After a finished game, does it predict the correct
               result token (#X / #O / #=)? Tests that it tracks board
               state well enough to *recognize* wins.
  strength     Win/draw/loss rates against a random player and against
               the perfect solver, plus how often its chosen move is
               one of the solver's optimal moves.

Run: python -m minillm.evaluate --out runs/eval.json   (or `make eval`)
Compare the pretrained vs the finetuned checkpoint with --ckpt to see
what each training stage contributed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from .dataset import read_jsonl, split_games
from .game import Game
from .solver import best_moves
from .utils import (default_checkpoint, greedy_unit, legal_move_logprobs,
                    load_model, pick_device, sample_unit, tokenizer_for_checkpoint)


def model_move_strict(model, tokenizer, game: Game, device) -> str:
    """The model's favourite LEGAL move (argmax over legal moves only).

    Moves are ranked by their joint token log-probability, which at
    move level is the familiar argmax over legal tokens and at char
    level chains both characters' conditionals: p("B")·p("2"|"B").
    """
    legal = game.legal_moves()
    scores = legal_move_logprobs(model, tokenizer, game.history, legal, device)
    return legal[int(scores.argmax())]


# ----------------------------------------------------------------------
# Legality + refereeing on held-out games (teacher forcing)
# ----------------------------------------------------------------------
def eval_on_val_games(model, tokenizer, val_games, device) -> dict:
    n_moves = argmax_legal = 0
    legal_mass = 0.0
    n_results = result_correct = 0

    for g in val_games:
        game = Game()
        for recorded_move in g["moves"]:
            legal = game.legal_moves()
            # Does the model's raw favourite continuation form a legal
            # move? At char level a move only counts if BOTH greedily
            # decoded characters combine to a legal cell.
            if greedy_unit(model, tokenizer, game.history, device) in legal:
                argmax_legal += 1
            # Probability mass on legal moves, joint over each move's tokens.
            legal_mass += legal_move_logprobs(
                model, tokenizer, game.history, legal, device).exp().sum().item()
            n_moves += 1
            game.push(recorded_move)  # follow the recorded game, not the model
        result_correct += greedy_unit(model, tokenizer, game.history, device) == g["result"]
        n_results += 1

    return {
        "positions": n_moves,
        "argmax_legal_rate": argmax_legal / n_moves,
        "mean_legal_prob_mass": legal_mass / n_moves,
        "result_prediction_accuracy": result_correct / n_results,
    }


# ----------------------------------------------------------------------
# Free-running self-play legality
# ----------------------------------------------------------------------
def eval_rollout_legality(model, tokenizer, device, n_games: int, seed: int) -> dict:
    """Sample whole moves with no engine help and count the accidents.

    One "attempt" is one complete move — a single sampled token at move
    level, two at char level — so first_try_legal_rate stays comparable
    across tokenizers: it is always "of the moves the model tried to
    play, how many were legal on the first try".
    """
    generator = torch.Generator().manual_seed(seed)
    total = illegal = clean_games = 0

    for _ in range(n_games):
        game = Game()
        clean = True
        while not game.is_over():
            attempt = sample_unit(model, tokenizer, game.history, device, generator)
            total += 1
            if attempt not in game.legal_moves():
                illegal += 1
                clean = False
                # project onto the legal moves so the rollout can continue
                legal = game.legal_moves()
                legal_probs = legal_move_logprobs(
                    model, tokenizer, game.history, legal, device).exp().cpu()
                pick = int(torch.multinomial(legal_probs / legal_probs.sum(), 1,
                                             generator=generator))
                attempt = legal[pick]
            game.push(attempt)
        clean_games += clean

    return {
        "games": n_games,
        "first_try_legal_rate": 1 - illegal / total,
        "clean_game_rate": clean_games / n_games,
    }


# ----------------------------------------------------------------------
# Playing strength
# ----------------------------------------------------------------------
def play_one(model, tokenizer, device, model_side: str, opponent) -> str:
    game = Game()
    while not game.is_over():
        if game.to_move == model_side:
            move = model_move_strict(model, tokenizer, game, device)
        else:
            move = opponent(game)
        game.push(move)
    winner = game.winner()
    if winner is None:
        return "draw"
    return "win" if winner == model_side else "loss"


def eval_matches(model, tokenizer, device, opponent_factory, n_games: int, seed: int) -> dict:
    rng = random.Random(seed)
    opponent = opponent_factory(rng)
    outcomes = Counter(
        play_one(model, tokenizer, device, "X" if i % 2 == 0 else "O", opponent)
        for i in range(n_games)
    )
    return {
        "games": n_games,
        "win_rate": outcomes["win"] / n_games,
        "draw_rate": outcomes["draw"] / n_games,
        "loss_rate": outcomes["loss"] / n_games,
    }


def random_opponent(rng: random.Random):
    return lambda game: rng.choice(game.legal_moves())


def optimal_opponent(rng: random.Random):
    def policy(game: Game) -> str:
        _, moves = best_moves(tuple(game.stacks))
        return rng.choice(moves)  # any optimal move; random among ties
    return policy


# ----------------------------------------------------------------------
# Agreement with the solver
# ----------------------------------------------------------------------
def eval_expert_agreement(model, tokenizer, device, n_rollouts: int, seed: int) -> dict:
    """Visit positions via random play, ask the model for its move, and
    check whether the solver counts it among the optimal ones.

    Note: the model conditions on the move *history*, the solver on the
    resulting *position* — several histories can share a position. We
    keep the first history seen per position.
    """
    rng = random.Random(seed)
    histories: dict[tuple, list[str]] = {}
    for _ in range(n_rollouts):
        game = Game()
        while not game.is_over():
            histories.setdefault(tuple(game.stacks), list(game.history))
            game.push(rng.choice(game.legal_moves()))

    agree = 0
    for stacks, history in histories.items():
        move = model_move_strict(model, tokenizer, Game.from_moves(history), device)
        _, optimal = best_moves(stacks)
        agree += move in optimal
    return {"positions": len(histories), "optimal_move_rate": agree / len(histories)}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--ckpt", default=None, help="default: finetune, else pretrain")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-games", type=int, default=200)
    parser.add_argument("--games-vs-random", type=int, default=400)
    parser.add_argument("--games-vs-optimal", type=int, default=200)
    parser.add_argument("--agreement-rollouts", type=int, default=300)
    parser.add_argument("--out", default=None, help="also write results as JSON")
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt_path = Path(args.ckpt or default_checkpoint())
    model, ckpt = load_model(ckpt_path, device)
    tokenizer = tokenizer_for_checkpoint(ckpt)
    # Reuse the val fraction recorded at training time so "held-out"
    # really means held out, even after a non-default --val-frac run.
    _, val_games = split_games(read_jsonl(Path(args.data_dir) / "all_games.jsonl"),
                               val_frac=ckpt.get("val_frac", 0.1))

    print(f"evaluating {ckpt_path} (stage {ckpt.get('stage')}, "
          f"tokenizer {tokenizer.name}, "
          f"val loss {ckpt.get('val_loss', float('nan')):.4f})\n")

    results = {
        "checkpoint": str(ckpt_path),
        "stage": ckpt.get("stage"),
        "tokenizer": tokenizer.name,
        "legality_teacher_forced": eval_on_val_games(model, tokenizer, val_games, device),
        "legality_free_running": eval_rollout_legality(
            model, tokenizer, device, args.rollout_games, args.seed),
        "vs_random": eval_matches(
            model, tokenizer, device, random_opponent, args.games_vs_random, args.seed),
        "vs_optimal": eval_matches(
            model, tokenizer, device, optimal_opponent, args.games_vs_optimal, args.seed),
        "solver_agreement": eval_expert_agreement(
            model, tokenizer, device, args.agreement_rollouts, args.seed),
    }

    tf = results["legality_teacher_forced"]
    fr = results["legality_free_running"]
    vr, vo = results["vs_random"], results["vs_optimal"]
    ag = results["solver_agreement"]
    print(f"legality   argmax legal          {tf['argmax_legal_rate']:8.1%}   "
          f"({tf['positions']} held-out positions)")
    print(f"           legal prob mass       {tf['mean_legal_prob_mass']:8.1%}")
    print(f"           free-running 1st try  {fr['first_try_legal_rate']:8.1%}   "
          f"({fr['games']} self-play games)")
    print(f"           clean games           {fr['clean_game_rate']:8.1%}")
    print(f"refereeing result prediction     {tf['result_prediction_accuracy']:8.1%}")
    print(f"strength   vs random   W/D/L     {vr['win_rate']:.1%} / "
          f"{vr['draw_rate']:.1%} / {vr['loss_rate']:.1%}   ({vr['games']} games)")
    print(f"           vs optimal  W/D/L     {vo['win_rate']:.1%} / "
          f"{vo['draw_rate']:.1%} / {vo['loss_rate']:.1%}   ({vo['games']} games)")
    print(f"           optimal-move rate     {ag['optimal_move_rate']:8.1%}   "
          f"({ag['positions']} positions)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
