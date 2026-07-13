"""Arena: the unified harness for exploring any trained checkpoint.

play.py lets you sit down against the model. evaluate.py measures its
behaviour against a random mover and the perfect solver, aggregated over
many games. Those two scripts answer different questions with different
entry points; arena.py is the ONE command a newcomer needs to reach for
instead — it ties both together and adds the option neither has: pitting
two checkpoints against each other.

    python -m minillm.arena --model <ckpt> --vs human
    python -m minillm.arena --model <ckpt> --vs random  [--games N]
    python -m minillm.arena --model <ckpt> --vs solver  [--games N]
    python -m minillm.arena --model <ckpt> --vs <path-to-other-ckpt> [--games N]

All four modes load the SAME checkpoint the SAME way (utils.load_model +
utils.tokenizer_for_checkpoint) and drive the model with the SAME move
picker (play.py's `model_move`, strict-legal argmax by default; legal
sampling if --temperature > 0) — so what you see in `--vs human` is
exactly the policy being measured in the other three modes, not a
lookalike. Nothing here reimplements the rules (game.py), the perfect
player (solver.py, via evaluate.py's opponent factories), or the
checkpoint/tokenizer/move-probability plumbing (utils.py) — arena.py only
adds the game loop, the win/draw/loss bookkeeping, and the CLI that
routes between opponents.

The model never sees a board: every policy here — including the
opponent, when it is a second checkpoint — only ever conditions on the
move sequence so far, exactly as during training. Model moves are always
drawn from `game.legal_moves()` (never the full vocabulary), so an
"illegal move" from a model is impossible by construction; `play_game`
still checks for it defensively in case a future/custom opponent policy
misbehaves, and fails loudly rather than silently corrupting the board.

Run: python -m minillm.arena --model runs/pretrain/model.pt --vs random
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import torch

from .evaluate import optimal_opponent, random_opponent
from .game import COLS, Game, IllegalMoveError, other
from .play import model_move, referee_verdict, show_distribution
from .utils import (default_checkpoint, load_model, next_token_logits,
                    pick_device, set_seed, tokenizer_for_checkpoint)

Policy = Callable[[Game], str]

ARENA_HELP = """commands:
  A / B / C     drop a piece into that column
  A1 .. C3      same, naming the exact landing cell
  why           show the model's probability distribution over its next move
  u             undo your last move (and the model's reply)
  ?             this help
  q             quit"""


# ----------------------------------------------------------------------
# Policies: every opponent (model, random, solver, human) reduces to a
# function Game -> move string picked from game.legal_moves().
# ----------------------------------------------------------------------
def make_model_policy(
    model, tokenizer, device: torch.device, temperature: float, generator: torch.Generator
) -> Policy:
    """Wrap a loaded checkpoint as a Policy.

    Reuses play.py's `model_move` (strict=True) directly rather than
    reimplementing move selection: strict-legal argmax when
    temperature <= 0, legal sampling at that temperature otherwise —
    the same logic play.py uses for the model's turn against a human.
    """
    return lambda game: model_move(
        model, tokenizer, game, device, strict=True,
        temperature=temperature, generator=generator,
    )


def resolve_opponent(
    vs: str, device: torch.device, temperature: float,
    rng: random.Random, generator: torch.Generator,
) -> tuple[Policy, str]:
    """Build the opponent Policy for --vs, plus a friendly label.

    "random" and "solver" reuse evaluate.py's opponent factories
    (random_opponent / optimal_opponent — the latter built on
    solver.py's negamax, so a correct expert model should never lose to
    it). Anything else is treated as the path to a second checkpoint,
    loaded and driven exactly like the main model.
    """
    if vs == "random":
        return random_opponent(rng), "random"
    if vs == "solver":
        return optimal_opponent(rng), "solver (perfect negamax)"
    opp_path = Path(vs)
    if not opp_path.exists():
        raise SystemExit(
            f"--vs must be 'human', 'random', 'solver', or an existing checkpoint "
            f"path (got {vs!r}, and no such file exists)"
        )
    opp_model, opp_ckpt = load_model(opp_path, device)
    opp_tokenizer = tokenizer_for_checkpoint(opp_ckpt)
    return (
        make_model_policy(opp_model, opp_tokenizer, device, temperature, generator),
        str(opp_path),
    )


# ----------------------------------------------------------------------
# Game loop + W/D/L bookkeeping
# ----------------------------------------------------------------------
def play_game(policy_for: dict[str, Policy]) -> Game:
    """One complete game between two policies (X moves first).

    Raises IllegalMoveError with a readable message if a policy ever
    proposes a move outside game.legal_moves() — a safety net, since
    every policy in this file already restricts itself to legal moves
    by construction (model policies via play.model_move's strict mode,
    random/solver via evaluate.py's factories).
    """
    game = Game()
    while not game.is_over():
        legal = game.legal_moves()
        move = policy_for[game.to_move](game)
        if move not in legal:
            raise IllegalMoveError(
                f"policy for {game.to_move} proposed illegal move {move!r}; "
                f"legal moves were {legal}"
            )
        game.push(move)
    return game


def play_match(main_policy: Policy, opponent_policy: Policy, n_games: int) -> Counter:
    """Play n_games between the two policies, alternating who moves
    first each game so first-move advantage cancels out (the same
    alternation evaluate.eval_matches uses). Outcomes are counted from
    main_policy's perspective: Counter with keys "win"/"draw"/"loss".
    """
    assert n_games > 0, "--games must be positive"
    outcomes: Counter = Counter()
    for i in range(n_games):
        main_side = "X" if i % 2 == 0 else "O"
        game = play_game({main_side: main_policy, other(main_side): opponent_policy})
        winner = game.winner()
        if winner is None:
            outcomes["draw"] += 1
        elif winner == main_side:
            outcomes["win"] += 1
        else:
            outcomes["loss"] += 1
    return outcomes


def print_summary(label_a: str, label_b: str, n_games: int, outcomes: Counter, mode: str) -> None:
    """Friendly W/D/L report: the two players, N games, counts + percentages."""
    w, d, l = outcomes["win"], outcomes["draw"], outcomes["loss"]
    print(f"\n{label_a}  vs  {label_b}   |   {n_games} games   |   model plays: {mode}")
    print(f"  win  {w:4d} / {n_games}   ({w / n_games:6.1%})")
    print(f"  draw {d:4d} / {n_games}   ({d / n_games:6.1%})")
    print(f"  loss {l:4d} / {n_games}   ({l / n_games:6.1%})")


# ----------------------------------------------------------------------
# --vs human: interactive single game
# ----------------------------------------------------------------------
def read_human_input(game: Game) -> str:
    """Prompt until we get a move or a command.

    Mirrors play.py's read_human_move (same undo/help/quit semantics,
    same "validate by replaying on a copy" trick) but spells the
    probability-distribution command "why" per the arena spec, rather
    than play.py's "p" — the OUTPUT of that command is still exactly
    play.py's show_distribution, reused unchanged below.
    """
    while True:
        try:
            raw = input(f"  your move [{game.to_move}] (A/B/C, why, u, ?, q) > ").strip().upper()
        except EOFError:  # stdin closed (e.g. piped input ran out)
            raw = "Q"
        if raw == "Q":
            print("  bye!")
            raise SystemExit(0)
        if raw == "?":
            print(ARENA_HELP)
            continue
        if raw in ("WHY", "W"):
            return "WHY"
        if raw == "U":
            return "U"
        if len(raw) == 1 and raw in COLS:
            height = len(game.stacks[COLS.index(raw)])
            if height >= 3:
                print(f"  column {raw} is full")
                continue
            raw = f"{raw}{height + 1}"
        try:
            game.copy().push(raw)  # validate without mutating
            return raw
        except IllegalMoveError as err:
            print(f"  illegal: {err}")


def run_human_game(model, tokenizer, device: torch.device, temperature: float, seed: int) -> None:
    """Interactive human-vs-model game, ending with a readable move log."""
    generator = torch.Generator().manual_seed(seed)
    human, ai = "X", other("X")
    print(f"you: {human}, model: {ai}  |  model plays: "
          f"{'strict argmax' if temperature <= 0 else f'sampled @ T={temperature}'}\n"
          f"{ARENA_HELP}\n")

    game = Game()
    while not game.is_over():
        print(game.render())
        if game.to_move == human:
            move = read_human_input(game)
            if move == "WHY":
                logits = next_token_logits(model, tokenizer, game.history, device)
                show_distribution(logits, tokenizer, game)
                continue
            if move == "U":
                if len(game.history) >= 2:
                    game = Game.from_moves(game.history[:-2])
                    print("  undid your last move and the model's reply")
                else:
                    print("  nothing to undo yet")
                continue
        else:
            move = model_move(model, tokenizer, game, device, strict=True,
                              temperature=temperature, generator=generator)
            print(f"  model plays {move}")
        game.push(move)
        print()

    print(game.render())
    winner = game.winner()
    if winner is None:
        print("\ndraw!")
    elif winner == human:
        print("\nyou win!")
    else:
        print("\nthe model wins!")
    print(f"(model as referee predicts: "
          f"{referee_verdict(model, tokenizer, game, device)}, "
          f"actual result token: {game.result_token})")
    print(f"moves: {' '.join(game.history)}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arena: play or evaluate a checkpoint against any opponent "
                    "(human, random, the perfect solver, or a second checkpoint)"
    )
    parser.add_argument("--model", default=None,
                        help="checkpoint to load and drive (default: finetune, else pretrain)")
    parser.add_argument("--vs", required=True,
                        help="human | random | solver | path to a second checkpoint")
    parser.add_argument("--games", type=int, default=200,
                        help="number of games for non-interactive modes (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = strict argmax over legal moves (default); "
                             ">0 samples legally at this temperature")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)
    model_path = args.model or default_checkpoint()
    model, ckpt = load_model(model_path, device)
    tokenizer = tokenizer_for_checkpoint(ckpt)
    print(f"model: {model_path} (stage {ckpt.get('stage')}, tokenizer {tokenizer.name})")

    if args.vs == "human":
        run_human_game(model, tokenizer, device, args.temperature, args.seed)
        return

    generator = torch.Generator().manual_seed(args.seed)
    main_policy = make_model_policy(model, tokenizer, device, args.temperature, generator)

    # Decorrelated seed/generator for the opponent side so a second
    # model (or the solver's tie-breaking) does not just mirror the
    # main model's draws.
    opp_generator = torch.Generator().manual_seed(args.seed + 1)
    rng = random.Random(args.seed)
    opponent_policy, label_b = resolve_opponent(
        args.vs, device, args.temperature, rng, opp_generator
    )

    outcomes = play_match(main_policy, opponent_policy, args.games)
    mode = "strict argmax" if args.temperature <= 0 else f"sampled @ T={args.temperature}"
    print_summary(str(model_path), label_b, args.games, outcomes, mode)


if __name__ == "__main__":
    main()
