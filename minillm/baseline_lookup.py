"""A lookup-table baseline: memorization vs generalization (exercise 7).

The nastiest question you can ask any trained model: would a hash map
have done just as well? This module builds that hash map — a
no-learning baseline that records, for every transcript prefix in the
TRAINING split, the observed distribution of next tokens — and runs it
through the repo's regular measurements, plus the one that decides the
argument: prefix coverage on held-out games, and the table-vs-network
gap on exactly the prefixes the table has never seen.

The table wears the model's interface. `LookupModel.__call__` accepts a
`(1, T)` id tensor and returns `(logits, None)` shaped like
`GPT.forward`'s inference call, so `evaluate.eval_on_val_games`,
`evaluate.eval_expert_agreement` and the whole `utils` move-assembly
stack run on it unchanged — same code path, same seeds, honest
comparison. Seen prefix: logits are the log of the observed next-token
frequencies. Unseen prefix: uniform over the legal moves (the engine
provides legality, never a preference), or uniform over the three
result tokens when the replayed game is already over.

Move-level tokenizer only: with one token per move, "transcript prefix"
and "move history" are the same thing, which keeps the table honest.

Run: .venv/bin/python -m minillm.baseline_lookup   (needs `make data`
and a trained checkpoint for the network comparison)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .dataset import read_jsonl, split_games
from .evaluate import eval_on_val_games, eval_expert_agreement, model_move_strict
from .game import Game
from .solver import best_moves
from .tokenizer import Tokenizer
from .utils import default_checkpoint, load_model, pick_device, tokenizer_for_checkpoint

import random


class LookupModel:
    """The no-learning baseline, wearing GPT's inference interface."""

    def __init__(self, train_games: list[dict], tokenizer: Tokenizer) -> None:
        assert tokenizer.tokens_per_move == 1, (
            "the lookup baseline is defined for the move-level tokenizer"
        )
        self.tokenizer = tokenizer
        # Prefix (tuple of token ids, <bos> included) -> Counter of the
        # next token ids observed after it in the training split.
        self.table: dict[tuple[int, ...], Counter] = {}
        for game in train_games:
            seq = tokenizer.encode_game(game["moves"], game["result"])
            for t in range(len(seq) - 1):
                prefix = tuple(seq[: t + 1])
                self.table.setdefault(prefix, Counter())[seq[t + 1]] += 1

    def seen(self, prefix: tuple[int, ...]) -> bool:
        return prefix in self.table

    def logits_for_prefix(self, prefix: tuple[int, ...]) -> torch.Tensor:
        """Log-probabilities over the vocabulary for the next token."""
        logits = torch.full((self.tokenizer.vocab_size,), float("-inf"))
        counts = self.table.get(prefix)
        if counts is not None:
            total = sum(counts.values())
            for token_id, n in counts.items():
                logits[token_id] = torch.log(torch.tensor(n / total))
            return logits
        # Unseen prefix: fall back to uniform over what the RULES allow —
        # legality comes from the engine, preference from nothing at all.
        moves = self.tokenizer.decode(list(prefix))[1:]  # drop <bos>
        game = Game.from_moves(moves)
        if game.is_over():
            ids = self.tokenizer.result_ids  # uniform over #X / #O / #=
        else:
            ids = [self.tokenizer.encode_move(m)[0] for m in game.legal_moves()]
        for token_id in ids:
            logits[token_id] = torch.log(torch.tensor(1.0 / len(ids)))
        return logits

    def __call__(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        """(1, T) ids -> ((1, 1, vocab) logits, None), like GPT's
        inference call: only the last position's scores."""
        assert x.size(0) == 1 and targets is None
        logits = self.logits_for_prefix(tuple(x[0].tolist()))
        return logits.view(1, 1, -1), None


# ----------------------------------------------------------------------
# The decisive measurement: coverage, and the gap where coverage ends
# ----------------------------------------------------------------------
def prefix_coverage(table: LookupModel, tokenizer: Tokenizer, val_games: list[dict]) -> dict:
    """How many held-out positions has the table literally seen?"""
    seen = total = 0
    for g in val_games:
        game = Game()
        for move in g["moves"]:
            seen += table.seen(tuple(tokenizer.encode_prompt(game.history)))
            total += 1
            game.push(move)
    return {"positions": total, "seen": seen, "coverage": seen / total}


def agreement_by_coverage(model, table: LookupModel, tokenizer: Tokenizer,
                          device, n_rollouts: int, seed: int) -> dict:
    """Optimal-move rate for the network and the table, split by whether
    the table has seen the position's history — the same seeded rollout
    scheme evaluate.eval_expert_agreement uses, so the position set is
    identical to the regular eval."""
    rng = random.Random(seed)
    histories: dict[tuple, list[str]] = {}
    for _ in range(n_rollouts):
        game = Game()
        while not game.is_over():
            histories.setdefault(tuple(game.stacks), list(game.history))
            game.push(rng.choice(game.legal_moves()))

    buckets = {True: {"n": 0, "net": 0, "table": 0},
               False: {"n": 0, "net": 0, "table": 0}}
    for stacks, history in histories.items():
        covered = table.seen(tuple(tokenizer.encode_prompt(history)))
        _, optimal = best_moves(stacks)
        game = Game.from_moves(history)
        bucket = buckets[covered]
        bucket["n"] += 1
        bucket["net"] += model_move_strict(model, tokenizer, game, device) in optimal
        bucket["table"] += model_move_strict(table, tokenizer, game, device) in optimal
    return {
        ("seen" if covered else "unseen"): {
            "positions": b["n"],
            "network_optimal_rate": b["net"] / b["n"] if b["n"] else None,
            "table_optimal_rate": b["table"] / b["n"] if b["n"] else None,
        }
        for covered, b in buckets.items()
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Lookup-table baseline (exercise 7)")
    parser.add_argument("--ckpt", default=None, help="network to compare against "
                        "(default: finetune, else pretrain)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agreement-rollouts", type=int, default=300)
    parser.add_argument("--out", default=None, help="also write results as JSON")
    args = parser.parse_args()

    device = pick_device("cpu")
    ckpt_path = Path(args.ckpt or default_checkpoint())
    model, ckpt = load_model(ckpt_path, device)
    tokenizer = tokenizer_for_checkpoint(ckpt)

    games = read_jsonl(Path(args.data_dir) / "all_games.jsonl")
    # Same split seed and fraction training used, so "training split"
    # means exactly what it meant to the network.
    train_games, val_games = split_games(games, val_frac=ckpt.get("val_frac", 0.1))
    table = LookupModel(train_games, tokenizer)

    print(f"lookup table: {len(table.table):,} distinct training prefixes "
          f"({len(train_games):,} games); network: {ckpt_path}\n")

    results = {
        "table_prefixes": len(table.table),
        "legality_teacher_forced": eval_on_val_games(table, tokenizer, val_games, device),
        "solver_agreement": eval_expert_agreement(
            table, tokenizer, device, args.agreement_rollouts, args.seed),
        "prefix_coverage": prefix_coverage(table, tokenizer, val_games),
        "agreement_by_coverage": agreement_by_coverage(
            model, table, tokenizer, device, args.agreement_rollouts, args.seed),
    }

    tf = results["legality_teacher_forced"]
    cov = results["prefix_coverage"]
    ag = results["solver_agreement"]
    by = results["agreement_by_coverage"]
    print(f"table      argmax legal          {tf['argmax_legal_rate']:8.1%}   "
          f"({tf['positions']} held-out positions)")
    print(f"           result prediction     {tf['result_prediction_accuracy']:8.1%}")
    print(f"           optimal-move rate     {ag['optimal_move_rate']:8.1%}   "
          f"({ag['positions']} positions)")
    print(f"coverage   held-out prefixes seen {cov['coverage']:7.1%}   "
          f"({cov['seen']}/{cov['positions']})")
    for split in ("seen", "unseen"):
        b = by[split]
        if b["positions"]:
            print(f"{split:>10} positions {b['positions']:4d}   "
                  f"network optimal {b['network_optimal_rate']:6.1%}   "
                  f"table optimal {b['table_optimal_rate']:6.1%}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
