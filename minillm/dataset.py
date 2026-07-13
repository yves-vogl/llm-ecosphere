"""Corpus building: enumerate games, write JSONL files, build tensors.

Real LLM pipelines scrape terabytes of text; ours *enumerates its entire
universe* — every game of Drop-Tac-Toe that can possibly be played. Two
corpora come out of it:

  data/all_games.jsonl     every complete game. Pretraining on this
                           teaches the model what games LOOK like:
                           legal moves, gravity, when a game ends and
                           who won. The "grammar" of the language.
  data/expert_games.jsonl  games where one side plays perfectly (the
                           solver) against every possible opponent
                           reply. Finetuning on this — with the loss
                           masked to the expert's moves only — teaches
                           the model to play WELL. The SFT stage.
  data/meta.json           corpus statistics, for the docs and sanity
                           checks.

Run `python -m minillm.dataset --out data` (or `make data`).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from .game import O, X
from .solver import EMPTY, describe_root_value, enumerate_all_games, enumerate_expert_games, negamax
from .tokenizer import MAX_GAME_TOKENS, Tokenizer


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def split_games(games: list[dict], val_frac: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle, then train/val split.

    The val games are complete games the model never sees in training.
    (In a world this small most *positions* still occur in some training
    game via shared prefixes — docs/05-training.md discusses why val
    loss is nevertheless meaningful.)
    """
    shuffled = list(games)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


# ----------------------------------------------------------------------
# Tensorization
# ----------------------------------------------------------------------
def build_tensors(
    games: list[dict],
    tokenizer: Tokenizer,
    block_size: int,
    expert_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Games -> (x, y) training tensors of shape (N, block_size).

    Next-token prediction setup: y is x shifted one position left.
    Position t of x sees tokens 0..t and must predict y[t] = token t+1.

    Targets set to -1 are skipped by the loss (ignore_index):
      * padding after <eos>, always;
      * with expert_only=True additionally every OPPONENT move — the
        finetuning trick borrowed from real SFT, where the user's turns
        are masked and only the assistant's turns are imitated.
    """
    x = torch.full((len(games), block_size), tokenizer.pad_id, dtype=torch.long)
    y = torch.full((len(games), block_size), -1, dtype=torch.long)

    for i, game in enumerate(games):
        seq = tokenizer.encode_game(game["moves"], game["result"])
        assert len(seq) <= block_size + 1, "game longer than block_size + 1"
        x[i, : len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
        for t in range(len(seq) - 1):
            target = seq[t + 1]
            if expert_only and tokenizer.is_move_id(target):
                # The target sits at seq index t+1; seq[0] is <bos>, and
                # each move occupies tokens_per_move ids after it (one at
                # move level, two at char level — both characters of a
                # move belong to the same move number, so both get masked
                # or trained together). X plays the odd moves (1st, 3rd,
                # ...), O the even ones.
                move_no = t // tokenizer.tokens_per_move + 1
                mover = X if move_no % 2 == 1 else O
                if mover != game["expert"]:
                    continue  # leave -1: do not imitate the opponent
            y[i, t] = target
    return x, y


# ----------------------------------------------------------------------
# JSONL I/O
# ----------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — generate the datasets first with `make data`"
        )
    return [json.loads(line) for line in path.read_text().splitlines()]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate games and write datasets")
    parser.add_argument("--out", default="data", help="output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Enumerating every possible game ...")
    all_games = enumerate_all_games()
    print("Enumerating expert games (solver plays X, then O) ...")
    expert_games = enumerate_expert_games(X) + enumerate_expert_games(O)

    write_jsonl(out / "all_games.jsonl", all_games)
    write_jsonl(out / "expert_games.jsonl", expert_games)

    lengths = [len(g["moves"]) for g in all_games]
    meta = {
        "n_all_games": len(all_games),
        "n_expert_games": len(expert_games),
        "results_all": dict(Counter(g["result"] for g in all_games)),
        "results_expert": dict(Counter(g["result"] for g in expert_games)),
        "shortest_game_moves": min(lengths),
        "longest_game_moves": max(lengths),
        "root_value": describe_root_value(),
        "positions_solved": negamax.cache_info().currsize,
        "max_sequence_tokens": MAX_GAME_TOKENS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))
    print(f"\nWrote {out}/all_games.jsonl, {out}/expert_games.jsonl, {out}/meta.json")


if __name__ == "__main__":
    main()
