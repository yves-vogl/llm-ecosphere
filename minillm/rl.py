"""REINFORCE self-play: the "proper" win-maximising gambler (issue #35).

Where the SFT gambler (`train.py --objective gambler`) *imitates*
whoever happened to win a batch of pre-generated games, this module
*optimizes* wins directly: the learner plays complete games against a
uniform-random opponent and its own move probabilities are pushed up
or down by how the game actually turned out. No dataset, no fixed
targets — the objective is the game's outcome itself.

Algorithm — REINFORCE with a baseline:

  1. The learner is a GPT policy initialized from a checkpoint (default
     `runs/pretrain/model.pt`, so it already plays legally). It samples
     ONLY legal moves, using the same chain-rule log-probability math as
     `utils.legal_move_logprobs` (reimplemented here without the
     `@torch.no_grad()` decorator — REINFORCE needs gradients through
     exactly that quantity, which the inference-only helper deliberately
     forecloses).
  2. Each iteration plays a batch of complete games. The learner
     alternates sides (X on even games, O on odd) so it learns to
     both open and respond; the opponent always samples uniformly at
     random among its legal moves.
  3. Every game yields one scalar return from the LEARNER's perspective:
     +1 win, -1 loss, 0 draw — the same number whichever side it played.
     Every learner move in that game is credited with that one return
     (no discounting: the game is short and every move contributed to
     the outcome equally in this formulation).
  4. A running mean of returns (over all games played so far, BEFORE
     this iteration) serves as the baseline, subtracted from the return
     to form the advantage. This does not bias the gradient — the
     baseline is a constant w.r.t. the current iteration's samples — but
     shrinks its variance.
  5. Objective: gradient ASCENT on expected return, i.e. minimize
         loss = -(1/N) * sum_i (return_i - baseline) * logpi(a_i | s_i)
     over N learner moves in the batch. Opponent moves never appear in
     this sum: only states where the LEARNER was to move are recorded.

Run: python -m minillm.rl --iters 60 --games-per-iter 40   (a few minutes on CPU)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .game import Game, O, X
from .model import GPT
from .tokenizer import Tokenizer
from .utils import load_model, pick_device, set_seed, tokenizer_for_checkpoint

SIDES = (X, O)


# ----------------------------------------------------------------------
# Grad-enabled move log-probabilities
# ----------------------------------------------------------------------
def logits_for_ids(model: GPT, ids: list[int], device: torch.device) -> torch.Tensor:
    """Grad-enabled counterpart of utils.logits_for_ids (that one is
    @torch.no_grad() and would sever the graph we need for REINFORCE)."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    return logits[0, -1]


def legal_move_logprobs(
    model: GPT,
    tokenizer: Tokenizer,
    moves: list[str],
    legal_moves: list[str],
    device: torch.device,
) -> torch.Tensor:
    """log p(move | history) for each legal move, gradient-attached.

    Identical chain-rule math to `utils.legal_move_logprobs` (a move may
    span several tokens; its log-prob is the sum of its tokens'
    conditional log-probs, with per-prefix caching so moves sharing a
    prefix share the forward pass) — the only difference is the absence
    of `@torch.no_grad()`, which utils' version carries because it is
    used purely for inference/evaluation.
    """
    base = tuple(tokenizer.encode_prompt(list(moves)))
    cache: dict[tuple[int, ...], torch.Tensor] = {}

    def logprobs_after(prefix: tuple[int, ...]) -> torch.Tensor:
        if prefix not in cache:
            cache[prefix] = F.log_softmax(logits_for_ids(model, list(prefix), device), dim=-1)
        return cache[prefix]

    scores = []
    for move in legal_moves:
        prefix, logp = base, torch.zeros((), device=device)
        for token_id in tokenizer.encode_move(move):
            logp = logp + logprobs_after(prefix)[token_id]
            prefix = prefix + (token_id,)
        scores.append(logp)
    return torch.stack(scores)


# ----------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------
def game_return(winner: str | None, learner_side: str) -> float:
    """+1 if `learner_side` won, -1 if it lost, 0 for a draw — correct
    regardless of which side (X or O) the learner played."""
    if winner is None:
        return 0.0
    return 1.0 if winner == learner_side else -1.0


def random_move(game: Game, rng: random.Random) -> str:
    return rng.choice(game.legal_moves())


# ----------------------------------------------------------------------
# Self-play episode
# ----------------------------------------------------------------------
@dataclass
class MoveRecord:
    """One learner decision: its sampled move's log-probability, still
    attached to the autograd graph of the forward pass that produced
    it."""
    logp: torch.Tensor


def play_episode(
    model: GPT,
    tokenizer: Tokenizer,
    device: torch.device,
    learner_side: str,
    rng: random.Random,
    generator: torch.Generator,
) -> tuple[list[MoveRecord], float]:
    """Play one complete game: `learner_side` samples from the policy
    (legal moves only, via `legal_move_logprobs`), the other side plays
    uniformly at random. Returns (learner's MoveRecords, game_return).

    Only positions where the learner is to move produce a MoveRecord —
    the random opponent's moves are pushed onto the board but never
    scored, so they cannot enter the REINFORCE loss.
    """
    game = Game()
    records: list[MoveRecord] = []
    while not game.is_over():
        legal = game.legal_moves()
        if game.to_move == learner_side:
            logprobs = legal_move_logprobs(model, tokenizer, game.history, legal, device)
            # Sample from the legal-renormalized distribution. softmax over the
            # legal-move log-probs is the numerically stable form of
            # exp()/sum() and cannot divide by ~0; .cpu() keeps multinomial's
            # CPU generator valid under --device mps/cuda.
            probs = F.softmax(logprobs, dim=0).detach().cpu()
            idx = int(torch.multinomial(probs, 1, generator=generator))
            move = legal[idx]
            # REINFORCE's score must be log pi(a|s) for the distribution the
            # move was actually sampled from — the legal-renormalized one — so
            # subtract the legal-set normalizer (logsumexp). The raw
            # logprobs[idx] is normalized over the FULL vocabulary instead, so
            # its gradient omits the softmax term and distorts the relative
            # preference update across the competing legal moves.
            records.append(MoveRecord(logp=logprobs[idx] - torch.logsumexp(logprobs, dim=0)))
        else:
            move = random_move(game, rng)
        game.push(move)
    return records, game_return(game.winner(), learner_side)


# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------
class RunningMean:
    """Mean of every return seen so far. Queried as the baseline BEFORE
    the current iteration's returns are folded in, so the baseline used
    in a loss is always a constant w.r.t. that loss's own samples —
    variance reduction without biasing the gradient."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def update(self, values) -> None:
        for v in values:
            self.total += v
            self.count += 1


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def policy_gradient_loss(
    logps: torch.Tensor, returns: torch.Tensor, baseline: float
) -> torch.Tensor:
    """-(1/N) * sum (return - baseline) * logpi(a|s) — gradient DESCENT
    on this equals gradient ASCENT on expected return (REINFORCE)."""
    advantage = returns - baseline
    return -(advantage * logps).mean()


# ----------------------------------------------------------------------
# One training iteration
# ----------------------------------------------------------------------
def run_iteration(
    model: GPT,
    tokenizer: Tokenizer,
    device: torch.device,
    games_per_iter: int,
    rng: random.Random,
    generator: torch.Generator,
    baseline: RunningMean,
) -> tuple[torch.Tensor, dict]:
    """Play `games_per_iter` games (learner sides alternating) and
    return (loss, stats). Does not step the optimizer."""
    all_records: list[MoveRecord] = []
    all_returns: list[float] = []
    game_returns: list[float] = []

    for i in range(games_per_iter):
        learner_side = SIDES[i % 2]
        records, ret = play_episode(model, tokenizer, device, learner_side, rng, generator)
        all_records.extend(records)
        all_returns.extend([ret] * len(records))
        game_returns.append(ret)

    baseline_value = baseline.mean
    logps = torch.stack([r.logp for r in all_records])
    returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
    loss = policy_gradient_loss(logps, returns_t, baseline_value)

    # Baseline tracks the per-GAME mean return (one value per game, not per
    # move) — a move-weighted mean would over-count long games and is not the
    # simple average the baseline is meant to be.
    baseline.update(game_returns)

    wins = sum(1 for r in game_returns if r > 0)
    draws = sum(1 for r in game_returns if r == 0)
    losses = sum(1 for r in game_returns if r < 0)
    stats = {
        "games": games_per_iter,
        "mean_return": sum(game_returns) / games_per_iter,
        "win_rate": wins / games_per_iter,
        "draw_rate": draws / games_per_iter,
        "loss_rate": losses / games_per_iter,
        "baseline": baseline_value,
    }
    return loss, stats


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="REINFORCE self-play vs a random opponent: the win-maximising gambler"
    )
    parser.add_argument("--init-from", default="runs/pretrain/model.pt")
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--games-per-iter", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    parser.add_argument("--out-dir", default="runs/exp-rl-gambler")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)

    init_from = Path(args.init_from)
    if not init_from.exists():
        raise FileNotFoundError(
            f"{init_from} not found — run `make pretrain` first (see README)"
        )
    model, ckpt = load_model(init_from, device)
    tokenizer = tokenizer_for_checkpoint(ckpt)
    model.train()  # dropout on, matching how it will run once merged back in

    optimizer = model.configure_optimizer(args.weight_decay, args.lr)
    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    baseline = RunningMean()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"init         {init_from} (stage {ckpt.get('stage')}, "
          f"tokenizer {tokenizer.name})")
    print(f"objective    rl (REINFORCE vs uniform-random opponent, alternating sides)")
    print(f"schedule     {args.iters} iters x {args.games_per_iter} games/iter, "
          f"lr {args.lr:.0e}")
    print()

    last_stats = None
    for it in range(args.iters):
        loss, stats = run_iteration(
            model, tokenizer, device, args.games_per_iter, rng, generator, baseline
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        last_stats = stats
        print(f"iter {it:4d} | loss {loss.item():+.4f} | mean return {stats['mean_return']:+.3f} "
              f"| win/draw/loss vs random {stats['win_rate']:.1%}/"
              f"{stats['draw_rate']:.1%}/{stats['loss_rate']:.1%}")

    torch.save(
        model.checkpoint_dict(
            stage="rl",
            objective="rl",
            tokenizer=tokenizer.name,
            init_from=str(init_from),
            iters=args.iters,
            games_per_iter=args.games_per_iter,
            lr=args.lr,
            seed=args.seed,
            final_win_rate_vs_random=last_stats["win_rate"] if last_stats else None,
        ),
        out_dir / "model.pt",
    )
    print(f"\nsaved {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
