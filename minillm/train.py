"""The training loop — where the model actually learns.

Two stages, selected with --stage:

  pretrain  next-token prediction on EVERY possible game
            (data/all_games.jsonl). After this the model knows the
            "grammar" of the game: which moves are legal, that pieces
            stack bottom-up, when a game ends, who won. It does NOT
            particularly try to win — it imitates the average game.

  finetune  continues from the pretrained checkpoint on expert games
            (data/expert_games.jsonl), with the loss masked so only the
            solver's perfect moves are imitated (opponent moves get
            target -1 and are ignored). This is the exact shape of
            supervised finetuning (SFT) in real LLM pipelines: same
            objective, curated data, masked non-assistant turns.

The mechanics in both stages are the textbook recipe used from GPT-2 to
today's frontier models: AdamW, linear warmup + cosine decay, gradient
clipping, evaluate on held-out data, keep the checkpoint with the best
validation loss.

Run: python -m minillm.train --stage pretrain   (or `make pretrain`)
     python -m minillm.train --stage finetune   (or `make finetune`)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from .config import ModelConfig
from .dataset import build_tensors, read_jsonl, split_games
from .model import GPT
from .tokenizer import TOKENIZERS, get_tokenizer
from .utils import pick_device, set_seed

STAGE_DEFAULTS = {
    #             steps    lr      corpus file
    "pretrain": (3000, 1e-3, "all_games.jsonl"),
    "finetune": (1500, 2e-4, "expert_games.jsonl"),
}


def lr_at(step: int, max_steps: int, max_lr: float, warmup: int, min_lr: float) -> float:
    """Linear warmup to max_lr, then cosine decay to min_lr.

    Warmup protects the freshly initialized (or freshly re-purposed)
    model from huge early gradients; the cosine tail lets it settle
    into a minimum instead of bouncing around it.
    """
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def full_split_loss(model: GPT, x: torch.Tensor, y: torch.Tensor, batch: int = 1024) -> float:
    """Exact mean loss over an entire split (it is tiny, so we can).

    Chunked forward passes; each chunk's mean loss is re-weighted by its
    number of real (non-ignored) targets so the result is the true mean
    over all predicted tokens, not a mean of chunk means.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, x.size(0), batch):
        xb, yb = x[i : i + batch], y[i : i + batch]
        n_valid = int((yb != -1).sum())
        _, loss = model(xb, yb)
        total += loss.item() * n_valid
        count += n_valid
    if was_training:
        model.train()
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Drop-Tac-Toe GPT")
    parser.add_argument("--stage", required=True, choices=("pretrain", "finetune"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=None, help="default: runs/<stage>")
    parser.add_argument("--steps", type=int, default=None, help="default: 3000 / 1500")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="default: 1e-3 / 2e-4")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda | auto")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint to start from (finetune default: runs/pretrain/model.pt)")
    parser.add_argument("--tokenizer", default=None, choices=sorted(TOKENIZERS),
                        help="pretrain only, default: move; finetune inherits "
                             "the checkpoint's tokenizer")
    # architecture knobs (pretrain only; finetune inherits the checkpoint's)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    default_steps, default_lr, corpus_file = STAGE_DEFAULTS[args.stage]
    steps = args.steps or default_steps
    max_lr = args.lr or default_lr
    min_lr = max_lr * 0.1
    out_dir = Path(args.out_dir or f"runs/{args.stage}")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = pick_device(args.device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    games = read_jsonl(Path(args.data_dir) / corpus_file)
    train_games, val_games = split_games(games, val_frac=args.val_frac)
    expert_only = args.stage == "finetune"

    # ------------------------------------------------------------------
    # Model: fresh for pretraining, loaded from checkpoint for finetuning.
    # The tokenizer travels with the model: chosen at pretraining time,
    # recorded in the checkpoint, inherited by finetuning — a model's
    # embedding table is meaningless under any other vocabulary.
    # ------------------------------------------------------------------
    if args.stage == "pretrain":
        tokenizer = get_tokenizer(args.tokenizer or "move")
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
        model = GPT(config)
        init_note = "fresh weights"
    else:
        init_from = Path(args.init_from or "runs/pretrain/model.pt")
        if not init_from.exists():
            raise FileNotFoundError(
                f"{init_from} not found — run `make pretrain` before `make finetune`"
            )
        ckpt = torch.load(init_from, map_location="cpu", weights_only=True)
        tokenizer = get_tokenizer(ckpt.get("tokenizer", "move"))
        if args.tokenizer is not None and args.tokenizer != tokenizer.name:
            raise SystemExit(
                f"--tokenizer {args.tokenizer} conflicts with {init_from}, "
                f"which was pretrained with the {tokenizer.name!r} tokenizer"
            )
        config = ModelConfig(**ckpt["config"])
        model = GPT(config)
        model.load_state_dict(ckpt["model"])
        init_note = f"initialized from {init_from}"
    if tokenizer.max_game_tokens > config.block_size + 1:
        raise SystemExit(
            f"block_size {config.block_size} cannot hold the longest game: the "
            f"{tokenizer.name!r} tokenizer needs up to {tokenizer.max_game_tokens} "
            f"tokens — pass --block-size {tokenizer.max_game_tokens - 1} or larger"
        )
    model.to(device)
    model.train()

    x_train, y_train = build_tensors(train_games, tokenizer, config.block_size, expert_only)
    x_val, y_val = build_tensors(val_games, tokenizer, config.block_size, expert_only)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    optimizer = model.configure_optimizer(args.weight_decay, max_lr)

    n_train_tokens = int((y_train != -1).sum())
    print(f"stage        {args.stage} ({init_note})")
    print(f"tokenizer    {tokenizer.name} (vocab {tokenizer.vocab_size}, "
          f"up to {tokenizer.max_game_tokens} tokens per game)")
    print(f"device       {device.type}")
    print(f"parameters   {model.num_params():,}")
    print(f"games        {len(train_games):,} train / {len(val_games):,} val")
    print(f"targets      {n_train_tokens:,} trainable target tokens "
          f"(~{steps * args.batch_size / max(1, len(train_games)):.0f} epochs)")
    print(f"schedule     {steps} steps, lr {max_lr:.0e} -> {min_lr:.0e}, "
          f"warmup {args.warmup}, batch {args.batch_size}")
    print()

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    best_val = float("inf")
    log_rows: list[dict] = []
    started = time.time()

    for step in range(steps):
        lr = lr_at(step, steps, max_lr, args.warmup, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Sample a random batch of games (with replacement — simpler than
        # epoch bookkeeping and statistically equivalent at this scale).
        ix = torch.randint(0, x_train.size(0), (args.batch_size,), device=device)
        _, loss = model(x_train[ix], y_train[ix])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # backprop: d(loss)/d(every parameter)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()  # AdamW nudges every parameter downhill

        if step % args.eval_interval == 0 or step == steps - 1:
            train_loss = full_split_loss(model, x_train, y_train)
            val_loss = full_split_loss(model, x_val, y_val)
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    model.checkpoint_dict(stage=args.stage, step=step,
                                          val_loss=val_loss, val_frac=args.val_frac,
                                          tokenizer=tokenizer.name),
                    out_dir / "model.pt",
                )
                marker = "  <- saved"
            print(f"step {step:5d} | lr {lr:.2e} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f}{marker}")
            log_rows.append({"step": step, "lr": lr,
                             "train_loss": train_loss, "val_loss": val_loss})

    with (out_dir / "log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "lr", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\ndone in {time.time() - started:.0f}s — best val loss {best_val:.4f}")
    print(f"checkpoint: {out_dir / 'model.pt'}   loss curve: {out_dir / 'log.csv'}")


if __name__ == "__main__":
    main()
