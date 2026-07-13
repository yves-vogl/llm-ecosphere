"""Look inside the model: print the attention pattern of every head.

For a given game prefix, this shows — per layer and head — how much
each position (query, rows) attends to each earlier position (key,
columns). Values in a row sum to 1. The upper-right triangle is blank:
that is the causal mask, the future is invisible.

Things to look for once the model is trained:
  * heads that attend to the *previous* move (local/turn tracking),
  * heads that attend from a move back to earlier moves in the SAME
    column (they track stack heights — the gravity rule!),
  * the <bos> position acting as a "no-op" attention sink, a
    well-known phenomenon in real transformers too.

Run: python -m minillm.inspect_attention --moves "B1 A1 B2" [--layer 2]
"""

from __future__ import annotations

import argparse

import torch

from .utils import default_checkpoint, load_model, pick_device, tokenizer_for_checkpoint


def print_head(att: torch.Tensor, labels: list[str]) -> None:
    """att: (T, T) attention weights for one head."""
    width = 6
    print(" " * 7 + "".join(f"{label:>{width}}" for label in labels))
    for i, label in enumerate(labels):
        cells = []
        for j in range(len(labels)):
            cells.append(f"{att[i, j].item():>{width}.2f}" if j <= i else " " * (width - 1) + "·")
        print(f"{label:>6} " + "".join(cells))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print attention matrices")
    parser.add_argument("--ckpt", default=None, help="default: finetune, else pretrain")
    parser.add_argument("--moves", default="B1 A1 B2",
                        help='game prefix, e.g. "B1 A1 B2"')
    parser.add_argument("--layer", type=int, default=None, help="only this layer (0-based)")
    parser.add_argument("--head", type=int, default=None, help="only this head (0-based)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_model(args.ckpt or default_checkpoint(), device)
    tokenizer = tokenizer_for_checkpoint(ckpt)

    moves = args.moves.split()
    ids = tokenizer.encode_prompt(moves)
    labels = tokenizer.decode(ids)  # per-token labels: moves, or single chars
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        model(x, record_attn=True)  # fills block.attn.last_attn everywhere

    print(f"attention for prefix: {' '.join(moves)}   "
          f"(rows = queries, columns = keys, rows sum to 1)\n")
    for layer_idx, block in enumerate(model.transformer.h):
        if args.layer is not None and layer_idx != args.layer:
            continue
        attn = block.attn.last_attn[0]  # (n_head, T, T)
        for head_idx in range(attn.size(0)):
            if args.head is not None and head_idx != args.head:
                continue
            print(f"--- layer {layer_idx}, head {head_idx} ---")
            print_head(attn[head_idx], labels)


if __name__ == "__main__":
    main()
