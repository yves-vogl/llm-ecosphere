"""Free-running generation: let the model dream up complete games.

Start from <bos> alone and sample token by token until <eos>. This is
the purest check of what the model absorbed in pretraining — there is
no game engine steering it, so every rule it follows (gravity, turn
order, correct result token) it follows because it *learned* to.

Each transcript is then replayed through the real game engine and
annotated: was every move legal, and does the claimed result match?

Run: python -m minillm.sample --num 5      (or `make sample`)
"""

from __future__ import annotations

import argparse

import torch

from .game import Game, IllegalMoveError
from .tokenizer import EOS, MOVE_TOKENS
from .utils import default_checkpoint, load_model, pick_device, tokenizer_for_checkpoint


def verify_transcript(tokens: list[str]) -> str:
    """Replay `tokens` (without <bos>) in the engine. Returns "ok" or a
    description of the first rule violation.

    Works on transcript UNITS ("B2", "#X", "<eos>") — char-level output
    must be re-assembled with tokenizer.group_units first, so a botched
    pair like "A#" simply shows up here as an unexpected token."""
    game = Game()
    saw_result = False
    for i, token in enumerate(tokens):
        if token in MOVE_TOKENS:
            if game.is_over():
                return f"move {token} played after the game was over"
            try:
                game.push(token)
            except IllegalMoveError as err:
                return f"illegal move {token}: {err}"
        elif token in ("#X", "#O", "#="):
            if saw_result:
                return f"duplicate result token {token}"
            if not game.is_over():
                return f"result {token} claimed while the game was still running"
            if token != game.result_token:
                return f"wrong result: claimed {token}, actually {game.result_token}"
            saw_result = True
        elif token == EOS:
            if not saw_result:
                return "<eos> before a result token"
            return "ok" if i == len(tokens) - 1 else "tokens after <eos>"
        else:
            return f"unexpected token {token}"
    return "ran out of tokens without <eos>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample complete games from the model")
    parser.add_argument("--ckpt", default=None, help="default: finetune, else pretrain")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt_path = args.ckpt or default_checkpoint()
    model, ckpt = load_model(ckpt_path, device)
    tokenizer = tokenizer_for_checkpoint(ckpt)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print(f"model: {ckpt_path} (stage {ckpt.get('stage')}, "
          f"tokenizer {tokenizer.name}, "
          f"val loss {ckpt.get('val_loss', float('nan')):.4f})")
    print(f"sampling {args.num} games at temperature {args.temperature}\n")

    ok = 0
    for i in range(args.num):
        idx = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=tokenizer.max_game_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_id=tokenizer.eos_id,
            generator=generator,
        )
        tokens = tokenizer.decode(out[0].tolist())[1:]  # drop <bos>
        # Char-level streams are re-assembled into moves before the
        # engine judges them; at move level this is the identity.
        verdict = verify_transcript(tokenizer.group_units(tokens))
        ok += verdict == "ok"
        print(f"game {i + 1}: {' '.join(tokens)}")
        print(f"         -> {verdict}\n")

    print(f"{ok}/{args.num} transcripts are fully legal with the correct result")


if __name__ == "__main__":
    main()
