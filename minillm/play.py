"""Play Drop-Tac-Toe against the model, in the terminal.

The model never sees the board — only the move sequence so far, exactly
as during training. Each turn it predicts a distribution over the next
token; we sample its move from that.

Two modes:
  strict (default)  sampling is restricted to legal moves. The model
                    picks among what the rules allow.
  --raw             sampling is unrestricted. If the model proposes an
                    illegal token you get to SEE it (educational!); it
                    then re-samples among the legal moves so the game
                    can continue.

In-game commands:  A / B / C or A1..C3 to move,  p = show the model's
probability distribution for the current position,  u = undo your last
move,  ? = help,  q = quit.

Run: python -m minillm.play [--human O] [--raw] [--show-probs]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .game import COLS, Game, IllegalMoveError, other
from .tokenizer import Tokenizer
from .utils import (default_checkpoint, greedy_unit, legal_move_logprobs,
                    load_model, next_token_logits, pick_device, sample_unit,
                    tokenizer_for_checkpoint)

HELP = """commands:
  A / B / C     drop a piece into that column
  A1 .. C3      same, naming the exact landing cell
  p             show the model's next-token probabilities
  u             undo your last move (and the model's reply)
  ?             this help
  q             quit"""


def show_distribution(logits: torch.Tensor, tokenizer: Tokenizer, game: Game) -> None:
    """Print the model's top next-token candidates as a bar chart.

    With the char tokenizer this is the distribution over the FIRST
    character of the next move — the model commits to a column letter
    before it has said which row the piece lands in.
    """
    probs = F.softmax(logits, dim=-1)
    legal = set(game.legal_moves())
    print("  model's next-token distribution:")
    for token_id in probs.argsort(descending=True)[:8].tolist():
        token = tokenizer.id_to_token[token_id]
        p = probs[token_id].item()
        if p < 0.001:
            break
        if token in legal:
            note = "legal move"
        elif tokenizer.tokens_per_move > 1 and any(m.startswith(token) for m in legal):
            note = "starts a legal move"
        elif tokenizer.is_move_id(token_id):
            note = "ILLEGAL move" if tokenizer.tokens_per_move == 1 else "move character"
        elif tokenizer.is_result_id(token_id):
            note = "result token" if tokenizer.tokens_per_move == 1 else "result character"
        else:
            note = "special token"
        print(f"    {token:>5}  {'#' * max(1, round(p * 40)):<40} {p:6.1%}  {note}")


def model_move(model, tokenizer, game: Game, device, *, strict: bool,
               temperature: float, generator: torch.Generator) -> str:
    """Sample the model's move; in raw mode, narrate illegal attempts.

    Moves are whole units here: at char level, "one move" means two
    sampled tokens assembled back into a cell name before the rules
    get a say.
    """
    legal = game.legal_moves()

    def sample_legal() -> str:
        # Joint log-prob per legal move; softmax over moves = the same
        # distribution the old single-token masking produced at move level.
        scores = legal_move_logprobs(model, tokenizer, game.history, legal, device)
        if temperature <= 0:
            return legal[int(scores.argmax())]
        probs = F.softmax(scores / temperature, dim=-1).cpu()
        return legal[int(torch.multinomial(probs, 1, generator=generator))]

    if strict:
        return sample_legal()
    # Raw mode: anything in the vocabulary, mistakes included.
    attempt = sample_unit(model, tokenizer, game.history, device, generator,
                          temperature=temperature)
    if attempt in legal:
        return attempt
    print(f"  (model proposed '{attempt}' — illegal here; re-sampling among legal moves)")
    return sample_legal()


def referee_verdict(model, tokenizer, game: Game, device) -> str:
    """After the game: which result does the model itself predict?"""
    return greedy_unit(model, tokenizer, game.history, device)


def read_human_move(game: Game) -> str | None:
    """Prompt until we get a move or a command. Returns the move, or None
    for undo, or raises SystemExit on quit."""
    while True:
        try:
            raw = input(f"  your move [{game.to_move}] (A/B/C, p, u, ?, q) > ").strip().upper()
        except EOFError:  # stdin closed (e.g. piped input ran out)
            raw = "Q"
        if raw == "Q":
            print("  bye!")
            raise SystemExit(0)
        if raw == "?":
            print(HELP)
            continue
        if raw in ("U", "P"):
            return raw  # handled by the caller
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against the model")
    parser.add_argument("--ckpt", default=None, help="default: finetune, else pretrain")
    parser.add_argument("--human", default="X", choices=("X", "O"),
                        help="which side you play (X moves first)")
    parser.add_argument("--raw", action="store_true",
                        help="do not restrict the model to legal moves")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="0 = deterministic best move")
    parser.add_argument("--show-probs", action="store_true",
                        help="show the model's distribution before each of its moves")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    ckpt_path = args.ckpt or default_checkpoint()
    model, ckpt = load_model(ckpt_path, device)
    tokenizer = tokenizer_for_checkpoint(ckpt)
    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    human, ai = args.human, other(args.human)
    print(f"model: {ckpt_path} (stage {ckpt.get('stage')}, "
          f"tokenizer {tokenizer.name})  |  "
          f"mode: {'raw' if args.raw else 'strict'}  |  "
          f"you: {human}, model: {ai}\n{HELP}\n")

    game = Game()
    while not game.is_over():
        print(game.render())
        if game.to_move == human:
            move = read_human_move(game)
            if move == "P":
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
            if args.show_probs:
                logits = next_token_logits(model, tokenizer, game.history, device)
                show_distribution(logits, tokenizer, game)
            move = model_move(model, tokenizer, game, device, strict=not args.raw,
                              temperature=args.temperature, generator=generator)
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
    print(f"(model as referee predicts: {referee_verdict(model, tokenizer, game, device)}, "
          f"actual result token: {game.result_token})")


if __name__ == "__main__":
    main()
