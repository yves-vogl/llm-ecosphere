"""Small shared helpers: device selection, seeding, checkpoint loading,
and the functions every "use the model" script builds on —
`next_token_logits` for raw next-token scores, plus the move-assembly
trio (`legal_move_logprobs`, `greedy_unit`, `sample_unit`) that hides
whether a move is one token (move tokenizer) or several (char
tokenizer) from evaluate.py and play.py.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import Tokenizer, get_tokenizer


def pick_device(name: str = "cpu") -> torch.device:
    """Resolve a device string. "auto" prefers cuda > mps > cpu.

    The model is so small that plain CPU is entirely adequate (and often
    faster than paying GPU launch overhead for tiny kernels) — hence the
    conservative default.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def default_checkpoint() -> Path:
    """The checkpoint most scripts want: finetuned if it exists, else
    pretrained. Raises with a helpful message if neither exists."""
    for candidate in (Path("runs/finetune/model.pt"), Path("runs/pretrain/model.pt")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no checkpoint found — train one first: `make pretrain` (then `make finetune`)"
    )


def load_model(path: str | Path, device: torch.device) -> tuple[GPT, dict]:
    """Load a checkpoint saved by train.py. Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT.from_checkpoint(ckpt, device)
    return model, ckpt


def tokenizer_for_checkpoint(ckpt: dict) -> Tokenizer:
    """The tokenizer a checkpoint was trained with (recorded by
    train.py; checkpoints from before the char-tokenizer exercise
    default to move-level)."""
    return get_tokenizer(ckpt.get("tokenizer", "move"))


@torch.no_grad()
def logits_for_ids(model: GPT, ids: list[int], device: torch.device) -> torch.Tensor:
    """One forward pass over an id sequence; raw logits (1-D tensor,
    vocab_size) for whatever comes next."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    return logits[0, -1]


@torch.no_grad()
def next_token_logits(
    model: GPT, tokenizer: Tokenizer, moves: list[str], device: torch.device
) -> torch.Tensor:
    """Raw logits (1-D tensor, vocab_size) for the token after `moves`.

    Encodes <bos> + moves, runs one forward pass, returns the last
    position's scores. Softmax over this vector is the model's full
    "opinion" about what comes next — play.py, evaluate.py and
    sample.py all interpret it in their own way. With the char
    tokenizer this is the model's view before the FIRST character of
    the next move; the assembly helpers below handle the rest.
    """
    return logits_for_ids(model, tokenizer.encode_prompt(list(moves)), device)


@torch.no_grad()
def legal_move_logprobs(
    model: GPT,
    tokenizer: Tokenizer,
    moves: list[str],
    legal_moves: list[str],
    device: torch.device,
) -> torch.Tensor:
    """log p(move | history) for each legal move, as one 1-D tensor.

    A move may span several tokens, so its probability is the product
    of its tokens' conditional probabilities — accumulated here as a
    sum of log-probs. At move level this collapses to one log_softmax
    lookup per move (a single forward pass); at char level, moves
    sharing a first character share its cached forward pass too.
    """
    base = tuple(tokenizer.encode_prompt(list(moves)))
    cache: dict[tuple[int, ...], torch.Tensor] = {}

    def logprobs_after(prefix: tuple[int, ...]) -> torch.Tensor:
        if prefix not in cache:
            cache[prefix] = F.log_softmax(
                logits_for_ids(model, list(prefix), device), dim=-1
            )
        return cache[prefix]

    scores = []
    for move in legal_moves:
        prefix, logp = base, torch.zeros((), device=device)
        for token_id in tokenizer.encode_move(move):
            logp = logp + logprobs_after(prefix)[token_id]
            prefix = prefix + (token_id,)
        scores.append(logp)
    return torch.stack(scores)


@torch.no_grad()
def greedy_unit(
    model: GPT, tokenizer: Tokenizer, moves: list[str], device: torch.device
) -> str:
    """Greedy-decode one transcript unit (a move or a result): argmax
    over the FULL vocabulary, tokens_per_move times, joined back into
    a string. Deliberately unrestricted — the point is to see whether
    the model's raw favourite continuation forms a legal move at all
    (at char level: only if BOTH characters combine to a legal cell).
    """
    ids = list(tokenizer.encode_prompt(list(moves)))
    parts = []
    for _ in range(tokenizer.tokens_per_move):
        next_id = int(logits_for_ids(model, ids, device).argmax())
        parts.append(tokenizer.id_to_token[next_id])
        ids.append(next_id)
    return "".join(parts)


@torch.no_grad()
def sample_unit(
    model: GPT,
    tokenizer: Tokenizer,
    moves: list[str],
    device: torch.device,
    generator: torch.Generator,
    temperature: float = 1.0,
) -> str:
    """Sample one transcript unit from the full vocabulary, token by
    token (temperature <= 0 means greedy). The free-running counterpart
    of `greedy_unit`: this is the model "playing blind", exactly as it
    would while generating a transcript with no engine to help.
    """
    ids = list(tokenizer.encode_prompt(list(moves)))
    parts = []
    for _ in range(tokenizer.tokens_per_move):
        logits = logits_for_ids(model, ids, device)
        if temperature <= 0:
            next_id = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1).cpu()
            next_id = int(torch.multinomial(probs, 1, generator=generator))
        parts.append(tokenizer.id_to_token[next_id])
        ids.append(next_id)
    return "".join(parts)
