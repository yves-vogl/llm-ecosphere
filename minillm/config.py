"""Model hyperparameters, kept in one dataclass so checkpoints can
store them and every script rebuilds exactly the same architecture.

The defaults give a ~0.8M-parameter model — deliberately overpowered
for a 15-token world, so that capacity is never the reason something
fails to be learned. For scale: GPT-2 small is 124M parameters with the
same architecture, GPT-3 is 175B. The shape of the code does not change,
only these numbers do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 15  # see tokenizer.VOCAB
    block_size: int = 16  # max sequence length; a full game needs 12 tokens
    n_layer: int = 4      # number of stacked Transformer blocks
    n_head: int = 4       # attention heads per block (head size = 128/4 = 32)
    n_embd: int = 128     # width of the residual stream
    dropout: float = 0.1  # regularization; set to 0.0 for deterministic runs
