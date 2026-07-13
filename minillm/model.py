"""A GPT — decoder-only Transformer — written from scratch.

This is the heart of the project. The architecture is the same family
as GPT-2/3/4, Llama or Claude use, shrunk to ~0.8M parameters:

    token ids ──► token embedding + position embedding
                        │
                        ▼
              ┌── Transformer block ──┐   } repeated n_layer times:
              │  LayerNorm            │   }
              │  causal self-attention │   }  "communicate":  tokens look
              │  (+ residual)          │   }  at earlier tokens
              │  LayerNorm            │   }
              │  MLP  (+ residual)     │   }  "compute": per-token thinking
              └───────────────────────┘
                        │
                        ▼
              final LayerNorm ──► linear head ──► logits over the vocab

Everything is spelled out with explicit tensor shapes in the comments;
`docs/04-model.md` walks through the math at reading speed. The
attention implementation is intentionally the naive, readable one (no
FlashAttention, no KV cache) — a worked example, not a speed record.

Shape glossary used below:  B = batch size, T = sequence length ("time"),
C = n_embd (channels), nh = n_head, hs = head size = C / nh.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask.

    Attention lets every token gather information from *earlier* tokens.
    Each token emits a query ("what am I looking for?"), a key ("what do
    I contain?") and a value ("what do I hand over if someone attends to
    me?"). The attention weight between token i and token j is the
    softmaxed dot product of q_i and k_j; the causal mask forbids j > i,
    because during generation the future does not exist yet.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # One fused linear layer produces Q, K and V for all heads at once.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection back into the residual stream.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Lower-triangular matrix of ones: position i may attend to j <= i.
        # A buffer, not a parameter: it is constant and never trained.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))
        # Filled with the last attention pattern when record_attn=True —
        # only used by inspect_attention.py.
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, record_attn: bool = False) -> torch.Tensor:
        B, T, C = x.size()
        nh, hs = self.n_head, C // self.n_head

        # Project to queries, keys, values: (B, T, C) -> 3 x (B, T, C),
        # then split C into nh heads of size hs: -> (B, nh, T, hs).
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # Attention scores: every query against every key.
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T).
        # The 1/sqrt(hs) scaling keeps the dot products in a range where
        # softmax still has usable gradients.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)

        # Causal mask: set scores for future positions to -inf so that
        # softmax gives them exactly zero weight.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if record_attn:
            self.last_attn = att.detach()
        att = self.attn_dropout(att)

        # Weighted sum of values: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs),
        # then glue the heads back together -> (B, T, C).
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, nonlinearity, project back.

    Attention moves information *between* positions; the MLP does the
    per-position processing of whatever attention gathered. The 4x
    expansion factor is a GPT-2 convention kept by most models since.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One Transformer block: attention, then MLP, each behind a residual.

    "Pre-norm" layout (LayerNorm *before* each sublayer) — this keeps the
    residual stream an unnormalized highway that gradients flow through
    unimpeded, which is what makes deep stacks trainable.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, record_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), record_attn=record_attn)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full model: embeddings, n_layer blocks, and a language-model head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the matrix that maps token -> vector is reused to
        # map vector -> token logits. Saves parameters and couples the
        # input and output "meaning" of each token (GPT-2 does the same).
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale the init of residual projections down by
        # sqrt(2 * n_layer) so the residual stream's variance stays ~1
        # no matter how many blocks add their contribution.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Trainable parameter count (position embeddings included)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        record_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids. Returns (logits, loss).

        Training call:   logits (B, T, vocab) and the mean cross-entropy
                         of predicting targets[b, t] from idx[b, :t+1].
                         Positions with target -1 are ignored — that is
                         how padding and (in finetuning) opponent moves
                         are excluded from the loss.
        Inference call:  targets=None; as an optimization only the last
                         position's logits are computed: (B, 1, vocab).
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence of length {T} exceeds block_size"
        pos = torch.arange(T, device=idx.device)

        tok_emb = self.transformer.wte(idx)          # (B, T, C): what each token means
        pos_emb = self.transformer.wpe(pos)          # (T, C): where each token sits
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)
        for block in self.transformer.h:
            x = block(x, record_attn=record_attn)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss
        logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab)
        return logits, None

    def configure_optimizer(
        self, weight_decay: float, learning_rate: float
    ) -> torch.optim.AdamW:
        """AdamW with weight decay only where it belongs.

        Weight decay pulls parameters toward zero — a sensible prior for
        big matmul matrices, but harmful for LayerNorm gains (should sit
        near 1) and biases. Standard practice: decay everything with
        ndim >= 2, leave the 1-D parameters alone.
        """
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        allowed_ids: list[int] | None = None,
        stop_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling: feed the sequence, take the logits of
        the last position, pick one token, append, repeat.

        temperature=0 means greedy argmax. top_k keeps only the k most
        likely tokens. allowed_ids restricts sampling to a whitelist of
        token ids (single-token legality masking; play.py's strict mode
        instead ranks whole moves via utils.legal_move_logprobs, which
        also covers multi-token moves). stop_id ends generation early
        (usually <eos>).
        """
        assert idx.size(0) == 1, "generate() is written for batch size 1, for clarity"
        assert top_k is None or top_k >= 1, "top_k must be >= 1"
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (1, vocab)

            if allowed_ids is not None:
                keep = torch.full_like(logits, float("-inf"))
                keep[:, allowed_ids] = logits[:, allowed_ids]
                logits = keep

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits[logits < kth] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                # Sample on the CPU regardless of model device: one CPU
                # generator then gives reproducible draws everywhere.
                next_id = torch.multinomial(
                    probs.cpu(), num_samples=1, generator=generator
                ).to(idx.device)

            idx = torch.cat((idx, next_id), dim=1)
            if stop_id is not None and next_id.item() == stop_id:
                break
        return idx

    # -- checkpoint helpers ------------------------------------------------
    def checkpoint_dict(self, **extra) -> dict:
        return {"model": self.state_dict(), "config": asdict(self.config), **extra}

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: torch.device) -> "GPT":
        model = cls(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        return model
