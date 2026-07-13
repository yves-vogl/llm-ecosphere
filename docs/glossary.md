# Glossary — every LLM concept, grounded in this repo

Language-model vocabulary in one alphabetical list, defined in plain English
and pinned to the file where the idea is touchable in this codebase. It is
for readers who already read the chapters and want a fast lookup instead of
another linear pass. Every entry ends with an **In this repo** line pointing
at running code, not just prose.

## A

#### Attention (self-attention, heads)

Attention lets a token look at other tokens and pull in information from
them, weighted by relevance. "Self"-attention means the tokens looking and
the tokens being looked at come from the same sequence. A model usually runs
several heads in parallel, each with its own relevance weights, so one
position can gather several kinds of information at once — one head might
track the previous move, another the same board column. Attention is the
only layer where information moves *between* positions.

**In this repo:** `CausalSelfAttention` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
splits one Q/K/V projection into 4 heads of size 32. `minillm/inspect_attention.py`
prints real per-head matrices — watch a head learn to track column stack
heights. See [04 — The model](04-model.md).

#### Autoregressive model

An autoregressive model generates a sequence one element at a time, each new
element conditioned on everything generated so far — predicting the next
token, feeding it back in, repeating. That loop is the mechanical meaning of
"the model writes one token at a time."

**In this repo:** `GPT.generate()` in `minillm/model.py` is exactly this
loop: crop context, forward pass, pick a token, append, repeat. A sampled
game is 8–12 tokens produced this way via `python -m minillm.sample`. See
[06 — Inference](06-inference.md).

## B

#### Batch

A batch is a group of training examples processed together in one
forward-and-backward pass, so the same matmuls handle many examples at once —
why training is fast on parallel hardware. Examples of different lengths are
usually padded to a common width so the batch forms one rectangular tensor.

**In this repo:** `train.py --batch-size` defaults to 64 games per step,
sampled with replacement. `build_tensors` in `minillm/dataset.py` pads every
game to `block_size = 16`, so a batch is a `(64, 16)` tensor. See
[05 — Training](05-training.md).

#### Block size / context window

Block size, or context window, is the maximum number of tokens a model can
look at when predicting — anything beyond it is invisible. It bounds both the
attention computation, whose cost scales with the square of sequence length,
and the position-embedding table, which needs one row per position.

**In this repo:** `ModelConfig.block_size = 16` in `minillm/config.py`,
comfortably above `MAX_GAME_TOKENS = 12`, the longest possible Drop-Tac-Toe
transcript. See [03 — Tokenization](03-tokenization.md).

## C

#### Causal mask

A causal mask stops a position from attending to positions after it, so the
prediction at position *t* depends only on tokens `0..t`. Without it,
next-token prediction would be trivial: the model could copy the answer out
of the future it is supposed to predict. It works by setting the score of any
forbidden future position to negative infinity before the softmax.

**In this repo:** built with `torch.tril` and applied via `masked_fill` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py).
`tests/test_model.py::test_causality_future_does_not_leak_into_past` proves
it: tampering with the last input token changes only the last output logit.
See [04 — The model](04-model.md).

#### Checkpoint

A checkpoint is a saved snapshot of a model's weights, usually with its
architecture config, written to disk so training can resume or a later stage
can start from it. Production teams typically keep the checkpoint with the
best validation performance, not the very last one.

**In this repo:** `GPT.checkpoint_dict()` in `minillm/model.py` bundles
weights and config; `train.py` writes `runs/pretrain/model.pt` and
`runs/finetune/model.pt` only when validation loss improves. Finetuning
loads its start weights from the pretrain checkpoint via `--init-from`. See
[05 — Training](05-training.md).

#### Cross-entropy loss

Cross-entropy loss measures how much probability a model assigned to the
token that actually came next: `-log p(correct token)`. A confident, correct
prediction scores near zero; a confident, wrong one pays a large penalty.
Minimizing it is equivalent to maximizing the likelihood the model assigns
to the real data.

**In this repo:** computed via `F.cross_entropy(..., ignore_index=-1)` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py).
A clueless model scores `ln(15) ≈ 2.708` nats over this 15-token vocabulary;
the pretrained checkpoint reaches a best validation loss of 0.7506. See
[05 — Training](05-training.md).

## D

#### Decoder-only Transformer

A decoder-only Transformer is a stack of attention-plus-feed-forward blocks
that only looks backward, via the causal mask, and is trained to predict the
next token — the architecture family behind GPT-2/3, Llama, and Claude,
unlike encoder-decoder architectures used for tasks like translation.

**In this repo:** the entire model in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
— 4 stacked blocks, causal self-attention, an MLP per block, a tied output
head. The module docstring calls it GPT-2's architecture "shrunk to ~0.8M
parameters." See [04 — The model](04-model.md).

## E

#### Embedding

An embedding is a learned lookup table mapping a discrete symbol — a token
id, or a position index — to a dense vector, turning "token 7" into a point
in a continuous space where distance and direction carry meaning. A
Transformer typically adds two: one for *what* a token is, one for *where*
it sits.

**In this repo:** `wte` is an `nn.Embedding(15, 128)` token table, `wpe` an
`nn.Embedding(16, 128)` position table, added together in `GPT.forward`
(`minillm/model.py`). Position embeddings matter here because attention
alone cannot tell `B1 A1 B2` from `B2 A1 B1`. See
[04 — The model](04-model.md).

#### Evaluation / benchmark

Evaluation measures what a trained model actually *does*, distinct from what
its training loss says it fits — behavioural tests like "is this move
legal", or, at production scale, standardized benchmarks such as MMLU,
GSM8K, HumanEval, or SWE-bench. Loss says the model fits held-out text; a
benchmark says it can perform a task.

**In this repo:**
[`minillm/evaluate.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/evaluate.py)
runs five metric groups — teacher-forced and free-running legality,
refereeing, match play versus random and versus the solver, and solver-move
agreement. See [07 — Evaluation](07-evaluation.md).

## F

#### Finetuning / SFT

Finetuning continues training an already-pretrained model on a smaller,
curated dataset to shift its behavior toward a goal. Supervised finetuning
(SFT) is the flavor that turns a base model into one that follows
instructions or performs a target task: it imitates curated examples, with
the loss usually masked so only the desired output is trained on.

**In this repo:** `--stage finetune` in `minillm/train.py` loads the
pretrained checkpoint and continues on `data/expert_games.jsonl` (334
solver-optimal games), with opponent moves masked from the loss so only the
expert's moves are imitated. See [05 — Training](05-training.md).

#### FLOPs

FLOPs (floating-point operations) count the basic multiplies and adds a
computation performs, the standard unit for raw computational cost. Rule of
thumb: training costs roughly 6 FLOPs per parameter per token — 2 for the
forward pass, 4 for backpropagation.

**In this repo:** [10 — Why GPUs?](10-gpu-cuda.md) applies that rule to this
project's pretraining run — roughly 0.8M parameters times 2M tokens times 6,
about 10 TFLOPs total — versus GPT-3's roughly 3 x 10²³ FLOPs, about thirty
billion times more.

## G

#### Gradient descent, optimizer, AdamW

Gradient descent computes how much each parameter contributed to the loss,
then nudges every parameter toward reducing it. An optimizer is the specific
update rule; AdamW, used by almost every Transformer, keeps a running
average of each parameter's gradient and squared gradient, giving each its
own step size, and applies weight decay as a direct pull toward zero.

**In this repo:** `GPT.configure_optimizer` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
builds AdamW with betas `(0.9, 0.95)`, decaying only 2-D parameters, never
biases or LayerNorm gains. See [05 — Training](05-training.md).

#### GPU / CUDA

A GPU is built around thousands of small arithmetic units running in
lock-step, optimized for the massive, regular, branch-free arithmetic that
matrix multiplication is. CUDA is NVIDIA's programming platform (2007) that
turned GPUs into general-purpose parallel computers: a kernel model, a
toolchain, and libraries (cuBLAS, cuDNN) of hand-tuned Transformer ops.

**In this repo:** everything runs on plain CPU on purpose — the model is
small enough that GPU launch overhead would outweigh any speedup, per
`pick_device` in `minillm/utils.py`. The same PyTorch code runs as CUDA
kernels via `--device cuda` on larger models, unchanged. See
[10 — Why GPUs?](10-gpu-cuda.md).

## H

#### Hallucination

Hallucination is when a model states something confidently and fluently that
is false or ungrounded — a fluent failure mode, not a lack of fluency. It
happens because a model is trained to produce plausible continuations, not
to check facts; next-token prediction never guarantees the output is *true*.

**In this repo:** no factual hallucination exists here — the world is 15
tokens and fully solved — but the toy analogue is a transcript that is
fluent yet physically wrong: claiming victory (`#X`) while the game is still
running. `verify_transcript()` in `minillm/sample.py` catches exactly this.
See [06 — Inference](06-inference.md).

## I

#### Inference

Inference is using a trained model to produce output, as opposed to
training it — forward passes only, no gradients or weight updates. For a
language model this means the autoregressive loop: encode a prompt, get a
distribution over the next token, sample or pick one, repeat.

**In this repo:** `next_token_logits` in `minillm/utils.py` and
`GPT.generate()` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
are the shared plumbing behind `play.py`, `sample.py`, and `evaluate.py`.
See [06 — Inference](06-inference.md).

## K

#### KV cache

A KV cache stores the key and value vectors attention computes for
already-processed tokens, so generating a new token needs only its own
query/key/value plus the cached past — turning per-step cost from quadratic
to linear in sequence length. At long contexts this is the single most
important serving optimization in real systems.

**In this repo:** deliberately absent — `GPT.generate()` recomputes the full
forward pass every step, fine at a 12-token maximum and microseconds of
cost. The
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
docstring calls the attention "naive ... no FlashAttention, no KV cache" on
purpose. Implementing one is exercise 5 in [08 — Exercises](08-exercises.md).

## L

#### Learning-rate schedule (warmup + cosine)

The learning rate controls how large each optimizer step is, and rarely
stays constant. Warmup ramps it up linearly from near zero, since early
gradients are large and unreliable; after warmup, cosine decay glides it
back down toward a small minimum — large steps early, small careful steps as
training converges.

**In this repo:** `lr_at()` in `minillm/train.py` implements linear warmup
over 100 steps then cosine decay to 10% of the peak. Pretraining peaks at
`1e-3`, finetuning at `2e-4`. See [05 — Training](05-training.md).

#### Logits / softmax

Logits are a model's raw, unnormalized output scores — one number per
vocabulary entry, with no constraint that they sum to anything. Softmax
converts logits into a probability distribution by exponentiating each score
and normalizing by the sum, so results are positive and sum to 1.

**In this repo:** `GPT.forward` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
returns logits of shape `(B, T, 15)`; `F.softmax` runs inside `generate()`
to turn those into a next-token distribution. See
[04 — The model](04-model.md).

#### Loss masking

Loss masking excludes specific positions from the training loss even though
the model still sees them as input — conditioning on context without being
rewarded for reproducing it. It covers padding (never real language) and, in
SFT, the "other side" of a conversation — here, the opponent's moves.

**In this repo:** targets set to `-1` in `build_tensors`
(`minillm/dataset.py`) are skipped by `ignore_index=-1` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py).
The same `-1` masks `<pad>` always, and, when `expert_only=True`, every
opponent move. See [05 — Training](05-training.md).

## O

#### Overfitting / validation split

Overfitting is when training performance keeps improving while performance
on unseen data gets worse — memorizing the training set instead of the
pattern behind it. A validation split, held out and never trained on, is the
standard way to detect this: track its loss, keep only the best checkpoint.

**In this repo:** `split_games` in `minillm/dataset.py` holds out 10% of
games with a fixed seed-42 shuffle. Finetuning is a textbook case:
validation loss bottoms out at 0.4771 at step 100, then climbs to 0.659 by
step 1499 while training loss keeps falling to 0.340 — why `train.py` saves
only the best-validation checkpoint. See [05 — Training](05-training.md).

## P

#### Parameter / weights

Parameters, also called weights, are the numbers a model learns during
training — every matrix and vector entry gradient descent adjusts. Parameter
count is the standard headline size of a model, though it says nothing on
its own about how well it was trained.

**In this repo:** exactly 797,312 parameters, computed by `GPT.num_params()`
in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
and broken down tensor by tensor — for scale, GPT-2 small is 124M parameters
with the identical architecture. See [04 — The model](04-model.md).

#### Pretraining

Pretraining is the first, largest-scale training stage: teaching a model the
broad structure of its domain — grammar, facts, style, or here, the rules of
a game — from the widest available data, with plain next-token prediction
and no task-specific curation.

**In this repo:** `--stage pretrain` in `minillm/train.py` trains from
scratch on `data/all_games.jsonl` (all 1,310 games) for 3,000 steps,
learning legality and turn order without learning to play *well*. See
[05 — Training](05-training.md) and [07 — Evaluation](07-evaluation.md).

#### Prompt

A prompt is the input text a model conditions on before generating — the
tokens that go in before it produces new tokens of its own. Everything the
model knows about the situation must be inferable from the prompt alone.

**In this repo:** the prompt is the move history so far. `encode_prompt` in
`minillm/tokenizer.py` wraps it as `<bos>` plus the moves played;
`next_token_logits` builds exactly this prompt on every turn of `play.py`.
See [06 — Inference](06-inference.md).

## R

#### Residual stream / residual connection

A residual connection adds a sublayer's output back onto its input
(`x = x + sublayer(x)`) instead of replacing it. Stacked across layers this
forms a "residual stream" — an unobstructed path every block reads from and
writes to by addition, which is what makes deep stacks trainable: gradients
flow back without passing through any weight matrix.

**In this repo:** every `Block.forward` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
is `x = x + self.attn(...)` then `x = x + self.mlp(...)`, pre-norm. [04 — The
model](04-model.md) calls it "a shared 128-lane bus."

#### RLHF

RLHF (reinforcement learning from human feedback) typically follows SFT in
production assistants: humans rank model outputs, a reward model learns
those preferences, and the base model is optimized with reinforcement
learning to produce outputs the reward model scores highly.

**In this repo:** not implemented. The pipeline stops at SFT-style
finetuning — no reward model, no RL loop. Exercise 10 in
[08 — Exercises](08-exercises.md) sketches a REINFORCE self-play stage on
top of the finetuned checkpoint as a "what's missing" exercise, not
something the codebase runs.

## S

#### Sampling / temperature / top-k

Sampling draws the next token from the model's predicted distribution
instead of always taking the most likely one, keeping generation varied.
Temperature rescales logits before softmax — below 1 sharpens, above 1
flattens, 0 means "always argmax." Top-k keeps only the k highest-probability
tokens before sampling, cutting the long tail.

**In this repo:** implemented in `GPT.generate()`
(`minillm/model.py`). `evaluate.py` uses temperature 0, `play.py` defaults
to 0.7, `sample.py` to 1.0 — the raw learned distribution. See
[06 — Inference](06-inference.md).

#### Seed / reproducibility

A random seed fixes a pseudo-random generator's starting point, so every
"random" choice — initialization, batching, dropout, sampling — becomes
deterministic and repeatable. Reproducibility must be designed in, including
which random streams get seeded.

**In this repo:** `set_seed()` in `minillm/utils.py` seeds Python's `random`
and `torch.manual_seed` (default `--seed 1337`); the train/validation split
uses its own fixed seed, 42, so `--seed` never changes which games are held
out. Sampling always draws on the CPU RNG even on GPU/MPS. See
[05 — Training](05-training.md).

## T

#### Token / tokenizer / vocabulary

A token is the atomic unit a language model reads and writes — not
necessarily a word or character, whatever the vocabulary defines. The
vocabulary is the fixed list of tokens a model knows; the tokenizer is the
two-way mapping between text and the integer ids that index into it and the
model's embedding table.

**In this repo:** the vocabulary is 15 tokens — 3 special (`<pad>`, `<bos>`,
`<eos>`), 9 move cells (`A1`..`C3`), 3 results (`#X`, `#O`, `#=`) —
hand-written in
[`minillm/tokenizer.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/tokenizer.py),
one token per whole move, deliberately with no `<unk>`. See
[03 — Tokenization](03-tokenization.md).

## W

#### Weight tying

Weight tying reuses a single matrix for two roles — making the token
embedding table, which turns input ids into vectors, and the output
projection, which turns final hidden states into per-token scores, literally
the same tensor. This saves parameters and links a token's input meaning to
its output meaning by construction.

**In this repo:** `self.transformer.wte.weight = self.lm_head.weight` in
[`minillm/model.py`](https://github.com/yves-vogl/llm-ecosphere/blob/main/minillm/model.py)
ties the 15x128 embedding and output matrices, checked at the pointer level
by `tests/test_model.py::test_weight_tying`. The saving is a rounding error
— 1,920 of 797,312 parameters — kept as the historically faithful choice
GPT-2 also made. See [04 — The model](04-model.md).
