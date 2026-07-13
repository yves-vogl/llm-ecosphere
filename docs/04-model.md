# 04 — The Transformer, spelled out

This is the core chapter. We walk through `minillm/model.py` top to bottom
with the actual tensor shapes this project runs at. Everything else in the
pipeline — data, training, evaluation — exists to feed and measure the ~300
lines in this file.

The fixed shapes, from `minillm/config.py` (`ModelConfig`):

| symbol | meaning | value here |
|---|---|---|
| `B` | batch size | whatever the caller passes |
| `T` | sequence length | ≤ `block_size = 16` (a full game is at most 12 tokens) |
| `C` | `n_embd`, width of the residual stream | 128 |
| `nh` | `n_head`, attention heads per block | 4 |
| `hs` | head size = `C / nh` | 32 |
| — | `n_layer`, stacked blocks | 4 |
| — | `vocab_size` | 15 |

The architecture is a decoder-only Transformer — the same family as GPT-2/3,
Llama, and Claude. The docstring at the top of `model.py` says it plainly:
this is GPT-2's architecture "shrunk to ~0.8M parameters". Nothing about the
code changes with scale; only the numbers in `ModelConfig` do.

## Embeddings: from integers to vectors

A token id is just an integer — `B2` is `7`, per the vocabulary in
`minillm/tokenizer.py`. Integers carry no geometry: id 7 is not "close to"
id 6 in any useful sense. The first thing the model does is trade each id for
a learned 128-dimensional vector:

```python
tok_emb = self.transformer.wte(idx)          # (B, T, C): what each token means
pos_emb = self.transformer.wpe(pos)          # (T, C): where each token sits
x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)
```

`wte` is an `nn.Embedding(15, 128)` — literally a 15-row lookup table. Rows
get pushed around by gradient descent until tokens that behave similarly
(say, the three column-A cells) end up in similar directions.

`wpe` is the same trick for *positions*: an `nn.Embedding(16, 128)` indexed
by `pos = torch.arange(T)`. Why is it needed at all? Because attention (next
section) is a weighted sum over a *set* — it is permutation-invariant. Without
position information, the model literally could not tell `B1 A1 B2` from
`B2 A1 B1`, and in this game move order is everything: whether `B2` is legal
depends on whether `B1` was already played. Adding a per-position vector
breaks the symmetry; the model learns what "being the third move" means the
same way it learns what "B2" means.

Note that the two embeddings are simply *added*, not concatenated. That works
because the model is free to allocate different subspaces of the 128
dimensions to "what" and "where" if that is useful.

> **In a real LLM:** GPT-2 does exactly this — learned absolute position
> embeddings added to token embeddings, just with `wte` of shape 50257×768
> and `wpe` of shape 1024×768. Modern models (Llama, Claude-class models)
> replaced learned absolute positions with rotary embeddings (RoPE), which
> encode *relative* position directly inside the attention dot product and
> extrapolate better to long contexts. The reason positions are needed at
> all — attention is permutation-invariant — is identical at every scale.

## The residual stream: the central highway

After the embeddings, the tensor `x` of shape `(B, T, 128)` is the *residual
stream*. Every block reads from it and adds its contribution back:

```python
def forward(self, x, record_attn=False):
    x = x + self.attn(self.ln_1(x), record_attn=record_attn)
    x = x + self.mlp(self.ln_2(x))
    return x
```

That `x = x + ...` shape is the single most important design decision in the
file. The stream itself is never transformed, only *added to* — so the
identity path from the embedding to the final head is unobstructed. Gradients
flow backward along that same path without passing through any weight matrix,
which is what makes stacks of dozens of blocks trainable. A useful mental
model: the residual stream is a shared 128-lane bus; attention and MLP are
peripherals that read the bus, compute something, and write their result back
by addition.

This is also the "pre-norm" layout: `LayerNorm` is applied to the *input* of
each sublayer (`self.ln_1(x)`), not to the sum afterwards. The `Block`
docstring in `model.py` gives the reason: pre-norm "keeps the residual stream
an unnormalized highway that gradients flow through unimpeded". The original
2017 Transformer used post-norm and was notoriously hard to train deep;
GPT-2 switched to pre-norm and essentially everyone followed.

## CausalSelfAttention, step by step

Attention is the only place where information moves *between* positions.
Everything else in the model operates on each position independently.

### Q, K, V

Each token emits three vectors, computed by one fused linear layer:

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
...
q, k, v = self.c_attn(x).split(C, dim=2)
```

The intuition, straight from the class docstring: a query is "what am I
looking for?", a key is "what do I contain?", a value is "what do I hand
over if someone attends to me?". Fusing the three projections into a single
`128 → 384` matmul instead of three `128 → 128` ones is purely an efficiency
idiom (one kernel launch instead of three); mathematically it is identical,
and it is the layout GPT-2's original code used, hence the name `c_attn`.

### Splitting into heads

```python
q = q.view(B, T, nh, hs).transpose(1, 2)   # (B, T, 128) -> (B, 4, T, 32)
```

Nothing is computed here — the 128 channels are just *reinterpreted* as 4
independent groups of 32. Each head will run the full attention mechanism in
its own 32-dimensional subspace. The point of multiple heads is that one
softmax produces one attention pattern per query; four heads let a token look
at four different things simultaneously (we will see this concretely in the
inspection section: one head tracks the previous move, another tracks the
same column).

### Scores, scaling, mask, softmax

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
att = F.softmax(att, dim=-1)
```

`q @ k^T` gives `(B, 4, T, T)`: for every head, a T×T matrix of raw
compatibility scores between every query position and every key position.

Why divide by `sqrt(hs)`? A dot product of two random 32-dimensional vectors
with unit-variance components has variance ≈ 32, i.e. standard deviation
≈ 5.7. Feed numbers that large into a softmax and it saturates — one entry
gets weight ≈ 1, the rest ≈ 0, and the gradient through the softmax collapses
to nearly zero. Dividing by `sqrt(hs)` normalizes the score variance back to
≈ 1, keeping the softmax in the regime where it still has, as the comment in
`model.py` puts it, "usable gradients".

The causal mask is a lower-triangular matrix of ones built once in
`__init__` with `torch.tril` and stored via `register_buffer` — a constant,
not a parameter; it is saved with checkpoints but never trained. The
`masked_fill(... , float("-inf"))` trick is worth pausing on: you cannot zero
out forbidden weights *after* softmax, because the remaining row would no
longer sum to 1. Setting the score to −∞ *before* softmax is the clean
solution — `exp(−inf) = 0` exactly, and the normalization only distributes
mass over the allowed (past) positions. Position *i* may attend to *j ≤ i*
and nothing else; during generation the future does not exist yet, and
training must match that constraint or the model would learn to cheat.

### Weighted sum and merge

```python
y = att @ v                                   # (B, 4, T, T) @ (B, 4, T, 32) -> (B, 4, T, 32)
y = y.transpose(1, 2).contiguous().view(B, T, C)   # glue heads -> (B, T, 128)
return self.resid_dropout(self.c_proj(y))
```

Each row of `att` is a probability distribution over past positions; `att @ v`
replaces each position's head-output with the corresponding weighted average
of value vectors. The transpose/`view` pair is the exact inverse of the head
split. Finally `c_proj` (`128 → 128`) mixes the four heads' outputs together
before the result is added to the residual stream — without it, head 0's
output could only ever land in channels 0–31.

> **In a real LLM:** this is the textbook implementation, kept naive on
> purpose ("no FlashAttention, no KV cache — a worked example, not a speed
> record", per the module docstring). Production stacks compute the same
> mathematics but never materialize the full T×T matrix: FlashAttention
> tiles the computation through GPU SRAM, turning memory cost from O(T²) to
> O(T), which is what makes 100k+ token contexts affordable. And at
> inference, a KV cache stores keys and values of past tokens so each new
> token costs O(T) instead of re-running the full O(T²) pass — here, with
> T ≤ 16, `generate()` simply recomputes everything each step.

## MLP: expand, GELU, contract

```python
self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)   # 128 -> 512
self.gelu = nn.GELU()
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # 512 -> 128
```

The division of labor is stated in the `MLP` docstring: "Attention moves
information *between* positions; the MLP does the per-position processing of
whatever attention gathered." Communicate, then compute. The MLP is applied
to every position independently and identically — it never looks sideways.

The 4× expansion (a GPT-2 convention "kept by most models since") gives the
nonlinearity room to work: project into a wider space, apply GELU, project
back. GELU is a smooth cousin of ReLU (roughly `x · Φ(x)`) that avoids
ReLU's hard zero-gradient region. Note that the two MLP matrices — 65,536
parameters each — are the biggest single tensors in the whole model.

## Final LayerNorm and the tied head

After the last block: `x = self.transformer.ln_f(x)`, then a linear head
maps each 128-vector to 15 logits. But look closely:

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.transformer.wte.weight = self.lm_head.weight
```

That second line is *weight tying*: the token embedding matrix and the
output head are literally the same 15×128 tensor playing two roles —
row *i* is both "what token *i* means as input" and "the direction whose dot
product with the final hidden state scores token *i* as output". It saves
parameters and couples the input and output semantics of each token.
`tests/test_model.py::test_weight_tying` pins this down at the pointer level:

```python
assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
```

> **In a real LLM:** weight tying comes from Press & Wolf (2016) and was
> adopted by GPT-2, where it saves 50257 × 768 ≈ 38.6M parameters — over 30%
> of the 124M model. At larger scales the fraction shrinks (embeddings grow
> linearly with width, blocks quadratically), and many recent large models
> untie the matrices again because the parameter savings no longer justify
> the coupling. Here the tied matrix is a rounding error (1,920 params) —
> it is kept because it is the historically faithful choice.

## Initialization: why 0.02 / sqrt(2·n_layer)

All linears and embeddings start as `Normal(0, 0.02)` (`_init_weights`).
Then one extra pass:

```python
for name, p in self.named_parameters():
    if name.endswith("c_proj.weight"):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

Every `c_proj` — the last matrix of each attention module and each MLP — is
exactly the matrix whose output is *added to the residual stream*. With 4
layers there are 2 × 4 = 8 such additions. Sums of independent contributions
grow in variance linearly with their count, so the stream's variance would
grow ~8× through the stack; scaling each contribution's init by
`1/sqrt(2·n_layer)` cancels that growth and keeps the stream's variance ≈ 1
at initialization, regardless of depth. This is lifted directly from the
GPT-2 paper's initialization note; the comment in `model.py` says exactly
this.

## Parameter count: where the 797,312 live

Computed from the code (`GPT(ModelConfig()).num_params()`):

| tensor | shape | params |
|---|---|---:|
| `wte` (tied with `lm_head`) | 15 × 128 | 1,920 |
| `wpe` | 16 × 128 | 2,048 |
| per block: `ln_1` (w+b) | 2 × 128 | 256 |
| per block: `attn.c_attn` (w+b) | 128×384 + 384 | 49,536 |
| per block: `attn.c_proj` (w+b) | 128×128 + 128 | 16,512 |
| per block: `ln_2` (w+b) | 2 × 128 | 256 |
| per block: `mlp.c_fc` (w+b) | 128×512 + 512 | 66,048 |
| per block: `mlp.c_proj` (w+b) | 512×128 + 128 | 65,664 |
| **one block subtotal** | | **198,272** |
| × 4 blocks | | 793,088 |
| `ln_f` (w+b) | 2 × 128 | 256 |
| **total** | | **797,312** |

Two things to notice. First, 99.5% of the parameters are in the blocks; the
vocabulary is so tiny that embeddings are negligible (in GPT-2 they are a
third of the model). Second, each block spends about two thirds of its
parameters on the MLP and one third on attention — the same ratio as GPT-2,
because both follow from `C` and the 4× expansion, not from scale.

## forward(): one function, two modes

```python
if targets is not None:
    logits = self.lm_head(x)                      # (B, T, vocab)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
    )
    return logits, loss
logits = self.lm_head(x[:, [-1], :])              # (B, 1, vocab)
return logits, None
```

**Training mode** (`targets` given): logits for *all* T positions, because
thanks to the causal mask every position is simultaneously a training example
— predicting `targets[b, t]` from `idx[b, :t+1]`. One forward pass over a
12-token game yields up to 12 next-token predictions. The `ignore_index=-1`
argument is load-bearing for this project: any target set to −1 contributes
zero loss. That is how `<pad>` positions are excluded, and — crucially, in
finetuning — how *opponent moves* are masked out so the model is only graded
on its own side's decisions (chapter 05).

**Inference mode** (`targets=None`): only the last position's logits are
computed, `(B, 1, vocab)`, since that is all sampling needs. Verified by
`test_inference_returns_last_position_only`.

## generate(): the autoregressive loop

`generate()` is the whole "LLMs write one token at a time" story in fifteen
lines: crop the context to `block_size`, forward, take the last position's
logits, pick a token, append, repeat. The knobs:

- `temperature=0` → greedy argmax; higher values flatten the distribution
  before sampling.
- `top_k` → keep only the k most likely tokens, drop the rest to −∞.
- `allowed_ids` → hard whitelist; anything else is set to −∞ before
  sampling. This powers the "strict" play mode where only legal moves are
  permitted (chapter 07).
- `stop_id` → end early, normally on `<eos>`.

Note the whitelist and the mask reuse the same trick as causal masking:
"forbidden" is spelled −∞-before-softmax, never zero-after.

> **In a real LLM:** the loop is identical — Claude produces its answers
> token by token through exactly this recurrence — but the sampling zoo is
> richer (top-p/nucleus, repetition penalties) and `allowed_ids` generalizes
> to *constrained decoding*: grammar-restricted sampling that forces valid
> JSON or a fixed tool-call schema by masking illegal tokens at each step.
> Same −∞ trick, fancier automaton deciding what is legal.

## Proof, not vibes: the causality test

`tests/test_model.py::test_causality_future_does_not_leak_into_past` is the
one test to internalize:

```python
x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size  # tamper with the last token
...
assert torch.allclose(logits1[0, :9], logits2[0, :9], atol=1e-5)
assert not torch.allclose(logits1[0, 9], logits2[0, 9], atol=1e-5)
```

Change token 9 and every logit at positions 0–8 must remain unchanged —
identical up to the 1e-5 tolerance the test allows — while position 9 itself
must change. If someone deleted the
`masked_fill` line, this test — not the training loss — is what would catch
it. A model with a leaky mask still trains, and its loss looks *fantastic*,
because it is copying answers from the future. This failure mode is silent
everywhere except here.

## Looking inside: a real attention matrix

`minillm/inspect_attention.py` runs a forward pass with `record_attn=True`
(which makes each `CausalSelfAttention` stash its post-softmax weights in
`last_attn`) and prints one T×T matrix per head. Against the finetuned
checkpoint:

```
$ python -m minillm.inspect_attention --ckpt runs/finetune/model.pt --moves "B1 A1 B2"

--- layer 1, head 1 ---
        <bos>    B1    A1    B2
 <bos>   1.00     ·     ·     ·
    B1   0.00  1.00     ·     ·
    A1   0.01  0.96  0.04     ·
    B2   0.00  0.80  0.16  0.04

--- layer 3, head 3 ---
        <bos>    B1    A1    B2
 <bos>   1.00     ·     ·     ·
    B1   1.00  0.00     ·     ·
    A1   1.00  0.00  0.00     ·
    B2   1.00  0.00  0.00  0.00
```

Rows are queries, columns are keys, each row sums to 1, and the blank
upper-right triangle *is* the causal mask made visible.

Layer 1 head 1 is the payoff: from the query position `B2`, 0.80 of the
attention mass goes back to `B1` — the earlier move in the *same column*.
This is not a one-off; in the same run, layer 0 head 0 gives `B2 → B1` 0.76
and layer 2 head 1 gives it 0.83. These heads are tracking column stack
heights, which is exactly the information required by the gravity rule: `B2`
is only a legal, meaningful token because `B1` is underneath it. Nobody
programmed a "same column" feature — gradient descent discovered that
attending along columns reduces next-token loss.

Layer 3 head 3 shows the opposite pattern: every query dumps ~1.00 onto
`<bos>`. That is an *attention sink* — softmax rows must sum to 1, so a head
that currently has nothing useful to say parks its mass on a semantically
empty token rather than injecting noise. The docstring of
`inspect_attention.py` flags this as "a well-known phenomenon in real
transformers too": the same sink behavior is documented in production LLMs,
and the StreamingLLM line of work exploits it by always keeping the first
few tokens in the KV cache.

## Next

Next: [05 — Training: pretraining and finetuning](05-training.md) — how this
model's 797,312 parameters are actually fitted: cross-entropy over all 1,310
games first, then solver-optimal games with opponent moves masked to −1.
