# Lab report: the KV cache

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 5](08-exercises.md#5-implement-a-kv-cache-in-generate--a-weekend).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

Exercise 5 names the crime scene precisely: every iteration of
`GPT.generate` re-runs the *entire* prefix through all four blocks, even
though the keys and values for positions `0..T-1` come out identical to
the previous iteration's — only position `T` is new. The fix is the KV
cache: store `k`/`v` per layer, feed only the newest token, attend
against the cached past. O(T) per generated token instead of O(T²), and
**greedy output must be token-for-token identical** — that is the
correctness gate, and it is now pinned by tests.

## Design: strictly opt-in

This repo's model code is deliberately the naive, readable version, and
[CONTRIBUTING.md](https://github.com/yves-vogl/llm-ecosphere/blob/main/CONTRIBUTING.md)
says optimizations that obscure the shape of the computation don't belong
in `model.py`. The cache therefore ships **off by default**:
`generate(use_cache=False)` — and everything else in the pipeline — runs
the exact code path that was always there. `use_cache=True` switches the
loop; nothing else changes. The diff touches exactly the three places the
exercise names:

**`CausalSelfAttention.forward`** gains a `kv_cache` dict. On the prefill
call the whole prompt's `k`/`v` land in the dict; on later calls the new
token's `k`/`v` are concatenated on:

```python
if kv_cache is not None:
    if "k" in kv_cache:
        k = torch.cat((kv_cache["k"], k), dim=2)
        v = torch.cat((kv_cache["v"], v), dim=2)
    kv_cache["k"], kv_cache["v"] = k, v
S = k.size(2)                       # keys span the full history
att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)          # (B, nh, T, S)
att = att.masked_fill(self.causal_mask[:, :, S - T : S, :S] == 0, ...)
```

The mask slice is the elegant part: query row `i` sits at absolute
position `S - T + i`, so `causal_mask[S-T:S, :S]` is always the right
window. Uncached, `S == T` and it collapses to the familiar `[:T, :T]`;
for a single cached query it selects a row of all-ones — no masking,
because every key is in that query's past.

**`GPT.forward`** gains `kv_caches` (one dict per layer) and
`pos_offset`: `torch.arange(T)` becomes
`torch.arange(pos_offset, pos_offset + T)`.

**`GPT.generate`** prefills once with the whole prompt, then feeds
`idx[:, -1:]` with `pos_offset=idx.size(1) - 1`.

## The two classic bugs, encountered as advertised

The exercise's hint promises two traps, and both are real:

1. **The position embedding.** A token fed alone is still at absolute
   position `T_so_far`. Embed it at position 0 and the model quietly
   computes garbage — no crash, just wrong logits.
   `test_pos_offset_embeds_tokens_at_their_absolute_position` constructs
   exactly this bug and asserts it *would have been caught*: the wrongly
   embedded step diverges from the full forward pass, the correct one
   matches it to `atol=1e-5`.
2. **The causal mask.** A single query attending over all cached keys
   needs no mask at all — every key is its past. Masking with the
   *unshifted* triangle (`[:1, :S]`) would let the query see only key 0.
   The `S - T` offset above makes the correct behaviour fall out of one
   expression instead of an if/else.

One trap the hint does not mention: the naive path can slide a context
window (`idx[:, -block_size:]`) when generation runs past `block_size`. A
cache of absolute positions cannot — so `use_cache=True` asserts up front
that the whole generation fits in `block_size`. Every transcript in this
repo does, with room to spare (12 ≤ 16 move-level, 22 ≤ 24 char-level).

## Correctness, measured

`tests/test_model.py` pins the contract down:

* incremental logits equal the full-prefix forward at **every** step
  (`atol=1e-5`), starting from a 4-token prefill;
* greedy generation is token-for-token identical with and without the
  cache, for both vocabulary shapes (15/16 move-level, 13/24 char-level);
* `allowed_ids` and `stop_id` behave identically under the cache;
* the block-size guard actually fires.

On the shipped finetuned checkpoint, 100 greedy games and 300 sampled
games (temperature 1.0, same seeded generator) were generated through
both paths: XXAGREE.

## Speed, measured honestly

Timed on one CPU core (the model is ~0.8M parameters, sequences ≤ 12
tokens — this is the regime the docs *warn* is too small to care about):

| decode path | 300 sampled games | 100 greedy games |
|---|---:|---:|
| naive (`use_cache=False`) | XXNAIVE | XXNAIVEG |
| KV cache (`use_cache=True`) | XXCACHED | XXCACHEDG |
| speedup | XXSPEED | XXSPEEDG |

XXSPEEDNOTE

> **In a real LLM:** the KV cache is not an optimization, it is *the*
> serving-cost driver. At 100k-token contexts the cached keys and values
> for one conversation run to gigabytes — often rivaling the weights —
> which is why grouped-query attention, multi-query attention, cache
> quantization and paged attention (vLLM) exist: all of them are KV-cache
> compression or management schemes. "Prefill vs. decode" as separate
> serving phases is likewise a KV-cache concept, and you just implemented
> both phases: the prefill pass that builds the cache in parallel, and
> the decode loop that extends it one token at a time. The `S - T` mask
> arithmetic you wrote is the same bookkeeping a paged-attention kernel
> does at industrial scale.

## Reproduce it

```bash
make test          # includes the five KV-cache tests
.venv/bin/python - <<'PY'
import time, torch
from minillm.utils import load_model, pick_device, tokenizer_for_checkpoint
model, ckpt = load_model("runs/finetune/model.pt", pick_device("cpu"))
tok = tokenizer_for_checkpoint(ckpt)
for use_cache in (False, True):
    g = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    for _ in range(300):
        model.generate(torch.tensor([[tok.bos_id]]), max_new_tokens=tok.max_game_tokens,
                       temperature=1.0, stop_id=tok.eos_id, generator=g,
                       use_cache=use_cache)
    print(f"use_cache={use_cache}: {time.perf_counter() - t0:.2f}s")
PY
```

Next: [10 — Why GPUs?](10-gpu-cuda.md) explains why this same mechanism
dominates serving economics at scale, or back to the
[exercises](08-exercises.md).
