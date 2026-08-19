# Lab report: the ablations

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 3](08-exercises.md#3-ablations-what-is-actually-load-bearing--an-afternoon).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

The default model (797,312 parameters, 4 layers × 4 heads, learned
position embeddings) is deliberately overpowered for a 15-token world —
so which parts of the architecture are actually load-bearing? Exercise 3
prescribes three cuts, each pretrained from scratch with the standard
recipe and measured with the standard eval:

```bash
.venv/bin/python -m minillm.train --stage pretrain --n-layer 1 --out-dir runs/exp-ablate-1layer
.venv/bin/python -m minillm.train --stage pretrain --n-head 1  --out-dir runs/exp-ablate-1head
# ablation 3 has no flag, by design - a two-line edit in minillm/model.py:
#   x = self.transformer.drop(tok_emb + pos_emb)   ->   ...drop(tok_emb)
.venv/bin/python -m minillm.train --stage pretrain --out-dir runs/exp-ablate-nopos
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-ablate-<name>/model.pt --out ...
```

(For ablation 3, evaluate with the edit still in place — loading the
checkpoint into the unedited model would add the *untrained, random*
position embeddings back in. Revert the edit afterwards; `git checkout
minillm/model.py`.)

## Predictions first

The exercise insists on predicting before training, so, on the record:

1. **1 layer** (202,496 parameters): legality mostly survives — the
   gravity rule is a local pattern. Result prediction and strength drop
   noticeably: recognizing a completed line and planning both look like
   multi-step relational work that wants depth.
2. **1 head** (same 797,312 parameters — only the factoring of attention
   changes): near-baseline everywhere, maybe small dips where the
   [taxonomy's](08-exercises.md#8-attention-head-taxonomy-catalogue-all-16-heads--an-afternoon)
   specialized roles (previous move, same column, sink) must now
   time-share one softmax.
3. **No position embeddings**: the interesting one. *Which cells are
   occupied* is determined by the multiset of moves alone (gravity!), so
   legality should largely survive a bag-of-moves view. But *who owns*
   each cell depends on whether a move was played 1st or 2nd — so result
   prediction should collapse toward guessing, and strength with it.
   The causal mask still leaks some order, so "collapse" should stop
   short of chance.

## Results

Baseline = the reference pretrain checkpoint (`runs/eval_pretrain.json`).
Same seeds, same eval protocol throughout:

| metric | baseline | 1 layer | 1 head | no pos emb |
|---|---:|---:|---:|---:|
| parameters | 797,312 | 202,496 | 797,312 | 797,312* |
| best val loss | 0.7506 | 0.7533 | 0.7467 | 0.8533 |
| argmax legal (teacher-forced) | 100.0% | 100.0% | 100.0% | 98.6% |
| legal probability mass | 99.6% | 99.4% | 99.6% | 98.0% |
| free-running 1st-try legal | 99.8% | 99.5% | 99.6% | 98.7% |
| clean self-play games | 98.0% | 96.0% | 96.5% | 90.0% |
| result prediction | 99.2% | 98.5% | 99.2% | **73.3%** |
| vs random W/D/L | 41.8/20.2/38.0% | 42.5/24.8/32.8% | 41.5/20.8/37.8% | **33.8/16.5/49.8%** |
| vs optimal W/D/L | 0/0/100% | 0/0/100% | 0/**9.5**/90.5% | 0/0/100% |
| optimal-move rate | 70.3% | 72.2% | 70.8% | 72.2% |

\* the `wpe` table still exists (unused and untrained — no gradient
flows into it), so the parameter count does not move.

## Reading the numbers

**1. Depth is not load-bearing here.** The 1-layer model — a quarter of
the parameters, one attention pass, one MLP — matches the 4-layer
baseline within noise on every metric, including result prediction
(98.5%). Whatever "recognize a completed line from a move sequence"
requires, one round of communicate-then-compute is enough at this board
size. The prediction above was wrong about depth, and the honest
conclusion cuts deeper: the default model is not just overpowered in
width, it is overpowered in *depth*, and eval — not architecture
aesthetics — is the only way to find out.

**2. Head count is not load-bearing either.** One head, forced to time-share the roles the taxonomy exercise finds specialized across four (previous move, same column, `<bos>` sink), lands within noise of the baseline on every metric - and its best val loss (0.7467) is nominally the better one. It even converts 9.5% draws against the perfect solver where the baseline scores zero, a reminder that pretrain-stage strength numbers wobble (see the multi-seed lesson in the char-tokenizer lab) rather than evidence that one head is *better*. The factoring of attention, at this scale, is a matter of taste, not capability.

**3. Position embeddings are the only cut that draws blood, and exactly
where predicted.** Legality barely notices (98.6% argmax-legal,
teacher-forced — occupancy really is order-invariant under gravity,
and the model still knows *where pieces are*). Ownership does not
survive: result prediction falls from 99.2% to **73.3%**, and playing
strength inverts from +3.8 points net vs random to −16 (the model now
*loses* to a random opponent more often than it wins). The two
metrics separate cleanly because they sit on opposite sides of the
symmetry: "is this cell full?" is a bag-of-moves question, "whose line
is that?" needs the order the embeddings carried. That 73.3% still
beats the ~45% a majority-class guess scores (X wins about 45% of the
held-out games) is the causal-mask leak at work: with strictly causal
attention, position i sees i+1 keys, and that alone smuggles some order
information back in.

The optimal-move-rate row deserves its footnote: it *rises* slightly
under two of the three cuts (72.2% vs 70.3%). It is a per-position
argmax metric dominated by common early positions where the average
game's move is often also an optimal one — a reminder that it measures
agreement, not competence; the W/D/L rows are where competence lives.

> **In a real LLM:** you just reproduced two standing results. First,
> ablation studies — not intuition — are how architecture decisions are
> actually justified; entire families of "obviously necessary"
> components (extra depth here) turn out replaceable when measured,
> which is why scaling-law papers sweep depth/width/heads instead of
> arguing about them. Second, the position-embedding result mirrors the
> real finding that decoder-only transformers *without* positional
> encodings (NoPE) still learn usable order information from the causal
> mask alone — attention over i+1 keys is itself a position signal —
> but degrade exactly on tasks that need precise order arithmetic. Your
> 73.3% referee is a NoPE transformer failing at order-critical work
> while passing order-invariant work, in a world small enough to prove
> which is which.

## Reproduce it

```bash
make data
.venv/bin/python -m minillm.train --stage pretrain --n-layer 1 --out-dir runs/exp-ablate-1layer
.venv/bin/python -m minillm.train --stage pretrain --n-head 1  --out-dir runs/exp-ablate-1head
# edit model.py as above, then:
.venv/bin/python -m minillm.train --stage pretrain --out-dir runs/exp-ablate-nopos
for d in runs/exp-ablate-*; do
  .venv/bin/python -m minillm.evaluate --ckpt "$d/model.pt" --out "$d/eval.json"
done
git checkout minillm/model.py     # put the position embeddings back
```

Next: the [attention-head taxonomy](08-exercises.md#8-attention-head-taxonomy-catalogue-all-16-heads--an-afternoon)
asks what those four layers of heads were doing all along, or back to
the [exercises](08-exercises.md).
