# Lab report: the lookup-table baseline

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 7](08-exercises.md#7-a-lookup-table-baseline-memorization-vs-generalization--an-afternoon).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

The nastiest question you can ask any trained model: **would a hash map
have done just as well?** With 1,310 total games sharing heavy prefix
overlap, the suspicion is legitimate — so we built the hash map.
`minillm/baseline_lookup.py` records, for every transcript prefix in the
training split, the observed distribution of next tokens; at eval time
it predicts from the table, and on a prefix it never saw it falls back
to uniform over the engine's legal moves (uniform over the three result
tokens when the game is over). No parameters, no gradients, no
generalization by construction.

The design trick that keeps the comparison honest: **the table wears
GPT's inference interface** (`LookupModel.__call__` returns
`(1, 1, vocab)` logits like `GPT.forward`'s inference call), so
`eval_on_val_games`, `eval_expert_agreement` and the whole `utils` move
assembly run on it *unchanged* — same code path, same seeds, same split
(`split_games` with seed 42 / `val_frac` 0.1, exactly what training
used).

```bash
.venv/bin/python -m minillm.baseline_lookup          # vs the finetune ckpt
.venv/bin/python -m minillm.baseline_lookup --ckpt runs/pretrain/model.pt
```

## Results

1,179 training games produce **4,227 distinct prefixes**. Network
baselines from `runs/eval_pretrain.json` / `runs/eval.json`:

| metric (held-out games) | lookup table | pretrain net | finetune net |
|---|---:|---:|---:|
| argmax legal (teacher-forced) | **100.0%** | 100.0% | 99.5% |
| result prediction | **45.0%** | 99.2% | 100.0% |
| optimal-move rate (414 rollout positions) | **72.0%** | 70.3% | 86.5% |

And the exercise's new measurement — **prefix coverage on held-out
games**: the table has seen **90.2%** of held-out positions (958 of
1,062). The prefix overlap really is that heavy. Splitting the
optimal-move comparison by coverage, on positions walked along the
held-out games themselves (298 distinct positions, of which 70 are
prefixes the table has *never* seen):

| held-out positions | n | table optimal | pretrain net | finetune net |
|---|---:|---:|---:|---:|
| prefix seen in training | 228 | 63.6% | 59.2% | 79.8% |
| prefix never seen | 70 | 98.6% | 97.1% | 98.6% |

## Reading the numbers

**1. The hash map ties pretraining on strength.** Table 72.0% vs
pretrain network 70.3% optimal-move rate — and on held-out seen
positions the table (63.6%) actually beats the pretrained network
(59.2%). No surprise, and the right lesson: pretraining's objective is
to imitate the *average* game, and the table IS the average game, stored
losslessly. If your training story ends at "fit the corpus", a hash map
is a fair rival. Finetuning is what buys real distance: 79.8% vs 63.6%
on the same seen positions — the network extracts *policy* from the
solver's demonstrations, the table can only replay frequency.

**2. The never-seen-prefix column refuses to play its assigned role —
and that is a finding.** The exercise predicts the gap should open where
the table is reduced to uniform guessing. It doesn't: on the 70
never-seen prefixes, everyone scores ~98%. The reason is structural in
this tiny world: prefixes escape the table only *deep* in rare games,
and deep positions have one or two legal moves, most of them
solver-optimal — uniform guessing is nearly perfect there. In a world
where every early position is shared with some training game, "unseen"
correlates with "trivial". The measured lesson: memorization-vs-
generalization cannot be settled on a strength metric here, because the
unseen set is not hard.

**3. Refereeing is the real never-seen test, and there the table
collapses: 45.0% vs 99–100%.** A complete game's full move sequence is
*by construction* absent from the table (a training game with the same
moves would *be* that game, and games in the corpus are unique), so
every result prediction is a fallback — uniform over `#X`/`#O`/`#=`,
argmax-resolved to the first id, which matches exactly the X-win base
rate of the held-out split. The network, on the same held-out games,
referees at 99–100%: it tracks board state well enough to *recognize* a
win it has never watched happen. That is generalization, measured where
the table provably cannot follow — 100% of these inputs are outside its
training set, not 6.6% of them.

**4. Legality is no differentiator — by design.** Both the table's
seen-prefix predictions (legal moves observed in real games) and its
fallback (uniform over legal moves) are legal by construction, so its
100.0% argmax-legal even nominally beats the finetuned network's 99.5%.
Knowing the rules is the cheapest thing in this world; the interesting
capabilities start above it.

> **In a real LLM:** this is the memorization-vs-generalization debate
> at lab scale, and the punchline is faithful to history. n-gram
> language models *were* this lookup table — smoothing tricks included
> (our uniform-over-legal fallback is a crude backoff) — and they ruled
> for decades precisely because, on frequent contexts, frequency is
> hard to beat. The transformer's margin shows up exactly where it
> showed up here: composing state from inputs that never co-occurred in
> training (our result prediction), not on contexts the corpus already
> covers densely. The methodology transfers too: probing on data
> *provably absent* from training — not merely held out, but
> structurally impossible to have memorized — is how modern
> contamination and memorization studies are run, and point 2 is the
> standing warning that a held-out set can quietly be too easy to
> settle the question.

## Reproduce it

```bash
make data                                            # corpus first
make test                                            # includes the table's unit tests
.venv/bin/python -m minillm.baseline_lookup --out runs/baseline-vs-finetune.json
.venv/bin/python -m minillm.baseline_lookup --ckpt runs/pretrain/model.pt \
    --out runs/baseline-vs-pretrain.json
```

Next: the [char-tokenizer lab](09-char-tokenizer-lab.md) applies the
same one-variable methodology to tokenization, or back to the
[exercises](08-exercises.md).
