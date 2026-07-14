# 09 — Lab report: the character-level tokenizer

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 1](08-exercises.md#1-character-level-tokenizer--an-afternoon).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

Exercise 1 asks a sharp question: `minillm/tokenizer.py` hands the model
moves on a silver platter — one token per move — so what does the model
*lose* when `B2` becomes the two tokens `B`, `2`? We built it, retrained,
and measured. The headline, in one sentence: **legality got slightly
better, not worse — and the interesting damage showed up somewhere the
question didn't point.**

## The new vocabulary

`CharTokenizer` in `minillm/tokenizer.py` shares the three specials with
the move-level vocabulary and replaces the twelve move/result tokens with
ten characters — 13 ids in total:

| ids | tokens | class |
|-----|--------|-------|
| 0–2 | `<pad>` `<bos>` `<eos>` | specials (unchanged) |
| 3–8 | `A B C 1 2 3` | move characters |
| 9–12 | `# X O =` | result characters |

The same game now reads

```
move level:  <bos> B1 A1 B2 C1 B3 #X <eos>                    (8 tokens)
char level:  <bos> B 1 A 1 B 2 C 1 B 3 # X <eos>              (14 tokens)
```

and the longest possible game grows from 12 to 22 tokens (`MAX_GAME_CHARS`),
which is why the char-level pretraining below passes `--block-size 24`
(finetuning and evaluation inherit it from the checkpoint's config).

## Implementation: hide the splitting, generalize the assembly

The exercise's hint turned out to be the whole design. Both tokenizers
share one interface — `tokens_per_move` (1 vs 2), `encode_move`,
`encode_prompt`, `encode_game`, `group_units` — and the character
splitting lives entirely inside `CharTokenizer`. Consequences downstream:

* **`dataset.build_tensors` survived almost unchanged.** The one casualty
  was the advertised trap: the finetuning loss mask assumed
  "move number = token index". With two tokens per move, move *k* occupies
  sequence indices 2k−1 and 2k (after `<bos>`), so the target at index
  t+1 belongs to move `t // tokens_per_move + 1` — and both characters of
  an opponent move are masked (or trained) together.
  `tests/test_dataset.py` pins this down for both tokenizers.
* **Inference needed a move-assembly step** (`minillm/utils.py`). Three
  helpers hide the granularity from `evaluate.py` and `play.py`:
  `legal_move_logprobs` scores each legal move by the chain rule —
  p(`B2`) = p(`B`)·p(`2` | `B`) — so "the model's favourite legal move"
  stays well-defined; `greedy_unit` and `sample_unit` decode one whole
  move (or result) unrestricted, one token at a time. Per the exercise's
  rule, **a char-level move counts as legal only if both characters
  combine to a legal cell** — sampling `#` then `X` mid-game is one
  illegal attempt, exactly like sampling a floating cell.
* **Checkpoints record their tokenizer.** `train.py --tokenizer char`
  stores the name in the checkpoint; finetuning inherits it and refuses a
  conflicting flag; `evaluate.py`, `play.py`, `sample.py` and
  `inspect_attention.py` reconstruct the right vocabulary from the file.
  Old checkpoints (no `tokenizer` key) keep meaning move-level.
* **Regression discipline.** Before training anything char-level, the
  refactored `evaluate.py` was re-run on the *move-level* baseline
  checkpoints: every metric reproduced the reference
  `runs/eval_pretrain.json` / `runs/eval.json` exactly (the sole
  difference, `mean_legal_prob_mass` at the 8th decimal, is
  `exp(log_softmax)` vs `softmax` floating-point noise). The comparison
  below is therefore apples to apples. The test suite grew from 39 to 56.

Training used the exercise's recommended hygiene:

```bash
.venv/bin/python -m minillm.train --stage pretrain --tokenizer char \
    --block-size 24 --out-dir runs/exp-char-pretrain          # 130 s CPU
.venv/bin/python -m minillm.train --stage finetune \
    --init-from runs/exp-char-pretrain/model.pt --out-dir runs/exp-char-finetune
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-char-pretrain/model.pt \
    --out runs/exp-char-pretrain/eval.json                    # and finetune alike
```

The model is architecturally identical (798,080 parameters vs 797,312 —
eight more position embeddings, two fewer vocabulary rows).

## Results

Same seeds, same eval protocol, same 414 solver-agreement positions.
Baselines from `runs/eval_pretrain.json` / `runs/eval.json`, char numbers
from `runs/exp-char-*/eval.json`:

| metric | move pretrain | char pretrain | move finetune | char finetune |
|---|---:|---:|---:|---:|
| argmax legal (teacher-forced) | 100.0% | 100.0% | 99.5% | 99.2% |
| legal probability mass | 99.6% | 99.8% | 99.1% | 98.8% |
| free-running 1st-try legal | 99.8% | **99.9%** | 98.8% | **99.8%** |
| clean self-play games | 98.0% | **99.0%** | 90.5% | **98.0%** |
| result prediction | 99.2% | 100.0% | 100.0% | 97.7% |
| vs random W/D/L | 41.8/20.2/38.0% | 41.2/21.5/37.2% | **79.2**/14.5/6.2% | 68.8/22.2/9.0% |
| vs optimal solver W/D/L | 0/0/100% | 0/**36**/64% | 0/61/39% | 0/**95**/5% |
| optimal-move rate | 70.3% | 72.9% | 86.5% | 83.1% |

## Reading the numbers

**1. The predicted damage never arrived.** The exercise's framing ("what
does the model lose?") suggests legality should suffer once the model has
to assemble moves from characters. It didn't — free-running legality
*improved*, dramatically so after finetuning (98.0% vs 90.5% clean games).
The game's structure explains why. Under gravity, choosing a move
factorizes into an easy decision and a forced one: the *column* is the
only real choice (a column character is illegal only when its stack is
full), and given the column, the landing *row* is fully determined by the
stack height. The char model spends its first token on the easy 3-way
choice and learns the nearly deterministic conditional p(row | column,
history) almost perfectly. The move-level model must rank all nine cells
in one softmax and can leak probability onto floating cells of the right
column. Add that the char corpus carries ~1.9× more training targets per
game (19.3 vs 10.2 across the corpus), and the "harder" tokenization turns
out to give the model *more* signal about an *easier* factorization of the
same problem.

**2. The real casualty is strength-vs-random — and it is a trade, not a
loss.** After finetuning, the char model wins fewer games against the
random player (68.8% vs 79.2%) and agrees with the solver slightly less
often across random positions (83.1% vs 86.5%) — but against the *perfect*
solver it converts 95% of games into draws, versus 61% for the move-level
model, and a draw is the theoretical ceiling of this game. The two models
sit at different points on a sharp-vs-solid frontier: the move-level model
is the better shark (punishes weak opponents), the char-level one the
better wall (survives the strongest opponent). One caveat belongs in every
lab report: this is a single seeded run per configuration; the vs-optimal
gap (95% vs 61%) is far outside sampling noise at 200 games, but the
few-point differences elsewhere are not. Disentangling *why* — is the
factorized policy inherently more drawish, or did the earlier
best-val-loss checkpoint (step 200 vs a 1,500-step run) simply overfit
less? — needed multi-seed reruns. We ran them; see "Multi-seed: which of
these numbers survive a reseed?" below for the answer. (Exercise 4's
temperature sweep got its own report,
[temperature-sweep.md](temperature-sweep.md).)

**3. Result refereeing dipped (97.7% vs 100.0%).** The result is now two
tokens too, and the referee must get both right; the errors are the
model's, not the harness's (the same greedy assembly scores 100% on the
char pretrain checkpoint). A small, honest cost of the longer format.

**4. Never compare losses across tokenizers.** The char pretrain run's
best validation loss (0.3835) looks *half* the move-level one (0.7506).
Meaningless: cross-entropy is per *token*, and the tokenizations have
different token counts and different per-token difficulty (the row
character is nearly free). Normalize per *game* — loss × targets-per-game,
both measured on the validation split — and the two models are almost
indistinguishable: 0.7506 × 10.1 ≈ 7.6 nats/game vs 0.3835 × 19.2 ≈ 7.4
nats/game. Same knowledge, different denominators.

## Multi-seed: which of these numbers survive a reseed?

Everything above is one seeded run per config. We reran the finetuning
stage — the stage with all the interesting deltas — at three training
seeds (1337, 1338, 1339) for both tokenizers; 1337 is the shipped
default, so its column reproduces the numbers in the table above exactly.
Each checkpoint was evaluated with the same fixed eval seed, so only
training-init variance moves the metrics below — the random opponent, the
solver positions, and the sampling draws are held constant across all six
runs.

| metric | move finetune (mean [min–max]) | char finetune (mean [min–max]) |
|---|---:|---:|
| vs optimal solver, draw rate | 82.2% [61–100] | 95.0% [95–95] |
| vs random, win rate | 77.3% [73–80] | 76.3% [69–82] |
| clean self-play games | 91.2% [88–96] | 98.2% [98.0–98.5] |
| optimal-move rate | 85.3% [83–86] | 83.7% [83–85] |

Per-seed, in training-seed order (1337 / 1338 / 1339): move draw-vs-solver
runs 61.0 / 85.5 / 100.0, char draw-vs-solver runs 95.0 / 95.0 / 95.0;
move win-vs-random runs 79.2 / 73.2 / 79.5, char win-vs-random runs 68.8 /
81.5 / 78.8.

**F1. The char wall is real and inherent.** Draw-vs-solver lands on
exactly 95.0% for all three seeds — zero variance. This settles the open
question point 2 raised above: the char model's drawishness against the
perfect solver is not a lucky, less-overfit checkpoint. It is a property
of the factorized column-then-row policy itself.

**F2. The move model's headline number was a checkpoint lottery.**
Draw-vs-solver swings from 61.0 to 85.5 to 100.0 across seeds — a mean of
82.2%, but the shipped 61% was the pessimistic tail, not the typical
outcome. The dramatic "95 vs 61" gap in the single-seed table was
partly a single-seed artifact. The honest gap is char 95.0% (stable) vs
move 82.2% (mean, high variance).

**F3. As attackers they are tied.** Win-vs-random averages 77.3% for
move and 76.3% for char, with fully overlapping ranges (73–80 vs 69–82).
The single-seed "move is the sharper shark" story (79.2 vs 68.8) does not
survive a reseed — 68.8% was char's low tail, and one char seed (81.5%)
outright beat every move seed.

**Revised conclusion.** The axis that separates these two models is not
sharp-vs-solid — they attack random opponents equally well on average.
It's *consistency*. The character/factorized policy is a reliable,
low-variance drawer against the solver; the move policy is high-variance,
its defensive strength hostage to which best-val-loss checkpoint you
happen to land on. The legality gain does hold up across seeds (clean
self-play 98.2% vs 91.2%, in the same direction on every seed). Stated
plainly, the meta-lesson: one seed told two stories — move is sharper,
char is a far better wall — and multi-seed shows one of those was real
(char reliably draws) and the other was noise (the attacking strengths
are actually tied). That's the whole argument for reporting seeds instead
of a single run.

**Reproduce it:** 3 seeds x {move, char} x {pretrain, finetune} —
`python -m minillm.train --seed S [--tokenizer char --block-size 24] ...`
then `python -m minillm.evaluate --ckpt .../model.pt --out .../eval.json`
per checkpoint, aggregated as mean and [min-max] across the three seeds.

> **In a real LLM:** you just reproduced both halves of the tokenization
> debate. BPE vocabularies (GPT-2's 50k, Llama's 32k–128k) buy short
> sequences at the price of opacity inside tokens — the "count the r's in
> strawberry" failure — and byte/character-level architectures (ByT5,
> MambaByte) buy transparency at the price of long sequences. In our
> 22-token world the length penalty is invisible, so transparency wins
> outright on legality. At 100k-token contexts, where attention cost and
> KV-cache memory scale with sequence length, the economics reverse — and
> that, not model quality, is why production models tokenize coarsely.
> The per-game loss normalization in point 4 is the same reason serious
> tokenizer comparisons report bits-per-byte, never raw per-token loss.

## Reproduce it

```bash
make test                                     # 56 tests, both tokenizers
.venv/bin/python -m minillm.sample --ckpt runs/exp-char-pretrain/model.pt --num 5
```

The sampled transcripts come out as character streams —
`C 1 B 1 C 2 C 3 B 2 A 1 A 2 # X <eos>` — and `sample.py` reassembles the
pairs before the engine verifies them, so a dangling half-move is caught
and named. Play against it with
`python -m minillm.play --ckpt runs/exp-char-finetune/model.pt` and watch
the `p` command: the distribution you see is over the *first character* of
the next move — the model commits to a column before it says the row.

Next: [10 — Why GPUs?](10-gpu-cuda.md), or back to the
[exercises](08-exercises.md) — number 8 (attention-head taxonomy) is twice
as interesting on the char checkpoint, where heads must track
column-letter/row-digit pairing on top of everything else.
