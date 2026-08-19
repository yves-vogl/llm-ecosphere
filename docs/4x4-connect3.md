# Lab report: 4×4 Connect-3

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 9](08-exercises.md#9-scale-the-world-44-connect-3--a-weekend).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

Every constant in this repo is downstream of `game.py`'s `N = 3` and
`COLS = "ABC"`. Exercise 9 widens the world to a 4×4 board with win
length 3 and forces the whole pipeline to follow. Two things made the
worked solution smaller than the exercise advertises — because the
generalizations it demands are now committed:

* **`LINES` is derived.** `game.py` now builds its winning lines from
  `_win_segments(n, win)` — every straight `win`-cell window on an
  `n × n` board. With the shipped `N = 3, WIN = 3` this is provably the
  classic 8 lines (a test pins it); at `N = 4, WIN = 3` it yields all
  **24** windows (8 horizontal, 8 vertical, 8 diagonal) that full-length
  lines would miss.
* **State is N-agnostic.** `solver.py`'s `State`/`EMPTY` are N-tuples,
  the `Game.stacks` default, row validation and the board footer derive
  from `N`/`COLS`, and `play.py`'s column-full check uses `N`.

What remains of "touch the entire pipeline" is deliberately tiny, and
that is the demonstration of how the pieces connect:

```diff
 # minillm/game.py
-COLS = "ABC"
-N = 3
+COLS = "ABCD"
+N = 4
```

plus `--block-size 20` at pretraining time (`MAX_GAME_TOKENS` grows to
1 + 16 + 1 + 1 = 19, and the tokenizer's 16 cell tokens follow from
`COLS`/`N` automatically — vocab 15 → 22).

## Enumeration dies; sampling replaces it

The full 3×3 enumeration is 1,310 games. At branching factor ≤ 4 and
depth ≤ 16 the 4×4 tree has billions of leaves — `enumerate_all_games`
is over. The committed replacement is `solver.sample_random_games`
(uniform random legal rollouts, deduplicated, seeded) and
`sample_expert_games` (solver-optimal expert vs random opponent — the
solver still provides ground truth; sampling only replaces the *corpus*),
wired into the dataset CLI:

```bash
.venv/bin/python -m minillm.dataset --out data --sample 20000
.venv/bin/python -m minillm.train --stage pretrain --block-size 20 \
    --out-dir runs/exp-4x4-pretrain
.venv/bin/python -m minillm.train --stage finetune \
    --init-from runs/exp-4x4-pretrain/model.pt --out-dir runs/exp-4x4-finetune
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-4x4-<stage>/model.pt --out ...
```

20,000 rollouts produced **15,408 unique complete games** (8,957 X wins,
6,376 O wins, and only **75 draws** — 0.5%; win length 3 on a 4×4 board
leaves almost no room for a full, line-free board) and an 11,582-game
sampled expert corpus. The exact solver still works perfectly one size
up: **41,750 reachable positions**, solved in about a second, and the
verdict is news:

> **4×4 Connect-3 is a first-player win.** `describe_root_value()`:
> "X (the first player) wins with perfect play." The extra column kills
> the draw — and with it the 3×3 chapter's tidy "a draw against the
> solver is the ceiling" story. Against a perfect opponent the best
> possible outcome is now *win as X, lose as O*: the eval's
> `vs_optimal` ceiling becomes 50 / 0 / 50, and 0 / 0 / 100 is what a
> merely mortal O-side defense looks like.

## Results

Reference numbers are the 3×3 pipeline
(`runs/eval_pretrain.json` / `runs/eval.json`); the 4×4 numbers come
from the same seeded recipe on the sampled corpus:

| metric | 3×3 pretrain | 4×4 pretrain | 3×3 finetune | 4×4 finetune |
|---|---:|---:|---:|---:|
| best val loss | 0.7506 | 1.0915 | 0.4771 | 0.7649 |
| argmax legal (teacher-forced) | 100.0% | 99.9% | 99.5% | 99.8% |
| legal probability mass | 99.6% | 99.4% | 99.1% | 99.2% |
| free-running 1st-try legal | 99.8% | 99.0% | 98.8% | 98.9% |
| clean self-play games | 98.0% | 90.5% | 90.5% | 88.5% |
| result prediction | 99.2% | 99.2% | 100.0% | 99.2% |
| vs random W/D/L | 41.8/20.2/38.0% | 41.5/1.5/57.0% | 79.2/14.5/6.2% | 91.0/1.0/8.0% |
| vs optimal W/D/L | 0/0/100% | 0/0/100% | 0/61/39% | **44.0/1.0/55.0%** |
| optimal-move rate | 70.3% | 65.8% | 86.5% | 93.3% |

## Reading the numbers

**1. The ceiling moved — and the model climbed straight to it.** The
headline row is vs optimal: **44.0 / 1.0 / 55.0**. Remember what perfect
play permits here: X wins, O loses, so a model alternating sides against
the solver has a theoretical ceiling of 50 / 0 / 50. The finetuned model
converts the forced win in 88% of its games as X *against perfect
defense*, and as O loses as theory demands. At 3×3 the equivalent
headline was "draws the solver 61% of the time"; one extra column turned
the best-possible story from "survives" into "executes a proven win",
and the model actually tells it.

**2. Legality scales through sampling.** 99.9% argmax-legal over 15,614
held-out positions — positions that now routinely include states no
training game visited — and 99.0% free-running first-try legality. The
gravity grammar is learned from a *sample* just as well as it was from
the closed enumeration; coverage was never what legality needed.

**3. The draw is functionally extinct, and the eval quietly says so.**
0.5% draws in the corpus become 1.0–1.5% draw rates everywhere. Result
prediction holds at 99.2%, but read it knowingly: the `#=` class is so
rare that the referee is graded almost entirely on wins. Class imbalance
does not announce itself — you have to notice the 75-games-in-15,408
line in `meta.json`.

**4. Strength-vs-random jumps to 91.0%** (from 79.2% at 3×3). A wider
board gives a competent attacker more simultaneous threats against a
random opponent, and the 93.3% optimal-move rate (over 1,290 distinct
positions) says the finetuned policy tracks the solver closely across a
41,750-position world.

**5. Do not compare the val losses to 3×3.** 1.0915 / 0.7649 vs 0.7506 /
0.4771 — different vocabulary, different sequence lengths, different
per-token difficulty, and (the new part) a val split that contains
genuine novelty rather than recombinations. Per-token cross-entropy only
ever compares within one world.

## The epistemics change — that was the point

At 3×3, "held-out games" was a polite fiction: every *position* in the
val split also occurs in some training game via shared prefixes, so eval
measured recombination, not extrapolation (the
[lookup-baseline lab](08-exercises.md#7-a-lookup-table-baseline-memorization-vs-generalization--an-afternoon)
quantifies exactly this). At 4×4 the corpus is a 15k-game sample of a
game space with billions of leaves: held-out games routinely contain
positions **no training game ever visited**, and every eval number above
is partly a statement about extrapolation. That is what real-LLM eval
numbers mean — the corpus is always a vanishing sample of the input
space — and the exercise's deepest lesson is that this change arrives
not with a bigger model but with a barely bigger *world*.

> **In a real LLM:** the corpus decision you just made — "we cannot
> enumerate, so we sample, and now coverage is a variable" — is data
> engineering's entire job description. Web corpora are samples with
> selection bias (crawl policy, dedup, quality filters), and what the
> model never saw silently defines what eval can and cannot claim. The
> draw-vanishing surprise has a production twin too: change the domain
> slightly and the label distribution shifts under you (75 draws in
> 15,408 games would starve any "predict the draw" capability — class
> imbalance you must notice yourself, because no solver will tell you).

## Reproduce it

```bash
# 1. the two-constant flip in minillm/game.py (diff above)
# 2. sampled corpora + the pipeline:
.venv/bin/python -m minillm.dataset --out data --sample 20000
.venv/bin/python -m minillm.train --stage pretrain --block-size 20 --out-dir runs/exp-4x4-pretrain
.venv/bin/python -m minillm.train --stage finetune --init-from runs/exp-4x4-pretrain/model.pt --out-dir runs/exp-4x4-finetune
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-4x4-finetune/model.pt
git checkout minillm/game.py       # back to the shipped 3x3
make test                          # the suite pins the 3x3 world
```

Next: [outlook](outlook.md) for where the scaling road leads from here,
or back to the [exercises](08-exercises.md).
