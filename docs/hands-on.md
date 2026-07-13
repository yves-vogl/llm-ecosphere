# Hands-on: change the training, watch what happens

The fastest way to understand a system is to perturb it and observe what
changes. That is normally bad advice for a language model — a real
pretraining run costs months of GPU time, so nobody "just tries" flipping a
knob to see what happens. This repo is different: `make pretrain` finishes
in about 2 minutes on a laptop CPU and `make finetune` in about 1 (see the
README's quickstart), and every claim about what the model
learned has a ground-truth answer via the exact solver. That combination —
cheap to rebuild, cheap to verify — makes "change it and measure it" a
completely valid way to learn this material, not a shortcut around learning
it.

This page is not a reading exercise. It is a loop, repeated four times:

1. **Change ONE thing** — an objective, a tokenizer, an architecture knob,
   or a decoding setting.
2. **Rebuild** — retrain (or, for decoding, skip straight to step 3 — no
   retraining needed).
3. **Test** — in the arena against the perfect solver, or with
   `minillm.evaluate`.
4. **Compare** against the ground truth: the README's numbers, or the
   solver's proof that perfect play is always a draw.

Every command below was run against this repository before being written
down — the numbers under "what you should see" are real measured output,
not predictions. Your own run will land close (training is seeded) but not
always identical down to the last game, since a few of the runs below
retrain a fresh checkpoint.

One piece of hygiene before you start: never point an experiment's
`--out-dir` at `runs/pretrain` or `runs/finetune` — that would overwrite the
reference checkpoints the README's numbers describe. Every command below
uses a `runs/exp-*` directory instead, exactly as
[08 — Exercises](08-exercises.md) recommends.

## Experiment 1 — change the objective: expert vs. gambler

This is the flagship experiment: same architecture, same tokenizer, same
pretrained starting point — only *what the model is taught to imitate*
changes. `minillm/train.py` supports two finetuning objectives:

- **`expert`** (the default) — imitates the solver's optimal moves, with the
  opponent's moves masked out of the loss. Minimax-optimal, unbeatable.
- **`gambler`** — imitates the *winning* side of decisive games from the
  full enumeration (draws dropped), with the *losing* side's moves masked
  out instead. This imitates whoever happened to win — aggressive,
  exploitable play, not optimal play.

### The change and the rebuild

```bash
make finetune         # the expert objective — ~1 min CPU
make model-gambler     # the gambler objective — writes runs/exp-gambler-move
```

`make model-gambler` runs
`.venv/bin/python -m minillm.train --stage finetune --objective gambler --out-dir runs/exp-gambler-move`
— same finetuning stage, same corpus size class, only `--objective` and
`--out-dir` differ from `make finetune`.

### The test

Play each checkpoint against the perfect solver, 200 games, alternating who
moves first:

```bash
.venv/bin/python -m minillm.arena --model runs/finetune/model.pt --vs solver --games 200
.venv/bin/python -m minillm.arena --model runs/exp-gambler-move/model.pt --vs solver --games 200
```

### What you should see

The expert model, holding the theoretical ceiling against perfect play:

```
runs/finetune/model.pt  vs  solver (perfect negamax)   |   200 games   |   model plays: strict argmax
  win     0 / 200   (  0.0%)
  draw  122 / 200   ( 61.0%)
  loss   78 / 200   ( 39.0%)
```

The gambler model, losing most games to the same perfect opponent:

```
runs/exp-gambler-move/model.pt  vs  solver (perfect negamax)   |   200 games   |   model plays: strict argmax
  win     0 / 200   (  0.0%)
  draw   60 / 200   ( 30.0%)
  loss  140 / 200   ( 70.0%)
```

Neither model ever *wins* — the solver proved the game is a draw with
perfect play, so a perfect opponent cannot be beaten, only matched or lost
to. The expert model draws 61% of the time (matching `runs/eval.json`); the
gambler model, trained on exactly the same *amount* of supervision but on
whoever-won rather than whoever-played-optimally, loses 70% of the time. If
you want the second half of the picture, `.venv/bin/python -m minillm.evaluate --ckpt runs/exp-gambler-move/model.pt`
also shows it winning *more* against a weak random opponent (63% vs the
expert's ~79% — closer than you might expect, but still behind) while being
far more exploitable by a strong one. This is the entire imitation-learning
lesson in one pair of numbers: what you imitate is what you become, and
"trained to convergence" says nothing about which behavior it converged to.

## Experiment 2 — change the tokenizer: move-level vs. character-level

The shipped tokenizer hands the model one token per move (`B2` is a single
id). What happens if the model has to *spell* its moves instead — `B`, then
`2` — the way real BPE tokenizers force models to spell rare words out of
sub-word pieces?

### The change and the rebuild

```bash
.venv/bin/python -m minillm.train --stage pretrain --tokenizer char \
    --block-size 24 --out-dir runs/exp-char-pretrain
.venv/bin/python -m minillm.train --stage finetune \
    --init-from runs/exp-char-pretrain/model.pt --out-dir runs/exp-char-finetune
```

`--block-size 24` is required, not optional: a full game grows from up to
12 move-level tokens to up to 22 character-level tokens
(`MAX_GAME_CHARS` in `minillm/tokenizer.py`), and `train.py` refuses to
start if `block_size` can't hold the longest game. The finetuning command
needs no `--tokenizer` flag — it inherits `char` from the checkpoint it
initializes from, and `train.py` would reject a conflicting explicit value.

### The test

```bash
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-char-pretrain/model.pt
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-char-finetune/model.pt
```

### What you should see

The full measured comparison already lives in
[09 — Lab report: the character-level tokenizer](09-char-tokenizer-lab.md)
— read it after your own run, it is written as a spoiler. The headline
number: after finetuning, the character-level model draws against the
perfect solver **95% of the time**, versus **61%** for the move-level
model — a large jump in robustness against the strongest possible opponent,
at the cost of a lower win rate against a random one (68.8% vs 79.2%).
Legality does *not* get worse with the harder tokenizer, which is the
counter-intuitive result the lab report spends most of its length
explaining: dropping into a column is an easy 3-way choice, and the row a
piece lands on is then almost forced by gravity, so splitting a move into
two tokens gives the model an easier sub-problem per token rather than a
harder one.

## Experiment 3 — change the architecture: attention heads

[Anatomy of a tiny GPT](anatomy.md#your-turn-the-five-knobs-in-configpy)
lists five knobs in `ModelConfig`, each with a matching `train.py` flag.
`--n-head` is a good first one to try because it changes *nothing* about
parameter count — `c_attn`/`c_proj` stay the same size regardless of how
many heads the 128-wide stream is split into — so any effect you see is
purely about how many independent attention patterns run per layer, not
about model capacity.

### The change and the rebuild

```bash
.venv/bin/python -m minillm.train --stage pretrain --n-head 1 --out-dir runs/exp-onehead
```

The default is `--n-head 4` (head size 128/4 = 32); this collapses every
layer to a single attention pattern of size 128 instead of four running in
parallel.

### The test

```bash
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-onehead/model.pt
```

(`make eval` will not help here — it always evaluates the default
finetuned checkpoint at `runs/finetune/model.pt`; pointing `--ckpt`
explicitly at your experiment's checkpoint is how every command in this
guide reaches a non-default model.)

### What you should see

Comparing against the 4-head pretrain baseline from the README:

| metric | 4 heads (baseline) | 1 head |
|---|---:|---:|
| argmax-legal (held-out) | 100.0% | 100.0% |
| clean self-play games | 98.0% | 96.5% |
| result prediction | 99.2% | 99.2% |
| vs random W/D/L | 41.8 / 20.2 / 38.0% | 41.5 / 20.8 / 37.8% |
| optimal-move rate | 70.3% | 70.8% |

Most numbers barely move — at the *pretraining* stage the model is not yet
being pushed to play well, only to imitate the corpus faithfully, and one
128-wide attention pattern turns out to be nearly enough for a 15-token
world. The clearest signal is the clean-game rate: free-running self-play
drops from 98.0% to 96.5% clean games, a legality regression consistent
with what [anatomy.md](anatomy.md) predicts — fewer parallel relevance
patterns per layer means the model has to compress more distinctions into
each one. Try `--n-layer 1` next (drops to 202,496 parameters, a much
blunter cut) if you want a starker effect on the same metrics.

## Experiment 4 — change decoding only: no retraining required

Not every manipulation needs a rebuild. `minillm.arena` (and `play.py`,
`sample.py`) can sample from the *existing* finetuned checkpoint at any
temperature — 0 is strict legal argmax, the model's single favorite move
every time; above 0 it samples from the full softmax restricted to legal
moves, so the same weights can play differently from one game to the next.

### The change

Nothing to rebuild — reuse `runs/finetune/model.pt` exactly as `make
finetune` left it.

### The test

```bash
.venv/bin/python -m minillm.arena --model runs/finetune/model.pt --vs solver --games 200 --temperature 1.5
```

### What you should see

Strict argmax (temperature 0, the README's baseline) draws 61.0% of the
time against the solver and loses 39.0%. Sampling at temperature 1.5:

```
runs/finetune/model.pt  vs  solver (perfect negamax)   |   200 games   |   model plays: sampled @ T=1.5
  win     0 / 200   (  0.0%)
  draw   90 / 200   ( 45.0%)
  loss  110 / 200   ( 55.0%)
```

Draws fall from 61.0% to 45.0% and losses climb from 39.0% to 55.0% — the
exact same weights, playing visibly worse, purely because decoding stopped
always picking its most confident move. This is the cheapest experiment in
this guide (seconds, no training) and the one that best isolates what
"decoding strategy" even means: everything the model *knows* is unchanged,
only how it turns that knowledge into an actual move. For the free-running
generation side of the same knob, `python -m minillm.sample --temperature 1.5`
shows `verify_transcript` flagging more illegal or malformed games as the
distribution gets hotter.

## Where to go next

These four experiments only move one knob at a time. [08 —
Exercises](08-exercises.md) is the same loop scaled up — ablate the
positional embedding entirely, build a lookup-table baseline to check
whether the model is generalizing or memorizing, catalogue all 16 attention
heads, even scale the whole game to a 4x4 board — each exercise states the
files to touch and a difficulty tag, and several link to a matching "In a
real LLM" note connecting the miniature result to production-scale
practice.

For the *why* behind what you just perturbed: [the models in
detail](the-models.md) is the character study of Experiment 1's base,
expert, and gambler checkpoints — the same architecture trained three
different ways, with the full spectrum of numbers behind each one. [Anatomy
of a tiny GPT](anatomy.md) is the fast-reference version of the
architecture itself — every part of `minillm/model.py`, where it lives, and
how to change it, organized as the four questions this page's Experiment 3
borrowed from directly.
