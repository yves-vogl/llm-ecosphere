# Lab report: the temperature sweep

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 4](08-exercises.md#4-temperature-sweep-vs-playing-strength--30-minutes).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

Exercise 4 asks a pointed question: `evaluate.py` measures the model with
`model_move_strict`, a pure argmax — temperature 0 — while `play.py` defaults
to `--temperature 0.7`. Someone is leaving strength on the table, or picking
up variety they didn't pay for. We added a `--temperature` flag to
`evaluate.py`, swept it, and measured. The headline, in one sentence: **the
move-level and char-level checkpoints respond to temperature in mirror
image, because each one's greedy policy already sits at an opposite extreme
of the sharp-vs-solid frontier from
[09](09-char-tokenizer-lab.md) — and the same knob nudges both of them
toward the middle.**

## The change

`model_move_strict` in `minillm/evaluate.py` now takes a `temperature`
argument. At `temperature == 0` it is byte-for-byte the original
behavior — rank the legal moves by joint log-probability and take the
argmax. At `temperature > 0` it instead softmaxes those same legal-move
scores at the given temperature and samples, seeded off `--seed` (default
0) so every run in this report is reproducible:

```python
if temperature and temperature > 0.0:
    probs = torch.softmax(scores / temperature, dim=0)
    return legal[int(torch.multinomial(probs, 1, generator=generator))]
return legal[int(scores.argmax())]
```

This is the strict analogue of the sampling `play.py` already does, borrowed
into the evaluation harness rather than the play loop. One thing does *not*
move with temperature: `optimal-move rate`, the metric that checks agreement
with the solver over the 414-position set, is computed from teacher-forced
argmax regardless of `--temperature` — only the model's own move in
self-play games (vs random, vs solver) is affected. Temperature is a
strength knob here, not a legality or agreement knob.

Swept `T` in `{0, 0.3, 0.7, 1.0, 1.5}` on both finetuned checkpoints, eval
seed 0, 400 games vs random and 200 vs the optimal solver per point — 20
evaluation runs in total.

## The T=0 anchor

The `T=0` row on each checkpoint is not new data — it's the same regression
anchor as [09](09-char-tokenizer-lab.md)'s finetune numbers, reproduced here
as the first point of the sweep:

| checkpoint | vs random W/D/L (T=0) | vs optimal W/D/L (T=0) |
|---|---:|---:|
| move-level (`runs/finetune`) | 79.2 / 14.5\* / 6.2\*% | 0 / 61.0 / 39.0% |
| char-level (`runs/exp-char-finetune`) | 68.8 / 22.2\* / 9.0\*% | 0 / 95.0 / 5.0% |

\* draw/loss vs random are carried from 09's table for completeness; the
sweep below only tracks win-vs-random, since that is the metric exercise 4
asks for and the one that moves.

This is the frontier from 09 in two numbers: the move-level model wins 79.2%
against random but draws only 61.0% against the solver; the char-level model
wins less against random (68.8%) but draws 95.0% against the solver — the
theoretical ceiling of this game. Everything below asks what happens to
those two numbers as greedy stops being mandatory.

> One aside on the 61.0%: the companion multi-seed report
> ([09](09-char-tokenizer-lab.md)) notes the move-level draw-vs-solver rate
> is a single-seed value with real run-to-run variance. This report is about
> the temperature axis on the checkpoints as shipped, not a re-estimate of
> that variance — the 61.0% here is the same fixed anchor 09 reports, used
> as a baseline for the sweep below, not a new measurement of its spread.

## The sweep

Win vs random / draw vs optimal / loss vs optimal, percent:

**move-level (`runs/finetune`)**

| T | win vs random | draw vs optimal | loss vs optimal |
|---:|---:|---:|---:|
| 0.0 | **79.2** | 61.0 | 39.0 |
| 0.3 | 70.2 | **76.0** | **24.0** |
| 0.7 | 64.5 | 63.0 | 37.0 |
| 1.0 | 63.0 | 54.5 | 45.5 |
| 1.5 | 56.0 | 45.0 | 55.0 |

**char-level (`runs/exp-char-finetune`)**

| T | win vs random | draw vs optimal | loss vs optimal |
|---:|---:|---:|---:|
| 0.0 | 68.8 | **95.0** | **5.0** |
| 0.3 | **74.2** | 80.5 | 19.5 |
| 0.7 | 69.5 | 76.5 | 23.5 |
| 1.0 | 69.0 | 70.5 | 29.5 |
| 1.5 | 59.5 | 59.0 | 41.0 |

## Reading the numbers

**1. The move model's shark bite dulls monotonically, and its wall improves
non-monotonically.** Win-vs-random falls in a straight line as temperature
rises — 79.2 to 70.2 to 64.5 to 63.0 to 56.0 — confirming that greedy is
this model's sharpest attack; every degree of randomness against a weak,
unstructured opponent only costs it wins. But draw-vs-solver does *not*
fall with it. It rises first, from 61.0% at T=0 to **76.0% at T=0.3** (losses
falling from 39.0% to 24.0%), before decaying back down through 63.0, 54.5,
and 45.0 at higher temperatures. A little stochasticity steps the model off
whichever deterministic lines its argmax walks straight into against a
perfect opponent; more than a little just adds noise on top of a weakening
policy.

**2. The char model is the mirror image.** Draw-vs-solver falls
monotonically and hard — 95.0 to 80.5 to 76.5 to 70.5 to 59.0 — confirming
that greedy is this model's best defense; sampling only ever gives the
solver more chances to punish a move the model would not have picked on
its own. But win-vs-random peaks off T=0: 68.8% at T=0, up to **74.2% at
T=0.3**, before falling back through 69.5, 69.0, and down to 59.5 at T=1.5.
A little heat makes the wall a slightly better attacker, for the same
reason it costs the shark against the solver — some of the char model's
greedy lines against random are themselves not the best available response,
and mild sampling escapes them.

**3. Same knob, opposite effect, because they start in opposite corners.**
Both models sit at an extreme of the sharp-vs-solid frontier at T=0: the
move-level model is the sharper shark (79.2% vs random, 61.0% draws), the
char-level model is the more solid wall (68.8% vs random, 95.0% draws).
Temperature nudges both toward the middle of that frontier rather than
pushing either further into its own corner — the shark trades some bite for
a bit of wall (win falls, draws rise, at least initially), and the wall
trades a little of its defense for a bit of bite (draws fall, win rises,
at least initially). Neither model becomes the other; both become slightly
less extreme versions of themselves, then simply get worse at everything
once temperature climbs past T=0.3–0.7 and sampling stops being a nudge and
starts being noise.

**4. Honesty about a single eval seed.** Every number above is one seeded
run (`--seed 0`, 400 games vs random, 200 vs the solver). The two monotone
trends — move-level win-vs-random falling, char-level draw-vs-solver falling
— are robust: they are long, one-directional runs across five temperature
points, not a single delta. The two T=0.3 peaks are not equally solid. At
p≈0.7 and n=200, the binomial standard error on the solver-draw rate is
about 3.2 percentage points; the move model's 61.0% → 76.0% jump is roughly
15 points, about 4–5 standard errors — likely real. The char model's
68.8% → 74.2% win-vs-random jump is smaller (about 5.4 points against an SE
around 2.3 points at n=400) — real direction, softer magnitude; call it
suggestive rather than certain without a multi-seed rerun.

> **In a real LLM:** temperature is the same strength-vs-diversity knob
> every chat product exposes to users. Greedy decoding is strongest on
> tasks with a single right answer — code completion, arithmetic, this
> game's optimal-move rate — but it collapses diversity, and pushed far
> enough it degenerates into repetition or the same rehearsed phrasing
> every time. Production stacks rarely ship raw temperature alone; they
> layer top-k or top-p truncation on top to keep sampling from wandering
> into the genuinely bad tail of the distribution (top-k already exists in
> this repo's `GPT.generate`). It's exactly why `play.py` defaults to
> `--temperature 0.7` for a more natural opponent while `evaluate.py`
> measured strength at `0` — the two files were never disagreeing, they
> were answering different questions.

## Reproduce it

```bash
for T in 0 0.3 0.7 1.0 1.5; do
  .venv/bin/python -m minillm.evaluate --ckpt runs/finetune/model.pt --temperature $T
  .venv/bin/python -m minillm.evaluate --ckpt runs/exp-char-finetune/model.pt --temperature $T
done
```

Each invocation reuses the default 400 games vs random / 200 vs the solver
and seed 0; pass `--out runs/exp-.../eval-tN.json` per run to keep the ten
result files apart, and add `--seed` variants if you want to test the T=0.3
peaks against the honesty caveat above.

Back to [exercise 4](08-exercises.md#4-temperature-sweep-vs-playing-strength--30-minutes)
in the exercises, or [09 — the character-tokenizer lab](09-char-tokenizer-lab.md)
for the sharp-vs-solid frontier this sweep runs across.
