# 07 — Evaluation: what did it learn?

Training ended with a small validation loss. That number tells you the model
assigns high probability to held-out token sequences — and nothing else. It
does not tell you whether the model would ever play an illegal move, whether
it can recognize a finished game, or whether it plays *well*. Loss measures
fit to a distribution; we care about behaviour in the world that distribution
describes. The docstring at the top of `minillm/evaluate.py` states the split
directly:

```
Loss numbers say "the model fits the data"; these metrics say what that
means in the world the data describes
```

Two concrete failure modes make the point. First, a model could reach a
decent loss by nailing the easy, forced late-game moves while being confused
in the openings that decide games — average loss hides where the errors are.
Second, our finetuning corpus contains only solver-optimal expert games, so a
low finetune loss means "predicts what an expert would do on expert
trajectories". Whether that transfers to positions an expert would never
reach (say, after the model itself blundered) is an off-distribution question
loss cannot answer. Only playing the game answers it.

> **In a real LLM:** this is exactly why production labs maintain large eval
> harnesses (MMLU, GSM8K, HumanEval, SWE-bench, plus swarms of internal
> behavioural suites) separate from the loss curve. Cross-entropy on held-out
> web text is a *capability proxy*; whether the model refuses harmful
> requests, follows instructions, or writes code that compiles are
> behavioural properties that must be measured by making the model act and
> scoring the action. Our advantage here: Drop-Tac-Toe has a perfect solver,
> so "ground truth behaviour" is available for every position — a luxury no
> natural-language eval has.

`minillm/evaluate.py` measures three families of behaviour — legality,
refereeing, strength — with five metric groups. Run it with
`python -m minillm.evaluate --out runs/eval.json` (or `make eval`); pass
`--ckpt runs/pretrain/model.pt` to score the pretrained checkpoint instead of
the default finetuned one.

## The strict-argmax move rule

Every strength metric uses the same move-selection rule, so it is worth
pinning down first. `model_move_strict` takes the model's raw next-token
logits and picks the argmax **restricted to legal moves**:

```python
def model_move_strict(model, tokenizer, game: Game, device) -> str:
    """The model's favourite LEGAL move (argmax over legal tokens only)."""
    logits = next_token_logits(model, tokenizer, game.history, device)
    legal_ids = tokenizer.encode(game.legal_moves())
    best_id = max(legal_ids, key=lambda i: logits[i].item())
    return tokenizer.id_to_token[best_id]
```

Deterministic (no sampling temperature), and it never crashes the game even
if the model's global argmax were illegal. This deliberately separates two
questions: "does the model *know* the rules?" (measured by the legality
metrics, which look at the unconstrained distribution) and "how strong is its
*policy*?" (measured with the legality guardrail on, so a rare rule slip does
not contaminate the strength numbers).

## Metric 1: teacher-forced legality on held-out games

`eval_on_val_games` replays the validation games — the same 10% split that
was held out during training (`split_games(..., val_frac=ckpt.get("val_frac", 0.1))`,
reusing the fraction recorded in the checkpoint so "held-out" stays honest
even after a non-default run). At every position *before* each recorded move,
it asks the model for its next-token distribution, scores it, then pushes the
**recorded** move, not the model's:

```python
if tokenizer.id_to_token[int(logits.argmax())] in legal:
    argmax_legal += 1
...
game.push(recorded_move)  # follow the recorded game, not the model
```

This is teacher forcing: the model is always evaluated on positions from real
games, never on positions of its own making. Across the 131 validation games
that yields **1062 positions** and two numbers:

- `argmax_legal_rate` — fraction of positions where the single most likely
  token is a legal move.
- `mean_legal_prob_mass` — the softmax probability the model puts on the set
  of legal moves, averaged over positions
  (`probs[legal_ids].sum()`). This is stricter than the argmax rate: a model
  can have a legal argmax while still leaking 10% of its probability onto
  impossible moves, occupied cells, or floating cells like C3 with C2 empty.

## Metric 2: free-running legality

Teacher forcing only visits positions from the data. `eval_rollout_legality`
lets the model play **both sides against itself** for 200 games, sampling
from the full softmax (multinomial, seeded generator — no legality mask on
the first draw):

```python
if token not in game.legal_moves():
    illegal += 1
    clean = False
    # project onto the legal moves so the rollout can continue
```

When an illegal token is sampled it is counted, then the distribution is
re-normalized over legal moves only and re-sampled, so the game continues and
one early mistake cannot inflate the count by aborting rollouts. Two numbers
come out: `first_try_legal_rate` (per-move: sampled token was legal without
help) and `clean_game_rate` (per-game: an entire game with zero projections).
The second is the harsher one — at 9 moves a 99% per-move rate compounds to
roughly a 91% clean-game rate.

This matters because self-play drifts off the training distribution: the
model faces positions produced by its own sampled (sometimes suboptimal)
moves. It is the closest analogue to "exposure bias" this lab has — models
trained only on ground-truth prefixes can compound errors when fed their own
output.

## Metric 3: refereeing (result-token prediction)

The same `eval_on_val_games` loop ends each replayed game with one more
query: after the final move, is the model's argmax token the correct result
marker (`#X`, `#O`, or `#=`)?

```python
result_correct += tokenizer.id_to_token[int(logits.argmax())] == g["result"]
```

This tests something distinct from playing: to emit `#X` the model must have
tracked nine moves of implicit board state well enough to *recognize* that a
3-in-a-row exists and whose it is. It is a probe of the model's internal
world model, evaluated on 131 game endings.

## Metric 4: match play — W/D/L vs random and vs the solver

`eval_matches` plays full games with `model_move_strict` against an opponent
policy, alternating which side the model takes:

```python
play_one(model, tokenizer, device, "X" if i % 2 == 0 else "O", opponent)
```

Alternation matters: X has a large first-move advantage in the raw statistics
(616 of the 1310 enumerated games are X wins per `data/meta.json`), so
scoring only as X would flatter the model. Two opponents:

- `random_opponent` — uniform over legal moves. 400 games.
- `optimal_opponent` — queries the exact solver (`best_moves` from
  `minillm/solver.py`) and picks uniformly among the optimal moves, so the
  model sees varied perfect play rather than one canonical line. 200 games.

Because the solver proved the root value of Drop-Tac-Toe is a **draw**, a
perfect player can never lose — and therefore our model can never *win*
against `optimal_opponent`. Its draw rate against the solver is the honest
"distance from perfect play" number, with 100% draws as the theoretical
ceiling.

## Metric 5: solver agreement (optimal-move rate)

`eval_expert_agreement` collects positions by running 300 random-vs-random
rollouts, deduplicates them by board state, then asks: is the model's
strict-argmax move among the solver's optimal moves for that position? Random
rollouts reach 414 distinct positions — including messy, unbalanced ones an
expert game would never contain, which is the point: it probes generalization
beyond the finetuning distribution.

One subtlety, flagged in the code's own comment:

```python
Note: the model conditions on the move *history*, the solver on the
resulting *position* — several histories can share a position. We
keep the first history seen per position.
```

The model never sees a board; it sees a token sequence. Two different move
orders can produce the identical board, and the model's answer may differ
between them (its attention pattern is over history, not state). The metric
arbitrarily keeps the first history encountered per position, so it measures
"agreement for *some* history reaching this position", a mild simplification
worth knowing about.

## Pretrain vs finetune: the full table

Numbers from `runs/eval_pretrain.json` (checkpoint `runs/pretrain/model.pt`)
and `runs/eval.json` (checkpoint `runs/finetune/model.pt`), identical seeds
and game counts:

| Metric | Pretrained | Finetuned |
|---|---|---|
| Argmax-legal rate (1062 held-out positions) | 100.0% | 99.53% |
| Mean legal probability mass | 99.64% | 99.06% |
| Free-running first-try legality (200 games) | 99.75% | 98.85% |
| Clean-game rate | 98.0% | 90.5% |
| Result-token prediction (refereeing) | 99.24% | 100.0% |
| vs random, W/D/L (400 games) | 41.8% / 20.2% / 38.0% | 79.2% / 14.5% / 6.2% |
| vs optimal, W/D/L (200 games) | 0% / 0% / 100% | 0% / 61.0% / 39.0% |
| Solver-agreement optimal-move rate (414 positions) | 70.3% | 86.5% |

### Reading the pretrained column: grammar without judgement

Pretraining on 1179 of the 1310 enumerated games (10% held out for
validation) taught the model the *rules*
essentially perfectly: a 100% argmax-legal rate, 99.6% of probability mass on
legal moves, 98% of self-play games completely clean, and 99.2% correct
result calls. It learned the "grammar" of Drop-Tac-Toe — which sequences are
well-formed — because that is what next-token prediction on the full game
corpus rewards.

But its play is exactly as strong as its corpus: average. 41.8% wins against
a random opponent versus 38.0% losses is barely better than a coin flip,
and it loses **all 200 games** against the solver. The 70.3% optimal-move
rate is not evidence of skill — in a 3x3 game many positions have several
optimal moves among few legal ones, so even a mediocre policy agrees with
the solver often. The pretrained model imitates the *average* move in its
corpus, and the corpus contains every game, including terrible ones.

> **In a real LLM:** this is precisely the base-model phenomenon. GPT-3
> after pretraining could produce fluent, grammatical text — and would
> happily continue a bad answer with more bad answer, because it modeled
> the *distribution* of internet text, not the *best* of it. Fluency
> (legality) comes from pretraining; the preference for good outputs over
> typical ones has to come from a second stage.

### Reading the finetuned column: judgement, at a small price

Finetuning on 301 of the 334 solver-optimal expert games (33 held out as
validation) — with the opponent's
moves masked out of the loss, the SFT analogy from chapter 05 — moves every
strength number sharply:

- **vs random: 79.2% wins, 6.2% losses** (up from 41.8% / 38.0%).
- **vs optimal: 61% draws.** Since a draw is the best achievable result
  against perfect play, 61% is "the model plays a full game perfectly, from
  either side, 61% of the time". The remaining 39% losses are the residual
  imperfection: at least one move per lost game deviated from the solver,
  and the solver never forgives.
- **Optimal-move rate: 86.5%** (up from 70.3%), measured largely on scrappy
  random-rollout positions outside the expert distribution — so the policy
  improvement generalizes beyond the exact games it was finetuned on.
- **Refereeing: 100%**, up from 99.24%.

### The legality dip: a miniature alignment tax

Every legality number got slightly *worse* after finetuning: argmax-legal
100% → 99.53%, legal mass 99.64% → 99.06%, first-try legality 99.75% →
98.85%, and clean games 98.0% → 90.5%. The mechanism is plain: finetuning
continued optimizing on a corpus 4x smaller (334 vs 1310 games) and heavily
skewed (many positions never appear in optimal play), so the weights drifted
away from the broad rule-knowledge the full corpus had instilled. The model
traded a sliver of grammatical certainty for a large gain in judgement.
Note the compounding: a per-move dip from 99.75% to 98.85% looks tiny, but
over whole games it cuts the clean-game rate from 98.0% to 90.5%.

> **In a real LLM:** this trade-off is called the *alignment tax*.
> Instruction tuning and RLHF reliably improve helpfulness and preference-win
> rates while slightly degrading some raw pretraining capabilities or
> calibration — InstructGPT's paper measured exactly this and mixed
> pretraining gradients back in to reduce it. The mitigation ideas transfer
> directly to this lab: mix some all-games data into finetuning, lower the
> finetune learning rate, or stop earlier. Trying these is a natural extra
> exercise.

## What these evals do not cover

Honesty section. This harness is good, and it still has blind spots:

- **Coverage is sampled, not exhaustive.** 414 agreement positions out of the 505
  reachable positions with a move to make (694 total positions solved per
  `positions_solved` in `data/meta.json`, 189 of them terminal), 400 games
  vs random and 200 vs the solver, one seed (`--seed 0`). The numbers carry sampling noise — a 61.0%
  draw rate over 200 games has a standard error of roughly ±3.5 points. The
  game is small enough that a fully exhaustive eval is feasible; this one
  just does not do it.
- **History-vs-position aliasing** (metric 5): one history per position is
  scored. A model that answers well for one move order and badly for another
  reaching the same board is not fully characterized.
- **Strength is only measured under strict argmax.** How robust the policy is
  under temperature sampling — the mode `play.py` and `sample.py` can use —
  is not evaluated here.
- **No calibration metric.** We check *where* the probability mass sits
  (legal vs illegal) but never whether 80% confidence means being right 80%
  of the time.
- **No probing of internals.** These are black-box behavioural tests. Whether
  the model has a linear board-state representation in its residual stream
  (the kind of question `inspect_attention.py` starts poking at) is out of
  scope.
- **Evals ran on the data the pipeline generated.** Since the game is solved
  and enumerable this is fine here — but the habit of grading a model with
  machinery from its own training pipeline is exactly how real-world eval
  contamination happens, and it deserves a raised eyebrow on principle.

The gap the table leaves open is the interesting one: 86.5% of individual
moves are optimal, yet only 61% of full games against the solver are — errors
compound, and a single sub-optimal move against a perfect opponent is fatal.
Closing that gap is a training problem, not an evaluation problem.

Next: [08 — Exercises: make it yours](08-exercises.md), where you shrink the
model, rebuild the tokenizer, and bolt an RL stage onto the pipeline
yourself.
