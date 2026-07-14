# RL gambler — lab report: does policy gradient beat imitation?

> Companion to issue #35 and the SFT gambler
> (`train.py --objective gambler`, docs/05). Read that one first: it
> shows that *imitating winners* produces an aggressive-but-exploitable
> player that still loses only 63% of the time to random — worse than
> the expert's 79.3%. This report asks the obvious follow-up: what if,
> instead of imitating whoever happened to win a pre-generated corpus,
> the model directly *optimizes* for winning?

## The hypothesis

Filtered SFT is win-maximising only by proxy — it copies moves from
games a stronger-than-random opponent happened to win, which is not
the same as moves that *maximise the probability of winning against
random*. A policy-gradient agent trained with the actual win/loss/draw
signal as its objective should close that gap and become the
**dominant-yet-exploitable** gambler: the strongest player against a
weak opponent, and — because it never faces anything but random during
training — still hopeless against the perfect solver.

## The method: REINFORCE with a baseline

`minillm/rl.py` starts from the pretrained checkpoint (legal, aimless)
and turns it into a win-seeking policy with no dataset at all:

1. **Self-play vs. random.** Each iteration plays a batch of complete
   games. The learner samples moves from its own policy, restricted to
   legal moves (`legal_move_logprobs`, the same chain-rule move-assembly
   math as `utils.py`, reimplemented without `@torch.no_grad()` so the
   sampled action's log-probability stays attached to the graph). The
   opponent samples uniformly at random. The learner alternates sides
   every game so it has to learn to both open (X) and respond (O).
2. **Reward.** One scalar per finished game, from the learner's own
   perspective: **+1 win, −1 loss, 0 draw** — the same number whichever
   side it played. Every learner move in that game receives that same
   return (no discounting: nine plies is too short to need it, and every
   move in a 3-in-a-row game genuinely shares the credit or the blame).
3. **Baseline.** A running mean of every return seen in *previous*
   iterations (not the current batch — so it's a constant with respect
   to the gradient being computed, which keeps the estimator unbiased
   while cutting its variance).
4. **Objective.** Gradient ascent on expected return, phrased as descent
   on
   ```
   loss = -(1/N) * sum over learner moves of (return - baseline) * logpi(move | state)
   ```
   Only states where the learner was to move contribute a term — the
   random opponent's moves are pushed onto the board but never scored,
   so they cannot enter the sum. AdamW, lr 5e-4, gradient clipped at 1.0,
   seeded (1337 by default).

`logpi(move | state)` in that formula is `legal_move_logprobs`'s score
for the sampled action: the log-probability *after* renormalizing over
just the legal moves available at that state, `logprobs[idx] -
logsumexp(logprobs)` — not the raw `log_softmax` over the full 12/13-token
vocabulary. That distinction is not cosmetic. The first working version
of this scorer normalized over the *full* vocabulary, which silently
credits (or blames) the model for probability mass it never actually
controlled — the action was already sampled from the legal subset
before the log-prob was read off. An adversarial review caught the
mismatch before these results were trusted; the model below was
retrained from scratch on the corrected gradient, and — as the
alignment-tax section further down shows — the correction mattered.

Training: `python -m minillm.rl --iters 80 --games-per-iter 40` — 3,200
games, well under a minute of wall clock on CPU. Win rate against random
climbed steadily and noisily (REINFORCE has no value function, so
variance stays visible iteration to iteration) from ~50% at iteration 0
to a stable 80–90% band by iteration 60, printed live as
`iter N | loss L | mean return R | win/draw/loss vs random`.

## Results

All four checkpoints evaluated identically with `minillm.evaluate`
(400 games vs. random, 200 vs. the perfect solver, both seeded;
optimal-move rate over the same 414-position solver-agreement set used
in docs/09):

| checkpoint | vs random W/D/L | vs solver W/D/L | optimal-move rate | free-run clean games |
|---|---|---|---|---|
| base (pretrain only) | 41.8 / 20.3 / 38.0 | 0 / 0 / 100 | 70.3% | 98.0% |
| expert (SFT, optimal-move imitation) | 79.3 / 14.5 / 6.3 | 0 / 61 / 39 | 86.5% | 90.5% |
| SFT-gambler (SFT, winner imitation) | 63.0 / 15.5 / 21.5 | 0 / 30 / 70 | 78.3% | 97.5% |
| **RL-gambler (REINFORCE vs. random)** | **87.75 / 5.75 / 6.5** | **0 / 45 / 55** | **85.7%** | **0.0%** |

(`runs/exp-rl-gambler/eval.json` has the full breakdown, including
teacher-forced and free-running legality.)

## The alignment tax, measured

The most important number in this report is not a win rate. It's the
last column above: **0.0%**. Left to generate its own tokens with no
legality restriction, the RL-gambler produced zero fully legal games
across the free-running evaluation. Not "regressed" — collapsed.

| legality (RL-gambler) | teacher-forced | free-running |
|---|---|---|
| argmax / first-try legal | 58.3% | 23.0% |
| clean full game | — | **0.0%** |
| mean legal-probability mass | 64.5% | — |
| result prediction | 92.4% | — |

(`runs/exp-rl-gambler/eval.json`, `legality` and `legality_free_running`.)

Teacher-forced argmax-legal — is the model's single favourite next
token legal, given a real game prefix? — fell to 58.3%, against ≥90%
for every other checkpoint in the table above. Free-running — sample a
token, then the next, and see whether nine unconstrained samples
assemble into a legal finished game — is 0.0%.

And yet **the win-rate numbers above are completely real and completely
trustworthy.** `evaluate.py` and the arena never sample from the
model's raw token distribution; they always rank the legal moves by
`legal_move_logprobs` and choose among *those*. The RL-gambler is the
strongest player in the lab because REINFORCE's score,

```
score = logprobs[idx] - logsumexp(logprobs)   # over legal-move logits only
```

only ever looks at the *ranking within the legal set*. Nothing in the
loss has a term that says "and also, don't raise P(illegal token)" —
the raw, unconstrained token distribution is simply never consulted by
the objective. Eighty iterations of pure win-rate reward taught the
model an excellent legal-move ranking and, at the same time, let its
unconstrained distribution drift wherever gradient noise pushed it,
because nothing was holding it in place.

> **This is the alignment tax, in miniature and exactly measured.**
> Optimizing a reward that never mentions a capability is not neutral
> toward that capability. Here, optimizing win-rate under a
> legality-restricted action space silently destroyed the model's
> *unrestricted* legality — a property the reward function never once
> observed or scored. It's a small, cleanly-instrumented instance of
> what "catastrophic forgetting under RL fine-tuning" means at scale: a
> policy chased hard against a narrow reward drifts off the
> distribution the reward was never watching. The standard fix is the
> one this experiment is missing: a legality-preserving term, either a
> KL penalty pulling the policy back toward the pretrained/SFT
> checkpoint, or literally mixing SFT/legality batches into the RL loop
> so the objective can't drift away from "stay legal" while it chases
> "win more." That's the same mechanism real RLHF uses to keep a
> reward-optimized policy on-distribution. Neither is implemented here
> — measuring the gap honestly is the point of this report, not closing
> it.

One more thing worth being honest about: the *buggy* run reported a
softer version of this same collapse — 82.4% teacher-forced legal and
1.5% free-running clean games, not 58.3%/0.0%. The bug normalized the
score over the *entire* vocabulary, so the softmax competition the
buggy gradient was climbing already included the illegal tokens —
pushing up a legal move's full-vocabulary log-prob incidentally
suppressed illegal-token logits too, as a side effect of shared
normalization. Fixing the bug removed that accidental legality shaping
along with the incorrect credit assignment. The corrected gradient is a
purer read of what pure win-rate reward actually does to legality, and
it is worse than the leaky, buggy version suggested — the bug was,
unintentionally, a broken legality regularizer.

## Verdict: partially confirmed, and more interesting than predicted

**RL did produce the best-vs-random gambler** — 87.75% wins, ahead of
even the expert (79.3%) and nearly 25 points ahead of the SFT gambler it
was meant to upgrade (63.0%). That part of the hypothesis holds exactly:
optimizing the win signal directly beats imitating winners.

**It did *not* become the "worst vs. solver" gambler**, which is where
the clean dominant-yet-exploitable story breaks. Against the perfect
solver the RL-gambler loses 55% of the time and draws 45% — worse than
the expert (39% losses) as expected, but *better* than the SFT gambler
it was supposed to be the more-exploitable sibling of (70% losses). In
other words: RL-gambler **strictly dominates** the SFT gambler — better
against random *and* better against the solver — rather than trading
one for the other. The predicted trade-off (best vs. weak, worst vs.
strong) never had to materialize, because SFT-imitation-of-winners
turned out to be a strictly worse policy along both axes, not a
different point on a Pareto frontier.

Reading between the numbers: filtered SFT on winner moves is a noisy
label — it imitates a move because the game was *eventually* won, not
because that particular move helped win it, and it never sees its own
mistakes corrected. REINFORCE's credit assignment is just as coarse
(every move in a game gets the same return), but the *training
distribution* is the one that matters at eval time — self-play against
the actual opponent it's being scored against — so its "coarse" signal
still out-performs SFT's "coarse but off-distribution" one.

**The one real cost has its own section above, not a footnote here.**
See "The alignment tax, measured": free-running legality collapsed to a
0.0% clean-game rate while every win-rate number in this report stayed
completely trustworthy, because match play never leaves the legal-move
ranking. It is the central finding of this report, not a caveat — a
small, exactly-measured instance of reward optimization degrading a
capability the reward never mentioned.

## Honest summary

- **The headline finding**: pure win-rate optimization collapsed
  free-running legality to a 0.0% clean-game rate (teacher-forced
  argmax-legal fell to 58.3%) — because the REINFORCE score only ever
  ranks legal moves against each other and never touches the raw,
  unconstrained token distribution. A small, exactly-measured instance
  of the alignment tax: optimizing a reward silently degrades a
  capability the reward never mentioned. The fix would be a
  legality-preserving term — a KL penalty to the pretrained policy, or
  mixed-in SFT/legality batches — the same idea real RLHF uses to stay
  on-distribution.
- Hypothesis confirmed: constrained to legal moves — the way
  `evaluate.py` and the arena actually use it — the RL-gambler is the
  strongest-vs-random checkpoint in the lab (87.75%), beating both SFT
  objectives and the expert.
- Hypothesis not confirmed as stated: it isn't the "worst vs. solver"
  checkpoint — it beats the SFT gambler on that axis too (45% draws /
  55% losses vs. 30% / 70%). It dominates the SFT-gambler on *both*
  axes. The win-maximiser vs. draw-seeker trade-off the hypothesis
  predicted is real, but it sits between *expert* and *RL-gambler*, not
  between *SFT-gambler* and *RL-gambler*.
- A note on trust: an adversarial review caught the policy-gradient
  normalization bug (log-prob scored over the full vocabulary instead
  of the legal-move set the action was sampled from) before these
  numbers were trusted — measure, then verify, before you believe a
  result.
