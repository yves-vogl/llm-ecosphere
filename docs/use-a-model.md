# Use a trained model — the fastest way in

You don't need to understand a Transformer to use one. This page is the
short path: get a model, play it, try to beat it, and see what it was
"thinking" when it moved — no prior setup, no prior chapter required. If
you like what you see, the rest of the documentation explains *why* it
works; this page only gets you *playing*.

## Get a model — two ways

### Download it (coming soon)

The intended one-click path is a pretrained model attached to a [GitHub
Release](https://github.com/yves-vogl/llm-ecosphere/releases) — download
`model.pt`, point `--ckpt` at it, and skip training entirely.

**This isn't wired up yet.** The automated model zoo (a CI workflow that
trains the base, expert and gambler checkpoints and attaches them to each
release) is still being built. Check the
[Releases page](https://github.com/yves-vogl/llm-ecosphere/releases) — once
it ships, this section will point you straight at the file to grab. Until
then, use the path below; it takes a few minutes and runs on a plain
laptop CPU.

### Train it yourself (works today)

```bash
make setup      # create .venv and install torch + pytest
make data       # enumerate every game        (seconds)
make pretrain   # learn the rules             (~2 min CPU)
make finetune   # learn to play well          (~1 min CPU)
```

That's it — `runs/finetune/model.pt` now exists, and every command below
will find it automatically. `make setup` is the only step that needs a
network connection (it downloads `torch`); everything after that runs
offline. Training is seeded, so your numbers will land close to the ones
quoted on this page.

Want the beatable one too (see the flagship demo below)?

```bash
make model-gambler
```

This writes a second checkpoint to `runs/exp-gambler-move/model.pt` in
about a minute. You now have both models this page plays with.

## Play the strong one

```bash
make play
```

You play `X`, the finetuned model plays `O`, and it moves second the way
a game of Tic-Tac-Toe normally starts. In-game:

```
A / B / C     drop a piece into that column
A1 .. C3      same, naming the exact landing cell
p             show the model's next-token probabilities
u             undo your last move (and the model's reply)
?             this help
q             quit
```

Columns are `A B C` left to right, rows `1 2 3` bottom to top, and pieces
fall the way they would in Connect Four — you can't place `C3` until `C1`
and `C2` are already occupied. Type a bare column letter (`B`) and the
game figures out the landing row for you.

## The flagship demo: beat the gambler, fail to beat the expert

This is the single most convincing five minutes in the repo, and it needs
both checkpoints from above.

**Beat the gambler.** It was finetuned to imitate whoever *won* each
decisive game, regardless of whether their moves were objectively sound —
so it plays aggressively and can be punished for it. Look for a chance to
set up two threats at once; it usually walks into one.

```bash
.venv/bin/python -m minillm.arena --model runs/exp-gambler-move/model.pt --vs human
```

**Now fail to beat the expert.** Same interface, the other checkpoint:

```bash
.venv/bin/python -m minillm.arena --model runs/finetune/model.pt --vs human
```

(`make play` plays the same checkpoint, since `runs/finetune/model.pt` is
the default `play.py` and `arena.py` both fall back to when you don't pass
`--model`/`--ckpt`.)

**Why the difference?** Drop-Tac-Toe is small enough to solve exactly, and
the solver proves that with perfect play on both sides the game always
ends in a draw — nobody can force a win against a truly optimal opponent.
The two checkpoints were finetuned toward opposite goals from that same
fact:

- The **expert** was finetuned on solver-optimal games with the losing
  side's moves masked out of the loss — it imitates the solver-approved
  side only. It agrees with the solver's own choice of move 86.5% of the
  time and, in a 200-game match against the actual solver, draws 61% of
  games (`runs/eval.json`). Against a human it plays well enough that a
  slip on your side gets punished the way its 79.2% win rate against a
  random player suggests — so treat a draw as a good result, and expect
  a loss if you make a mistake it can see. It is not literally
  mathematically invincible (the perfect solver itself still beats it in
  the other 39% of games by finding its rare non-optimal moves), but
  finding those requires solver-level precision most human opponents
  won't bring to the board.
- The **gambler** was finetuned the opposite way: on decisive games only
  (draws dropped), imitating whichever side *won* — sound or not. That
  buys it aggression at the cost of soundness. In the same benchmark it
  only draws the solver 30% of the time (losing the other 70%), and it is
  exploitable enough that it loses 21.5% of its games even against a
  *random* mover (`runs/exp-gambler-move/eval.json`) — the expert loses
  only 6.2% of those. A model that occasionally loses to random moves is
  not going to hold up against a human looking for the trap.

Same architecture, same size, same training pipeline — the only
difference is which games the finetuning stage told each one to imitate.
That's the whole story of "alignment": what you optimize for is what you
get.

## See what it's thinking — the probability display

Every move is really the model naming its favorite token out of a
15-token vocabulary. `play.py`'s `p` (and `arena.py`'s `why`) print that
distribution before the model moves, so it stops being a black box:

```bash
.venv/bin/python -m minillm.play --raw --show-probs
```

`--show-probs` prints the distribution before every model move
automatically; `--raw` additionally lets the model attempt illegal moves
so you can see when it gets it wrong (it then quietly re-samples among
the legal ones so the game continues). Sample output:

```
  model's next-token distribution:
     B2  ########################  61.2%  legal move
     A1  #######  18.7%  legal move
     C1  ###  7.4%  legal move
     #X  #  1.2%  result token
```

Watch it across a game and the pattern is the point: probability mass
sits almost entirely on legal moves throughout, and the instant a line of
three completes, it jumps onto the correct result token (`#X`, `#O`, or
`#=`) — even though nobody ever told it the rules of winning. That's a
next-token predictor that learned to recognize a finished game purely
from move sequences.

## Watch models fight — the arena harness

`arena.py` is the one command that reaches every opponent: a human, a
random mover, the perfect solver, or a second checkpoint. All four modes
drive the model with the exact same move-picking logic, so what you see
in `--vs human` is the same policy the benchmark numbers above measure —
not a lookalike.

```bash
# model vs model — gambler against expert
.venv/bin/python -m minillm.arena --model runs/exp-gambler-move/model.pt --vs runs/finetune/model.pt

# benchmark against the perfect solver
.venv/bin/python -m minillm.arena --model runs/finetune/model.pt --vs solver

# benchmark against a random mover
.venv/bin/python -m minillm.arena --model runs/finetune/model.pt --vs random
```

`--vs solver` and `--vs random` play 200 games by default (override with
`--games N`) and print a win/draw/loss summary instead of a board — this
is how the percentages quoted earlier in this page were produced.
`make arena` runs the first of these two as a one-word shortcut
(`runs/finetune/model.pt` vs the solver). Add `--temperature 0.7` to any
of these to make a model sample among legal moves instead of always
taking its single best one, which is what makes repeated matches between
the same two checkpoints vary.

## Where to go next

You've now done the two things that matter most: gotten a model running,
and watched it win, lose, and explain itself. From here:

- **[Hands-on: change the training](hands-on.md)** — change the training,
  not just the play. Swap the objective, the tokenizer, or an architecture
  knob, retrain in minutes, and measure the effect against what you just
  watched happen on the board. The [exercises](08-exercises.md) scale the
  same loop up.
- **[The models in detail](the-models.md)** — the character study behind
  base, expert and gambler: what each one trains on, the full numbers
  behind the win/draw/loss claims on this page, and how to build every
  variant yourself.
- **[Learning paths & workshop guide](learning-paths.md)** — three
  reading routes through the full documentation depending on how much
  code you want to touch, plus a script for running this repo as a live
  group session.
