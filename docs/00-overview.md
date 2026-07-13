# Overview: a complete LLM pipeline in miniature

This repository contains every stage of a modern language-model pipeline —
corpus construction, tokenization, a decoder-only Transformer, pretraining,
supervised finetuning, decoding, evaluation, interpretability — shrunk until
the whole thing runs on a laptop CPU in about twenty minutes and every single
claim about the model can be checked against ground truth.

The model is a real GPT: the same architecture family as GPT-2/3, Llama and
Claude, implemented from scratch in `minillm/model.py`, at 797,312 parameters
(`GPT.num_params()`; the docstrings round this to "~0.8M"). Its entire world
is **Drop-Tac-Toe**: Tic-Tac-Toe on a 3x3 board with Connect-Four gravity.
Columns are `A B C`, rows `1 2 3` bottom-up; a move drops a piece into a
column, it lands on the lowest free cell and is written *as that cell* — so
`C3` is only legal once `C1` and `C2` are occupied. `X` moves first, three in
a row wins. The model never sees a board. It sees text: sequences of move
tokens, the way a real LLM sees language.

## Why a toy world

Three properties make this game an unusually honest laboratory:

1. **Closed.** The 15-token vocabulary in `minillm/tokenizer.py` covers the
   entire universe: `<pad>`, `<bos>`, `<eos>`, the nine cells `A1`..`C3`, and
   three result tokens `#X`, `#O`, `#=`. There is no out-of-distribution
   text, no scraping bias, no contamination question.
2. **Fully enumerable.** `data/meta.json` records the census: exactly
   **1,310 complete games** over **694 reachable positions**, games between
   5 and 9 moves long, a full transcript at most 12 tokens
   (`max_sequence_tokens`). The *enumerated corpus is the whole population*,
   not a sample — a luxury no real LLM ever has. (Training still holds out
   10% of it for validation: 1,179 games train, 131 validate; chapter 02.)
3. **Exactly solvable.** `minillm/solver.py` is a negamax solver with
   memoization; it proves the root value ("perfect play ends in a draw",
   again in `data/meta.json`). Every position has a game-theoretically
   correct move, so "did the model learn to play well?" has a numeric,
   non-vibes answer.

That last point is the whole reason this repo exists. With real LLMs, "did
it learn X?" is fought over with benchmarks and anecdotes. Here, "does the
model know the rules?" is a legality rate against the engine in
`minillm/game.py`, and "is it any good?" is a win rate against the perfect
solver. When the docs later claim that pretraining teaches *form* and
finetuning teaches *intent*, you can rerun `make eval` and check.

> **In a real LLM:** nothing is enumerable and nothing is solvable. GPT-3
> was pretrained on ~300B tokens sampled from a filtered web crawl; nobody
> can enumerate "all English", and there is no negamax for "the correct next
> sentence". That is precisely why production labs invest so heavily in
> evals — they are the substitute for the ground truth we get here for free.

## Repo map

The package docstring in `minillm/__init__.py` indexes the core modules
(`config.py` and `utils.py` are supporting additions beyond it); the full list:

| Module | Role |
| --- | --- |
| `minillm/game.py` | the "world" the training data describes — rules only, deliberately free of any ML code |
| `minillm/solver.py` | exact negamax solver: ground truth, plus the two game enumerators that generate both corpora |
| `minillm/tokenizer.py` | text <-> token ids; the whole 15-entry vocabulary written down by hand |
| `minillm/dataset.py` | corpus building, train/val splits, tensorization, SFT loss masking |
| `minillm/config.py` | `ModelConfig` dataclass — the handful of numbers that define the architecture |
| `minillm/model.py` | the Transformer itself, from scratch and heavily commented |
| `minillm/train.py` | pretraining + finetuning loops (AdamW, warmup + cosine decay, grad clipping, best-val checkpointing) |
| `minillm/sample.py` | free-running generation, each transcript replayed through the engine and verified |
| `minillm/evaluate.py` | behavioural metrics: legality, refereeing, playing strength |
| `minillm/play.py` | interactive human-vs-model games in the terminal |
| `minillm/inspect_attention.py` | print per-layer, per-head attention matrices for a game prefix |
| `minillm/utils.py` | device selection, seeding, checkpoint loading, `next_token_logits` |

Alongside the package: `tests/` (39 tests covering rules, solver, tokenizer,
masking and causality), `docs/` (this guided tour), and the generated
artifacts in `data/` and `runs/` — both reproducible from scratch, and
training is seeded, so your numbers will closely match the ones quoted here.

## The pipeline, stage by stage

Each row is one stage of this repo mapped onto its production counterpart:

| This repo | Real LLM pipeline |
| --- | --- |
| `enumerate_all_games` writes all 1,310 games to `data/all_games.jsonl` | Scrape and filter a web-scale corpus |
| Hand-written 15-token move-level vocabulary (`tokenizer.py`) | Learned BPE tokenizer, ~100k subword tokens |
| Pretrain on *all* games -> model learns legality, gravity, turn order, results | Pretraining -> model learns grammar, facts, style |
| Finetune on solver-optimal games with opponent moves masked from the loss (`target -1`, ignored) | SFT — imitate the assistant turns, mask the user turns |
| Temperature / top-k sampling; optional legality masking in `play.py` | Decoding strategies, sampling parameters, guardrails |
| Legality rate, result-token refereeing, win rate vs random and vs the solver | Evals and benchmarks |
| `inspect_attention.py` | Interpretability research |

The two training stages deserve one sentence each, because the split is the
conceptual heart of the project. From the `minillm/train.py` docstring:
pretraining on every possible game means the model "does NOT particularly
try to win — it imitates the average game"; finetuning continues from that
checkpoint on `data/expert_games.jsonl` (334 games, per `data/meta.json`)
"with the loss masked so only the solver's perfect moves are imitated".
Same next-token objective, curated data, masked non-expert turns — the exact
shape of SFT.

> **In a real LLM:** the analogy is precise down to the tensor level. In
> chat SFT, the tokens of the *user's* turns are present in the input (the
> model must condition on them) but their loss targets are set to an ignore
> index so the model is never trained to *produce* them. Here the opponent's
> moves play the user role: `dataset.py` sets their targets to `-1`, and
> `train.py` counts only `(y != -1)` positions toward the loss. Swap
> "opponent move" for "user message" and the code is the same.

## Quickstart

Everything is driven by the `Makefile`; run the targets in order. All
commands use the project venv (`PY := .venv/bin/python`), run on plain CPU,
and the timings below are from the README:

```bash
make setup      # one-time: create .venv via uv, install torch + pytest
make test       # 39 unit tests
make data       # enumerate every game                   (seconds)
make pretrain   # stage 2: learn the rules               (~15 min CPU)
make finetune   # stage 3: learn to play well            (~5 min CPU)
make eval       # stage 4: measure what it learned       (~2 min)
make play       # play against it
```

`make all` chains `data -> pretrain -> finetune -> eval`. Two extra targets
for looking around: `make sample` (the model dreams up five complete games,
each verified against the engine) and `make attention` (attention matrices
for the prefix `B1 A1 B2`). No `uv`? The README shows the plain
`python3.12 -m venv` alternative.

Training leaves an audit trail: `runs/pretrain/log.csv` shows validation
loss falling from 2.815 at step 0 (roughly uniform over the vocabulary) to
0.764 at step 2999; `runs/finetune/log.csv` starts already at 0.622 —
inherited competence from the pretrained checkpoint — and the eval JSONs
described below are written by `make eval`.

## Headline results

Both checkpoints were evaluated with `minillm/evaluate.py`; the numbers live
in `runs/eval_pretrain.json` and `runs/eval.json`.

The **pretrained** model is a flawless *grammarian* and a mediocre *player*:
its argmax choice is a legal move on 100.0% of 1,062 held-out teacher-forced
positions (`argmax_legal_rate: 1.0`), it puts 99.6% of its probability mass
on legal moves, and it predicts the correct result token 99.2% of the time —
it can referee a game it just watched. But it wins only 41.8% against a
*random* opponent (near coin-flip: 41.8 / 20.2 / 38.0 win/draw/loss) and
loses **100%** of games against the perfect solver. It learned what games
look like, not how to win them — exactly what "imitate the average game"
predicts.

The **finetuned** model wins **79.2%** against random (6.2% losses) and
holds the perfect solver to a draw in **61%** of games — and since the
solver-proven root value of the game is a draw, *drawing is the theoretical
ceiling*: no player can ever beat the solver. Its solver-agreement rate
(choosing a game-theoretically optimal move) rises from 70.3% to 86.5%.
The cost: argmax legality dips from 100% to 99.5%, and clean free-running
games from 98.0% to 90.5%.

> **In a real LLM:** that dip is the famous *alignment tax*. Finetuning on a
> narrow, goal-directed distribution (expert games; helpful assistant
> replies) buys capability on the target behaviour at a small cost in the
> broad-distribution competence pretraining bought — production pipelines
> see the same trade-off between instruction-following gains and regressions
> on base-model perplexity or breadth benchmarks, and mitigate it with
> techniques like mixing pretraining data into the SFT stage.

Chapter 07 (`docs/07-evaluation.md`) dissects every one of these metrics;
the table in the README summarizes them side by side.

## Reading order

The chapters follow the pipeline; each explains one stage against the actual
code, with "In a real LLM" asides connecting it to production scale:

1. `docs/00-overview.md` — this chapter.
2. `docs/01-the-game.md` — the rules engine and the exact solver.
3. `docs/02-data.md` — from enumerated games to JSONL corpora and tensors.
4. `docs/03-tokenization.md` — the 15-token vocabulary, and why move-level.
5. `docs/04-model.md` — the Transformer, spelled out layer by layer.
6. `docs/05-training.md` — pretraining and finetuning, including the loss mask.
7. `docs/06-inference.md` — sampling, temperature, and playing the model.
8. `docs/07-evaluation.md` — legality, refereeing, strength; what the numbers mean.
9. `docs/08-exercises.md` — modifications to try yourself.

Read them in order the first time; each assumes the vocabulary of the ones
before it.

Next: [The game and its exact solver](01-the-game.md)
