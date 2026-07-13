# 02 — From games to a training corpus

The previous chapter ended with a solver that can enumerate and evaluate every
position in Drop-Tac-Toe. This chapter turns that closed world into the two
things a language model actually consumes: **text corpora** (JSONL files of
game transcripts) and **training tensors** (integer matrices with
shifted-by-one targets). Everything here lives in `minillm/dataset.py`, with
the token vocabulary it depends on in `minillm/tokenizer.py` and the
behavioral contract pinned down in `tests/test_dataset.py`.

The comment at the top of `minillm/dataset.py` states the inversion that makes
this lab possible:

```python
"""Real LLM pipelines scrape terabytes of text; ours *enumerates its entire
universe* — every game of Drop-Tac-Toe that can possibly be played."""
```

We do not sample the world; we exhaust it.

## The two corpora

Running `make data` (i.e. `python -m minillm.dataset --out data`) calls the two
enumerators from `minillm/solver.py` and writes three files:

| File | Rows | Role in the pipeline |
| --- | ---: | --- |
| `data/all_games.jsonl` | 1,310 | Pretraining corpus: every complete game. Teaches what games *look like* — legal moves, gravity, when a game ends, who won. The grammar. |
| `data/expert_games.jsonl` | 334 | Finetuning corpus: games where one side plays solver-perfectly. Teaches how to play *well*. The SFT stage. |
| `data/meta.json` | — | Corpus statistics, used for sanity checks and this documentation. |

Each line is one complete game as a small JSON object. Real first lines from
each file:

```json
// data/all_games.jsonl
{"moves": ["A1", "A2", "A3", "B1", "B2", "B3", "C1"], "result": "#X"}

// data/expert_games.jsonl
{"moves": ["A1", "A2", "B1", "A3", "B2", "B3", "C1"], "result": "#X", "expert": "X"}
```

The formats differ by exactly one field: expert games carry `"expert": "X"`
or `"expert": "O"`, recording which side the solver played. That field is
metadata for the loss function, not part of the text the model sees — it
drives the masking described below and is never tokenized.

### all_games.jsonl: the full universe

`enumerate_all_games()` walks the game tree depth-first from the empty board,
branching over every legal column at every turn, and records each leaf. From
`data/meta.json`:

```json
"n_all_games": 1310,
"results_all": { "#X": 616, "#=": 308, "#O": 386 },
"shortest_game_moves": 5,
"longest_game_moves": 9
```

Note the skew: X wins 47% of *all possible* games, O wins 29%, and only 24%
are draws — even though the solver proves the game is a draw under perfect
play (`"root_value": "perfect play ends in a draw"`). The pretraining corpus
is a census of everything that *can* happen, not of what *should* happen.
That is exactly why a model pretrained on it plays legally but not well
(chapter 07 measures this: roughly coin-flip against a random opponent).

> **In a real LLM:** this is the pretraining-data dilemma in miniature. Web
> corpora for GPT-3 or Llama contain brilliant proofs next to confidently
> wrong forum posts; next-token pretraining teaches the model to imitate the
> *distribution* of all of it, not the best of it. "Knows the rules, plays the
> average game" is the precise toy analog of "fluent but not yet helpful".

### expert_games.jsonl: demonstrations worth imitating

`enumerate_expert_games(expert)` runs a different tree walk, visible in
`minillm/solver.py`: whenever it is the expert's turn it only branches over
solver-optimal moves (`best_moves`, with `ties="all"` so every equally good
move is demonstrated), and whenever it is the opponent's turn it branches over
**every** legal reply:

```python
if to_move(stacks) == expert:
    _, options = best_moves(stacks)
    if ties == "first":
        options = options[:1]
else:
    options = [notation(stacks, c) for c in legal_columns(stacks)]
```

This matters: the expert's perfect answers are demonstrated against every
situation any opponent could create, including bad opponents — not just
against perfect counter-play. `main()` builds the corpus for both seats:

```python
expert_games = enumerate_expert_games(X) + enumerate_expert_games(O)
```

The per-side breakdown (countable from the `"expert"` field, aggregated in
`meta.json` as `"results_expert"`):

| Expert side | Games | Expert wins | Draws | Expert losses |
| --- | ---: | ---: | ---: | ---: |
| X | 248 | 222 (`#X`) | 26 | 0 |
| O | 86 | 60 (`#O`) | 26 | 0 |
| total | 334 | 282 | 52 | **0** |

The expert never loses a single game — perfect play in a drawn game
guarantees at least a draw from either seat. X's tree is larger because the
first player has more winning lines to demonstrate; O, moving second, more
often gets funneled into the narrow drawing lines.

> **In a real LLM:** `expert_games.jsonl` is the SFT dataset. At Claude or
> InstructGPT scale this is tens of thousands to millions of curated
> demonstrations written or selected by humans — expensive, small (orders of
> magnitude smaller than pretraining data, exactly like 334 vs 1,310 here),
> and defining *behavior* rather than *knowledge*. Our advantage: the solver
> is an infinitely patient, provably perfect annotator.

## split_games: train/val, and what "held-out" honestly means

```python
def split_games(games: list[dict], val_frac: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    shuffled = list(games)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]
```

A dedicated `random.Random(seed)` instance — not the global RNG — makes the
split a pure function of `(games, val_frac, seed)`: every script that calls it
gets byte-identical splits, and `test_split_is_disjoint_and_deterministic` in
`tests/test_dataset.py` pins that down. With the default `val_frac=0.1` this
yields 1,179 train / 131 val games for pretraining and 301 train / 33 val
for finetuning (334 → 33 val).

Now the honest part. The 131 validation games are complete games the model
never trains on — but they are *not* unseen text in the way a held-out web
page is. Every game starts from the same empty board, and with only 694
reachable positions in the entire game, nearly every *prefix* of a validation
game also occurs as a prefix of some training game. The docstring in
`split_games` flags this openly rather than hiding it:

```python
"""(In a world this small most *positions* still occur in some training
game via shared prefixes — docs/05-training.md discusses why val
loss is nevertheless meaningful.)"""
```

The short version: validation loss here does not measure "can the model
handle novel positions" (there barely are any); it measures "did the model
learn the *distribution* rather than memorizing which exact continuations
appear in the training file". A model that memorized training games verbatim
would assign too little probability to the held-out continuations of shared
prefixes, and val loss would diverge from train loss. Chapter 05 shows the
actual curves from `runs/pretrain/log.csv`.

> **In a real LLM:** train/test contamination is one of the field's chronic
> headaches. Benchmark questions leak into web-scraped pretraining corpora,
> inflating scores; deduplication pipelines (MinHash, suffix arrays) exist
> precisely because "held out" is a property you must engineer, not assume.
> Our toy makes the failure mode impossible to ignore: the overlap is total
> and structural, so we are forced to state precisely what val loss can and
> cannot tell us — a discipline worth keeping at any scale.

## build_tensors: from transcripts to (x, y)

Training consumes two integer tensors of shape `(N, block_size)` built by
`build_tensors(games, tokenizer, block_size, expert_only=False)` in
`minillm/dataset.py`. `block_size` is 16 (from `ModelConfig` in
`minillm/config.py`); the longest possible game sequence is
`MAX_GAME_TOKENS = 1 + 9 + 1 + 1 = 12` tokens, so everything fits with room
to spare.

Each game is first encoded by `Tokenizer.encode_game` as

```
<bos> move_1 ... move_k result <eos>
```

(the 15-token vocabulary gets its own chapter next). Then two tensors are
filled:

```python
x = torch.full((len(games), block_size), tokenizer.pad_id, dtype=torch.long)
y = torch.full((len(games), block_size), -1, dtype=torch.long)
```

`x` holds the input tokens, pre-filled with `<pad>` (id 0). `y` holds the
targets, pre-filled with `-1` — and `-1` is the `ignore_index` that
`F.cross_entropy(..., ignore_index=-1)` in `minillm/model.py` skips entirely.
Padding therefore costs nothing and teaches nothing; a target of `-1` means
"no gradient from this position".

### The shift by one, worked on a real game

Next-token prediction means: position `t` of `x` sees tokens `0..t` and must
predict `y[t]`, which is token `t+1`. So `y` is simply `x` shifted one to the
left. Take the shortest game shape in the corpus, a 5-move column-A win that
really is one of the 24 five-move games in `data/all_games.jsonl`:

```json
{"moves": ["A1", "B1", "A2", "B2", "A3"], "result": "#X"}
```

X drops into column A three times (A1, A2, A3 — a vertical win); O answers in
column B twice. Encoded, the game is 8 tokens: ids
`[1, 3, 6, 4, 7, 5, 12, 2]` for
`<bos> A1 B1 A2 B2 A3 #X <eos>`. With `block_size = 16` the tensors row is:

```
t          0     1     2     3     4     5     6     7     8   ...  15
x[t]    <bos>   A1    B1    A2    B2    A3    #X  <pad> <pad>  ... <pad>
y[t]      A1    B1    A2    B2    A3    #X  <eos>   -1    -1   ...  -1
```

Read column by column: given only `<bos>`, predict X's opening `A1`; given
the game so far, predict each next move; after `A3` completes the vertical
line, predict the verdict `#X`; after the verdict, predict `<eos>`. Note that
`x` ends at `#X` — the final `<eos>` appears only as a target, never as an
input (`x[i, : len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)` in the
code), because nothing comes
after it that would need predicting. `test_targets_are_inputs_shifted_left`
and `test_padding_is_ignored_in_targets` in `tests/test_dataset.py` assert
exactly this layout.

Two non-obvious consequences fall out of this one layout:

- **Predicting `#X` is the "understanding" task.** To emit the right result
  token the model must know, from the move sequence alone, that three X
  pieces are stacked in column A. No board is ever shown to it.
- **Every position trains simultaneously.** One 8-token game contributes 7
  supervised predictions in a single forward pass — the efficiency trick that
  makes decoder-only training viable at any scale.

### expert_only: masking the opponent, the SFT trick

The one flag that separates pretraining from finetuning tensors is
`expert_only`. `minillm/train.py` sets it from the stage name:

```python
expert_only = args.stage == "finetune"
...
x_train, y_train = build_tensors(train_games, tokenizer, config.block_size, expert_only)
```

Inside `build_tensors`, when the flag is on and the target token is a move,
the mover is derived from the move *number* and compared against the game's
`"expert"` field:

```python
if expert_only and tokenizer.is_move_id(target):
    # seq[1] is move 1, seq[2] move 2, ... target sits at index t+1, so
    # it is move number t+1. X plays the odd moves (1st, 3rd, ...), O the
    # even ones.
    mover = X if (t + 1) % 2 == 1 else O
    if mover != game["expert"]:
        continue  # leave -1: do not imitate the opponent
```

X always moves first, so odd move numbers are X and even ones are O — the
mover is a pure function of position in the sequence, no board simulation
needed. If the mover is not the expert, the target stays `-1` and the loss
never sees it. The result token and `<eos>` are *not* moves
(`is_move_id` is false), so they stay supervised in every game —
`test_expert_masking_hides_opponent_moves` checks both halves: opponent moves
masked, expert moves plus `#X`/`<eos>` trained.

Why bother? Remember how the expert corpus was built: the opponent branches
over **every** legal reply, including terrible ones. Without masking, the
model would be trained to imitate those terrible moves whenever it plays the
opponent's seat — the finetuning data would teach bad play alongside perfect
play. With masking, the opponent's moves still appear in `x` (the model
conditions on them; it must handle any opponent), but gradient only flows
from the expert's answers.

> **In a real LLM:** this is user-turn masking in SFT, line for line. A chat
> training example contains both the user's message and the assistant's
> reply, but the loss is computed only on the assistant's tokens: the model
> must *condition on* arbitrary, possibly adversarial user input without
> being trained to *produce* it. Swap "opponent move" for "user turn" and
> "expert move" for "assistant turn" and this `continue` statement is the
> same mechanism that keeps a chat model from imitating its users.

## The token budget, and why this corpus is microscopic

Counting `<bos> + moves + result + <eos>` per game:

| Corpus | Games | Total tokens | Supervised targets |
| --- | ---: | ---: | ---: |
| all_games (pretrain) | 1,310 | 14,630 | 13,320 |
| expert_games (finetune) | 334 | 3,596 | 2,076 (after opponent masking) |

The masking is aggressive: of the expert corpus's 3,262 possible targets,
only 2,076 survive — roughly a third of the supervised signal is deliberately
thrown away to avoid imitating the opponent.

Fourteen thousand tokens. For scale: GPT-3 pretrained on ~300 billion tokens
and Llama 3 on ~15 trillion — this corpus is about seven to nine orders of
magnitude smaller, and the ~0.8M-parameter model has more than 50 parameters
per pretraining token where real LLMs have tens to hundreds of tokens per
parameter (up to ~2,000 for aggressively over-trained small models like
Llama 3 8B).
That inversion is deliberate (see the comment in `minillm/config.py`:
"deliberately overpowered ... so that capacity is never the reason something
fails to be learned"). We are not studying data efficiency; we are studying
mechanism, and a closed, fully enumerated world is what lets every later
chapter replace "the model seems to" with a measured number.

One practical consequence of the tiny budget: no dataloader, no streaming, no
sharding. `build_tensors` materializes the entire corpus as two tensors of
shape `(1310, 16)` — about 335 KB — and training samples random rows from
them. The concept ("batches of token blocks with shifted targets") is
identical to a production pipeline; only the plumbing evaporates.

Next: [Tokenization](03-tokenization.md) — the 15-token vocabulary, why
move-level tokens instead of characters, and what a tokenizer really is.
