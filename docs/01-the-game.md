# The game and its exact solver

Before there is a model, there is a world. This chapter covers the two files
that define that world completely: `minillm/game.py` (the rules) and
`minillm/solver.py` (perfect knowledge of the rules). Everything downstream —
corpus, tokenizer, training, evaluation — is derived from these two modules,
and neither of them contains a single line of ML code. As the docstring in
`minillm/game.py` puts it, this is "the 'physics' of the tiny world our
language model will learn purely from transcripts."

## The rules

Drop-Tac-Toe is Tic-Tac-Toe played on a 3x3 grid with Connect-Four gravity:

- Columns are labelled `A`, `B`, `C` (left to right); rows are `1`, `2`, `3`
  (bottom to top).
- A move drops a piece into a column. The piece falls to the lowest free cell
  of that column.
- `X` moves first; `X` and `O` alternate.
- Three of your own pieces in any row, column, or diagonal win.
- If all nine cells fill without a line, the game is a draw.

Here is the board rendering from `Game.render()`, exactly as the interactive
player prints it (row 3 on top, so gravity visually pulls pieces downward):

```
 3 | . . .
 2 | . X .
 1 | O X .
   +------
     A B C
```

This position arose from the move sequence `B1 A1 B2`: X dropped into column
B (landed at B1), O dropped into column A (landed at A1), X dropped into
column B again (landed on top of the first piece, at B2).

## Notation and the gravity constraint

A move is written as the cell the piece *lands on*, not the column it was
dropped into. That is a deliberate design choice with consequences for the
model, so it deserves precision:

- **Legal:** on an empty board, `A1`, `B1`, `C1` — every column is empty, so
  every piece lands on row 1.
- **Illegal:** on an empty board, `C3` — the piece would float. `C3` only
  becomes legal once `C1` and `C2` are already occupied.
- **Illegal:** `A2` as an opening move, for the same reason.

`Game.push()` in `minillm/game.py` enforces this with an exact height check
and a human-readable error:

```python
if row != height:
    raise IllegalMoveError(
        f"{move} would float: the next free cell in column {move[0]} "
        f"is {COLS[col]}{height + 1}"
    )
```

`tests/test_game.py::test_gravity_pieces_cannot_float` pins this behavior:
pushing `C3` or `A2` on an empty board raises `IllegalMoveError` matching
`"float"`, while `A1` is fine.

A corollary of gravity: at any moment there is **exactly one legal cell per
non-full column** — the cell right on top of the current stack. So the set of
legal moves always has size 0 to 3. Apart from an `if self.is_over():
return []` guard — which is why the legal-move set can be empty on a won
board — `Game.legal_moves()` is a three-line comprehension:

```python
return [
    f"{COLS[c]}{len(stack) + 1}"
    for c, stack in enumerate(self.stacks)
    if len(stack) < N
]
```

## Why gravity makes this interesting for a language model

Plain Tic-Tac-Toe notation would let a model get away with shallow pattern
matching: any cell name that has not appeared yet is legal. Gravity breaks
that. The row digit of a legal move is not a free choice — it is a
*function of the entire game history*. To know whether the next token can be
`C3`, you must know how many pieces have been dropped into column C so far,
which means counting occurrences of `C1` and `C2` across the whole preceding
sequence.

In other words: the transcript notation carries a **hidden state** (three
column heights) that is never written down explicitly, and the grammar of
valid continuations depends on that state. A model that emits `C3` after only
`C1` has been played has not merely made a weak move — it has made a
*physically impossible* one. The evaluation chapter
(`docs/07-evaluation.md`) measures exactly this: the fraction
of sampled moves that are legal is a direct read-out of whether the model
internally tracks column heights.

> **In a real LLM:** this is the toy version of state tracking in natural
> language and code. Whether the next token may be a closing brace depends on
> how many braces are currently open; whether "she" is a valid coreference
> depends on entities introduced hundreds of tokens ago. Transformers have no
> external memory — any such state must be reconstructed on the fly from the
> context window by the attention layers, at every single token position.
> Gravity gives us a minimal, fully checkable instance of that phenomenon.

## The numbers: a fully enumerable world

The pipeline's data-generation step writes `data/meta.json`. These are the
actual numbers of this world:

| Quantity | Value |
|---|---|
| Complete games (`n_all_games`) | 1310 |
| Reachable positions solved (`positions_solved`) | 694 |
| Shortest game (`shortest_game_moves`) | 5 moves |
| Longest game (`longest_game_moves`) | 9 moves |
| X wins (`results_all["#X"]`) | 616 |
| O wins (`results_all["#O"]`) | 386 |
| Draws (`results_all["#="]`) | 308 |
| Root value (`root_value`) | "perfect play ends in a draw" |

A few observations worth internalizing:

- **1310 games is the entire language.** Not a sample — the complete set of
  every game that can possibly be played. There is no held-out distribution
  shift, no long tail. This closed world is what lets later chapters ask
  sharp questions like "did the model assign probability mass to sequences
  outside the language?"
- **The shortest game is 5 moves**: X's three moves plus O's two, e.g.
  `A1 B1 A2 B2 A3` — X wins column A while O politely stacks column B
  (this exact line is `tests/test_game.py::test_vertical_win`).
- **X wins 616 of 1310 random-ish games (47%)**, an artifact of moving first.
  But the *root value is a draw*: with perfect play from both sides, nobody
  wins. The gap between "X usually wins in the full game tree" and "X cannot
  force a win" is precisely the gap between pretraining data (all games) and
  finetuning data (expert games) that Chapter 5 exploits.
- **694 positions** is why we can afford an exact solver at all — see below.

> **In a real LLM:** there is no `meta.json` for English. The pretraining
> corpus of a GPT-3- or Llama-class model is hundreds of billions to
> trillions of tokens sampled
> from an open-ended distribution that nobody can enumerate, and "is this
> sentence in the language?" has no ground-truth oracle. Our lab deliberately
> inverts that: because the world is closed and solved, every later claim
> about what the model learned can be checked against exact truth rather
> than benchmarks and vibes.

## `game.py`: the rules as code

### The stacks representation

`Game` stores the board as one string per column, bottom to top:

```python
stacks: list[str] = field(default_factory=lambda: ["", "", ""])
```

`stacks == ["XO", "", "X"]` means A1=X, A2=O, C1=X. This representation makes
gravity structural rather than checked: you can only ever append to a string,
so a piece physically cannot be inserted above a gap. Column height is
`len(stack)`, the landing row of a drop is `len(stack) + 1`, and the side to
move needs no stored flag at all:

```python
@property
def to_move(self) -> str:
    """X moves on even ply counts (0, 2, ...), O on odd ones."""
    return X if sum(len(s) for s in self.stacks) % 2 == 0 else O
```

Because X always moves first and turns strictly alternate, the total piece
count determines whose turn it is. State that can be derived is not stored —
one less invariant to break.

### `LINES`: the eight ways to win

All winning lines are precomputed once, as 0-based `(column, row)`
coordinates with row 0 at the bottom:

```python
LINES: list[list[tuple[int, int]]] = (
    [[(c, r) for r in range(N)] for c in range(N)]  # verticals
    + [[(c, r) for c in range(N)] for r in range(N)]  # horizontals
    + [[(i, i) for i in range(N)], [(i, N - 1 - i) for i in range(N)]]  # diagonals
)
```

3 verticals + 3 horizontals + 2 diagonals = 8 lines. `Game.winner()` just
checks whether any line is uniformly one piece. All four line orientations
are covered by dedicated tests in `tests/test_game.py` (`test_vertical_win`
through `test_anti_diagonal_win`), plus `test_o_can_win_too` to make sure the
win check is not accidentally X-only.

### `push()` validation, in order

`Game.push(move)` rejects, with distinct `IllegalMoveError` messages:

1. moves after the game is over ("the game is already over"),
2. malformed strings — anything not matching `[A-C][1-3]` after
   `strip().upper()` (`tests/test_game.py::test_malformed_moves_rejected`
   tries `"D1"`, `"A4"`, `"AA"`, `"1A"`, `""`, `"A"`),
3. drops into a full column ("column C is full"),
4. floating cells (the `row != height` check quoted above).

Only then does it mutate: `self.stacks[col] += self.to_move`. The engine is
strict on purpose — during interactive play (`minillm/play.py`) the model's
proposed moves go through this same `push()`, so an illegal generation is
caught by the physics, never silently absorbed.

## `solver.py`: perfect play in twenty lines

The solver operates on an immutable mirror of the game state:

```python
State = tuple[str, str, str]
EMPTY: State = ("", "", "")
```

Same stacks idea, but a tuple of strings — hashable, so `functools.lru_cache`
can memoize on it. The module reimplements `to_move`, `winner`, etc. as free
functions on `State` because, per its own comment, "these run millions of
times" during enumeration and a `Game` object per node would be pure
overhead.

### Negamax, line by line

```python
@lru_cache(maxsize=None)
def negamax(stacks: State) -> int:
    if winner(stacks) is not None:
        return -1  # the previous player just completed a line: I lost
    if is_full(stacks):
        return 0
    return max(-negamax(apply_move(stacks, c)) for c in legal_columns(stacks))
```

The value is always **from the perspective of the player to move**: `+1` =
I win with perfect play, `0` = draw, `-1` = I lose. Three cases:

- **`winner(stacks) is not None` returns `-1`.** This is the sign convention
  that trips people up. If a completed line exists when it is my turn, then
  my *opponent* made the last move and completed it — so I, the player to
  move, have already lost. A position is never evaluated with a line of the
  player-to-move's own color, because the game would have ended one ply
  earlier. `tests/test_solver.py::test_double_threat_is_lost_for_the_defender`
  exercises the convention: with `("XX", "O", "XX")` and O to move, both `A3`
  and `C3` win for X, O can only block one, and `negamax(state) == -1` — the
  player to move loses.
- **`is_full` returns `0`.** No line, no free cell: draw.
- **The recursion.** My value is the best I can achieve over my (at most
  three) moves, and after any move the resulting position is worth the
  *negation* of what it is worth for my opponent — the zero-sum identity
  that lets one function serve both players, instead of the mutually
  recursive min/max pair of textbook minimax.

Memoization is one decorator: `@lru_cache(maxsize=None)`. Different move
orders reach identical positions (`A1 B1 C1` and the transposition
`C1 B1 A1` both leave X on A1 and C1 with O on B1), and the cache
collapses them. The cache ends
up holding exactly the 694 reachable positions recorded as
`positions_solved` in `data/meta.json` — the whole game, solved, in memory.

> **In a real LLM:** the solver plays the role that human labelers, reward
> models, and curated demonstration data play at production scale. Someone
> has to define what "good" output is. For Claude- or GPT-4-class assistants
> that oracle is expensive, noisy, and itself learned (RLHF reward models
> trained on human preference comparisons). Here it is 20 lines of negamax
> that are *provably* correct — which means when the evaluation
> chapter (`docs/07-evaluation.md`) says "the model plays optimally in N% of
> positions," that percentage is measured against
> mathematical truth, not against another model's opinion.

### `best_moves`: values plus all optimal actions

```python
def best_moves(stacks: State) -> tuple[int, list[str]]:
    value = negamax(stacks)
    moves = [
        notation(stacks, c)
        for c in legal_columns(stacks)
        if -negamax(apply_move(stacks, c)) == value
    ]
    return value, moves
```

It re-derives which moves achieve the position's value and returns *all* of
them. Ties are common — the empty board is left-right symmetric, so if `A1`
draws, `C1` draws too (`tests/test_solver.py::test_mirror_symmetry` checks
that mirroring columns never changes a value). Returning the full tie set
rather than an arbitrary winner matters for data diversity, as the expert
enumerator shows next.

## From solver to corpora

`solver.py` ends with the two generators that produce the training data.

### `enumerate_all_games`: the pretraining corpus

```python
def dfs(stacks: State, moves: list[str]) -> None:
    if winner(stacks) is not None or is_full(stacks):
        games.append({"moves": list(moves), "result": result_token(stacks)})
        return
    for col in legal_columns(stacks):
        moves.append(notation(stacks, col))
        dfs(apply_move(stacks, col), moves)
        moves.pop()
```

A plain depth-first search that branches over *every* legal column at *every*
node, recording a game whenever it hits a terminal state. Note it records
complete *games*, not deduplicated positions: two games sharing a prefix are
distinct corpus entries, because the model will be trained on full
transcripts. The append/recurse/pop pattern keeps one shared move list
instead of copying a path per node. The traversal order is deterministic
(columns in A, B, C order), so the corpus is reproducible byte for byte.
`tests/test_solver.py::test_enumerate_all_games_are_legal_and_complete` pins
`len(games) == 1310`, replays a sample through the strict `Game.push()`, and
asserts there are no duplicates.

Every game ends with a result token — `#X`, `#O`, or `#=` (constants
`RESULT_X`, `RESULT_O`, `RESULT_DRAW` in `game.py`). Those are part of the
transcript language: the model will learn to *announce the outcome* as its
final token, which later becomes a free probe of whether it understands
positions, not just move legality.

### `enumerate_expert_games`: the finetuning corpus

`enumerate_expert_games(expert, ties="all")` runs the same DFS with one
asymmetry at each node:

```python
if to_move(stacks) == expert:
    _, options = best_moves(stacks)
    if ties == "first":
        options = options[:1]
else:
    options = [notation(stacks, c) for c in legal_columns(stacks)]
```

- When it is the **expert's** turn, only solver-optimal moves are explored.
  With `ties="all"` (the default used for the corpus) the expert branches
  over *all* equally good moves — more diverse demonstrations; `ties="first"`
  would collapse to a single canonical line.
- When it is the **opponent's** turn, every legal move is explored — including
  terrible ones. This matters: the expert's perfect answers are demonstrated
  against every situation an opponent can create, not only against perfect
  counter-play. A model finetuned on perfect-vs-perfect games alone would
  never see how to punish a blunder.
  `tests/test_solver.py::test_expert_games_opponent_branches_everything`
  verifies that after each optimal opening, the corpus contains every legal
  O reply.

Each resulting game dict carries an extra `"expert": "X"` or `"expert": "O"`
field. That field is the hook for the finetuning loss mask (Chapter 5): the
model is trained to imitate only the expert's moves, while the opponent's
arbitrary moves are excluded from the loss — visible as context, but never
imitated. Both sides are generated (`enumerate_expert_games("X")` plus
`enumerate_expert_games("O")`), giving the 334 expert games recorded in
`data/meta.json`, with results skewed heavily toward the expert:
222 `#X`, 60 `#O`, 52 `#=` — compare the much less skewed pretraining mix
(616 / 386 / 308).

> **In a real LLM:** this is supervised finetuning (SFT) in miniature, down
> to the loss masking. Assistant models are finetuned on conversations where
> the loss is computed only on the assistant's turns — the user's messages
> are context to condition on, not behavior to imitate. The `"expert"` field
> here is the analogue of the role labels in a chat template, and "opponent
> branches over everything" is the analogue of demonstrating good responses
> to the full messy range of user inputs, not just well-posed ones.

## Why exactness matters

It is worth pausing on what the solver buys us, because it shapes every
chapter that follows:

1. **Ground truth for data.** The finetuning corpus is optimal by
   construction, not "pretty good heuristic play."
2. **Ground truth for evaluation.** `best_moves()` gives, for any position,
   the complete set of acceptable answers. Model evaluation reduces to set
   membership.
3. **Ground truth for the narrative.** The root value is a draw
   (`describe_root_value()` in `solver.py` renders `negamax(EMPTY) == 0` as
   `"perfect play ends in a draw"`, the string stored in `data/meta.json`).
   So when the finetuned model later holds draws against perfect play, that
   is the best achievable outcome — not a mediocre one.

The test suites are part of the contract: `tests/test_game.py` pins the
physics (gravity, alternation, all four win directions, draw, no moves after
game over), and `tests/test_solver.py` pins the mathematics (tactical values,
symmetry, exact corpus sizes 1310 and 334, expert optimality). If either file
ever fails, nothing downstream — data, training, evaluation — can be trusted.

Next: [From games to a training corpus](02-data.md) — how these enumerated
games become shuffled, serialized training documents in `data/`.
