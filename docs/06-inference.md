# 06 — Inference: sampling and playing

Training produced a function: give it a token sequence, get back a probability
distribution over the next token. That is all a GPT is. Everything this
chapter covers — sampling games, playing interactively, refereeing — is just
different ways of *interpreting* that one distribution. The shared plumbing
lives in `minillm/utils.py`:

```python
@torch.no_grad()
def next_token_logits(model, tokenizer, moves, device):
    ids = tokenizer.encode([BOS] + list(moves))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    return logits[0, -1]
```

Encode the history, one forward pass, take the last position's scores.
`play.py`, `sample.py` and `evaluate.py` all build on this vector; they only
differ in what they do with it.

## The autoregressive loop

`GPT.generate()` in `minillm/model.py` is the canonical inference loop, and it
is short enough to quote almost whole:

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -self.config.block_size :]
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :]  # (1, vocab)
    ...pick next_id...
    idx = torch.cat((idx, next_id), dim=1)
    if stop_id is not None and next_id.item() == stop_id:
        break
```

Four steps, repeated: **condition** on everything generated so far (cropped to
`block_size`, though no Drop-Tac-Toe game can exceed it — `MAX_GAME_TOKENS` is
12 and `block_size` is sized to fit), **take the last position's logits**,
**pick one token**, **append it**. The model has no memory between iterations;
each pass re-reads the whole sequence from scratch. `stop_id` — always
`tokenizer.eos_id` in this project — ends generation early, which is why
sampled games vary in length — roughly 8–12 tokens including `<bos>`, depending
on how long the game runs — rather than always padding out to `max_new_tokens`.

Note the asymmetry with training: `forward()` with `targets=None` computes
logits only for the last position (`self.lm_head(x[:, [-1], :])`), because
during generation the predictions at earlier positions are already history.
During training all `T` positions are scored at once — that is the parallelism
that makes teacher-forced training cheap and generation comparatively
expensive.

> **In a real LLM:** this loop is exactly how GPT-4 or Claude produce text —
> one token per forward pass, appended and fed back. The reason streaming
> chat UIs show words trickling in is not a UI affectation; it is the
> autoregressive loop made visible. Serving economics are dominated by it:
> prompt processing ("prefill") is one parallel pass, but every generated
> token costs a full sequential model pass, which is why output tokens are
> priced several times higher than input tokens.

## Picking a token: temperature and top-k

With the last position's logits in hand, `generate()` picks:

```python
if temperature <= 0:
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
else:
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits[logits < kth] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).to(idx.device)
```

**Temperature** divides the logits before softmax. Because softmax
exponentiates, dividing by `T < 1` widens the gaps between scores and sharpens
the distribution; `T > 1` flattens it; `T -> 0` degenerates to argmax, which
is why `temperature=0` is special-cased as greedy. Intuition for this project:

- `temperature 0` — deterministic. Ask "what is your single best move?" Use it
  when evaluating strength (`evaluate.py` does) or when you want the model at
  its most serious opponent setting.
- `temperature 0.7` — `play.py`'s default. Enough randomness that games vary,
  low enough that the model rarely takes moves it considers clearly worse.
- `temperature 1.0` — `sample.py`'s default: the raw learned distribution,
  untouched. The right setting when the question is "what did the model
  actually learn?" rather than "how well can it play?"

**Top-k** truncates before sampling: keep the k highest logits, set the rest
to `-inf` so softmax gives them exactly zero. Its job is to cut off the long
tail of low-probability junk that temperature sampling would otherwise
occasionally hit. In a 15-token vocabulary the tail is short and the pretrained
model already concentrates 99.6% of its mass on legal moves (see
`runs/eval_pretrain.json`: `mean_legal_prob_mass` 0.9964), so `--top-k` exists
in `sample.py` mostly so you can experiment with it — try `--top-k 1` (argmax
by another name) versus no cap.

One deliberate detail: the multinomial draw happens on `probs.cpu()` even when
the model runs on CUDA or MPS. Different devices have different RNG streams;
sampling on one CPU generator means `--seed 0` reproduces the same games on
any machine. Reproducibility is a feature you design in, not one you get for
free.

> **In a real LLM:** temperature and top-k survive at every scale, joined by
> nucleus (top-p) sampling — keep the smallest token set whose cumulative
> probability exceeds p — which adapts the cutoff to the shape of each
> distribution instead of using a fixed k. With 100k+ token vocabularies the
> tail really does contain garbage, and untruncated `temperature 1.0`
> sampling visibly degrades long generations; production APIs expose exactly
> these knobs (`temperature`, `top_p`, `top_k`) for exactly these reasons.

## Constrained decoding: `allowed_ids`, strict vs `--raw`

`generate()` accepts a whitelist:

```python
if allowed_ids is not None:
    keep = torch.full_like(logits, float("-inf"))
    keep[:, allowed_ids] = logits[:, allowed_ids]
    logits = keep
```

Every token outside `allowed_ids` gets `-inf`, so after softmax it has
probability zero — the model *cannot* emit it. `play.py` uses this idea (via
its own local `sample()` in `model_move()`) to implement its two modes:

- **strict** (default): `legal_ids = tokenizer.encode(game.legal_moves())`,
  and sampling is restricted to those ids. The model chooses *among* the legal
  moves; the rules are enforced from outside. The game engine is the source of
  truth, the model only supplies preference.
- **`--raw`**: sampling runs over the entire vocabulary, mistakes included. If
  the model proposes an illegal token, `play.py` narrates it and recovers:

  ```python
  attempt = sample(None)  # anything in the vocabulary, mistakes included
  if attempt in game.legal_moves():
      return attempt
  print(f"  (model proposed '{attempt}' — illegal here; re-sampling among legal moves)")
  return sample(legal_ids)
  ```

`--raw` is the educational mode: it shows you what the network alone believes,
without a safety net. That the message rarely appears is itself a result — the
finetuned model's first attempt is legal 98.8% of the time in free running
(`runs/eval.json`: `first_try_legal_rate` 0.9885; the pretrained model is even
cleaner at 0.9975, because finetuning traded a little grammar for a lot of
strength — chapter 07 unpacks that trade).

> **In a real LLM:** masking logits to a whitelist is constrained decoding,
> and it is how "JSON mode", function-calling schemas, and grammar-guided
> generation work in production APIs. At each step the serving stack computes
> which tokens can legally extend the output under a grammar (JSON, a regex,
> a tool-call schema) and sets all other logits to `-inf` — structurally
> identical to `allowed_ids`, with a grammar automaton standing in for
> `game.legal_moves()`. The lesson transfers directly: when you must have
> valid output, don't hope the model learned the format — enforce it in the
> decoder.

## `play.py`: a tour

`python -m minillm.play` (or `make play`) loads the best available checkpoint
— `default_checkpoint()` in `utils.py` prefers `runs/finetune/model.pt` and
falls back to `runs/pretrain/model.pt` — and drops you into a loop against
the model. Flags: `--human O` to let the model open (X moves first), `--raw`,
`--temperature` (default 0.7, `0` = deterministic best move), `--show-probs`
to see the distribution before each model move, `--seed` for reproducible
games.

In-game commands, from the `HELP` string:

```
A / B / C     drop a piece into that column
A1 .. C3      same, naming the exact landing cell
p             show the model's next-token probabilities
u             undo your last move (and the model's reply)
?             this help
q             quit
```

Column shorthand is resolved by `read_human_move()`: typing `B` looks up the
current stack height and expands to the landing cell (`B1` on an empty column,
`B3` on a two-high one), then validates the move on a *copy* of the game
(`game.copy().push(raw)`) so an illegal entry never corrupts state. Undo pops
two plies — yours and the model's reply — via
`Game.from_moves(game.history[:-2])`, which is trivially correct because the
whole game state is just the move list.

The `p` command is the most instructive: `show_distribution()` renders the
top candidates as a bar chart, annotating each token as `legal move`,
`ILLEGAL move`, `result token`, or `special token`:

```
  model's next-token distribution:
     B2  ########################  61.2%  legal move
     A1  #######  18.7%  legal move
     ...
```

Watch it mid-game: probability mass sits almost entirely on legal moves, and
after a winning move it jumps to the correct result token. You are looking
directly at what the model knows.

When the game ends, the referee footer runs `referee_verdict()` — one more
`next_token_logits()` call on the finished move history, argmax over the whole
vocabulary:

```
(model as referee predicts: #X, actual result token: #X)
```

Nobody told the model who won; predicting the result token was just part of
the pretraining sequence. The finetuned model gets this right on 100% of held
out positions (`runs/eval.json`: `result_prediction_accuracy` 1.0). A next-token
predictor that reliably emits `#X` after a line X just completed has, in every
observable sense, learned what "winning" means.

## `sample.py`: free-running generation and the grammar check

`python -m minillm.sample --num 5` asks a different question: with no game
engine steering it, no human opponent, no legality mask — starting from
`<bos>` alone — can the model produce entire well-formed games? This is the
purest probe of pretraining:

```python
out = model.generate(
    idx, max_new_tokens=MAX_GAME_TOKENS,
    temperature=args.temperature, top_k=args.top_k,
    stop_id=tokenizer.eos_id, generator=generator,
)
```

Every transcript is then replayed through the real engine by
`verify_transcript()` — a small state machine that checks the full grammar of
a game, not just per-move legality. It rejects, with a specific message, each
way a transcript can go wrong: an illegal move (gravity or occupancy
violation, caught by `game.push()` raising `IllegalMoveError`), a move played
after the game was over, a result token claimed while the game was still
running, the *wrong* result token, a **duplicate** result token, `<eos>`
before any result, tokens after `<eos>`, or running out of tokens without
`<eos>`. Only a transcript that survives every check earns `"ok"`:

```
game 1: B1 A1 B2 C1 B3 #X <eos>
         -> ok
```

The duplicate-result check matters more than it looks: a model that emits
`#X #X <eos>` has learned "games end with result tokens" without learning
"exactly one" — a grammar error invisible to per-move legality metrics.
Across 200 free-running rollouts of the pretrained model, every first-attempt
move was legal in 98% of games (`runs/eval_pretrain.json`: `clean_game_rate`
0.98) — per-move legality learned purely from next-token prediction on
transcripts. (No logged metric covers the full transcript grammar that
`verify_transcript()` checks; `clean_game_rate` stops at move legality.)

> **In a real LLM:** free-running sampling plus an external verifier is the
> standard shape of generation-quality evaluation. Code models sample
> programs and run the tests (HumanEval's pass@k); math models sample
> derivations and check the final answer. The verifier is always some analog
> of `verify_transcript()`: a cheap, exact oracle that the expensive,
> approximate model is graded against. And "duplicate result token" has a
> famous large-scale cousin — degenerate repetition, the failure mode that
> repetition penalties and nucleus sampling exist to suppress.

## What the model conditions on — and what it doesn't

Read `model_move()` again: the model receives `game.history` — the token
sequence of moves — and nothing else. There is no board tensor, no 3x3 grid,
no feature plane saying "X occupies B1". `play.py` renders a board for *you*
(`game.render()`); the model never sees it. To know that column B is two high
and `B3` is the only remaining cell there, the model must reconstruct that
fact internally from the raw sequence `B1 A1 B2 ...` — tracking gravity, turn
order, and occupancy inside its residual stream, because nothing else will do
the bookkeeping for it.

That is the point of the whole project. A production LLM asked "whose turn is
it after 1. e4 e5 2. Nf3?" faces the same situation: no board, just tokens,
and any state it needs it must compute from the sequence. Chapter 04's
`inspect_attention.py` (see `minillm/inspect_attention.py`) lets you watch
heads attending back to the moves in the column currently being played —
state tracking made visible.

## No KV cache (on purpose)

Each iteration of `generate()` reruns the full forward pass over the whole
sequence. The keys and values that attention computes for positions
`0..T-1` are identical to what the previous iteration computed — only
position `T` is new. A **KV cache** stores those keys and values and, on each
step, feeds only the newest token through the model, attending against the
cached past. That turns per-step cost from O(T²) to O(T) and is the single
most important serving optimization in real systems.

Here it is deliberately absent — `minillm/model.py`'s module docstring says so
up front: "The attention implementation is intentionally the naive, readable
one (no FlashAttention, no KV cache) — a worked example, not a speed record."

With sequences capped at 12 tokens and ~0.8M parameters, the recompute costs
microseconds; a cache would double the code in `CausalSelfAttention` for no
observable benefit. But sequences of 100k tokens change the arithmetic
entirely.

> **In a real LLM:** the KV cache dominates serving memory. For a
> Llama-70B-class model at long context, the cached keys and values for a
> single conversation run to gigabytes — often rivaling the weights — which
> is why techniques like grouped-query attention (fewer KV heads),
> multi-query attention, cache quantization, and paged attention (vLLM)
> exist: they are all KV-cache compression or management schemes. "Prefill
> vs. decode" as separate serving phases is likewise a KV-cache concept:
> prefill builds the cache for the prompt in one parallel pass; decode
> extends it token by token.

Adding a KV cache to `generate()` — and verifying it produces bit-identical
output to the naive loop — is one of the exercises in
`docs/08-exercises.md`.

Next: [07 — Evaluation: what did it learn?](07-evaluation.md) — teacher-forced
legality, free-running cleanliness, tournament strength versus random and
optimal opponents, and what the numbers in `runs/eval.json` actually measure.
