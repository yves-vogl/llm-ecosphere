# 08 — Exercises: make it yours

Reading about a pipeline is not the same as owning one. Every exercise below
breaks something on purpose, then asks you to measure the damage — because the
repo ships with a complete measurement kit (`minillm/evaluate.py`,
`minillm/sample.py`, `minillm/inspect_attention.py`) and a ground-truth oracle
(`minillm/solver.py`) that most real ML projects would kill for. Use them.

Baseline numbers to beat or explain, from the reference runs (`runs/` is
gitignored but reproducible — training is seeded):

| metric | pretrain (`runs/eval_pretrain.json`) | finetune (`runs/eval.json`) |
|---|---|---|
| argmax legal (teacher-forced) | 100.0% | 99.5% |
| clean self-play games | 98.0% | 90.5% |
| optimal-move rate | 70.3% | 86.5% |
| vs optimal solver W/D/L | 0 / 0 / 100% | 0 / 61% / 39% |

Every exercise states the files to touch, a difficulty tag
(**30 minutes** / **an afternoon** / **a weekend**), and a hint — not a
solution. A full retrain is cheap: `make pretrain` is 3000 CPU steps on a
1,310-game corpus and finishes in minutes, so "just try it" is always a valid
research strategy here.

Ground rules: keep your experiments out of `runs/pretrain` and `runs/finetune`
(pass `--out-dir runs/exp-<name>` to `minillm.train`), and run `make test`
after touching `game.py`, `tokenizer.py` or `dataset.py` — the tests encode
the invariants the rest of the pipeline assumes.

---

## 1. Character-level tokenizer — *an afternoon*

**Motivation.** `minillm/tokenizer.py` hands the model moves on a silver
platter: one token per move, so "which cell" and "which token" are the same
question. Real tokenizers are not that kind. What does the model lose when
`B2` becomes the two tokens `B`, `2`?

**Task.** Build a character-level vocabulary (`A B C 1 2 3 # X O =` plus the
three specials) and re-run the whole pipeline. A full game grows from at most
12 tokens (`MAX_GAME_TOKENS` in `tokenizer.py`) to roughly 22, so raise
`block_size` (a `--block-size` flag already exists on `minillm.train` —
default 16, hardcoded in its argparse; `minillm/config.py` documents the
matching `ModelConfig` default, so pass the flag rather than editing config).
Then compare `first_try_legal_rate` (baseline: 99.8% pretrain / 98.8%
finetune, under `legality_free_running` in `runs/eval_pretrain.json` /
`runs/eval.json`) and `clean_game_rate` against the table above — but count a move as legal
only if *both* characters form a legal cell.

**Files.** `minillm/tokenizer.py`, `minillm/config.py`; the consumers
`minillm/utils.py` (`next_token_logits`), `minillm/evaluate.py` and
`minillm/play.py` all assume one token per move and will need a
move-assembly step.

**Hint.** Keep the `encode_game(moves, result)` interface intact and hide the
character splitting inside the tokenizer — then `dataset.build_tensors` works
unchanged. The finetuning loss mask in `build_tensors` is the tricky part: the
"move number = token index" arithmetic in its comment breaks when a move is
two tokens.

> **In a real LLM:** this trade-off is live and unresolved. BPE tokenizers
> (GPT-2's 50,257 merges, Llama's 32k–128k, Claude's comparable vocabularies)
> exist precisely to shorten sequences, at the price of the model having to
> learn what characters are *inside* a token — which is why models famously
> struggle to count the r's in "strawberry". You are about to reproduce that
> entire debate in a 15-symbol universe.

## 2. Mirror symmetry: augmentation in reverse — *30 minutes*

**Motivation.** Drop-Tac-Toe is symmetric under swapping columns A and C.
The classic move would be data augmentation — but here it is a no-op, and
understanding *why* is the exercise: `enumerate_all_games` in
`minillm/solver.py` already emits every game, so the mirror of each transcript
is already in the corpus. The expert corpus is mirror-closed too (the mirror
of an optimal move is optimal, and `enumerate_expert_games` branches over all
optimal ties). In a closed, enumerated world there is nothing to augment.

**Task.** So run the experiment backwards: *deduplicate* mirror pairs.
Write a filter that keeps only the lexicographically smaller of
`(game, mirror(game))`, pretrain on the roughly half-sized corpus, and check
whether legality and strength survive. If they do, the model is genuinely
generalizing across the symmetry rather than memorizing both halves.

**Files.** A small filter in or next to `minillm/dataset.py` (the `main()`
that writes `data/all_games.jsonl` is the natural hook); nothing else changes.

**Hint.** `mirror(move)` is a 1-line translation table A→C, C→A, B→B on
`move[0]`. In this corpus no complete game is self-symmetric (a transcript
equal to its mirror would need every move in column B, which only holds 3
pieces), so the dedup yields exactly half — 655 games — but handle the
`game == mirror(game)` case anyway for correctness.

> **In a real LLM:** augmentation is alive and well where the world is *not*
> enumerable — back-translation for machine translation, code with renamed
> variables, images flipped and cropped. The deeper production analogue of
> this exercise is deduplication: GPT-3's and Llama's corpora are aggressively
> deduplicated because near-duplicate documents inflate memorization and
> quietly contaminate held-out evaluation — the same effect you are probing
> here in miniature.

## 3. Ablations: what is actually load-bearing? — *an afternoon*

**Motivation.** The default model (797,312 parameters — `GPT.num_params()`)
is deliberately overpowered so capacity is never the excuse. Which parts of
the architecture does a 15-token world actually need?

**Task.** Three ablations, each trained with
`python -m minillm.train --stage pretrain --out-dir runs/exp-...` and measured
with `python -m minillm.evaluate --ckpt runs/exp-.../model.pt`:

1. `--n-layer 1` (drops the model to 202,496 parameters),
2. `--n-head 1` (same parameter count — only the *factoring* of attention changes),
3. no position embeddings: in `minillm/model.py`, delete `pos_emb` from
   `x = self.transformer.drop(tok_emb + pos_emb)` (this one needs a code edit;
   there is no flag).

**Files.** None for (1) and (2) — flags exist in `minillm/train.py`.
`minillm/model.py` for (3).

**Hint for (3), think before you train.** Without `wpe`, attention sees a
bag of tokens (the causal mask still leaks *some* order). Now note which
board facts are order-invariant: which cells are occupied is determined by
the multiset of moves alone — but who *owns* each cell depends on whether the
move was played 1st or 2nd or 3rd. Predict which eval metrics survive
(legality?) and which collapse (result prediction, strength), then check.

## 4. Temperature sweep vs playing strength — *30 minutes*

**Motivation.** `docs/06-inference.md` explains temperature; here you put a
number on it. `minillm/evaluate.py` plays with `model_move_strict` — a pure
argmax, i.e. temperature 0. `minillm/play.py` defaults to `--temperature 0.7`.
Someone is leaving strength on the table. Who?

**Task.** Add a `--temperature` flag to `evaluate.py` that makes
`model_move_strict` sample instead of argmax (borrow the `sample()` closure
from `model_move` in `play.py`). Sweep 0, 0.3, 0.7, 1.0, 1.5 and tabulate
`win_rate` vs random and `draw_rate` vs the optimal solver.

**Files.** `minillm/evaluate.py` only.

**Hint.** Expect a monotone story for strength and the opposite story for
variety: at temperature 0 the model plays its single favourite game over and
over. Cross-check with `python -m minillm.sample --temperature 1.5` — watch
`verify_transcript` verdicts deteriorate as sampling gets hotter.

> **In a real LLM:** the same tension is why chat products expose temperature
> at all — greedy decoding is strongest on tasks with one right answer, but
> collapses diversity, and at scale it produces degenerate repetition.
> Production systems layer top-k / top-p truncation on top (top-k is already
> implemented in `GPT.generate`; top-p is not — adding it is a nice bonus
> exercise) for exactly the failure mode your
> temperature-1.5 samples will show.

## 5. Implement a KV cache in `generate()` — *a weekend*

**Motivation.** `minillm/model.py` says it out loud: the attention is "the
naive, readable one (no FlashAttention, no KV cache)". Each of the up-to-12
steps in `GPT.generate` re-runs the *entire* prefix through all four blocks —
O(T²) work per token where O(T) suffices. At T ≤ 12 nobody cares; the point
is that after this exercise you will never again nod vaguely when someone
says "KV cache".

**Task.** Give `CausalSelfAttention.forward` an optional cache: on the first
call store `k` and `v` (per layer); on later calls feed only the *new* token,
compute its `q, k, v`, append `k, v` to the cache, and attend against the full
cached keys/values. Then rewrite the loop in `GPT.generate` to pass single
tokens. Correctness gate: greedy generation (`temperature=0`) from the same
prompt must produce token-for-token identical output with and without cache.

**Files.** `minillm/model.py` (`CausalSelfAttention.forward`, `GPT.forward`,
`GPT.generate`).

**Hint.** Two classic bugs await. First, the position embedding: a token fed
alone is still at absolute position `T_so_far`, so `torch.arange(T)` in
`GPT.forward` must become an offset. Second, the causal mask: a single query
attending to all cached keys needs no mask at all — masking it anyway is the
other classic bug. Verify with `torch.allclose` on logits at every step
before trusting sampled output.

> **In a real LLM:** the KV cache is not an optimization, it is *the*
> serving-cost driver. For a 100k-token context, recomputing the prefix per
> token would be ruinously quadratic; instead inference engines keep gigabytes
> of cached keys and values per request, and techniques like grouped-query
> attention (Llama 2 70B onward), multi-query attention, and paged attention
> (vLLM) exist mainly to shrink or manage this cache.

## 6. Plot the loss curves — *30 minutes*

**Motivation.** `runs/finetune/log.csv` contains a textbook drama that a
single "best val loss" number hides completely.

**Task.** Plot `train_loss` and `val_loss` against `step` for both
`runs/pretrain/log.csv` and `runs/finetune/log.csv`
(`pip install matplotlib` into `.venv` — it is deliberately not in
`requirements.txt`). Then explain, in one paragraph: in the finetune run,
validation loss bottoms out at **0.4771 at step 100** and climbs back to
0.659 by step 1499 while train loss falls to 0.340. Why is the shipped
checkpoint still good? Find the exact line in `minillm/train.py` that saved
the day (look for `if val_loss < best_val`).

**Files.** New plotting script anywhere you like; read-only on `runs/`.

**Hint.** The gap has a size explanation: finetuning sees 334 expert games
minus the val split, and the loss mask in `build_tensors` throws away the
opponent-move targets on top — count the trainable target tokens the training
banner printed. Overfitting is not a bug at this corpus size; *not
checkpointing on val* would have been the bug.

## 7. A lookup-table baseline: memorization vs generalization — *an afternoon*

**Motivation.** The nastiest question you can ask any trained model: would a
hash map have done just as well? With 1,310 total games (`data/meta.json`)
sharing heavy prefix overlap, the suspicion is legitimate.

**Task.** Build a no-learning baseline: iterate over the training-split games
(reuse `split_games` from `minillm/dataset.py` — same default seed 42 and
`val_frac=0.1` as training, so the split matches), and record, for every
transcript prefix, the observed distribution of next tokens. At eval time,
predict from the table; when a prefix was never seen, fall back to uniform
over legal moves. Then run this baseline through the same measurements the repo already
makes — argmax-legal rate (`eval_on_val_games`) and optimal-move rate
(`eval_expert_agreement`) — plus one you add yourself: prefix coverage on
held-out games. The gap between the table and the network — on
exactly the prefixes the table has *never seen* — is a measured quantity of
generalization.

**Files.** New script (suggested: `minillm/baseline_lookup.py`); read-only
everywhere else.

**Hint.** Key the table on `tuple(moves_so_far)` with a
`collections.Counter` as value. Report the never-seen-prefix rate separately
— that subset is where the comparison gets interesting, because there the
table is guessing uniformly and the network is not.

> **In a real LLM:** this is the memorization-vs-generalization debate at lab
> scale. n-gram language models *were* this lookup table, smoothing tricks
> included, and they ruled for decades; the reason transformers displaced
> them is exactly what your never-seen-prefix column will show. The same
> methodology — probe on data provably absent from training — is how
> memorization and data-contamination studies are run on production models.

## 8. Attention-head taxonomy: catalogue all 16 heads — *an afternoon*

**Motivation.** The model has 4 layers × 4 heads = 16 attention heads, and
`minillm/inspect_attention.py` prints every one of them. Its docstring
promises three species: previous-move heads, same-column (stack-height)
heads, and `<bos>`-sink heads. Nobody has checked whether the promise holds
for your checkpoint. That is now your job.

**Task.** Run
`python -m minillm.inspect_attention --moves "B1 A1 B2 C1 B3"` (and at least
two other prefixes) and build a 16-row table: layer, head, dominant pattern,
confidence, evidence. Every head gets a row, including the boring ones —
"diffuse / no clear role" is a legitimate finding, and at this scale probably
a frequent one.

**Files.** None to modify; `--layer` and `--head` flags narrow the output.

**Hint.** One prefix cannot separate a "looks 2 positions back" head from a
"looks at the same column" head — in `B1 A1 B2` those coincide. Choose
prefixes that break the tie, e.g. compare `"B1 A1 B2"` with `"B1 C1 A1 B2"`:
a positional head keeps its offset, a column-tracking head follows the B's.
Also compare the pretrain and finetune checkpoints (`--ckpt
runs/pretrain/model.pt`) — did finetuning repurpose any head?

## 9. Scale the world: 4×4 Connect-3 — *a weekend*

**Motivation.** Every constant in this repo is downstream of `game.py`'s
`N = 3` and `COLS = "ABC"`. Widening the board to 4×4 (win = 3 in a row)
forces you to touch the entire pipeline in order — the best possible proof
that you understand how the pieces connect.

**What must change, in dependency order.**

- `minillm/game.py`: `COLS = "ABCD"`, `N = 4`, and — the real work — `LINES`.
  It is derived from `N`, but only generates *full-length* lines (at `N = 4`:
  4 verticals + 4 horizontals + 2 diagonals); with win-length 3 on a 4×4
  board you need every 3-cell segment: horizontal, vertical, and both
  diagonal directions. Also unhardcode `"123"` in `push()`, the
  `Game.stacks` default_factory (three empty strings), the `render()` footer
  (`"   +------"` / `"     A B C"`), and the `height >= 3` check in
  `play.py`'s `read_human_move`.
- `minillm/tokenizer.py`: `MOVE_TOKENS` grows to 16 automatically (it is
  computed from `COLS` and `N`), vocab 15 → 22, `MAX_GAME_TOKENS` 12 → 19.
- `minillm/config.py`: `block_size` 16 → at least 19 (round up to 20 or 24).
- `minillm/solver.py`: one small edit — `EMPTY` hardcodes three column
  stacks (`("", "", "")`) and the `State` alias is `tuple[str, str, str]`;
  make it a 4-tuple, ideally derived from `N`
  (`tuple("" for _ in range(N))`). The algorithms are otherwise unchanged —
  but check
  `negamax.cache_info().currsize` (694 positions today) after a solve; it
  will grow a lot and still fit in memory.

**The trap, and the actual lesson.** `enumerate_all_games` is a full DFS over
the game tree — fine at 1,310 games, but at branching factor ≤ 4 and depth
≤ 16 the 4×4 tree has potentially billions of leaves. Complete enumeration
dies. You must replace "pretrain on *everything*" with "pretrain on a
*sample*" (random rollouts are the simplest corpus generator), which changes
the epistemics of the whole project: held-out games may now contain genuinely
unseen positions, so your eval numbers finally mean what real-LLM eval
numbers mean.

**Hint.** Do it in two commits: first make `make test` pass with the new
constants (the tests in `tests/` will point at every hardcoded 3), then
replace the enumerator. Ask the solver for the new root value —
`describe_root_value()` will tell you whether 4×4 Connect-3 is still a draw.
Do not trust your intuition; that is what the solver is for.

## 10. An RL stage: REINFORCE self-play after SFT — *a weekend*

**Motivation.** The pipeline stops at SFT, and the eval table shows the
ceiling: 86.5% optimal-move rate, 39% losses against the solver. Imitation
can only be as good as its data and its coverage. The missing third stage of
the modern pipeline is reinforcement learning: let the model play, and reward
*outcomes* instead of imitating *moves*.

**Task.** Sketch (and, if brave, implement in a new `minillm/rl.py`) a
REINFORCE loop on top of the finetuned checkpoint:

1. Roll out a batch of games — model vs a frozen opponent (random, a frozen
   copy of itself, or the solver at your peril), sampling with
   `temperature > 0` so there is exploration to reinforce.
2. Score each finished game with the *engine*: +1 win, 0 draw, −1 loss from
   the model's side. `game.winner()` is the referee — never the model's own
   result token.
3. Loss = `−(reward − baseline) · Σ log π(move)` summed over the *model's*
   moves only — the same masking discipline `build_tensors` applies with
   `expert_only=True`, now applied to the policy gradient. A running mean
   of recent rewards is a sufficient baseline.
4. Small learning rate (start below finetuning's 2e-4), and evaluate with
   `minillm/evaluate.py` every few hundred games — vs both opponents, not
   just the training one.

**Reward hacking — read before training.** REINFORCE maximizes whatever you
measure, not what you meant. Concrete failure modes at this scale: reward the
model for *emitting* `#X` and it will learn to claim victory mid-game
(exactly the transcript crime `sample.verify_transcript` flags as "result
claimed while the game was still running"); train only against a random
opponent and it will learn traps that lose to the solver; skip the strict
legal-move projection and it can learn that illegal tokens end bad games
early. Keep the engine as the sole source of reward, and keep an eval the
policy is *not* trained on.

**Hint.** You need per-move log-probabilities from the rollout. Simplest
correct approach: replay each finished game through one teacher-forced
forward pass called *with targets* (`model(x, targets=y)` — dummy targets
work; without `targets`, `GPT.forward` takes its inference shortcut and
returns only the last position's logits),
take `log_softmax`, and gather the logits at the model's move positions.
That is one batched forward instead of bookkeeping during generation.

> **In a real LLM:** this is the RLHF/RLAIF stage that turns an SFT model
> into an assistant — with one luxury you have that production never does:
> a perfect reward function. Real pipelines must first *train a reward model*
> on human preference data, and reward hacking stops being a thought
> experiment: policies learn sycophancy, verbose padding, and confident
> fabrication because the reward model scores them well. Your engine referee
> is the ground truth those pipelines can only approximate — which is why
> your RL run will be the easy version, and why it is still worth doing.

---

## Where to go from here

If you climbed the whole ladder you have touched every file in `minillm/` and
rebuilt every stage of the pipeline at least once. The repo is now yours:
the same scaffolding — enumerable world, exact solver, behavioural evals —
carries any small combinatorial game you care about.

Exercise 1 has a worked lab report with measured results in
[09 — the character-tokenizer lab](09-char-tokenizer-lab.md) — read it
*after* your own attempt; it spoils the fun otherwise. And when you wonder
why all of this runs on a CPU while the real thing needs a datacenter,
[10 — Why GPUs?](10-gpu-cuda.md) connects this repo's matmuls to the
hardware story (CUDA included).

Next: back to [00 — Overview](00-overview.md), which reads differently the
second time.
