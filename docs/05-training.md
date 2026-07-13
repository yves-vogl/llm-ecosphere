# 05 — Training: pretraining and finetuning

Everything before this chapter was setup: a solved game, two corpora, a
15-token vocabulary, and a 797,312-parameter GPT that so far contains only
random numbers. This chapter is where those numbers change. The file to read
alongside is `minillm/train.py` — the whole loop is ~220 lines, and every one
of them corresponds to something a frontier-model training run also does.

Two stages, selected with `--stage`:

| Stage | Corpus | Steps | Peak LR | Analogy |
|---|---|---|---|---|
| `pretrain` | `data/all_games.jsonl` (all 1,310 games) | 3000 | 1e-3 | base-model pretraining |
| `finetune` | `data/expert_games.jsonl` (334 solver games) | 1500 | 2e-4 | supervised finetuning (SFT) |

The defaults live in one small table at the top of `train.py`:

```python
STAGE_DEFAULTS = {
    #             steps    lr      corpus file
    "pretrain": (3000, 1e-3, "all_games.jsonl"),
    "finetune": (1500, 2e-4, "expert_games.jsonl"),
}
```

## The objective: next-token prediction with teacher forcing

The model is trained on exactly one task: given tokens 0..t of a game
sequence, predict token t+1. `build_tensors` in `minillm/dataset.py` sets this
up by making `y` a copy of `x` shifted one position left — "position t of x
sees tokens 0..t and must predict y[t] = token t+1", as its docstring puts it.

Note what is *not* happening: the model never plays a game during training. It
never sees its own predictions. At every position it is conditioned on the
*true* prefix from the corpus, predicts one token, gets scored, and moves on.
This is **teacher forcing**, and it is why training is a single parallel
forward pass over all 16 positions instead of a 16-step sequential rollout.
The causal mask inside the attention layers (chapter 04) is what makes this
legal: position t physically cannot see tokens t+1..15, so all 16 predictions
of a sequence can be computed — and scored — simultaneously.

The mismatch this creates (training always conditions on correct prefixes;
generation conditions on the model's own, possibly wrong, prefixes) is called
exposure bias. You will see its footprint in chapter 07: teacher-forced
legality is 100%, free-running legality is slightly lower.

> **In a real LLM:** identical. GPT-2, GPT-3, Llama and Claude are all trained
> with teacher-forced next-token prediction. The entire pretraining phase of a
> frontier model is this one objective applied to trillions of tokens — no
> game-playing, no reinforcement, no explicit "understanding" target. Every
> capability the base model has falls out of getting better at this one score.

## Cross-entropy, and what 2.71 means

The loss is plain cross-entropy over the vocabulary, computed in
`GPT.forward` (`minillm/model.py`):

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
)
```

Cross-entropy at one position is `-log p(correct token)` — the negative log of
the probability the model assigned to what actually came next. It is measured
in *nats* (natural-log units). Two anchor points make the raw numbers legible:

- **A clueless model.** If the model spreads probability uniformly over all 15
  tokens, every correct token gets p = 1/15, and the loss is
  `-ln(1/15) = ln 15 ≈ 2.708`. That is the score for knowing nothing at all.
- **An omniscient model.** Loss 0 would mean p = 1.0 on the correct token at
  every position — as we will see below, that is not merely hard here, it is
  mathematically impossible.

Now the real numbers. First and last logged rows of `runs/pretrain/log.csv`:

```
step,lr,train_loss,val_loss
0,1e-05,2.8151355380811625,2.8154780864715576
...
2999,0.0001000002640500325,0.7074194437575444,0.7643032073974609
```

Step 0 sits at 2.815 — essentially the clueless baseline of 2.708, plus a
little, because randomly initialized logits are not exactly uniform (random
weights produce small arbitrary preferences, and an arbitrary preference is
slightly *worse* than no preference). Within the first 100 steps the loss
collapses to ~1.03: the model has learned the cheap structure — the token
after `<bos>` is always a move, and sequences end with a result token then
`<eos>` (padding after `<eos>` is masked out of the loss entirely). The
remaining 2,900 steps grind out the expensive structure: which
of the ≤3 columns is actually played, and when. The best validation loss,
0.7506, occurs at step 1700; the checkpoint kept on disk is from that step,
not from step 2999 (see best-val checkpointing below).

### Perplexity aside

Perplexity is `exp(loss)` — the loss re-expressed as "the model is as
uncertain as if it were choosing uniformly among *this many* options."

| Loss | Perplexity | Reading |
|---|---|---|
| 2.815 (init) | 16.7 | worse than uniform over the 15-token vocab |
| 0.7506 (pretrain best val) | 2.12 | ~2.1 effective choices per token |
| 0.4771 (finetune best val) | 1.61 | ~1.6 effective choices per token |

Perplexity ~2.1 after pretraining is remarkably close to the floor: a typical
mid-game position has 2–3 legal columns, and the corpus plays all of them.
The model cannot know *which* branch this particular training game takes —
nobody could.

> **In a real LLM:** the same anchors apply, scaled up. With a ~50k-token
> vocabulary, a clueless model starts near ln(50,000) ≈ 10.8 nats; GPT-2-scale
> models land in the 3–4 range on held-out web text, frontier models lower
> still. Scaling-law papers (Kaplan 2020, Chinchilla 2022) are precisely
> studies of how this one number falls as a smooth power law in parameters,
> data, and compute — and, exactly as here, it asymptotes toward the
> irreducible entropy of the text distribution, never zero.

## Irreducible entropy: why loss 0 is impossible

This point deserves its own section because it reframes what the loss curve
means. The pretraining corpus is not 1,310 samples from some deterministic
function — it is the *complete enumeration of a branching tree*. After the
prefix `<bos> B1`, the corpus contains continuation games through A1, B2, and
C1. All three follow that identical prefix. No model, however large, can put
probability 1 on all three at once; the best possible model matches the
empirical branch frequencies exactly and still pays the entropy of the
distribution at every fork.

So the "perfect" pretraining loss is not 0 — it is the average branch entropy
of the game tree, which is why train loss flatlines around 0.707 and simply
refuses to go lower. That plateau is not a failure to converge. It is the
model hitting the information-theoretic floor of the task. (The finetune
loss can fall further because expert positions branch only over the usually
small set of solver-optimal moves — built with `ties="all"`, so genuine ties
remain — while opponent-move targets are masked out of the loss entirely.)

## Memorization vs generalization in a 1,310-game world

`split_games` in `minillm/dataset.py` holds out 10% of games (131 of 1,310,
via `--val-frac 0.1`) with a deterministic shuffle (`random.Random(seed)`,
seed 42). Its docstring is honest about the caveat: "In a world this small
most *positions* still occur in some training game via shared prefixes."
A held-out game's first five moves almost certainly appear inside some
training game; only the specific full trajectory is new.

So what does val loss actually tell you here?

- **What it can tell you:** whether the model learned position-conditional
  *rules* rather than a lookup table of specific sequences. A pure memorizer
  would ace train games and produce garbage probabilities on held-out
  trajectories; the observed gap (train 0.707 vs val 0.764 at the end of
  pretraining) is small, so the model is genuinely modeling the game, not the
  file.
- **What it cannot tell you:** anything about truly out-of-distribution
  behavior — there is no out-of-distribution. The training set contains every
  legal game. This is the fundamental un-realism of the lab: real LLMs
  generalize to prompts nothing like their training data; ours generalizes
  only across an exhaustively covered space. Chapter 08's exercises poke at
  this boundary.

> **In a real LLM:** the same tension exists but flipped. Web-scale corpora
> are so large that no model can memorize them, yet held-out "validation"
> documents still share phrases, templates, and near-duplicates with training
> data — which is why serious evaluations go beyond val loss to contamination-
> checked benchmarks. And memorization is not hypothetical at scale: models
> demonstrably memorize rare sequences seen a handful of times, which becomes
> a privacy problem, not just a metrics one.

## The optimizer: AdamW, grouped weight decay

One line creates it — `optimizer = model.configure_optimizer(args.weight_decay, max_lr)` —
but two decisions hide inside `GPT.configure_optimizer` (`minillm/model.py`).

**AdamW** maintains, for every one of the 797,312 parameters, a running mean
of its gradient (momentum, `beta1=0.9`) and a running mean of its squared
gradient (`beta2=0.95`). The update divides the momentum by the square root of
the second: parameters with consistently large gradients get smaller steps,
rarely-touched parameters get relatively larger ones. That per-parameter
adaptive scaling is why AdamW tolerates a single global learning rate across
embeddings, attention matrices, and LayerNorm gains that receive wildly
different gradient magnitudes — and why it, not plain SGD, is the default for
Transformers.

**Weight decay** (`--weight-decay 0.1`) pulls parameters toward zero each
step — a regularizer. But it is only applied to some parameters:

```python
decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
```

Every 2-D parameter (embedding tables, attention and MLP weight matrices) is
decayed; every 1-D parameter (all biases, all LayerNorm gains) is not. The
docstring gives the why: decay toward zero is "a sensible prior for big matmul
matrices, but harmful for LayerNorm gains (should sit near 1) and biases."
A LayerNorm gain dragged toward 0 would progressively mute its layer's output;
there is no regularization benefit to compensate, since 1-D parameters are a
negligible fraction of capacity anyway.

## Learning-rate schedule: warmup then cosine

The learning rate is not a constant; `lr_at` in `train.py` recomputes it every
step and writes it into the optimizer's param groups:

```python
if step < warmup:
    return max_lr * (step + 1) / warmup
progress = (step - warmup) / max(1, max_steps - warmup)
return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Linear **warmup** over the first 100 steps (`--warmup 100`) ramps from ~0 to
the peak. Early in training, gradients are large and AdamW's running averages
are uncalibrated; full-size steps then can fling the weights somewhere hard to
recover from. After warmup, a **cosine decay** glides from `max_lr` down to
`min_lr = max_lr * 0.1` (set in `main`) — large steps while far from a
minimum, small steps to settle into one instead of orbiting it. You can watch
it in `log.csv`: the `lr` column reads 1e-05 at step 0, 0.001 at step 100,
and 0.0001000 at step 2999.

**Gradient clipping** is the last safety net before each update:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

If the global gradient norm exceeds `--grad-clip 1.0`, the whole gradient is
rescaled to norm 1.0 — direction preserved, magnitude capped. A single freak
batch cannot blow up the run.

> **In a real LLM:** warmup + cosine + clipping is *the* recipe, essentially
> unchanged since GPT-2/GPT-3 (which also used AdamW betas of 0.9/0.95, the
> same values in `configure_optimizer`). At scale the stakes are higher —
> a loss spike 40% into a month-long run on thousands of GPUs is an incident,
> and teams respond by skipping bad data batches or rolling back checkpoints.
> Warmup and clipping exist precisely to make those events rarer.

## Evaluation inside the loop: exact loss, best-val checkpointing

Every `--eval-interval 100` steps, the loop calls `full_split_loss` on both
splits. Two details matter:

1. **It is exact, not sampled.** The dataset is tiny, so instead of estimating
   val loss from a random batch (the necessity at scale), it iterates over the
   *entire* split in chunks and re-weights each chunk's mean by its count of
   real targets — "the true mean over all predicted tokens, not a mean of
   chunk means," per the docstring. Positions whose target is `-1` (padding,
   and masked opponent moves in finetuning) are excluded from the count.
2. **The checkpoint on disk is always the best-so-far by val loss.** Only when
   `val_loss < best_val` does the loop call `torch.save(...)` on
   `model.checkpoint_dict(...)` — you can see it happen live as the
   `<- saved` marker in the console output. Training may continue past the
   best point (and in finetuning it very much does), but `runs/<stage>/model.pt`
   never gets worse.

This second detail is a free, brutally effective form of early stopping, and
the finetune log below is the demonstration of why it exists.

## The finetune stage

`--stage finetune` changes four things and nothing else:

1. **Initialization.** Instead of fresh weights, the model is loaded from
   `runs/pretrain/model.pt` (override with `--init-from`). Architecture flags
   are ignored — "finetune inherits the checkpoint's" config, reconstructed
   from the `config` dict stored inside the checkpoint. If the checkpoint is
   missing, `train.py` fails fast with "run `make pretrain` before
   `make finetune`".
2. **Data.** 334 expert games where the solver plays one side perfectly
   against every possible opponent reply (`data/meta.json`:
   `"n_expert_games": 334`).
3. **Masked loss.** `expert_only = args.stage == "finetune"` flows into
   `build_tensors`, which leaves the target at `-1` for every *opponent* move
   — `cross_entropy(..., ignore_index=-1)` then skips those positions
   entirely. The model still *reads* opponent moves as context (they are in
   `x`), it just is not trained to *produce* them. Only the solver's moves are
   imitated.
4. **A gentler schedule.** 1500 steps at peak LR 2e-4 — five times lower than
   pretraining. The weights already encode the rules of the game; the point is
   to shift the move distribution toward expert play without bulldozing that
   knowledge. A large LR here would cause exactly the catastrophic forgetting
   the lower LR avoids.

The numbers from `runs/finetune/log.csv` tell a textbook overfitting story:

```
step,lr,train_loss,val_loss
0,2e-06,0.606268584728241,0.6221192479133606
100,0.0002,0.43748903274536133,0.47713005542755127
200,0.00019774351209636413,0.3760433793067932,0.4984547197818756
...
1499,2.0000226597965142e-05,0.33990639448165894,0.6590251326560974
```

Val loss bottoms out at **0.4771 at step 100** and then climbs for the
remaining 1,400 steps, ending at 0.659 — while train loss keeps falling to
0.340. With only ~300 training games, the model starts memorizing solver
trajectories instead of the solver's *policy*, and its probabilities on the 33
held-out expert games degrade. The best-val checkpointing rule is what saves
the run: `runs/finetune/model.pt` is the step-100 model, and everything after
step 100 is, in hindsight, wasted compute that cost nothing because it never
touched the saved checkpoint.

> **In a real LLM:** this stage *is* SFT. Chat models are finetuned on curated
> conversations with the loss masked to assistant turns only — the user's
> messages are context, never targets — which is byte-for-byte the
> `expert_only` trick in `build_tensors`. The lower learning rate, the init
> from the base checkpoint, and the small-corpus overfitting risk all carry
> over directly. Production SFT runs are similarly short and similarly prone
> to val-loss U-curves.

### The measured alignment tax

Finetuning bought skill and paid for it in reliability. Compare
`runs/eval_pretrain.json` and `runs/eval.json` (full analysis in chapter 07):

| Metric | Pretrain | Finetune |
|---|---|---|
| Teacher-forced argmax legality | 1.000 | 0.9953 |
| Free-running clean-game rate | 0.980 | 0.905 |
| Win rate vs random | 0.4175 | 0.7925 |
| Draw rate vs optimal | 0.00 | 0.61 |
| Solver move agreement | 0.7029 | 0.8647 |

Legality — the "grammar" pretraining had mastered at 100% — slipped to 99.5%
after finetuning, while playing strength improved sharply on every measure — win
rate vs random nearly doubled, draws against the optimal player went from 0%
to 61%, and solver move agreement rose from 70% to 86%.
Narrowing the model's distribution onto expert lines slightly eroded its
command of rarely-expert regions of the game. This trade-off has a name in
the LLM literature: the **alignment tax** — base-model capabilities that get
slightly worse when the model is specialized toward preferred behavior. Here
it is small, quantified, and reproducible, which is exactly what makes it a
good specimen.

## Reproducibility

`set_seed(args.seed)` (`minillm/utils.py`) seeds Python's `random` and
`torch.manual_seed` with `--seed 1337` before anything else happens; the
train/val split uses its own fixed seed 42 inside `split_games`, so changing
`--seed` reshuffles batch sampling and dropout but not which games are held
out. On the same machine and PyTorch build, two runs with the same flags
produce the same `log.csv`. (Bit-exact reproducibility across different
hardware is a harder promise — floating-point reduction order differs across
devices — which is why the numbers quoted in these docs come from the actual
run in `runs/`, not from a spec.)

## CLI tour of train.py

All flags, with defaults:

| Flag | Default | Meaning |
|---|---|---|
| `--stage` | (required) | `pretrain` or `finetune`; picks corpus, steps, LR defaults |
| `--data-dir` | `data` | where the JSONL corpora live |
| `--out-dir` | `runs/<stage>` | where `model.pt` and `log.csv` go |
| `--steps` | 3000 / 1500 | optimizer steps |
| `--batch-size` | 64 | games per step, sampled with replacement |
| `--lr` | 1e-3 / 2e-4 | peak learning rate (min is peak x 0.1) |
| `--warmup` | 100 | linear warmup steps |
| `--weight-decay` | 0.1 | applied to ndim >= 2 parameters only |
| `--grad-clip` | 1.0 | global gradient-norm cap |
| `--val-frac` | 0.1 | fraction of games held out |
| `--eval-interval` | 100 | steps between exact split evaluations |
| `--seed` | 1337 | RNG seed for batching/dropout |
| `--device` | `cpu` | `cpu` / `mps` / `cuda` / `auto` (CPU is fine at this size) |
| `--init-from` | — | finetune-only; defaults to `runs/pretrain/model.pt` |
| `--n-layer` / `--n-head` / `--n-embd` / `--block-size` / `--dropout` | 4 / 4 / 128 / 16 / 0.1 | architecture (pretrain only; finetune inherits the checkpoint's) |

One detail worth noting from the loop itself: batches are drawn with
`torch.randint` — sampling games *with replacement* rather than shuffling
epochs, because as the comment says it is "simpler than epoch bookkeeping and
statistically equivalent at this scale." At 3,000 steps x 64 games over 1,179
training games, each game is seen ~163 times either way.

Next: [Inference: sampling and playing](06-inference.md) — how a trained
checkpoint turns logits into moves, and what temperature and top-k actually do.
