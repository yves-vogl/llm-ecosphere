# Lab report: the mirror dedup

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 2](08-exercises.md#2-mirror-symmetry-augmentation-in-reverse).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

Drop-Tac-Toe is symmetric under reflecting the board across its vertical
axis: swap columns A and C, leave B alone, and every legal game stays a
legal game with the same result. The classic ML move — augment the corpus
with mirrored copies — is a no-op here, because `enumerate_all_games`
already emits *every* game, mirrors included. So exercise 2 runs the
experiment backwards: **delete** the redundancy. Keep one game per mirror
pair, pretrain on the half-sized corpus, and see what survives. If the
model plays mirrored positions as well as originals it never saw even in
mirrored form, it is generalizing across the symmetry; if it needed both
halves, it was memorizing them separately.

## The filter

`dedup_mirror_games` in `minillm/dataset.py` (with helpers `mirror_move`
and `mirror_game`) keeps a game exactly when its move sequence is
lexicographically no larger than its mirror's:

```python
MIRROR_COLS = dict(zip(COLS, reversed(COLS)))       # A<->C, B->B

def mirror_move(move: str) -> str:
    return MIRROR_COLS[move[0]] + move[1]           # rows are untouched

def dedup_mirror_games(games: list[dict]) -> list[dict]:
    return [g for g in games if g["moves"] <= mirror_game(g)["moves"]]
```

Because the corpora are mirror-closed (the mirror of an optimal move is
optimal, and both enumerators branch over everything), every discarded
game's partner is guaranteed to survive, so the output is exactly half:
1,310 → 655 complete games, 334 → 167 expert games. No complete game in
this corpus is its own mirror (that would need every move in column B,
which holds only three pieces), but the `<=` keeps such a game once
anyway. `tests/test_dataset.py` pins all of this down.

The flag `--dedup-mirror` on `minillm.dataset` writes the filtered
corpora, so the experiment is four commands away from the standard
pipeline:

```bash
.venv/bin/python -m minillm.dataset --out data-mirror --dedup-mirror
.venv/bin/python -m minillm.train --stage pretrain \
    --data-dir data-mirror --out-dir runs/exp-mirror-pretrain
.venv/bin/python -m minillm.train --stage finetune \
    --init-from runs/exp-mirror-pretrain/model.pt \
    --data-dir data --out-dir runs/exp-mirror-finetune
.venv/bin/python -m minillm.evaluate --ckpt runs/exp-mirror-pretrain/model.pt \
    --data-dir data-mirror --out runs/exp-mirror-pretrain/eval.json  # and finetune alike
```

Finetuning deliberately uses the *full* expert corpus, so the only
variable under test is what pretraining saw. One honest caveat on the
teacher-forced rows below: the deduped corpus has its own train/val split
(seed 42 on 655 games), so "held-out positions" is a different — and
harder — set than the baseline's: for a deduped-val game, *neither* the
game *nor its mirror* was trainable. All the free-running and strength
metrics are corpus-independent and compare directly.

## Results

Same seeds, same eval protocol as the reference runs. Baselines from
`runs/eval_pretrain.json` / `runs/eval.json`:

| metric | pretrain (full) | pretrain (deduped) | finetune (full) | finetune (deduped pretrain) |
|---|---:|---:|---:|---:|
| argmax legal (teacher-forced) | 100.0% | 99.1% | 99.5% | 99.6% |
| legal probability mass | 99.6% | 98.2% | 99.1% | 97.6% |
| free-running 1st-try legal | 99.8% | 98.7% | 98.8% | 95.4% |
| clean self-play games | 98.0% | 91.0% | 90.5% | **70.0%** |
| result prediction | 99.2% | 93.8% | 100.0% | 93.8% |
| vs random W/D/L | 41.8/20.2/38.0% | 28.2/23.5/48.2% | 79.2/14.5/6.2% | 75.0/15.5/9.5% |
| vs optimal solver W/D/L | 0/0/100% | 0/0/100% | 0/61/39% | 0/**63.5**/36.5% |
| optimal-move rate | 70.3% | 74.6% | 86.5% | 85.0% |

## A measurement the eval suite does not have: symmetry consistency

The question the exercise actually asks — "is the model generalizing
across the symmetry?" — has a sharper probe than aggregate legality: does
the *policy commute with mirroring*? For every position visited by the
same seeded random rollouts `eval_expert_agreement` uses (414 positions),
ask for the strict argmax move on the history `h` and on `mirror(h)`, and
check `argmax(mirror(h)) == mirror(argmax(h))`. A policy that has truly
internalized the symmetry scores ~100% (exact floating-point ties are the
only excuse); independent memorization of the two halves does not.

```python
import random

from minillm.dataset import mirror_move
from minillm.evaluate import model_move_strict
from minillm.game import Game
from minillm.utils import load_model, pick_device, tokenizer_for_checkpoint

def symmetry_consistency(ckpt_path, n_rollouts=300, seed=0):
    device = pick_device("cpu")
    model, ckpt = load_model(ckpt_path, device)
    tokenizer = tokenizer_for_checkpoint(ckpt)
    rng, histories = random.Random(seed), {}
    for _ in range(n_rollouts):
        game = Game()
        while not game.is_over():
            histories.setdefault(tuple(game.stacks), list(game.history))
            game.push(rng.choice(game.legal_moves()))
    agree = 0
    for h in histories.values():
        move = model_move_strict(model, tokenizer, Game.from_moves(h), device)
        mirrored = model_move_strict(
            model, tokenizer, Game.from_moves([mirror_move(m) for m in h]), device)
        agree += mirrored == mirror_move(move)
    return agree / len(histories), len(histories)
```

| checkpoint | mirror-consistent positions |
|---|---:|
| pretrain, full corpus | 81.2% |
| finetune, full corpus | 82.1% |
| pretrain, mirror-deduped | 60.1% |
| finetune from deduped pretrain | 68.1% |

## Reading the numbers

**1. The headline: the model was memorizing both halves.** The sharp
probe answers first. Symmetry consistency *drops* when the corpus is
deduplicated — 81.2% → 60.1% at pretrain, 82.1% → 68.1% after
finetuning. If the full-corpus model had internalized the A↔C
invariance, deleting the redundant mirrors would have cost nothing;
instead, the policy of the deduped model is visibly *handed*, favouring
the lexicographically-smaller half it was trained on. Which means the
original ~81% consistency was never an internalized symmetry at all: it
was the data's symmetry, reflected. Every mirrored pair taught both
halves separately, and behavioural symmetry emerged from double
memorization. The exercise's proposition — "if legality and strength
survive, the model is genuinely generalizing across the symmetry" —
resolves against the model on the symmetry half.

**2. What does survive: the rules, and (after SFT) the strength.**
Legality barely moves (99.1% argmax-legal, 98.7% free-running first-try)
— gravity is a local, column-wise pattern and every column still appears
plentifully in the kept half. And the finetuned-from-deduped model plays
nearly as well as the reference: 75.0% wins vs random (79.2 reference),
63.5% draws against the perfect solver (61.0), 85.0% optimal moves
(86.5). The finetuning stage used the full, mirror-closed expert corpus
in both runs, so this is the SFT data doing symmetric repair work on an
asymmetric prior — and mostly succeeding, at least on the strength
metrics. (The pretrain-stage vs-random dip to 28.2% wins and the bump to
74.6% optimal moves are two faces of the same fact: "imitate the average
game" now means imitating a *biased* average; the multi-seed lesson from
the char lab applies to reading too much into either number.)

**3. The costs concentrate where coverage is the product: fluency and
refereeing.** Clean self-play games fall from 98.0% to 91.0% at
pretrain and — the largest single delta in the table — from 90.5% to
70.0% after finetuning; result prediction drops to 93.8% on both
checkpoints. Free-running rollouts and full-game refereeing are exactly
the tasks that wander across the *whole* game space, mirrored half
included, and there the deduped model is a partial stranger. The
comparable val losses agree: finetuning from the deduped prior bottoms
out at 0.5835 against the reference 0.4771 on the *same* expert corpus
and split — the prior lost half the world, and SFT does not fully buy
it back.

**The verdict on "augmentation in reverse".** In an enumerated, closed
world the mirror games were never redundant duplicates — they were the
training signal for the symmetry itself. Halving the corpus kept the
rules and (with symmetric SFT) most of the strength, but the invariance
the duplicates encoded went with them.

> **In a real LLM:** the production twin of this experiment is corpus
> deduplication. GPT-3's and Llama's training sets are aggressively
> deduplicated because near-duplicate documents inflate memorization and
> quietly contaminate held-out evaluation — and the lesson generalizes:
> what a model does on data it *provably never saw* (here: the discarded
> mirror halves) is the only clean measurement of generalization.
> Symmetry itself has the same double life at scale: real corpora encode
> "translational" redundancies (paraphrases, translations, code with
> renamed variables), and whether a model treats them as one fact or as
> many is exactly the memorization-vs-generalization axis this miniature
> makes measurable.

## Reproduce it

```bash
make test                       # includes the mirror/dedup invariants
.venv/bin/python -m minillm.dataset --out data-mirror --dedup-mirror
# then the train/evaluate commands above; the symmetry_consistency
# snippet runs as-is from the repo root.
```

Next: back to the [exercises](08-exercises.md), or the
[char-tokenizer lab](09-char-tokenizer-lab.md) for the same
"one variable, same seeds" methodology on a different axis.
