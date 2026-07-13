# Contributing

Thank you for considering a contribution to llm-ecosphere. This guide
covers the development loop, the testing conventions, and the commit/PR
discipline this repository enforces.

---

## Code of conduct

Be respectful. Discuss code, not people. Assume good intent. If you find a
security issue, do not open a public issue — follow the disclosure path in
[`SECURITY.md`](SECURITY.md).

---

## What this repo is (and is not)

llm-ecosphere is an **educational lab**: a from-scratch GPT whose entire
world is an exactly-solvable game, so every claim about the model can be
measured against ground truth. Contributions should preserve the three
properties that make it teachable:

1. **CPU-only and fast.** Every experiment retrains in minutes on a plain
   CPU. Nothing may require a GPU (chapter
   [10](docs/10-gpu-cuda.md) explains why that is a feature).
2. **Minimal dependencies.** `torch`, `numpy`, `pytest` — that's the list.
   A contribution that adds a dependency needs a very good reason.
3. **Readable over clever.** The model code is deliberately the naive,
   heavily commented version. Optimizations that obscure the shape of the
   computation belong in a doc chapter or an exercise, not in `model.py`.

Good first contributions: solving an [exercise](docs/08-exercises.md) and
writing it up as a lab report (see
[docs/09](docs/09-char-tokenizer-lab.md) for the pattern), improving doc
clarity, new evaluation metrics, fixing genuine bugs.

---

## Development loop

### Prerequisites

- Python 3.12 — via [uv](https://docs.astral.sh/uv/) (recommended) or a
  plain venv (see README).
- `gpg` for signing commits, with a published public key associated with
  your committer email.

### Setup and test

```bash
make setup      # create .venv and install dependencies
make test       # run the unit test suite — must stay green
make all        # full pipeline: data -> pretrain -> finetune -> eval
```

### Experiment hygiene

Training runs write to `runs/`. The reference checkpoints live in
`runs/pretrain` and `runs/finetune` — **never overwrite them** with
experimental configurations. Experiments go to `runs/exp-<name>`:

```bash
.venv/bin/python -m minillm.train --stage pretrain --tokenizer char \
    --block-size 24 --out-dir runs/exp-char-pretrain
```

Everything under `runs/` and `data/` is gitignored and reproducible —
training is seeded, so results are directly comparable.

### Docs

The documentation site is MkDocs Material:

```bash
pip install -r requirements/docs.lock.txt --require-hashes --no-deps
mkdocs serve    # live preview at http://127.0.0.1:8000
```

`mkdocs build --strict` must pass — broken internal links fail CI.

---

## Tests

- The suite lives in `tests/` and runs with plain `pytest`.
- CI enforces a **85% coverage floor** over the core library
  (`.coveragerc` scopes the measurement; the CLI entry points are
  exercised by the pipeline itself, not by unit tests).
- If you change model, tokenizer or dataset code, the causality/masking
  tests are the contract — read them before editing.
- Behavioural claims in docs (legality percentages, win rates) come from
  `python -m minillm.evaluate` on seeded runs. If your change shifts
  them, re-run the evaluation and update the numbers in the same PR.

---

## Commits and PRs

- **Conventional Commits** are enforced by CI on every PR:
  `<type>(<scope>): <subject>` with types
  `feat fix docs test refactor chore ci build perf style revert`.
- **Sign your commits.** `main` requires verified signatures.
- PRs go against `main`; merges are squash-only, so keep one logical
  change per PR.
- All repository artifacts — code, comments, commits, PRs, docs — are
  written in English.
- No AI-attribution trailers or badges in commits or code; a CI gate
  rejects them.
