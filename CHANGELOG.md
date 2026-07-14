# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The RL gambler** (`minillm.rl`): a REINFORCE self-play gambler that
  maximises wins directly against a random opponent — the strongest player
  in the lab when constrained to legal moves, and a measured demonstration
  of the "alignment tax" (optimizing purely for wins collapses free-running
  legality). Lab report in `docs/rl-gambler.md`.
- Scenario Makefile aliases `model-base`, `model-expert`,
  `model-rl-gambler`, and `zoo` (the local model-zoo matrix).
- A `--temperature` flag on `minillm.evaluate` (exercise 4): the strength
  matches can now sample the model's move instead of taking the argmax, so
  playing strength can be measured across a temperature sweep. Temperature 0
  is the exact previous behaviour.
- **Two lab reports for the round-one experiments.** `docs/temperature-sweep.md`
  (exercise 4) shows the move-level and char-level checkpoints respond to
  temperature in mirror image — each greedy policy sits at an opposite
  extreme of the sharp-vs-solid frontier, and temperature nudges both toward
  the middle. A new multi-seed section in `docs/09-char-tokenizer-lab.md`
  reruns the char-vs-move claim across three training seeds and revises the
  single-seed conclusion: the char "wall" is a stable 95% draw rate versus
  the solver on every seed, the move model's draw rate is a high-variance
  checkpoint lottery (61–100%), and the two are statistically tied as
  attackers — the distinguishing axis is consistency, not sharpness.

## [0.1.0] — 2026-07-14

First public release.

### Added

- **The lab.** Drop-Tac-Toe game engine with an exact negamax solver
  (1,310 enumerable games, solver-proven draw), a from-scratch
  ~0.8M-parameter GPT, pretraining plus SFT-style finetuning with
  opponent-move loss masking, a behavioural evaluation suite, interactive
  play, and attention inspection.
- **Two interchangeable tokenizers**: move-level (15 tokens) and
  character-level (13 tokens), selectable via `--tokenizer`; checkpoints
  record their tokenizer.
- **Training scenarios** beyond the base model: the minimax *expert*
  (finetune) and the *gambler* (`--objective gambler`, winner-imitation
  SFT — aggressive and exploitable).
- **The arena** (`minillm.arena`): pit any checkpoint against a human, a
  random player, the perfect solver, or another checkpoint.
- **Model zoo CI** (`models.yml`): trains the full
  {base, expert, gambler} × {move, char} matrix and attaches each model
  plus its `eval.json` to the release.
- **Documentation**: a guided tour (`docs/00`–`docs/10`) plus an on-ramp
  (use a model, the models in detail, a hands-on tutorial), deep dives
  (model anatomy, the mathematics, interpretability lenses including
  Anthropic's 2026 J-space, a frontier outlook), lab reports (character
  tokenizer, RL gambler), a glossary, learning paths, and a motivation
  essay — published via MkDocs Material (with MathJax) to GitHub Pages.
- **Repository infrastructure**: CI (pytest + 85% coverage gate + corpus
  smoke test + no-AI-attribution and SHA-pin gates + a self-hosted coverage
  badge), Semgrep SAST, gitleaks secret scan, commit-message lint,
  OpenSSF Scorecard, Dependabot, and a Pages deploy — all with SHA-pinned
  actions, plus branch protection on `main`.
