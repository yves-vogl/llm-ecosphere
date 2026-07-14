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
