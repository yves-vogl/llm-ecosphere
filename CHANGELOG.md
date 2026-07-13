# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The complete lab: Drop-Tac-Toe game engine with exact negamax solver
  (1,310 enumerable games, solver-proven draw), from-scratch ~0.8M-parameter
  GPT, pretraining + SFT-style finetuning with opponent-move loss masking,
  behavioural evaluation suite, interactive play, attention inspection.
- Two interchangeable tokenizers: move-level (15 tokens) and
  character-level (13 tokens), selectable via `--tokenizer`; checkpoints
  record their tokenizer.
- Guided documentation in eleven chapters (`docs/00`–`docs/10`), including
  a worked lab report on the character-level tokenizer and a chapter on
  GPU/CPU/CUDA fundamentals, published via MkDocs Material to GitHub Pages.
- Repository infrastructure: CI (pytest + coverage gate + corpus smoke
  test), Semgrep SAST, gitleaks secret scan, commit-message lint,
  OpenSSF Scorecard, Dependabot, SHA-pinned workflows.
