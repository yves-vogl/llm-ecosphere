"""minillm — a complete, minimal LLM lab.

A from-scratch GPT that learns to play Drop-Tac-Toe (Tic-Tac-Toe with
gravity) purely from game transcripts. Every stage of a real LLM
pipeline exists here in miniature:

    game.py      the "world" the training data describes
    solver.py    exact negamax solver (data generator + gold policy)
    tokenizer.py text <-> token ids
    dataset.py   corpus building, splits, tensorization
    model.py     the Transformer itself
    train.py     pretraining + finetuning loops
    sample.py    free-running generation
    evaluate.py  behavioural metrics
    play.py      interactive human-vs-model games
    inspect_attention.py  look inside the model's heads

Read docs/00-overview.md for the guided tour.
"""

__version__ = "0.1.0"
