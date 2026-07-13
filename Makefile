PY := .venv/bin/python

.PHONY: setup test data pretrain finetune model-gambler eval play arena sample attention all

## One-time environment setup (uses uv; see README for a plain-venv alternative)
setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -r requirements.txt

## Run the unit test suite
test:
	$(PY) -m pytest -q

## Stage 1 of the pipeline: enumerate every possible game and write datasets
data:
	$(PY) -m minillm.dataset --out data

## Stage 2: pretrain on ALL games (the model learns the rules of the game)
pretrain:
	$(PY) -m minillm.train --stage pretrain

## Stage 3: finetune on expert games (the model learns to play WELL)
finetune:
	$(PY) -m minillm.train --stage finetune

## Stage 3, gambler variant: finetune on decisive games, imitating only the
## WINNING side (exploitable aggression, not minimax-optimal play)
model-gambler:
	$(PY) -m minillm.train --stage finetune --objective gambler --out-dir runs/exp-gambler-move

## Stage 4: measure legality, refereeing and playing strength
eval:
	$(PY) -m minillm.evaluate --out runs/eval.json

## Play against the model interactively
play:
	$(PY) -m minillm.play

## Arena: pit any model against a human, a random player, the solver, or another model
arena:
	$(PY) -m minillm.arena --model runs/finetune/model.pt --vs solver

## Generate a few complete game transcripts and check them for legality
sample:
	$(PY) -m minillm.sample --num 5

## Print the attention matrices for a given game prefix
attention:
	$(PY) -m minillm.inspect_attention --moves "B1 A1 B2"

## Full pipeline: data -> pretrain -> finetune -> eval
all: data pretrain finetune eval
