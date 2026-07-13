# 03 — Tokenization

A neural network cannot consume the string `"B1 A1 B2 C1 B3 #X"`. Its first
layer is an embedding table — a matrix indexed by integer — so everything the
model ever sees must first become a sequence of integer ids. The component
that does this mapping, in both directions, is the tokenizer. In llm-ecosphere it
lives in `minillm/tokenizer.py` and is small enough to hold in your head
completely, which is exactly the point: every design decision that real
tokenizers make under pressure of scale shows up here in miniature, with
nothing to hide behind.

## Why ids, not strings

The model's input layer is `nn.Embedding(vocab_size, n_embd)` (see
`minillm/model.py`): a lookup table with one learned vector per vocabulary
entry. Its output layer is a linear projection back to `vocab_size` logits —
one score per vocabulary entry. The vocabulary is therefore not a
preprocessing detail; it defines the model's entire input and output
interface. The tokenizer is the contract between text-land and tensor-land,
and both sides must agree on it exactly — which is why
`minillm/config.py` hard-codes:

```python
vocab_size: int = 15  # see tokenizer.VOCAB
```

## The full vocabulary

The vocabulary is built in three slices in `minillm/tokenizer.py`:

```python
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]
MOVE_TOKENS = [f"{c}{r}" for c in COLS for r in range(1, N + 1)]  # A1..C3
RESULT_TOKENS = [RESULT_X, RESULT_O, RESULT_DRAW]
VOCAB = SPECIAL_TOKENS + MOVE_TOKENS + RESULT_TOKENS
```

Enumerated, the whole language of Drop-Tac-Toe is 15 tokens:

| id | token   | class   | meaning |
|----|---------|---------|---------|
| 0  | `<pad>` | special | filler after the sequence ends; never a training target |
| 1  | `<bos>` | special | beginning of sequence |
| 2  | `<eos>` | special | end of sequence |
| 3  | `A1`    | move    | piece lands in column A, row 1 |
| 4  | `A2`    | move    | column A, row 2 (requires A1 occupied) |
| 5  | `A3`    | move    | column A, row 3 |
| 6  | `B1`    | move    | column B, row 1 |
| 7  | `B2`    | move    | column B, row 2 |
| 8  | `B3`    | move    | column B, row 3 |
| 9  | `C1`    | move    | column C, row 1 |
| 10 | `C2`    | move    | column C, row 2 |
| 11 | `C3`    | move    | column C, row 3 |
| 12 | `#X`    | result  | X won |
| 13 | `#O`    | result  | O won |
| 14 | `#=`    | result  | draw |

A complete game transcript becomes

```
<bos> B1 A1 B2 C1 B3 #X <eos>
```

and the longest possible sequence is capped by the geometry of the game:

```python
# Longest possible sequence: <bos> + 9 moves + result + <eos>
MAX_GAME_TOKENS = 1 + N * N + 1 + 1
```

That is 12, and `data/meta.json` confirms it end-to-end:
`"max_sequence_tokens": 12`, with games ranging from
`"shortest_game_moves": 5` to `"longest_game_moves": 9` across all
`"n_all_games": 1310` games.

## Move-level vs char-level vs BPE

Tokenizing one move as one token is a **word-level** tokenizer: the atomic
units of the language (moves, results) map one-to-one onto tokens. Two
alternatives were on the table:

- **Character-level** (`"B"`, `"1"`, `" "`, ...): the vocabulary would be
  even smaller, but the model would have to spend capacity learning that
  characters group into moves — that `"B"` is always followed by a digit,
  that `"B4"` is not a thing. The module docstring in
  `minillm/tokenizer.py` calls this out explicitly: it is a fine exercise
  (Exercise 1 in `docs/08-exercises.md` picks it up), but move-level tokens keep the mapping
  between model behaviour and game concepts one-to-one. When a logit for
  `B2` goes up, that is directly a statement about a move — no decoding
  layer of interpretation in between.
- **BPE (byte-pair encoding)**: learn a subword vocabulary from corpus
  statistics, merging frequent character pairs until a target vocab size is
  reached. For a 15-symbol closed language this is pointless — BPE exists to
  compress an *open-ended* language into a fixed-size vocabulary, and our
  language is already fixed-size.

The trade-off in general: larger tokens mean shorter sequences (cheaper
attention, longer effective context) but a bigger embedding table and softmax;
smaller tokens mean the opposite, plus more of the burden of "spelling"
pushed into the model. Word-level tokenization does not survive contact with
real text because real text has unbounded vocabulary — names, typos, code,
other languages. Our game does not have that problem, so we get word-level's
interpretability for free.

> **In a real LLM:** production models use learned subword vocabularies —
> BPE for GPT-2/3/4 (50,257 entries for GPT-2), byte-level BPE or
> SentencePiece/unigram for Llama (32k in Llama 2, 128k in Llama 3), and
> vocabularies in the 100k-200k range for recent frontier models. The vocab
> size is a genuine engineering trade-off: the embedding and output matrices
> are `vocab_size x n_embd` each, so at 128k tokens and 8k embedding width
> those two matrices alone are ~2B parameters. Multilingual coverage pushes
> vocab up; memory and softmax cost push it down.

### Why there is no `<unk>`

Classic NLP vocabularies reserve an `<unk>` (unknown) token to absorb
out-of-vocabulary words. llm-ecosphere deliberately does not, and `encode` makes
the alternative policy explicit — fail loudly:

```python
def encode(self, tokens: list[str]) -> list[int]:
    """Token strings -> ids. Unknown tokens raise (no <unk> here:
    in a 15-token world an unknown token is always a bug)."""
```

The language is closed: every legal move and every outcome has a token.
Anything else — `"D5"`, a typo, a corrupted file — is a programming error,
and mapping it to a catch-all id would silently convert a bug into bad
training data. `tests/test_tokenizer.py::test_unknown_token_raises` pins
this contract down with `pytest.raises(ValueError, match="unknown token")`.

> **In a real LLM:** modern tokenizers also avoid `<unk>`, but by the
> opposite route — byte fallback. Byte-level BPE (GPT-2 onward) and
> SentencePiece with byte fallback can encode *any* UTF-8 string, because in
> the worst case a character decomposes into raw bytes that are always in
> the vocabulary. Nothing is unknown, so nothing is lost. `<unk>` mostly
> survives as a vestigial slot in vocab files.

## Special tokens

Three tokens carry no game meaning; they are sequence plumbing:

- `<bos>` (id 1) marks the start of a sequence. It gives the model a
  position from which to predict the *first* move — without it, there would
  be no input position whose target is move 1. When generating a fresh game,
  sampling starts from a context of just `[bos_id]`.
- `<eos>` (id 2) marks the end. The model learns to emit it after the result
  token, which is how generation knows to stop.
- `<pad>` (id 0) is not part of the language at all — it is a batching
  artifact, covered below.

> **In a real LLM:** the same roles exist under different names — GPT-2
> famously used a single `<|endoftext|>` token as both document separator
> and stop signal; Llama has `<s>`/`</s>`; chat models add a whole grammar
> of control tokens (`<|im_start|>`, `<|im_end|>`, role markers) on top.
> These special tokens are also a security surface: user-supplied text must
> never be able to smuggle in the byte sequence of a control token, so
> production tokenizers encode user text with special-token recognition
> disabled.

## encode, decode, encode_game

The `Tokenizer` class is two dictionaries and a handful of helpers:

```python
self.token_to_id = {tok: i for i, tok in enumerate(VOCAB)}
self.id_to_token = {i: tok for i, tok in enumerate(VOCAB)}
```

`encode` maps token strings to ids, `decode` maps back, and `encode_game`
wraps a complete game in the sequence frame:

```python
def encode_game(self, moves: list[str], result: str) -> list[int]:
    """A complete game -> <bos> moves result <eos> as ids."""
    return self.encode([BOS] + list(moves) + [result, EOS])
```

Concretely, the five-move X win from the docstring:

```
encode_game(["B1", "A1", "B2", "C1", "B3"], "#X")
  -> [1, 6, 3, 7, 9, 8, 12, 2]
      │  │  │  │  │  │   │  └ <eos>
      │  B1 A1 B2 C1 B3  #X
      └ <bos>
```

Round-tripping is exact and tested
(`tests/test_tokenizer.py::test_roundtrip`), and helper properties expose
convenient id groups: `move_ids` (the nine cell tokens) and `result_ids`
(the three outcomes) — currently exercised only by the tests; the evaluator
instead builds its own legal-move id list per position with
`tokenizer.encode(game.legal_moves())`.

The tokenizer also persists itself: `save` writes `token_to_id` as JSON,
`load` reads it back and asserts it matches the code —
`"vocab file does not match this code version"`. A stale vocab file paired
with a checkpoint trained under a different mapping would scramble every
prediction while crashing nothing; the assert turns that silent corruption
into a hard failure. This mirrors how real models ship a `tokenizer.json` /
`vocab.json` next to their weights: the weights are meaningless without the
exact id mapping they were trained under.

## Padding vs the -1 loss mask: two mechanisms, two jobs

Games have different lengths (5 to 9 moves), but a training batch must be a
rectangular tensor. `build_tensors` in `minillm/dataset.py` reconciles this
with **two separate mechanisms** that beginners often conflate:

```python
x = torch.full((len(games), block_size), tokenizer.pad_id, dtype=torch.long)
y = torch.full((len(games), block_size), -1, dtype=torch.long)
```

- **`<pad>` in `x` keeps tensors rectangular.** Every input row is
  `block_size` (= 16, from `minillm/config.py`) columns wide; positions after
  the game's tokens hold `pad_id` (0). The shortest game occupies 7 of the 16
  input positions — the other 9 are filler that exists only so the batch has
  a shape.
- **`-1` in `y` keeps the loss honest.** `y` is `x` shifted one position
  left: position `t` of `x` must predict `y[t]`, the token at `t+1`. Any
  position whose target is left at `-1` contributes nothing to the loss,
  because the model computes

  ```python
  loss = F.cross_entropy(
      logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
  )
  ```

  (`minillm/model.py`). Padding positions after the last real input token
always keep `-1`
  targets — the model is never rewarded for predicting filler. During
  finetuning, `expert_only=True` reuses the *same* mechanism for a second
  purpose: opponent moves also keep `-1`, so the model only imitates the
  solver's side (covered in `docs/05-training.md`).

The distinction matters because they answer different questions. `<pad>` in
`x` answers "what does the model *see* at unused positions" (a harmless
dummy token). `-1` in `y` answers "which positions does the model get
*graded* on" (only real, wanted targets). You could pad `x` with any token
id and, as long as the `y` mask is right, training would be unaffected at
the positions that count.

### Why pad cannot leak backwards

One worry remains: pad tokens are garbage *inputs* — can they contaminate
the predictions at earlier, real positions? No, and the reason is the causal
attention mask, not the loss mask. From `minillm/model.py`:

```python
mask = torch.tril(torch.ones(config.block_size, config.block_size))
...
att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
```

The lower-triangular mask means position `t` attends only to positions
`<= t`. Pad tokens sit strictly *after* the last real input token — the
result token; `<eos>` itself only ever appears as a target in `y`, never as
an input in `x` — so no real position can ever attend to a pad. The garbage flows
only forward, into positions whose loss is already masked to `-1`. The two
masks close the loop from opposite sides: the causal mask guarantees pads
cannot corrupt real *predictions*, the `ignore_index` mask guarantees pads
cannot corrupt the *gradient*.

> **In a real LLM:** production pretraining largely sidesteps padding by
> packing — documents are concatenated with separator tokens and chopped
> into full-length blocks, so every position is real and the batch is dense
> by construction. Padding plus attention masking comes back at inference
> time (batched serving of variable-length prompts) and in finetuning,
> where the SFT loss mask over non-assistant turns is exactly the
> `expert_only` trick from `build_tensors`, scaled up.

## What the tests pin down

`tests/test_tokenizer.py` is short but each test guards a real failure
mode: `test_vocab_is_15_tokens` (the model/tokenizer size contract),
`test_roundtrip` (encode/decode are inverses), `test_encode_game_layout`
(`<bos>` first, `<eos>` last), `test_unknown_token_raises` (no silent
`<unk>` behaviour), `test_max_game_tokens` (the 12-token bound that
justifies `block_size = 16`), and `test_save_load` (the persisted vocab is
faithful). When the interface between text and tensors is this small, it is
cheap to test it exhaustively — and any bug here would poison everything
downstream while looking like a modeling problem.

Next: [04 — The model](04-model.md), where these ids meet the embedding
table and the transformer stack that turns a prefix of a game into a
distribution over what comes next.
