# Lab report: the attention-head taxonomy

> **Spoiler warning.** This chapter is a worked solution of
> [exercise 8](08-exercises.md#8-attention-head-taxonomy-catalogue-all-16-heads--an-afternoon).
> If you have not tried it yourself yet, go do that first — the exercise is
> the point, this report is the answer key.

`minillm/inspect_attention.py`'s docstring promises three species of
heads — previous-move heads, same-column (stack-height) heads, and
`<bos>`-sink heads — and exercise 8 asks the only fair question: does
the promise hold for the actual checkpoint? All 16 heads (4 layers × 4
heads), catalogued, including the boring ones.

## Method: stats first, matrices second

Eyeballing 16 matrices across several prefixes invites cherry-picking,
so the catalogue below rests on summary statistics over four prefixes
chosen to break ties between competing explanations:

```
B1 A1 B2 C1 B3      the exercise's default
B1 C1 A1 B2         previous-move vs same-column (in "B1 A1 B2" they coincide)
A1 A2 A3 B1 B2      a long same-column chain
C1 B1 A1 C2 B2      same-column pairs at several distances
```

Per head, averaged over all query positions of all four prefixes:

* **bos** — weight on the `<bos>` key;
* **prevd** — weight on the immediately preceding position *when that
  move is in a different column* (pure positional behaviour — queries at
  position ≥ 2, so `<bos>` never counts as "previous");
* **colf** — total weight on same-column moves at distance ≥ 2 (pure
  column-tracking, the previous position excluded);
* **ent** — row entropy normalized by ln(#visible keys): 1.0 is
  perfectly diffuse, 0.0 a delta function.

The measurement script is the `record_attn=True` machinery
`inspect_attention.py` already uses; the full listing is in
"Reproduce it" below.

## The catalogue (finetune checkpoint)

Stats are (bos / prevd / colf / ent). Confidence: **high** = one pattern
dominates on every prefix; medium = dominates on most; low = suggestive
only.

| L | H | dominant pattern | conf. | evidence |
|--:|--:|---|---|---|
| 0 | 0 | diffuse, mildly recent | low | .26/.32/.29/**.82** — no component reaches 1/3 |
| 0 | 1 | diffuse | low | .17/.24/.18/.70 — flattest profile in the model |
| 0 | 2 | diffuse (near-uniform) | high (that it has no role) | ent **.91**, the most uniform head anywhere |
| 0 | 3 | weak `<bos>` lean, else local | low | bos .37, ent .56 — a half-hearted sink |
| 1 | 0 | **previous move** | high | prevd **.50**; `A1→C1 0.92` in `B1 C1 A1 B2` |
| 1 | 1 | **previous move** (+ column tint) | medium | prevd .48, colf .27; keeps some self-attention late |
| 1 | 2 | **previous move**, soft | medium | prevd .50 but ent .79 — the diffuse sibling of H0 |
| 1 | 3 | **previous move** early; 2-back late | medium | prevd .56, ent .35; in `C1 B1 A1 C2 B2`: `C2→A1 0.98` (prev) but `B2→A1 0.90` (2-back) |
| 2 | 0 | **same column** | medium-high | colf **.38**; `B2→B1 0.53`, `B3→B1 0.70` in the default prefix |
| 2 | 1 | opening anchor / column mix | medium | colf .35, bos .35; early queries pin the first move (`A1→B1 0.90`) |
| 2 | 2 | recency mix | low | prevd .37, colf .11 — positional lean, no clean offset |
| 2 | 3 | self + previous | medium | `A1→A1 0.84`, `B2→B2 0.51`, `C1→B2 0.52` — a local window ending at itself |
| 3 | 0 | **same column** (stack geometry) | medium-high | colf **.50**; `B2→B1 0.96` in `B1 C1 A1 B2`; falls back to row-mates (`B2→A2 0.64` in the A-chain) |
| 3 | 1 | **`<bos>` sink** | high | bos **.74**, ent .26; rows are literally `1.00 0 0 0 0` |
| 3 | 2 | diffuse / self | low | .28/.19/.31/.71; occasional self-spikes (`A1→A1 0.63`) |
| 3 | 3 | **`<bos>` sink** | high | bos **.68**, ent .28; sink rows with rare self leaks |

**Verdict on the docstring's promise: it holds, and it is organized by
layer.** All three promised species exist — but not scattered anywhere:
layer 1 is *entirely* previous-move heads (all four), the same-column
heads live in layers 2–3 (L2H0, L3H0, half of L2H1), and the pure
`<bos>` sinks sit at the top (L3H1, L3H3). Layer 0 is undifferentiated
local mixing — four of the five "diffuse / no clear role" rows come from
the bottom of the stack. That ordering is the transformer cliché in
miniature: position before content, syntax before geometry, and a
dumping ground for attention that has nothing to say.

The tie-break prefixes earned their keep: in `B1 A1 B2` alone, L1H0
("looks one back") and L2H0 ("tracks the B column") are
indistinguishable — both point at the same key. `B1 C1 A1 B2` separates
them (`A1→C1` for the positional head, `B2→B1` across two intervening
moves for the column head), and the `A1 A2 A3` chain shows L3H0's
geometry is really "the cell under mine": same column when one exists at
distance ≥ 2, same *row* as a fallback.

## Pretrain vs finetune: did finetuning repurpose any head?

Mostly no — and where yes, in one direction. Stats that moved by ≥ 0.1
between the pretrain and finetune checkpoints:

| head | pretrain | finetune | reading |
|---|---|---|---|
| L2H1 | bos .53, colf .18 | bos .35, colf .35 | **a sink half-repurposed into a column tracker** |
| L3H0 | colf .37 | colf **.50** | column tracking sharpened |
| L1H1 | colf .16 | colf .27 | picked up a column tint |
| L3H1 | bos .67 | bos .74 | the sink got sinkier |

Everything else is stable within noise: the previous-move layer, the
diffuse bottom layer and the top-layer sinks survive finetuning nearly
untouched. The drift direction makes sense: finetuning teaches *where to
play*, and "where" in this game is a column decision informed by stack
heights — so the capacity that moved, moved toward column geometry.

> **In a real LLM:** all three species have famous production
> counterparts, which is exactly why this repo promises them.
> Previous-token heads are among the first circuits found in GPT-2, and
> they feed the induction heads of in-context-learning fame; our
> same-column heads are the miniature of content-based retrieval heads
> that track "the last mention of this entity"; and the `<bos>` sink is
> the attention-sink phenomenon — softmax rows must sum to 1, so heads
> park mass on a semantically empty token when they have nothing to say
> (the observation behind StreamingLLM's "keep the first token" trick,
> and one reason real systems pin a BOS/system token permanently in the
> KV cache). The layerwise ordering — positional early, semantic later,
> diffuse at the bottom — is the standard finding of interpretability
> work, reproduced here in a model small enough to audit every head by
> hand. And the honest sixth of the table — "diffuse / no clear role" —
> is faithful too: most heads in most transformers resist a crisp label,
> which is why mechanistic interpretability at scale is hard.

## Reproduce it

Matrices, one prefix at a time (rows = queries, columns = keys):

```bash
make attention                                             # B1 A1 B2, all heads
.venv/bin/python -m minillm.inspect_attention --moves "B1 C1 A1 B2" --layer 3
.venv/bin/python -m minillm.inspect_attention --moves "B1 C1 A1 B2" \
    --ckpt runs/pretrain/model.pt --layer 2 --head 1       # the repurposed head
```

The summary statistics:

```python
from math import log
import torch
from minillm.utils import load_model, pick_device, tokenizer_for_checkpoint

PREFIXES = ["B1 A1 B2 C1 B3", "B1 C1 A1 B2", "A1 A2 A3 B1 B2", "C1 B1 A1 C2 B2"]

def head_stats(ckpt_path):
    model, ckpt = load_model(ckpt_path, pick_device("cpu"))
    tok = tokenizer_for_checkpoint(ckpt)
    agg = {}
    for prefix in PREFIXES:
        moves = prefix.split()
        ids = tok.encode_prompt(moves)
        with torch.no_grad():
            model(torch.tensor([ids]), record_attn=True)
        cols = [None] + [m[0] for m in moves]
        for l, block in enumerate(model.transformer.h):
            att = block.attn.last_attn[0]
            for h in range(att.size(0)):
                a, d = att[h], agg.setdefault((l, h), {"bos": [], "prevd": [], "colf": [], "ent": []})
                for i in range(1, len(ids)):
                    row = a[i, : i + 1]
                    d["bos"].append(row[0].item())
                    d["ent"].append(-(row * (row + 1e-12).log()).sum().item() / log(i + 1))
                    if i >= 2 and cols[i - 1] != cols[i]:
                        d["prevd"].append(row[i - 1].item())
                    far = [j for j in range(1, i - 1) if cols[j] == cols[i]]
                    if far:
                        d["colf"].append(sum(row[j].item() for j in far))
    return agg

for path in ("runs/finetune/model.pt", "runs/pretrain/model.pt"):
    print(f"== {path} ==")
    for (l, h), d in sorted(head_stats(path).items()):
        m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"L{l} H{h}  bos {m(d['bos']):.2f}  prevd {m(d['prevd']):.2f}  "
              f"colf {m(d['colf']):.2f}  ent {m(d['ent']):.2f}")
```

Next: the [deep-dive lenses](deep-dive-lenses.md) look inside the same
model along a different axis (logit lens, Jacobian lens), or back to the
[exercises](08-exercises.md) — the char checkpoint's heads must track
column-letter/row-digit pairing on top of all this, and nobody has
catalogued *those* yet.
