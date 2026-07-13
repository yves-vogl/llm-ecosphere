# 10 — Why GPUs? The hardware under the pipeline

Everything in this repo runs on a plain CPU, on purpose: `make pretrain`
finishes 3,000 optimizer steps on a ~0.8M-parameter model in about two
minutes, and `minillm/utils.py` even *defaults* to CPU with a comment
explaining why. Real LLMs run the same code shape you read in
[04 — The model](04-model.md) — the exact same matmuls, softmaxes and
residual additions — at five to six orders of magnitude larger scale. The
gap between those two sentences is not bridged by smarter algorithms. It is
bridged by hardware, and this chapter explains how, ending at the name you
keep hearing: CUDA.

## What training actually computes

Strip the training loop in `minillm/train.py` to its arithmetic and almost
everything is matrix multiplication:

* attention: `q @ k.transpose(-2, -1)` and `att @ v` in
  `CausalSelfAttention.forward`,
* the MLP: two linear layers spanning `n_embd -> 4*n_embd -> n_embd`,
* the projections: `c_attn`, `c_proj`, and the `lm_head`.

A useful back-of-envelope rule (used throughout the scaling literature):
training costs roughly **6 FLOPs per parameter per token** — 2 for the
forward pass, 4 for the backward. Our pretraining run processes about
3,000 steps × 64 games × ~10 tokens ≈ 2 million target tokens, so:

```
6 × 0.8M params × 2M tokens  ≈  10^13 FLOPs  ≈  10 TFLOPs total
```

A modern CPU core sustains on the order of tens of GFLOP/s on dense linear
algebra, which predicts a couple of minutes — and indeed the run takes
about two. Now the same arithmetic for GPT-3 (175B parameters, ~300B
training tokens) gives ≈ 3 × 10²³ FLOPs: **about thirty billion of our
pretraining runs**. On this laptop's CPU that is geological time. That
factor — not any change in the code — is why the field runs on GPUs.

## CPU vs GPU: two opposite bets

A CPU and a GPU are both "processors", but they are engineered around
opposite bets about what work looks like.

**A CPU bets on serial, irregular, branchy work.** It spends its silicon on
a few fat cores (8–16 in a laptop), each with deep caches, branch
prediction, out-of-order execution — machinery whose entire purpose is to
make *one* instruction stream finish as early as possible. This repo
contains a perfect specimen of CPU-shaped work: `minillm/solver.py`. Negamax
recursion over a memoization cache (`functools.lru_cache`), branching on
game state at every node — unpredictable branches and pointer-chasing all
the way down, almost no arithmetic. No GPU would help; the work refuses to
line up.

**A GPU bets on parallel, regular, branch-free work.** It spends the same
silicon on *thousands* of small arithmetic units (a current data-center GPU
has on the order of 15,000+ FP32 lanes) marching in lock-step groups, fed by
memory designed for throughput: HBM stacks delivering terabytes per second,
versus the tens of GB/s a CPU gets from DRAM. Individual operations are not
faster — they are slower, and the GPU hides their latency by always having
thousands of other threads ready to run. The one workload that fits this
machine perfectly is the one training is made of: the *same*
multiply-accumulate applied across huge, regular arrays. A matmul has no
branches, no pointer-chasing, and total independence between output cells —
it is embarrassingly parallel, which is exactly the property a GPU pays for.

The division of labor in this repo mirrors the industry's: the game engine
and solver (irregular, branchy) generate data on the CPU; the model
(regular, dense) consumes tensors that would run identically on a GPU.

Two numbers decide which processor wins, and it is worth knowing both:

1. **Compute** (FLOP/s): GPUs quote hundreds of TFLOP/s in low-precision
   matmul (bf16/fp16 on tensor cores — dedicated matmul circuits) versus
   tens of GFLOP/s per CPU core. Three to four orders of magnitude.
2. **Memory bandwidth** (bytes/s): every parameter must travel from memory
   to the arithmetic units and back each step. For big models this — not
   compute — is the binding constraint, which is why GPU generations are
   marketed as much on HBM bandwidth as on FLOPs, and why inference
   engineering obsesses over the KV cache
   (see [exercise 5](08-exercises.md#5-implement-a-kv-cache-in-generate--a-weekend)).

## Why this repo still (rightly) uses the CPU

`pick_device` in `minillm/utils.py` says it plainly: the model is so small
that CPU is "entirely adequate (and often faster than paying GPU launch
overhead for tiny kernels)". Every operation dispatched to a GPU costs a
fixed overhead — the driver launches a *kernel* (a GPU program) and, for
batches this small, the launch costs more than the arithmetic it triggers.
Our forward pass over a `(64, 16)` batch of ids through 0.8M parameters is
microseconds of math; a GPU would spend longer starting each kernel than
running it, while 15,000 lanes sit idle behind the first hundred. The
crossover comes with scale: at GPT-2 small (124M parameters — same
architecture, see `docs/04-model.md`) a GPU is already the difference
between hours and weeks, and at frontier scale the question is not one GPU
but tens of thousands, connected by dedicated interconnects and coordinated
with data-, tensor- and pipeline-parallelism.

You can measure the small-model effect yourself — every model-touching
entry point in this repo takes `--device` (only `minillm.dataset` has none:
enumeration is solver work, and the solver is CPU-shaped):

```bash
time .venv/bin/python -m minillm.train --stage pretrain --out-dir runs/exp-device-cpu
time .venv/bin/python -m minillm.train --stage pretrain --device mps --out-dir runs/exp-device-mps
```

On an Apple-silicon Mac, `mps` (Metal Performance Shaders — Apple's GPU
backend, the role CUDA plays for NVIDIA) will likely *lose* to the CPU at
this model size. That is not a broken setup; it is the launch-overhead
lesson above, reproduced as a negative result — the most instructive kind.

## CUDA, finally

**CUDA** (Compute Unified Device Architecture, NVIDIA, 2007) is the
programming platform that made "GPU" mean "general-purpose parallel
computer" instead of "graphics card". It is three things at once:

1. **A programming model.** You write a *kernel* — a function in a C++
   dialect describing what ONE thread does — and launch it over a grid of
   thousands of threads. The hardware groups them into lock-step bundles
   ("warps" of 32) and schedules them over the chip's compute units. All the
   parallelism in the machine is expressed through this one abstraction.
2. **A toolchain**: compiler, profiler, debugger for those kernels.
3. **A library ecosystem**: cuBLAS (dense linear algebra), cuDNN (deep
   learning primitives), NCCL (multi-GPU communication), and years of
   hand-tuned kernels for exactly the operations transformers are made of.

The practical punchline for this codebase: **you already write CUDA
programs without knowing it.** PyTorch dispatches every tensor operation to
a precompiled kernel for wherever the tensor lives. Run

```bash
.venv/bin/python -m minillm.train --stage pretrain --device cuda   # on an NVIDIA machine
```

and not one line of `minillm/` changes: `model.to(device)` moves the weight
tensors to GPU memory, and from then on the same `q @ k.transpose(-2, -1)`
in `model.py` executes as a cuBLAS kernel instead of a CPU BLAS call. The
Python is a *description* of the computation; where it runs is a deployment
detail. That device-agnosticism is why `pick_device` can offer
`auto` → cuda > mps > cpu as a one-line policy.

The performance frontier, however, lives *below* PyTorch. The attention in
`minillm/model.py` calls itself "the naive, readable one (no
FlashAttention, no KV cache)" — FlashAttention is the canonical example of
what CUDA-level rethinking buys: mathematically the same softmax attention,
but restructured into a single kernel that tiles the computation through the
GPU's fast on-chip memory instead of materializing the full `(T, T)`
attention matrix in slow HBM. Same math, several times faster, purely by
respecting the memory hierarchy. When people say "kernel engineering" they
mean this.

> **In a real LLM:** hardware is the pipeline's binding constraint, and it
> shapes everything upstream. Training a frontier model occupies tens of
> thousands of GPUs for months, which is why FLOPs budgets, not ideas, set
> model sizes (the scaling-law literature is precisely the science of
> spending that budget). Serving is bound by memory bandwidth — moving
> weights and KV cache per token — which is why grouped-query attention,
> paged attention and quantization exist. And CUDA's decade head start in
> kernels and libraries is a real moat: ROCm (AMD), Metal/MPS (Apple) and
> compiler projects like Triton and `torch.compile` compete with it, but
> "does it run fast on NVIDIA?" is still the first question every ML system
> answers. The lesson of this chapter at production scale: the model is
> software, but the *product* is co-designed with silicon.

## Where this leaves the lab

Nothing in llm-ecosphere needs a GPU — that is a feature. Every experiment in
[08 — Exercises](08-exercises.md) retrains in minutes on a CPU, so "just
try it" stays a valid research strategy, which at real scale it never is.
But every concept this chapter introduced has a handle in this repo you can
touch: the matmuls of `model.py` are the FLOPs, `pick_device` is the
deployment seam, the naive attention is the kernel-engineering target, and
exercise 5's KV cache is the memory-bandwidth story in miniature.

Next: [09 — the character-tokenizer lab](09-char-tokenizer-lab.md) if you
have not read it yet, or back to [00 — Overview](00-overview.md), which
reads differently once you know what the hardware is doing.
