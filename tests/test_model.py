"""The Transformer: shapes, causality, generation, checkpoints."""

import torch

from minillm.config import ModelConfig
from minillm.model import GPT

CFG = ModelConfig(dropout=0.0)  # deterministic for testing


def test_forward_shapes_and_finite_loss():
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, 12))
    y = torch.randint(0, CFG.vocab_size, (2, 12))
    logits, loss = model(x, y)
    assert logits.shape == (2, 12, CFG.vocab_size)
    assert torch.isfinite(loss)


def test_inference_returns_last_position_only():
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (2, 12))
    logits, loss = model(x)
    assert logits.shape == (2, 1, CFG.vocab_size)
    assert loss is None


def test_causality_future_does_not_leak_into_past():
    """Changing token t must not change any logits before t."""
    model = GPT(CFG).eval()
    x1 = torch.randint(0, CFG.vocab_size, (1, 10))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size  # tamper with the last token
    dummy = torch.zeros(1, 10, dtype=torch.long)  # targets force full logits
    logits1, _ = model(x1, dummy)
    logits2, _ = model(x2, dummy)
    assert torch.allclose(logits1[0, :9], logits2[0, :9], atol=1e-5)
    assert not torch.allclose(logits1[0, 9], logits2[0, 9], atol=1e-5)


def test_generate_respects_allowed_ids_and_stop():
    model = GPT(CFG).eval()
    idx = torch.tensor([[1]])
    out = model.generate(idx, max_new_tokens=3, temperature=0.0, allowed_ids=[7])
    assert out[0, 1:].tolist() == [7, 7, 7]
    out = model.generate(idx, max_new_tokens=5, temperature=0.0, allowed_ids=[2], stop_id=2)
    assert out.shape[1] == 2  # stopped right after emitting the stop token


def test_generate_stays_within_vocab_and_block():
    model = GPT(CFG).eval()
    idx = torch.tensor([[1]])
    gen = torch.Generator().manual_seed(0)
    out = model.generate(idx, max_new_tokens=12, temperature=1.0, generator=gen)
    assert out.shape[1] <= 13
    assert out.min() >= 0 and out.max() < CFG.vocab_size


def test_checkpoint_roundtrip(tmp_path):
    model = GPT(CFG).eval()
    path = tmp_path / "model.pt"
    torch.save(model.checkpoint_dict(stage="test", step=0, val_loss=0.0), path)
    ckpt = torch.load(path, weights_only=True)
    restored = GPT.from_checkpoint(ckpt, torch.device("cpu"))
    x = torch.randint(0, CFG.vocab_size, (1, 8))
    dummy = torch.zeros(1, 8, dtype=torch.long)
    logits_a, _ = model(x, dummy)
    logits_b, _ = restored(x, dummy)
    assert torch.allclose(logits_a, logits_b)


def test_weight_tying():
    model = GPT(CFG)
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()


# ----------------------------------------------------------------------
# KV cache (exercise 5): cached and naive paths must agree exactly
# ----------------------------------------------------------------------
def test_kv_cache_logits_match_the_full_forward_at_every_step():
    """Incremental (prefill + one token at a time) logits must equal the
    naive full-prefix forward pass at every generation step."""
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (1, 12))

    kv_caches = [{} for _ in model.transformer.h]
    prompt = ids[:, :4]
    cached, _ = model(prompt, kv_caches=kv_caches)  # prefill
    full, _ = model(prompt)
    assert torch.allclose(cached[:, -1], full[:, -1], atol=1e-5)

    for t in range(4, ids.size(1)):
        cached, _ = model(ids[:, t : t + 1], kv_caches=kv_caches, pos_offset=t)
        full, _ = model(ids[:, : t + 1])
        assert torch.allclose(cached[:, -1], full[:, -1], atol=1e-5)
        # The cache now covers t+1 positions in every layer.
        assert all(c["k"].size(2) == t + 1 for c in kv_caches)


def test_kv_cache_greedy_generation_is_token_for_token_identical():
    """The exercise's correctness gate, for both vocabulary shapes."""
    for vocab, block in ((15, 16), (13, 24)):  # move-level and char-level
        torch.manual_seed(1)
        model = GPT(ModelConfig(vocab_size=vocab, block_size=block, dropout=0.0)).eval()
        prompt = torch.tensor([[1]])
        naive = model.generate(prompt, max_new_tokens=block - 1, temperature=0.0)
        cached = model.generate(prompt, max_new_tokens=block - 1, temperature=0.0,
                                use_cache=True)
        assert naive.tolist() == cached.tolist()


def test_kv_cache_respects_stop_and_allowed_ids():
    model = GPT(CFG).eval()
    idx = torch.tensor([[1]])
    out = model.generate(idx, max_new_tokens=3, temperature=0.0, allowed_ids=[7],
                         use_cache=True)
    assert out[0, 1:].tolist() == [7, 7, 7]
    out = model.generate(idx, max_new_tokens=5, temperature=0.0, allowed_ids=[2],
                         stop_id=2, use_cache=True)
    assert out.shape[1] == 2


def test_kv_cache_refuses_to_slide_the_context_window():
    """The cache stores absolute positions, so the whole generation must
    fit in block_size — the naive path crops a window instead, which a
    cache of absolute positions cannot do."""
    import pytest

    model = GPT(CFG).eval()
    idx = torch.tensor([[1]])
    with pytest.raises(AssertionError):
        model.generate(idx, max_new_tokens=CFG.block_size + 1, temperature=0.0,
                       use_cache=True)


def test_pos_offset_embeds_tokens_at_their_absolute_position():
    """A token fed alone is still at its absolute position — the first
    classic KV-cache bug is embedding it at position 0."""
    torch.manual_seed(2)
    model = GPT(CFG).eval()
    ids = torch.randint(0, CFG.vocab_size, (1, 6))
    kv_caches = [{} for _ in model.transformer.h]
    model(ids[:, :5], kv_caches=kv_caches)
    right, _ = model(ids[:, 5:6], kv_caches=kv_caches, pos_offset=5)
    wrong_caches = [{} for _ in model.transformer.h]
    model(ids[:, :5], kv_caches=wrong_caches)
    wrong, _ = model(ids[:, 5:6], kv_caches=wrong_caches, pos_offset=0)
    full, _ = model(ids)
    assert torch.allclose(right[:, -1], full[:, -1], atol=1e-5)
    assert not torch.allclose(wrong[:, -1], full[:, -1], atol=1e-4)
