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
