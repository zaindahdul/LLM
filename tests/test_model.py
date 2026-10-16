import math

import pytest
import torch
import torch.nn.functional as F

from model import GPT, CausalSelfAttention, GPTConfig

SMALL = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(SMALL).eval()


def test_output_shapes(model):
    idx = torch.randint(0, SMALL.vocab_size, (3, 20))
    logits, loss = model(idx)
    assert logits.shape == (3, 20, SMALL.vocab_size)
    assert loss is None


def test_initial_loss_is_near_uniform(model):
    idx = torch.randint(0, SMALL.vocab_size, (8, SMALL.block_size))
    # Targets must be independent of inputs: with weight tying, the input token's own
    # embedding is still in the residual stream, so "repeat the input" scores high at init.
    targets = torch.randint(0, SMALL.vocab_size, (8, SMALL.block_size))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(SMALL.vocab_size)) < 0.15


def test_causal_no_future_leakage(model):
    idx = torch.randint(0, SMALL.vocab_size, (2, SMALL.block_size))
    logits1, _ = model(idx)
    idx2 = idx.clone()
    idx2[:, 10:] = torch.randint(0, SMALL.vocab_size, (2, SMALL.block_size - 10))
    logits2, _ = model(idx2)
    # Positions 0..9 never saw tokens 10+, so their predictions must be identical.
    torch.testing.assert_close(logits1[:, :10], logits2[:, :10])
    assert not torch.allclose(logits1[:, 10:], logits2[:, 10:])


def test_attention_matches_pytorch_reference():
    # Our hand-written attention must match PyTorch's fused implementation exactly
    # (used only here, as an independent check of the math).
    torch.manual_seed(0)
    attn = CausalSelfAttention(SMALL).eval()
    x = torch.randn(2, 16, SMALL.n_embd)
    ours = attn(x)

    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, dim=2)
    q, k, v = (t.view(B, T, SMALL.n_head, C // SMALL.n_head).transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).reshape(B, T, C))
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_weight_tying(model):
    assert model.lm_head.weight is model.tok_emb.weight
    # The tied matrix must be counted once.
    unique = {id(p): p.numel() for p in model.parameters()}
    assert model.num_params() == sum(unique.values())


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    m = GPT(SMALL)
    idx = torch.randint(0, SMALL.vocab_size, (2, SMALL.block_size))
    _, loss = m(idx, idx)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient for {name}"


def test_rejects_sequence_longer_than_block_size(model):
    with pytest.raises(AssertionError):
        model(torch.zeros(1, SMALL.block_size + 1, dtype=torch.long))
