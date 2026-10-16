import torch

from model import GPT, GPTConfig, KVCache, sample_next

SMALL = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)


def _model():
    torch.manual_seed(0)
    return GPT(SMALL, verbose=False).eval()


def test_kv_cache_matches_full_forward():
    m = _model()
    idx = torch.randint(0, SMALL.vocab_size, (2, 24))
    full, _ = m(idx)
    # Prefill 10 tokens, then feed the rest one by one through the cache.
    cache = KVCache(SMALL.n_layer)
    first, _ = m(idx[:, :10], cache=cache)
    steps = [first]
    for t in range(10, 24):
        logits, _ = m(idx[:, t:t + 1], cache=cache)
        steps.append(logits)
    assert len(cache) == 24
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)


def test_greedy_generation_matches_naive_loop():
    m = _model()
    idx = torch.randint(0, SMALL.vocab_size, (2, 5))
    fast = m.generate(idx, max_new_tokens=20, temperature=0)
    # Reference: re-run the full context through the model for every token, no cache.
    slow = idx
    for _ in range(20):
        logits, _ = m(slow)
        slow = torch.cat([slow, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    torch.testing.assert_close(fast, slow)


def test_generation_past_block_size():
    m = _model()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=3 * SMALL.block_size,
                     temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 1 + 3 * SMALL.block_size)
    assert int(out.max()) < SMALL.vocab_size


def test_generation_restores_training_mode():
    m = _model().train()
    m.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=2)
    assert m.training


def test_stream_yields_valid_distributions():
    m = _model()
    for nxt, probs in m.stream(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=5):
        assert nxt.shape == (1,)
        assert probs.shape == (1, SMALL.vocab_size)
        torch.testing.assert_close(probs.sum(-1), torch.ones(1))


def test_top_k_one_and_tiny_top_p_are_greedy():
    logits = torch.randn(4, 50)
    g = torch.Generator().manual_seed(0)
    greedy = logits.argmax(-1)
    for _ in range(20):
        torch.testing.assert_close(sample_next(logits, top_k=1, generator=g), greedy)
        torch.testing.assert_close(sample_next(logits, top_p=1e-6, generator=g), greedy)


def test_top_k_only_samples_from_top_k():
    logits = torch.arange(10, dtype=torch.float).repeat(1000, 1)  # token 9 is best
    picks = sample_next(logits, temperature=5.0, top_k=3, generator=torch.Generator().manual_seed(0))
    assert set(picks.tolist()) == {7, 8, 9}


def test_top_p_keeps_the_nucleus():
    # probs 0.5, 0.3, 0.15, 0.05: p=0.75 needs the top two (0.5 alone is not enough).
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).repeat(2000, 1)
    picks = sample_next(probs.log(), top_p=0.75, generator=torch.Generator().manual_seed(0))
    assert set(picks.tolist()) == {0, 1}


def test_low_temperature_concentrates_mass():
    logits = torch.tensor([[2.0, 1.0, 0.0]]).repeat(2000, 1)
    g = torch.Generator().manual_seed(0)
    cold = (sample_next(logits, temperature=0.2, generator=g) == 0).float().mean()
    hot = (sample_next(logits, temperature=2.0, generator=g) == 0).float().mean()
    assert cold > 0.95 > hot


def test_attention_maps_are_causal_distributions():
    m = _model()
    maps = m.attention_maps(torch.randint(0, SMALL.vocab_size, (1, 12)))
    assert maps.shape == (SMALL.n_layer, 1, SMALL.n_head, 12, 12)
    torch.testing.assert_close(maps.sum(-1), torch.ones(SMALL.n_layer, 1, SMALL.n_head, 12))
    assert maps.triu(diagonal=1).abs().max() == 0  # nothing attends to the future
    assert all(not b.attn.record_attn and b.attn.last_attn is None for b in m.blocks)
