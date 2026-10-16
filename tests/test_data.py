import pytest
import torch

from data import get_batch, load_tokens, split
from tokenizer import BPETokenizer


def test_batch_targets_are_inputs_shifted_by_one():
    ids = torch.arange(1000)
    x, y = get_batch(ids, batch_size=8, block_size=16, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16)
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
    # With ids = arange, every window is consecutive integers.
    torch.testing.assert_close(y, x + 1)


def test_batch_windows_stay_in_bounds():
    ids = torch.arange(20)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        x, y = get_batch(ids, batch_size=4, block_size=19, generator=g)
        assert y.max() <= 19 and x.min() >= 0


def test_batch_rejects_tiny_corpus():
    with pytest.raises(ValueError):
        get_batch(torch.arange(16), batch_size=1, block_size=16)


def test_split_is_contiguous_and_complete():
    ids = torch.arange(100)
    train, val = split(ids, val_frac=0.1)
    assert len(train) == 90 and len(val) == 10
    torch.testing.assert_close(torch.cat([train, val]), ids)


def test_load_tokens_roundtrips_and_caches(tmp_path):
    text = "Now is the winter of our discontent\nMade glorious summer by this sun of York.\n" * 20
    corpus = tmp_path / "input.txt"
    corpus.write_text(text, encoding="utf-8")
    tok = BPETokenizer()
    tok.train(text, vocab_size=300)

    ids = load_tokens(corpus, tok, cache_dir=tmp_path)
    assert ids.dtype == torch.long
    assert tok.decode(ids.tolist()) == text
    assert len(list(tmp_path.glob("tokens-*.pt"))) == 1
    torch.testing.assert_close(load_tokens(corpus, tok, cache_dir=tmp_path), ids)

    # A different tokenizer must not hit the same cache entry.
    tok2 = BPETokenizer()
    tok2.train(text, vocab_size=280)
    load_tokens(corpus, tok2, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("tokens-*.pt"))) == 2
