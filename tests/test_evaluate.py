import torch
import torch.nn.functional as F

from evaluate import total_nll
from model import GPT, GPTConfig


def test_short_sequence_matches_single_forward():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0), verbose=False).eval()
    ids = torch.randint(0, 50, (12,))
    logits, _ = m(ids[None, :-1])
    expected = F.cross_entropy(logits[0], ids[1:], reduction="sum").item()
    assert abs(total_nll(m, ids) - expected) < 1e-3


def test_long_sequence_scores_every_token_exactly_once():
    # With no layers and zeroed position embeddings, each prediction depends only on the
    # current token, so the windowing must not matter: the result has to equal the plain
    # sum of per-token losses. Double counting or skipping any token would change it.
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, block_size=16, n_layer=0, n_head=2, n_embd=32, dropout=0.0)
    m = GPT(cfg, verbose=False).eval()
    with torch.no_grad():
        m.pos_emb.weight.zero_()
    for n in (17, 18, 40, 41, 100):
        ids = torch.randint(0, 50, (n,))
        logits = m.lm_head(m.ln_f(m.tok_emb(ids[:-1])))
        expected = F.cross_entropy(logits, ids[1:], reduction="sum").item()
        assert abs(total_nll(m, ids) - expected) < 1e-3, n
