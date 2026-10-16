"""Score a checkpoint on the full held-out split.

Usage:  python evaluate.py [--ckpt checkpoints/best.pt]

Reports loss per token (what training optimizes) and per character, which is comparable
across tokenizers: total negative log-likelihood of the val text / its length in characters.
The val tokens are scored in windows of block_size that overlap by half, and each token is
counted once, from the window where it has the most context (at least block_size/2 tokens,
except at the very start of the split).
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from data import load_tokens, split
from model import GPT
from tokenizer import BPETokenizer


@torch.no_grad()
def total_nll(model: GPT, ids: torch.Tensor) -> float:
    """Sum of -ln p(token | context) over ids[1:], using half-overlapping windows."""
    bs = model.cfg.block_size
    stride = bs // 2
    nll, scored = 0.0, 1  # ids[0] has no context and is never predicted
    start = 0
    while scored < len(ids):
        window = ids[start:start + bs + 1]
        logits, _ = model(window[None, :-1])
        losses = F.cross_entropy(logits[0], window[1:], reduction="none")
        # Position j in the window predicts ids[start + j + 1]; keep only not-yet-scored ones.
        first_new = scored - (start + 1)
        nll += losses[first_new:].sum().item()
        scored = start + len(window)
        start += stride
    return nll


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the validation split.")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--data", default="data/input.txt")
    args = p.parse_args()
    model, ckpt = GPT.from_checkpoint(args.ckpt)
    tok = BPETokenizer.load(ckpt["tokenizer"])
    _, val = split(load_tokens(args.data, tok))
    nll = total_nll(model, val)
    n_tok, n_chars = len(val) - 1, len(tok.decode(val[1:].tolist()))
    print(f"val tokens: {n_tok:,}  chars: {n_chars:,}  ({n_chars / n_tok:.2f} chars/token)")
    print(f"loss/token: {nll / n_tok:.4f} nats   perplexity: {math.exp(nll / n_tok):.1f}")
    print(f"loss/char:  {nll / n_chars:.4f} nats   ({nll / n_chars / math.log(2):.3f} bits per character)")


if __name__ == "__main__":
    main()
