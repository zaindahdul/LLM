"""Turn the raw corpus into batches of (input, target) token windows for training.

The whole corpus is tokenized once into a flat 1-D tensor of ids and cached on disk. A
training example is any window of block_size+1 consecutive tokens: the first block_size
are the input, and the same window shifted left by one is the target, so position t is
trained to predict token t+1.

The split is a contiguous cut (first 90% train, last 10% val) rather than a random
shuffle of windows: overlapping windows from the same passage would otherwise appear in
both sets and val loss would flatter the model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tokenizer import BPETokenizer


def load_tokens(text_path: str | Path, tok: BPETokenizer, cache_dir: str | Path = "data") -> torch.Tensor:
    """Encode the corpus to a 1-D int64 tensor, caching the result next to the data.

    The cache key hashes both the text and the tokenizer's merges, so retraining the
    tokenizer or changing the corpus can never silently reuse stale ids.
    """
    text = Path(text_path).read_text(encoding="utf-8")
    h = hashlib.sha256(text.encode("utf-8"))
    h.update(repr(list(tok.merges)).encode())
    cache = Path(cache_dir) / f"tokens-{h.hexdigest()[:16]}.pt"
    if cache.exists():
        return torch.load(cache)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache)
    return ids


def split(ids: torch.Tensor, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def get_batch(ids: torch.Tensor, batch_size: int, block_size: int,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows. Returns x, y of shape (B, T) with y = x shifted by one."""
    if len(ids) <= block_size:
        raise ValueError(f"need more than block_size={block_size} tokens, got {len(ids)}")
    starts = torch.randint(0, len(ids) - block_size, (batch_size,), generator=generator)
    # Gather all windows with one indexing op instead of a Python loop of slices.
    offsets = torch.arange(block_size + 1)
    windows = ids[starts[:, None] + offsets]  # (B, T+1)
    return windows[:, :-1], windows[:, 1:]
