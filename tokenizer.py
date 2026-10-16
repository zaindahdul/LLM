"""Byte-level Byte-Pair-Encoding (BPE) tokenizer, written from scratch.

The idea in four steps:
  1. Base vocabulary = the 256 possible byte values. Any string becomes UTF-8 bytes, so
     every input (emoji, Chinese, binary junk) is encodable and there is no <unk> token.
  2. train(): count every adjacent pair of tokens in the corpus, merge the most frequent
     pair into a brand-new token, and repeat until the vocabulary reaches the target size.
     The ordered list of merges IS the tokenizer.
  3. encode(): split text into chunks with a regex, turn each chunk into bytes, then
     replay the learned merges in the order they were learned.
  4. decode(): map each token id back to its bytes, concatenate, decode UTF-8.

Usage:
  python tokenizer.py --vocab-size 2048            # trains on data/input.txt -> tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path

# Pre-tokenization regex: a stdlib-`re` port of GPT-2's split pattern.
# Merges never cross chunk boundaries, which keeps "dog." and "dog!" sharing the token
# " dog" instead of wasting vocab on "dog." and "dog!" variants.
# Alternatives, in order:
#   '(?:[sdmt]|ll|ve|re)   English contractions: 's 't 'll 've 're ...
#    ?[^\W\d_]+            a run of letters, with an optional leading space (" hello")
#    ?\d{1,3}              1-3 digits (caps how many number tokens get learned)
#    ?(?:[^\s\w]|_)+       a run of punctuation / symbols / emoji (`_` is \w, so add it back)
#   \s+(?!\S)              whitespace, but leave the last space to attach to the next word
#   \s+                    any remaining whitespace (e.g. trailing at end of text)
# Every character falls into one of these classes, so the chunks always re-join to the
# original text, which is what makes encode->decode lossless.
SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""

_CACHE_LIMIT = 100_000  # max cached chunk encodings; bounds memory in a long-running server


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id` (left to right)."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2  # skip both halves of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None, pattern: str = SPLIT_PATTERN):
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self._reset()
        for pair in merges or []:
            self._add_merge(tuple(pair))

    def _reset(self) -> None:
        # merges: (left_id, right_id) -> new_id. New ids are handed out in order, so a
        # smaller id means the merge was learned earlier: the id doubles as merge priority.
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: id -> the raw bytes that token stands for.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._cache: dict[str, list[int]] = {}

    def _add_merge(self, pair: tuple[int, int]) -> int:
        new_id = len(self.vocab)
        self.merges[pair] = new_id
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return new_id

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
        self._reset()
        num_merges = vocab_size - 256

        # Pair statistics depend only on how often each distinct chunk occurs, so process each
        # unique chunk once and weight it by its frequency (" the" appears ~thousands of times
        # but is handled once). This is the main speedup over a naive implementation.
        chunk_counts = Counter(self._regex.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # pair_counts: total frequency of each adjacent pair across the corpus.
        # where: pair -> indices of words containing it, so each merge only touches the
        # words it affects instead of rescanning the whole corpus (incremental update).
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        for step in range(num_merges):
            if not pair_counts:
                print(f"Stopping early: corpus fully merged at vocab size {self.vocab_size}")
                break
            # Greedy: merge the most frequent pair. itemgetter keeps max() in C (fast).
            # Ties go to the pair seen first, which is deterministic for a given corpus.
            best, best_count = max(pair_counts.items(), key=itemgetter(1))
            new_id = self._add_merge(best)

            for wi in where.pop(best):
                old = words[wi]
                new = _merge(old, best, new_id)
                if len(new) == len(old):
                    continue  # stale index entry: an earlier merge already consumed this pair
                f = freqs[wi]
                # Simplest correct update: remove all of the word's old pairs, add its new ones.
                for pair in zip(old, old[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]  # keep the dict small so max() stays fast
                for pair in zip(new, new[1:]):
                    pair_counts[pair] += f
                    where[pair].add(wi)
                words[wi] = new

            if verbose and (step % 200 == 0 or step == num_merges - 1):
                print(f"merge {step + 1:5d}/{num_merges}: {self.vocab[best[0]]!r} + "
                      f"{self.vocab[best[1]]!r} -> id {new_id} {self.vocab[new_id]!r} "
                      f"(count {best_count})")

    # ------------------------------------------------------------------ encode / decode
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._regex.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached  # text is full of repeated words; caching makes encode ~10x faster
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the EARLIEST-learned merge available. Merges must be replayed in training
            # order because later merges were learned on top of earlier ones.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no adjacent pair is mergeable: done
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        # errors="replace": a token slice can end mid-character (e.g. half of a 4-byte emoji)
        # when decoding streamed/partial output. Decoding a full encode() result is still exact.
        return data.decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        return self.vocab[token_id]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        # Only the pattern and the ordered merges are saved; the vocab is rebuilt from them.
        payload = {"type": "byte-bpe", "pattern": self.pattern,
                   "merges": [list(pair) for pair in self.merges]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=[tuple(p) for p in payload["merges"]], pattern=payload["pattern"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the byte-level BPE tokenizer.")
    parser.add_argument("--input", default="data/input.txt")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--out", default="tokenizer.json")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    tok = BPETokenizer()
    t0 = time.perf_counter()
    tok.train(text, args.vocab_size, verbose=True)
    train_s = time.perf_counter() - t0
    tok.save(args.out)

    t0 = time.perf_counter()
    ids = tok.encode(text)
    encode_s = time.perf_counter() - t0
    assert tok.decode(ids) == text, "round-trip failed on the training corpus"

    n_bytes = len(text.encode("utf-8"))
    print(f"\nTrained vocab of {tok.vocab_size} in {train_s:.1f}s, saved to {args.out}")
    print(f"Corpus: {len(text):,} chars / {n_bytes:,} bytes -> {len(ids):,} tokens "
          f"({n_bytes / len(ids):.2f} bytes/token), encoded in {encode_s:.1f}s")
    longest = sorted(range(256, tok.vocab_size), key=lambda i: len(tok.vocab[i]), reverse=True)[:10]
    print("Longest tokens:", [tok.decode([i]) for i in longest])
    sample = "To be, or not to be, that is the question."
    pieces = [tok.decode([i]) for i in tok.encode(sample)]
    print(f"Sample: {sample!r}\n  -> {len(pieces)} tokens: {'|'.join(pieces)}")


if __name__ == "__main__":
    main()
