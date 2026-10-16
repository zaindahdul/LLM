"""Generate text from a trained checkpoint, streaming tokens to the terminal as they're sampled.

Usage:
  python sample.py                                   # continue "ROMEO:" with default settings
  python sample.py --prompt "JULIET:\nO Romeo," --tokens 300 --temperature 0.9
  python sample.py --temperature 0                   # greedy (deterministic)
  python sample.py --num 3 --seed 7                  # three independent samples
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from model import GPT
from tokenizer import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--prompt", default="ROMEO:\n", help=r"use \n for newlines")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--num", type=int, default=1, help="number of samples")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model, ckpt = GPT.from_checkpoint(args.ckpt)
    tok = BPETokenizer.load(ckpt["tokenizer"])
    prompt = args.prompt.replace("\\n", "\n")
    # An empty prompt has nothing to condition on; a newline is a natural "start of a line".
    ids = tok.encode(prompt) or tok.encode("\n")
    g = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    for n in range(args.num):
        if args.num > 1:
            print(f"\n{'=' * 30} sample {n + 1} {'=' * 30}")
        print(prompt, end="", flush=True)
        generated: list[int] = []
        printed = ""
        t0 = time.perf_counter()
        for nxt, _ in model.stream(torch.tensor([ids]), args.tokens, temperature=args.temperature,
                                   top_k=args.top_k, top_p=args.top_p, generator=g):
            generated.append(int(nxt))
            # Decode the whole tail and print only the new part: a single token can be half
            # of a multi-byte character, which only becomes printable once the rest arrives.
            text = tok.decode(generated)
            if not text.endswith("�"):
                print(text[len(printed):], end="", flush=True)
                printed = text
        dt = time.perf_counter() - t0
        print(f"\n\n[{args.tokens} tokens in {dt:.1f}s, {args.tokens / dt:.0f} tok/s | "
              f"checkpoint step {ckpt['step']}, val loss {ckpt['best_val']:.3f}]")


if __name__ == "__main__":
    main()
