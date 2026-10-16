"""Train the GPT on Tiny Shakespeare.

Usage:
  python train.py                      # 30-minute run with the default ~5.3M-param model
  python train.py --minutes 5          # quick run
  python train.py --steps 2000         # fixed step budget instead of a time budget
  python train.py --resume             # continue from checkpoints/last.pt

Outputs (in --out, default checkpoints/):
  best.pt     weights with the lowest validation loss so far (what sample.py/server.py load)
  last.pt     latest weights + optimizer state, for --resume
  log.jsonl   one JSON line per evaluation: step, time, losses, lr, throughput

The learning-rate schedule is linear warmup then cosine decay to 10% of the peak. With a
time budget, "progress" along the cosine is elapsed / budget rather than step / max_steps,
so the decay always finishes on time no matter how fast the machine is.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from data import get_batch, load_tokens, split
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    d = GPTConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--tokenizer", default="tokenizer.json")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--minutes", type=float, default=30.0, help="wall-clock budget (ignored if --steps)")
    p.add_argument("--steps", type=int, default=None, help="fixed number of optimizer steps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=200, help="warmup steps")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--log-every", type=int, default=50, help="print training loss every N steps")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-layer", type=int, default=d.n_layer)
    p.add_argument("--n-head", type=int, default=d.n_head)
    p.add_argument("--n-embd", type=int, default=d.n_embd)
    p.add_argument("--block-size", type=int, default=d.block_size)
    p.add_argument("--dropout", type=float, default=d.dropout)
    return p.parse_args()


def lr_at(step: int, progress: float, args: argparse.Namespace) -> float:
    if step < args.warmup:
        # Warmup: Adam's moment estimates are noisy for the first steps, and big updates
        # from random init can push the network somewhere it never recovers from.
        return args.lr * (step + 1) / args.warmup
    progress = min(max(progress, 0.0), 1.0)
    min_lr = 0.1 * args.lr
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, args: argparse.Namespace) -> torch.optim.AdamW:
    # Weight decay only on matrices (Linear weights, embeddings). Decaying biases and
    # LayerNorm gains just pulls them toward 0 for no regularization benefit.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor], args, cfg: GPTConfig) -> dict[str, float]:
    """Mean loss over a fixed set of batches per split (same batches every eval, so curves are smooth)."""
    model.eval()
    out = {}
    for name, ids in splits.items():
        g = torch.Generator().manual_seed(0)
        losses = [model(*get_batch(ids, args.batch_size, cfg.block_size, g))[1].item()
                  for _ in range(args.eval_batches)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def save(path: Path, model: GPT, cfg: GPTConfig, step: int, elapsed: float, best_val: float,
         args: argparse.Namespace, opt: torch.optim.Optimizer | None = None) -> None:
    ckpt = {"model": model.state_dict(), "config": asdict(cfg), "step": step,
            "elapsed": elapsed, "best_val": best_val, "tokenizer": args.tokenizer,
            "args": vars(args)}
    if opt is not None:
        ckpt["optimizer"] = opt.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a corrupt checkpoint


def main() -> None:
    args = parse_args()
    # Samples can contain any Unicode; the default Windows console codec (cp1252) can't.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = BPETokenizer.load(args.tokenizer)
    ids = load_tokens(args.data, tok)
    train_ids, val_ids = split(ids)
    print(f"Data: {len(ids):,} tokens ({len(train_ids):,} train / {len(val_ids):,} val)")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    step, elapsed_before, best_val = 0, 0.0, float("inf")
    resume = torch.load(out / "last.pt") if args.resume and (out / "last.pt").exists() else None
    if resume is not None:
        cfg = GPTConfig(**resume["config"])
    model = GPT(cfg)
    opt = make_optimizer(model, args)
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        step, elapsed_before, best_val = resume["step"], resume["elapsed"], resume["best_val"]
        print(f"Resumed from step {step} (best val {best_val:.4f})")
    elif args.resume:
        print("No checkpoint to resume from; starting fresh.")

    budget_s = args.minutes * 60
    log = (out / "log.jsonl").open("a" if resume else "w", encoding="utf-8")
    tokens_per_step = args.batch_size * cfg.block_size
    prompt = torch.tensor([tok.encode("ROMEO:\n")])

    def progress(elapsed: float) -> float:
        return step / args.steps if args.steps else elapsed / budget_s

    def done(elapsed: float) -> bool:
        return step >= args.steps if args.steps else elapsed >= budget_s

    def evaluate(elapsed: float, window_s: float, window_steps: int) -> None:
        nonlocal best_val
        losses = estimate_loss(model, {"train": train_ids, "val": val_ids}, args, cfg)
        tps = window_steps * tokens_per_step / max(window_s, 1e-9)
        improved = losses["val"] < best_val
        if improved:
            best_val = losses["val"]
            save(out / "best.pt", model, cfg, step, elapsed, best_val, args)
        save(out / "last.pt", model, cfg, step, elapsed, best_val, args, opt)
        lr = opt.param_groups[0]["lr"]
        print(f"step {step:6d} | {elapsed / 60:5.1f} min | train {losses['train']:.4f} | "
              f"val {losses['val']:.4f}{' *' if improved else '  '} | lr {lr:.2e} | {tps:,.0f} tok/s")
        log.write(json.dumps({"step": step, "time_s": round(elapsed, 1), "train_loss": losses["train"],
                              "val_loss": losses["val"], "lr": lr, "tokens_per_s": round(tps),
                              "tokens_seen": step * tokens_per_step}) + "\n")
        log.flush()
        sample = model.generate(prompt, 60, temperature=0.8, top_k=50,
                                generator=torch.Generator().manual_seed(step))
        print("   " + tok.decode(sample[0].tolist()).replace("\n", "\n   "))

    model.train()
    t0 = time.perf_counter()
    win_t, win_steps = t0, 0
    elapsed = elapsed_before
    try:
        while not done(elapsed):
            lr = lr_at(step, progress(elapsed), args)
            for group in opt.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_ids, args.batch_size, cfg.block_size)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Clip the global gradient norm: one weird batch can't blow up the weights.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            win_steps += 1
            elapsed = elapsed_before + time.perf_counter() - t0
            if step % args.log_every == 0 and step % args.eval_every:
                tps = win_steps * tokens_per_step / (time.perf_counter() - win_t)
                print(f"step {step:6d} | {elapsed / 60:5.1f} min | loss  {loss.item():.4f} | "
                      f"lr {lr:.2e} | {tps:,.0f} tok/s", flush=True)
            if step % args.eval_every == 0:
                now = time.perf_counter()
                evaluate(elapsed, now - win_t, win_steps)
                # Don't charge eval + sampling time against the throughput number.
                win_t, win_steps = time.perf_counter(), 0
    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint...")
    if win_steps:
        evaluate(elapsed, time.perf_counter() - win_t, win_steps)
    log.close()
    print(f"\nDone: {step} steps, {step * tokens_per_step:,} tokens, best val loss {best_val:.4f} "
          f"(perplexity {math.exp(best_val):.1f}). Weights: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
