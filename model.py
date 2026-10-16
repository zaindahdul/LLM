"""A small decoder-only (GPT-style) transformer, written from torch primitives.

Data flow for a batch of token ids `idx` with shape (B, T):

  token embedding (B,T,C) + position embedding (T,C)
    -> N x Block:  x = x + Attention(LayerNorm(x))      # tokens exchange information
                   x = x + MLP(LayerNorm(x))            # each token "thinks" on its own
    -> final LayerNorm -> linear head (tied to the token embedding) -> logits (B,T,vocab)

Logits at position t are the model's prediction for token t+1. The causal mask guarantees
position t only ever sees positions <= t, so one forward pass trains on T predictions at once.

Usage:  python model.py      # print parameter count, sanity checks, and a speed benchmark
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    # Defaults (~5.3M params) were picked by benchmarking on an 8-core laptop CPU: ~5k
    # tokens/sec trains for ~27 epochs of Tiny Shakespeare in 30 min. A 384-wide model
    # (11.5M) only manages ~9 epochs in the same time. See README for the comparison.
    vocab_size: int = 2048   # must match the tokenizer
    block_size: int = 128    # max context length (tokens the model can look back over)
    n_layer: int = 6         # number of transformer blocks
    n_head: int = 4          # attention heads per block; n_embd must divide evenly (64 each)
    n_embd: int = 256        # width of the residual stream (embedding dimension)
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position can only attend to itself and the past."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # One Linear producing Q, K and V together: same math as three separate layers,
        # but a single larger matmul is faster.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        # Output projection mixes the concatenated heads back into the residual stream.
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj.is_residual_proj = True  # flag for scaled init (see GPT._init_weights)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular matrix: mask[i, j] = 1 iff j <= i (query i may look at key j).
        # A buffer moves with .to(device) but is not a trainable parameter; non-persistent
        # keeps it out of checkpoints since it's trivially rebuilt.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)  # each (B, T, C)
        # Split the channels into heads and move the head axis forward so each head is an
        # independent attention computation: (B, T, C) -> (B, n_head, T, head_dim).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: how much each query matches each key. (B, nh, T, T)
        # Dividing by sqrt(head_dim) keeps the dot products ~unit variance; without it,
        # large scores saturate the softmax into near one-hot and gradients vanish.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask: set scores for future keys to -inf so softmax gives them weight 0.
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)  # each row now sums to 1: a weighting over the past
        att = self.attn_dropout(att)
        y = att @ v  # weighted average of value vectors: (B, nh, T, head_dim)

        # Re-assemble heads side by side: (B, nh, T, hd) -> (B, T, nh*hd = C).
        # contiguous() is needed because transpose only changes strides, and view needs
        # the memory laid out in the new order.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU non-linearity, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.proj.is_residual_proj = True
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformer block with pre-LayerNorm and residual connections."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize the *input* to each sub-layer, leave the residual path untouched.
        # The un-normalized "x +" highway lets gradients flow straight to early layers,
        # which makes deep stacks train stably without careful warmup tricks (vs. post-LN).
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # position -> vector (learned)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)  # pre-LN leaves the last block's output un-normalized
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the output head reuses the input embedding matrix. Both map between
        # "token" and "vector" space, so sharing them saves vocab*n_embd params and tends to
        # improve small models (rare tokens get gradient from both ends).
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        print(f"GPT: {self.num_params() / 1e6:.2f}M parameters "
              f"(n_layer={cfg.n_layer}, n_head={cfg.n_head}, n_embd={cfg.n_embd}, "
              f"block_size={cfg.block_size}, vocab={cfg.vocab_size})")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02  # GPT-2's init: small weights keep initial logits near uniform
            if getattr(module, "is_residual_proj", False):
                # Each block adds 2 outputs into the residual stream, so its variance grows
                # with depth; scale those projections by 1/sqrt(2*n_layer) to compensate.
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # .parameters() de-duplicates shared tensors, so the tied matrix is counted once.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. targets: (B, T) next-token ids, or None at inference."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        # (B,T,C) + (T,C): positions broadcast over the batch. Without position embeddings
        # attention is order-blind ("dog bites man" == "man bites dog").
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab): unnormalized next-token scores

        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab) vs (B*T,): every position is an independent
            # classification problem "which of vocab_size tokens comes next?".
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def _benchmark(cfg: GPTConfig, batch_size: int, steps: int = 5) -> float:
    """Training throughput (tokens/sec) for forward + backward + AdamW step."""
    import time
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))
    for i in range(steps + 1):  # step 0 is warm-up (allocations, kernel selection)
        if i == 1:
            t0 = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return steps * batch_size * cfg.block_size / (time.perf_counter() - t0)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = GPT(cfg)

    # Sanity 1: at init the model should be ~uniform over the vocab, so the expected
    # cross-entropy is -ln(1/vocab_size) = ln(vocab_size). Targets are drawn independently
    # of inputs: with weight tying, "repeat the input token" scores above uniform at init.
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    targets = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    with torch.no_grad():
        logits, loss = model(idx, targets)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"initial loss: {loss.item():.3f}  (ideal uniform: ln({cfg.vocab_size}) = "
          f"{math.log(cfg.vocab_size):.3f})")

    # Sanity 2: causality. Changing the last token must not change any earlier logits.
    idx2 = idx.clone()
    idx2[:, -1] = (idx2[:, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits2, _ = model(idx2)
    leak = (logits[:, :-1] - logits2[:, :-1]).abs().max().item()
    print(f"causality check: max change in earlier logits = {leak:.2e} (should be 0)")

    print(f"\ntraining throughput: {_benchmark(cfg, batch_size=16):,.0f} tokens/sec "
          f"(batch 16 x {cfg.block_size}, {torch.get_num_threads()} CPU threads)")
