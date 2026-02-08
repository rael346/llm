from dataclasses import dataclass
from typing import cast, final, override

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass()
class GPTConfig:
    vocab_size: int
    context_len: int
    n_embd: int
    n_head: int
    n_blocks: int


@final
class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"embedding dimension {config.n_embd} is not divisible by number of attention heads {config.n_head}"
        )
        self.c = config

        self.tok_embs = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embs = nn.Embedding(config.context_len, config.n_embd)
        self.register_buffer("pos_idx", torch.arange(config.context_len))

        self.transformer = nn.Sequential(
            *[
                TransformerBlock(config.n_embd, config.n_head)
                for _ in range(config.n_blocks)
            ],
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.vocab_size, bias=False),
        )

    @override
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # (B, T)
        seq_len = tokens.size(dim=-1)
        assert seq_len <= self.c.context_len, (
            f"Cannot forward sequence of length {seq_len} when context length is {self.c.context_len}"
        )

        # (B, T, C)
        tok_emb = self.tok_embs(tokens)

        # (B, T, C)
        pos_emb = self.pos_embs(cast(torch.Tensor, self.pos_idx)[:seq_len])

        return self.transformer(tok_emb + pos_emb)


@final
class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.head_dim = n_embd // n_head
        self.attn_norm = nn.LayerNorm(n_embd)
        self.attn_query = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_key = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_value = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_mlp = nn.Linear(n_embd, n_embd)

        self.mlp = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, context_len, n_embd)
        B, T, C = x.size()

        x_norm = self.attn_norm(x)
        q = self.attn_query(x_norm).view(B, T, -1, self.head_dim).transpose(1, 2)
        k = self.attn_key(x_norm).view(B, T, -1, self.head_dim).transpose(1, 2)
        v = self.attn_value(x_norm).view(B, T, -1, self.head_dim).transpose(1, 2)

        # (B, n_head, T, head_dim)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        #    (B, n_head, T, head_dim)
        # -> (B, T, n_head, head_dim)
        # -> (B, T, C)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        # (B, T, C)
        # residual for self-attention
        attn = x + self.attn_mlp(attn)

        # (B, T, C)
        # residual for mlp
        out = attn + self.mlp(attn)
        return out
