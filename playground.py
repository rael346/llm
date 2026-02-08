import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    # (B, n_head, T, head_dim)
    L, S = query.size(-2), key.size(-2)

    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(key.size(-1))

    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    attn_weight.masked_fill_(temp_mask.logical_not(), -torch.inf)

    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


def manual_attention(
    x: torch.Tensor,
    query: nn.Linear,
    key: nn.Linear,
    value: nn.Linear,
    n_head: int,
    head_dim: int,
):
    q = query(x).view(1, -1, n_head, head_dim).transpose(1, 2)
    k = key(x).view(1, -1, n_head, head_dim).transpose(1, 2)
    v = value(x).view(1, -1, n_head, head_dim).transpose(1, 2)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn = attn.transpose(1, 2).contiguous().view(1, -1, n_head * head_dim)

    return attn


def main():
    # n_embd = 128
    # n_head = 4
    # head_dim = n_embd // n_head
    #
    # attn_query = nn.Linear(n_embd, n_embd, bias=False)
    # attn_key = nn.Linear(n_embd, n_embd, bias=False)
    # attn_value = nn.Linear(n_embd, n_embd, bias=False)
    #
    # # (T, n_head, head_dim)
    # x_list: list[torch.Tensor] = [
    #     torch.rand((random.randint(5, 10), n_embd)) for _ in range(2)
    # ]
    #
    # start = time.perf_counter()
    # attn_exp: list[torch.Tensor] = [
    #     manual_attention(x, attn_query, attn_key, attn_value, n_head, head_dim)
    #     for x in x_list
    # ]
    # end = time.perf_counter()
    # print(f"manual {end - start:.2f}")

    # x_cat = torch.cat(x_list)
    # seq_lens = [0]
    # seq_lens.extend([x.size(0) for x in x_list])
    # max_seq = max(seq_lens)
    # cu_seq = np.cumsum(seq_lens)
    #
    # total_len = cu_seq[-1]
    # local_mask = torch.ones(max_seq, max_seq, dtype=torch.bool).tril(diagonal=0)

    # for i, exp in enumerate(attn_exp):
    #     l, r = cu_seq[i], cu_seq[i + 1]
    #     torch.testing.assert_close(exp[0], new_attn[l:r])

    pass


if __name__ == "__main__":
    main()
