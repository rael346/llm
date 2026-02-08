import math
import random
import time

import mlx.core as mx
import mlx.nn as nn


def main():
    mx.random.seed(1234)
    n_embd = 256 * 4
    n_head = 16
    head_dim = n_embd // n_head

    query = nn.Linear(n_embd, n_embd, bias=False)
    key = nn.Linear(n_embd, n_embd, bias=False)
    value = nn.Linear(n_embd, n_embd, bias=False)

    batch = 100
    seq_len = 1000
    x = mx.random.uniform(shape=(batch, seq_len, n_embd))

    q = query(x).reshape(batch, -1, n_head, head_dim).transpose(0, 2, 1, 3)
    k = key(x).reshape(batch, -1, n_head, head_dim).transpose(0, 2, 1, 3)
    v = value(x).reshape(batch, -1, n_head, head_dim).transpose(0, 2, 1, 3)

    scale = math.sqrt(head_dim)
    start = time.perf_counter()
    attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / scale, mask="causal")
    end = time.perf_counter()
    print(f"fast {end - start:.4f}")
    # attn = attn.transpose(0, 2, 1, 3).reshape(1, -1, n_head * head_dim)

    # (B, n_head, T, head_dim)

    start = time.perf_counter()
    L, S = q.shape[-2], k.shape[-2]
    attn_weight = (q @ k.transpose(0, 1, 3, 2)) / scale

    temp_mask = mx.broadcast_to(
        mx.triu(mx.ones((L, S), dtype=mx.bool_), k=1), attn_weight.shape
    )

    attn_weight[temp_mask] = -mx.inf
    attn_weight = mx.softmax(attn_weight, axis=-1)
    manual_attn = attn_weight @ v
    end = time.perf_counter()
    print(f"slow {end - start:.4f}")

    # print(attn[0][1][1])
    # print(manual_attn[0][1][1])
    print(mx.allclose(attn, manual_attn, atol=1e-7))

    source = """
        uint elem = thread_position_in_grid.x;
        // Utils from `mlx/backend/metal/kernels/utils.h` are automatically included
        uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
        T tmp = inp[loc];
        // Output arrays are always row contiguous
        out[elem] = metal::exp(tmp);
    """

    kernel = mx.fast.metal_kernel(
        name="myexp_strided",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )

    def exp_elementwise(a: mx.array):
        outputs = kernel(
            inputs=[a],
            template=[("T", mx.float32)],
            grid=(a.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
            verbose=True,
        )
        return outputs[0]

    a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
    b = exp_elementwise(a)
    assert mx.allclose(b, mx.exp(a))


if __name__ == "__main__":
    main()
