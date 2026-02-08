import time
from dataclasses import dataclass

import polars as pl
import regex
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from llm_py.tokenizer import Chunk, Pair, decode, encode_inplace


def line_to_pair(line: str) -> Pair:
    split = line.split(" ")
    return (int(split[0]), int(split[1]))


def main():
    with open("./cache/merges.txt", "r") as f:
        merges: list[Pair] = [line_to_pair(line) for line in f]

    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    for l, r in merges:
        vocab.append(vocab[l] + vocab[r])

    gpt4_split_pattern = r"""
        # shorten form of words like "will" ('ll), "have" ('ve), etc
        '(?i:[sdmt]|ll|ve|re)
        #
        |[^\r\n\p{L}\p{N}]?+\p{L}+
        # numbers between 0 - 999
        |\p{N}{1,3}
        |[ ]?[^\s\p{L}\p{N}]++[\r\n]*
        |\s*[\r\n]
        |\s+(?!\S)
        |\s+
        """
    split_pattern = regex.compile(gpt4_split_pattern, regex.VERBOSE)

    start = time.perf_counter()
    chunk_indexes: dict[str, int] = {}
    curr_chunk_idx: int = 0
    rows: list[list[int]] = []
    for i in range(0, 1):
        df = pl.read_parquet(f"./dataset/shard_{i:05d}.parquet")

        for row_text in df["text"]:
            row: list[int] = []
            for chunk_text in split_pattern.findall(row_text):
                if chunk_text not in chunk_indexes:
                    chunk_indexes[chunk_text] = curr_chunk_idx
                    curr_chunk_idx += 1
                row.append(chunk_indexes[chunk_text])
            rows.append(row)

    end = time.perf_counter()
    print(f"regex chunking {end - start:.2f}")

    start = time.perf_counter()
    chunks: list[list[int]] = [list(text.encode()) for text, _ in chunk_indexes.items()]
    encode_inplace(merges, chunks)
    end = time.perf_counter()
    print(f"encoding {end - start:.2f}")

    start = time.perf_counter()
    token_seqs: list[list[int]] = [
        [tok for chunk_idx in row for tok in chunks[chunk_idx]] for row in rows
    ]
    end = time.perf_counter()
    print(f"To tokens {end - start:.2f}")

    seq_lens = [len(seq) for seq in token_seqs]
    print("max seq len", max(seq_lens))
    print("max seq len", min(seq_lens))
    print("avg seq len", sum(seq_lens) / len(seq_lens))
    seq_bins: dict[int, list[int]] = {}
    bin_size = 10
    for seq_len in seq_lens:
        bin_idx = seq_len // bin_size
        if bin_idx not in seq_bins:
            seq_bins[bin_idx] = []
        seq_bins[bin_idx].append(seq_len)

    for bin_idx, seq_lens in seq_bins.items():
        print(f"bin {bin_idx} | {len(seq_lens)}")

    # df = pl.read_parquet(f"./dataset/shard_{0:05d}.parquet")
    # assert decode(vocab, [], tokens[0]) == df["text"][0]
    # ratio = len(df["text"][0].encode("utf-8")) / len(token_seqs[0])
    # print(f"{ratio:<7.2f}")


if __name__ == "__main__":
    main()
