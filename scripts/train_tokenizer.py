import multiprocessing
import os
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import regex
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from llm_py.tokenizer import train


def count_chunks(
    task_id: TaskID,
    index: int,
    stride: int,
    split_regex: str,
    q: queue.Queue[TaskID],
):
    split_pattern = regex.compile(split_regex, regex.VERBOSE)

    chunk_count: dict[str, int] = {}
    for i in range(index, index + stride):
        df = pl.read_parquet(f"./dataset/shard_{i:05d}.parquet")

        for row in df["text"]:
            for chunk in split_pattern.findall(row):
                chunk_count[chunk] = 1 + chunk_count.get(chunk, 0)

        q.put(task_id)

    return chunk_count


def calc_chunks(n_shards: int, split_regex: str):
    n_workers = os.process_cpu_count() or 1
    stride = n_shards // n_workers
    with (
        Progress(
            TextColumn("[bold blue]{task.fields[batch]}", justify="left"),
            BarColumn(),
            TaskProgressColumn(),
            "|",
            TimeRemainingColumn(),
        ) as progress,
        ProcessPoolExecutor(max_workers=n_workers) as executor,
        multiprocessing.Manager() as manager,
    ):
        progress_q: queue.Queue[TaskID] = manager.Queue()

        futures = [
            executor.submit(
                count_chunks,
                progress.add_task(
                    "download",
                    total=stride,
                    batch=f"{i} - {i + stride}",
                ),
                i,
                stride,
                split_regex,
                progress_q,
            )
            for i in range(0, n_shards, stride)
        ]

        while not progress.finished:
            task_id = progress_q.get()
            progress.advance(task_id, advance=1)

        results = [future.result() for future in as_completed(futures)]

    chunk_count: dict[str, int] = {}
    for local_chunk_count in results:
        for chunk, count in local_chunk_count.items():
            chunk_count[chunk] = count + chunk_count.get(chunk, 0)

    return chunk_count


def main():
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
    n_shards = 240

    chunk_count = calc_chunks(n_shards, gpt4_split_pattern)
    merges = train(chunk_count, 2**16, True)

    path = Path("cache/merges.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for m in merges:
            f.write(f"{m[0]} {m[1]}\n")


if __name__ == "__main__":
    main()
