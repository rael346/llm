import concurrent.futures
import os
from pathlib import Path
from typing import Callable

import requests
import rich
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

BASE_URL = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
MAX_N_SHARDS = 1823
DOWNLOAD_DIR = "dataset"
shard_name: Callable[[int], str] = lambda index: f"shard_{index:05d}.parquet"


def download_file(
    progress: Progress,
    task_id: TaskID,
    file_index: int,
) -> tuple[bool, str, str]:
    progress.update(task_id, visible=True)
    url = f"{BASE_URL}/{shard_name(file_index)}"
    filepath = os.path.join(DOWNLOAD_DIR, shard_name(file_index))

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        progress.update(task_id, total=total_size)

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(task_id, advance=len(chunk))

        return True, url, f"Successfully downloaded to {filepath}"
    except (requests.exceptions.RequestException, Exception) as e:
        return False, url, f"Failed: {str(e)}"
    finally:
        progress.update(task_id, visible=False)


def main():
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    n_shards = min(MAX_N_SHARDS, 240)

    with (
        Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="left"),
            # "[progress.percentage]{task.percentage:>3.1f}%",
            # "|",
            BarColumn(),
            DownloadColumn(),
            "|",
            TimeRemainingColumn(),
        ) as progress,
        concurrent.futures.ThreadPoolExecutor() as executor,
    ):
        futures = [
            executor.submit(
                download_file,
                progress,
                progress.add_task(
                    "download", start=False, visible=False, filename=shard_name(i)
                ),
                i,
            )
            for i in range(n_shards)
        ]

        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    console = rich.console.Console()
    # Print summary
    console.print("\n[bold]Download Summary:[/bold]")
    successful = sum(1 for success, _, _ in results if success)
    failed = len(results) - successful

    console.print(f"[green]✓ Successful: {successful}[/green]")
    console.print(f"[red]✗ Failed: {failed}[/red]\n")

    # Print details of failed downloads
    if failed > 0:
        console.print("[bold red]Failed downloads:[/bold red]")
        for success, filename, message in results:
            if not success:
                console.print(f"  [red]• {filename}: {message}[/red]")


if __name__ == "__main__":
    main()
