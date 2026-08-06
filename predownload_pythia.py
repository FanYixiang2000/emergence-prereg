"""Pre-download Pythia checkpoint revisions via huggingface_hub.

Network-bound helper so the GPU probe can run entirely from local files.
Downloads config.json, the safetensors index (if present) and every shard
listed in the index, per revision, into external_pythia_<size>/step<N>.
Uses hf_hub_download's built-in resume; retries transient failures.
Does not touch any experiment output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

HERE = Path(__file__).resolve().parent

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
         1000, 2000, 4000, 8000, 16000, 32000, 64000,
         96000, 128000, 143000]


def fetch(repo: str, rev: str, fname: str, dest: Path) -> Path:
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            return Path(hf_hub_download(
                repo_id=repo, filename=fname, revision=rev,
                local_dir=dest))
        except EntryNotFoundError:
            raise
        except Exception as exc:  # transient network failures
            last_exc = exc
            wait = 10 * (attempt + 1)
            print(f"  {rev}/{fname}: retry after {wait}s ({exc})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{rev}/{fname} failed after retries") from last_exc


def download_step(repo: str, size: str, step: int) -> None:
    rev = f"step{step}"
    dest = HERE / f"external_pythia_{size}" / rev
    t0 = time.time()
    fetch(repo, rev, "config.json", dest)
    try:
        index_path = fetch(repo, rev, "model.safetensors.index.json", dest)
        shards = sorted(set(json.loads(
            index_path.read_text(encoding="utf-8"))["weight_map"].values()))
    except EntryNotFoundError:
        shards = ["model.safetensors"]
    for shard in shards:
        fetch(repo, rev, shard, dest)
    total = sum(f.stat().st_size for f in dest.rglob("*")
                if f.is_file() and not str(f).count(".cache"))
    print(f"{rev}: complete ({total / 2**30:.2f} GiB, "
          f"{time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="1.4b")
    parser.add_argument("--steps", type=int, nargs="*", default=STEPS)
    args = parser.parse_args()
    repo = f"EleutherAI/pythia-{args.size}"
    for step in args.steps:
        download_step(repo, args.size, step)
    print("ALL CHECKPOINTS READY", flush=True)


if __name__ == "__main__":
    main()
