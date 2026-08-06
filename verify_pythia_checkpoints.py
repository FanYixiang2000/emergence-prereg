"""Integrity audit for locally cached Pythia checkpoints.

For each stepN directory: verify config + all index-listed shards exist,
record SHA-256 of every shard, and confirm that no two revisions share an
identical shard set (the failure mode of the quarantined 2.8B mirror run,
where early revisions silently resolved to one weight object).

Writes outputs/pythia_<size>_checkpoint_hashes.json. Read-only otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def sha256(path: Path, chunk: int = 32 * 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="1.4b")
    args = parser.parse_args()
    root = HERE / f"external_pythia_{args.size}"
    report: dict = {"size": args.size, "steps": {}}
    problems: list[str] = []

    for d in sorted(root.glob("step*"), key=lambda p: int(p.name[4:])):
        entry: dict = {"complete": False, "shards": {}}
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            shards = sorted(set(json.loads(
                idx.read_text(encoding="utf-8"))["weight_map"].values()))
        else:
            shards = ["model.safetensors"]
        missing = [s for s in shards
                   if not (d / s).exists() or (d / s).stat().st_size == 0]
        if missing or not (d / "config.json").exists():
            problems.append(f"{d.name}: missing {missing or 'config.json'}")
            report["steps"][d.name] = entry
            continue
        for s in shards:
            entry["shards"][s] = {
                "bytes": (d / s).stat().st_size,
                "sha256": sha256(d / s),
            }
        entry["complete"] = True
        report["steps"][d.name] = entry
        print(f"{d.name}: {len(shards)} shards hashed", flush=True)

    fingerprint_to_steps = defaultdict(list)
    for step, entry in report["steps"].items():
        if entry["complete"]:
            fp = tuple(sorted(v["sha256"] for v in entry["shards"].values()))
            fingerprint_to_steps[fp].append(step)
    duplicates = {tuple(v): list(v) for v in fingerprint_to_steps.values()
                  if len(v) > 1}
    report["n_complete"] = sum(
        e["complete"] for e in report["steps"].values())
    report["duplicate_revision_groups"] = [
        steps for steps in fingerprint_to_steps.values() if len(steps) > 1]
    report["problems"] = problems
    report["all_revisions_distinct"] = not report["duplicate_revision_groups"]

    out = OUTPUTS / f"pythia_{args.size}_checkpoint_hashes.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"complete: {report['n_complete']}, "
          f"distinct: {report['all_revisions_distinct']}, "
          f"problems: {problems or 'none'}")
    print(f"Wrote {out}")
    if problems or not report["all_revisions_distinct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
