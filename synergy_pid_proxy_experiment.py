"""PID-inspired synergy proxy experiment.

The goal is to connect possibility collapse to spatial emergence. We compare
four systems where future basin B is predicted by micro variables X1, X2:

- synergistic_xor: neither part predicts B alone; the joint state does.
- unique_x1: X1 alone predicts B.
- redundant: X1 and X2 both predict B.
- noise: neither individual nor joint state predicts B.

The synergy proxy is:

    max(I(X1,X2;B) - I(X1;B) - I(X2;B), 0)

This is not a full PID estimator, but it is a controlled demonstration of the
key idea: emergence requires joint structure, not just individual prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


OUTPUTS = Path(__file__).resolve().parent / "outputs"


def entropy_from_counts(counts: Mapping[object, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    result = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        result -= p * math.log(p, 2)
    return result


def mutual_information(samples: Sequence[Tuple[object, object]]) -> float:
    joint = Counter(samples)
    x_counts = Counter(x for x, _ in samples)
    y_counts = Counter(y for _, y in samples)
    return entropy_from_counts(x_counts) + entropy_from_counts(y_counts) - entropy_from_counts(joint)


def sample_system(system: str, n: int, seed: int) -> List[Tuple[int, int, int]]:
    rng = random.Random(seed)
    rows: List[Tuple[int, int, int]] = []
    for _ in range(n):
        x1 = rng.randint(0, 1)
        x2 = rng.randint(0, 1)
        if system == "synergistic_xor":
            b = x1 ^ x2
        elif system == "unique_x1":
            b = x1
        elif system == "redundant":
            x2 = x1
            b = x1
        elif system == "noise":
            b = rng.randint(0, 1)
        else:
            raise ValueError(f"unknown system: {system}")
        rows.append((x1, x2, b))
    return rows


def analyze_system(system: str, n: int, seed: int) -> Dict[str, float | str]:
    rows = sample_system(system, n, seed)
    i_x1 = mutual_information([(x1, b) for x1, _, b in rows])
    i_x2 = mutual_information([(x2, b) for _, x2, b in rows])
    i_joint = mutual_information([((x1, x2), b) for x1, x2, b in rows])
    synergy_proxy = max(i_joint - i_x1 - i_x2, 0.0)
    redundancy_proxy = max(i_x1 + i_x2 - i_joint, 0.0)
    b_entropy = entropy_from_counts(Counter(b for _, _, b in rows))
    collapse_from_joint = i_joint
    useful_value = 1.0 if system != "noise" else 0.0
    emergence_score = synergy_proxy * collapse_from_joint * useful_value
    return {
        "system": system,
        "h_basin": b_entropy,
        "i_x1_b": i_x1,
        "i_x2_b": i_x2,
        "i_joint_b": i_joint,
        "synergy_proxy": synergy_proxy,
        "redundancy_proxy": redundancy_proxy,
        "collapse_from_joint": collapse_from_joint,
        "useful_value": useful_value,
        "spatial_emergence_score": emergence_score,
    }


def run_all(n: int, seed: int, output_dir: Path) -> None:
    systems = ("synergistic_xor", "unique_x1", "redundant", "noise")
    rows = [analyze_system(system, n, seed + idx * 1000) for idx, system in enumerate(systems)]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "synergy_pid_proxy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (output_dir / "synergy_pid_proxy_summary.json").write_text(json.dumps({"summary": rows}, indent=2), encoding="utf-8")
    print("system,I(X1;B),I(X2;B),I(X1X2;B),synergy,score")
    for row in rows:
        print(
            f"{row['system']},{float(row['i_x1_b']):.4f},{float(row['i_x2_b']):.4f},"
            f"{float(row['i_joint_b']):.4f},{float(row['synergy_proxy']):.4f},"
            f"{float(row['spatial_emergence_score']):.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PID-inspired synergy proxy experiment.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(args.samples, args.seed, args.output_dir)
    print(f"\nWrote {args.output_dir / 'synergy_pid_proxy_summary.csv'}")
    print(f"Wrote {args.output_dir / 'synergy_pid_proxy_summary.json'}")


if __name__ == "__main__":
    main()
