"""Trajectory-level PTC adapter for decoy target-selection records.

The summary-level decoy adapter shows that role-aware controllers win while
nearest-only gets trapped by decoys. This script uses per-agent, per-time target
records to measure the trajectory-level collapse:

- target-role entropy across agents;
- KL from a broad role prior to the observed target-role distribution;
- collapse into decoy traps versus useful non-decoy targets;
- temporal burst in target-role collapse.

This is still not a full simulator rollout with value estimates, but it is a
real trajectory-level behavioral distribution from existing external records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SWARM = WORKSPACE / "examples_6.29_MARL_SWARM"
OUTPUTS = ROOT / "outputs"

ROLES = ("decoy", "threat", "fragile")
CONTROLLER_RECORDS = {
    "nearest_only": SWARM / "outputs" / "mined_from_nearest_records.json",
    "role_oracle": SWARM / "outputs" / "mined_from_role_oracle_records.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def normalize(counts: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(counts.get(role, 0.0), 0.0) for role in ROLES)
    if total <= 0:
        return {role: 1.0 / len(ROLES) for role in ROLES}
    return {role: max(counts.get(role, 0.0), 0.0) / total for role in ROLES}


def entropy(dist: Mapping[str, float]) -> float:
    eps = 1e-12
    return -sum(dist[role] * math.log(dist[role] + eps, 2) for role in ROLES if dist[role] > 0)


def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    eps = 1e-12
    return sum(p[role] * math.log((p[role] + eps) / (q[role] + eps), 2) for role in ROLES if p[role] > 0)


def records_from_path(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        records = []
    return [record for record in records if isinstance(record, dict)]


def group_distributions(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, float | int]]:
    grouped: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: {role: 0.0 for role in ROLES})
    for record in records:
        episode = int(record.get("episode", 0))
        t = int(record.get("t", 0))
        role = str(record.get("target_role", "unknown"))
        if role not in ROLES:
            continue
        grouped[(episode, t)][role] += 1.0

    rows: List[Dict[str, float | int]] = []
    prior = {role: 1.0 / len(ROLES) for role in ROLES}
    for (episode, t), counts in sorted(grouped.items()):
        dist = normalize(counts)
        h = entropy(dist)
        h_norm = h / math.log(len(ROLES), 2)
        collapse = kl(dist, prior)
        non_decoy = dist["threat"] + dist["fragile"]
        useful_consensus = non_decoy * (1.0 - h_norm)
        decoy_trap = dist["decoy"] * (1.0 - h_norm)
        rows.append(
            {
                "episode": episode,
                "t": t,
                "p_decoy": dist["decoy"],
                "p_threat": dist["threat"],
                "p_fragile": dist["fragile"],
                "entropy": h,
                "entropy_norm": h_norm,
                "role_collapse_kl": collapse,
                "non_decoy_rate": non_decoy,
                "useful_nondecoy_consensus": useful_consensus,
                "decoy_trap_consensus": decoy_trap,
            }
        )
    return rows


def burst_metrics(rows: Sequence[Mapping[str, float | int]]) -> Dict[str, float]:
    by_episode: Dict[int, List[Mapping[str, float | int]]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode"])].append(row)
    bursts: List[float] = []
    for ep_rows in by_episode.values():
        ordered = sorted(ep_rows, key=lambda row: int(row["t"]))
        values = [float(row["role_collapse_kl"]) for row in ordered]
        if len(values) < 2:
            bursts.append(0.0)
            continue
        bursts.append(max(max(values[i] - values[i - 1], 0.0) for i in range(1, len(values))))
    return {
        "mean_max_burst": mean(bursts),
        "max_burst": max(bursts) if bursts else 0.0,
    }


def summarize_controller(controller: str, records: Sequence[Mapping[str, Any]]) -> Dict[str, float | str]:
    rows = group_distributions(records)
    burst = burst_metrics(rows)
    mean_win = 0.0
    grid_path = SWARM / "outputs" / "decoy_robustness_grid_marl.json"
    if grid_path.exists():
        grid = load_json(grid_path)
        wins = []
        for record in grid.get("records", []):
            ctrl = record.get("controllers", {}).get(controller, {})
            if ctrl:
                wins.append(float(ctrl.get("mean_win", 0.0)))
        mean_win = mean(wins)

    useful = mean(float(row["useful_nondecoy_consensus"]) for row in rows)
    trap = mean(float(row["decoy_trap_consensus"]) for row in rows)
    non_decoy = mean(float(row["non_decoy_rate"]) for row in rows)
    decoy = mean(float(row["p_decoy"]) for row in rows)
    collapse = mean(float(row["role_collapse_kl"]) for row in rows)
    entropy_norm = mean(float(row["entropy_norm"]) for row in rows)
    trajectory_ptc = mean_win * useful * (1.0 - trap)
    return {
        "controller": controller,
        "n_timepoints": float(len(rows)),
        "mean_win": mean_win,
        "mean_p_decoy": decoy,
        "mean_non_decoy_rate": non_decoy,
        "mean_entropy_norm": entropy_norm,
        "mean_role_collapse_kl": collapse,
        "mean_useful_nondecoy_consensus": useful,
        "mean_decoy_trap_consensus": trap,
        "mean_max_burst": burst["mean_max_burst"],
        "max_burst": burst["max_burst"],
        "trajectory_level_ptc_score": trajectory_ptc,
    }


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_time_rows: List[Dict[str, float | int | str]] = []
    summaries: List[Dict[str, float | str]] = []
    for controller, path in CONTROLLER_RECORDS.items():
        records = records_from_path(path)
        time_rows = group_distributions(records)
        for row in time_rows:
            all_time_rows.append({"controller": controller, **row})
        summaries.append(summarize_controller(controller, records))

    summaries = sorted(summaries, key=lambda row: float(row["trajectory_level_ptc_score"]), reverse=True)
    with (output_dir / "external_decoy_trajectory_ptc_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        columns = list(all_time_rows[0].keys()) if all_time_rows else []
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in all_time_rows:
            writer.writerow(row)
    with (output_dir / "external_decoy_trajectory_ptc_summary.csv").open("w", newline="", encoding="utf-8") as f:
        columns = list(summaries[0].keys()) if summaries else []
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    (output_dir / "external_decoy_trajectory_ptc_summary.json").write_text(
        json.dumps({"summary": summaries}, indent=2),
        encoding="utf-8",
    )
    print("controller,win,p_decoy,non_decoy,collapse,useful_consensus,trap_consensus,score")
    for row in summaries:
        print(
            f"{row['controller']},{float(row['mean_win']):.4f},{float(row['mean_p_decoy']):.4f},"
            f"{float(row['mean_non_decoy_rate']):.4f},{float(row['mean_role_collapse_kl']):.4f},"
            f"{float(row['mean_useful_nondecoy_consensus']):.4f},"
            f"{float(row['mean_decoy_trap_consensus']):.4f},"
            f"{float(row['trajectory_level_ptc_score']):.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trajectory-level decoy PTC adapter.")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args.output_dir)
    print(f"\nWrote {args.output_dir / 'external_decoy_trajectory_ptc_summary.csv'}")
    print(f"Wrote {args.output_dir / 'external_decoy_trajectory_ptc_timeseries.csv'}")


if __name__ == "__main__":
    main()
