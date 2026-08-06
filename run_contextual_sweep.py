"""Multi-seed sweep for the contextual selective-trigger benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from contextual_sacrifice_gridworld import REGIMES, run_regime


METRICS = (
    "natural_team_return_mean",
    "rescue_success_rate",
    "bridge_success_rate",
    "over_sacrifice_rate",
    "selective_trigger_score",
    "potential_effective_modes",
    "natural_trigger_rate",
    "trigger_choice_tension",
    "counterfactual_necessity",
    "retrospective_importance",
    "endogenous_emergence_score",
)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stderr(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def t_critical_95(n: int) -> float:
    table = {1: 0.0, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    return table.get(n, 2.1 if n < 30 else 1.96)


def summarize(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_regime: Dict[str, List[Mapping[str, float]]] = {}
    for record in records:
        by_regime.setdefault(str(record["regime"]), []).append(record["ptc"])  # type: ignore[arg-type]
    rows: List[Dict[str, object]] = []
    for regime, metric_records in by_regime.items():
        row: Dict[str, object] = {"regime": regime, "n_seeds": len(metric_records)}
        for metric in METRICS:
            values = [float(item[metric]) for item in metric_records]
            metric_mean = mean(values)
            metric_stderr = stderr(values)
            row[f"{metric}_mean"] = metric_mean
            row[f"{metric}_stderr"] = metric_stderr
            row[f"{metric}_ci95"] = t_critical_95(len(values)) * metric_stderr
        rows.append(row)
    return sorted(rows, key=lambda item: str(item["regime"]))


def write_outputs(
    records: Sequence[Mapping[str, object]],
    summary: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contextual_sweep_records.json").write_text(
        json.dumps({"records": list(records), "summary": list(summary)}, indent=2),
        encoding="utf-8",
    )
    columns = ["regime", "n_seeds"]
    for metric in METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_stderr", f"{metric}_ci95"))
    with (output_dir / "contextual_sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def print_summary(summary: Sequence[Mapping[str, object]]) -> None:
    columns = (
        "regime",
        "n_seeds",
        "natural_team_return_mean_mean",
        "rescue_success_rate_mean",
        "bridge_success_rate_mean",
        "over_sacrifice_rate_mean",
        "selective_trigger_score_mean",
        "endogenous_emergence_score_mean",
    )
    print(",".join(columns))
    for row in summary:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print(",".join(values))


def parse_seeds(seed_text: str) -> List[int]:
    return [int(part.strip()) for part in seed_text.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run contextual benchmark sweep.")
    parser.add_argument("--seeds", type=str, default="53,59,61")
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--eval_episodes", type=int, default=2000)
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--regimes", nargs="*", default=list(REGIMES), choices=list(REGIMES))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    records: List[Mapping[str, object]] = []
    for seed in seeds:
        for idx, regime in enumerate(args.regimes):
            records.append(
                run_regime(
                    regime=regime,
                    train_episodes=args.train_episodes,
                    eval_episodes=args.eval_episodes,
                    seed=seed + idx * 10_000,
                    eval_temperature=args.eval_temperature,
                )
            )
    summary = summarize(records)
    write_outputs(records, summary, args.output_dir)
    print_summary(summary)
    print(f"\nWrote {args.output_dir / 'contextual_sweep_records.json'}")
    print(f"Wrote {args.output_dir / 'contextual_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
