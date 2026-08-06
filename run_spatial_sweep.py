"""Multi-seed statistics for the spatial PTC benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from spatial_sacrifice_gridworld import REGIMES, run_regime


METRICS = (
    "potential_effective_modes",
    "natural_trigger_rate",
    "trigger_choice_tension",
    "natural_team_return_mean",
    "sacrifice_basin_rate",
    "team_direct_basin_rate",
    "selfish_basin_rate",
    "collapse_bits",
    "trigger_effect_js_bits",
    "trigger_specificity_js_bits",
    "counterfactual_necessity",
    "retrospective_importance",
    "retrospective_gain_per_local_cost",
    "endogenous_emergence_score",
    "team_return_mean",
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
    """Two-sided 95% t critical value for small seed counts."""

    table = {
        1: 0.0,
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    if n in table:
        return table[n]
    if n < 30:
        return 2.1
    return 1.96


def summarize_runs(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
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
    rows = sorted(rows, key=lambda item: str(item["regime"]))
    add_score_margins(rows)
    return rows


def add_score_margins(rows: Sequence[Dict[str, object]]) -> None:
    score_key = "endogenous_emergence_score_mean"
    sorted_rows = sorted(rows, key=lambda item: float(item[score_key]), reverse=True)
    for rank, row in enumerate(sorted_rows, start=1):
        row["endogenous_score_rank"] = rank

    target = next((row for row in rows if row["regime"] == "uncertain_preference"), None)
    if target is None:
        return
    baselines = [row for row in rows if row["regime"] != "uncertain_preference"]
    if not baselines:
        target["score_margin_vs_best_baseline"] = 0.0
        return
    best_baseline = max(baselines, key=lambda item: float(item[score_key]))
    target["best_baseline_regime"] = best_baseline["regime"]
    target["score_margin_vs_best_baseline"] = (
        float(target[score_key]) - float(best_baseline[score_key])
    )


def write_outputs(
    records: Sequence[Mapping[str, object]],
    summary: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "spatial_sweep_records.json").write_text(
        json.dumps({"records": list(records), "summary": list(summary)}, indent=2),
        encoding="utf-8",
    )
    columns = ["regime", "n_seeds"]
    for metric in METRICS:
        columns.extend((f"{metric}_mean", f"{metric}_stderr", f"{metric}_ci95"))
    columns.extend(
        (
            "endogenous_score_rank",
            "best_baseline_regime",
            "score_margin_vs_best_baseline",
        )
    )
    with (output_dir / "spatial_sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def print_summary(summary: Sequence[Mapping[str, object]]) -> None:
    compact = (
        "regime",
        "n_seeds",
        "potential_effective_modes_mean",
        "natural_trigger_rate_mean",
        "trigger_choice_tension_mean",
        "natural_team_return_mean_mean",
        "sacrifice_basin_rate_mean",
        "counterfactual_necessity_mean",
        "retrospective_importance_mean",
        "endogenous_emergence_score_mean",
        "endogenous_emergence_score_ci95",
        "endogenous_score_rank",
    )
    print(",".join(compact))
    for row in summary:
        values = []
        for column in compact:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print(",".join(values))


def parse_seeds(seed_text: str) -> List[int]:
    return [int(part.strip()) for part in seed_text.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a spatial PTC multi-seed sweep.")
    parser.add_argument("--seeds", type=str, default="23,29,31")
    parser.add_argument("--train_episodes", type=int, default=30000)
    parser.add_argument("--eval_episodes", type=int, default=1500)
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
    summary = summarize_runs(records)
    write_outputs(records, summary, args.output_dir)
    print_summary(summary)
    print(f"\nWrote {args.output_dir / 'spatial_sweep_records.json'}")
    print(f"Wrote {args.output_dir / 'spatial_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
