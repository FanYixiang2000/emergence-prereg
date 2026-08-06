"""Robustness sweep for performance closure.

This script varies mismatch payoff and payoff asymmetry to test whether the
performance-closure result is a single handcrafted setting or a stable region.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from performance_closure_benchmark import frange, summarize, sweep


def run_setting(
    mismatch_payoff: float,
    trigger_payoff: float,
    direct_payoff: float,
) -> Dict[str, float]:
    rows = sweep(
        p_values=frange(0.05, 0.95, 0.05),
        cash_values=frange(0.0, 12.0, 0.5),
        cost_values=frange(0.0, 4.0, 0.5),
        trigger_payoff=trigger_payoff,
        direct_payoff=direct_payoff,
        mismatch_payoff=mismatch_payoff,
    )
    summary = summarize(rows)
    return {
        "mismatch_payoff": mismatch_payoff,
        "trigger_payoff": trigger_payoff,
        "direct_payoff": direct_payoff,
        **summary,
    }


def run_robustness(
    mismatch_values: Sequence[float],
    asymmetry_values: Sequence[float],
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for mismatch in mismatch_values:
        for asymmetry in asymmetry_values:
            trigger_payoff = 11.0 + asymmetry
            direct_payoff = 11.0 - asymmetry
            records.append(
                run_setting(
                    mismatch_payoff=mismatch,
                    trigger_payoff=trigger_payoff,
                    direct_payoff=direct_payoff,
                )
            )
    return records


def aggregate(records: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    def mean(key: str) -> float:
        return sum(float(record[key]) for record in records) / len(records)

    return {
        "n_settings": float(len(records)),
        "mean_performance_closure_rate": mean("performance_closure_rate"),
        "mean_full_best_return_rate": mean("full_best_return_rate"),
        "mean_full_return_gain_vs_myopic": mean("full_return_gain_vs_myopic"),
        "mean_full_return_gain_vs_no_context": mean("full_return_gain_vs_no_context"),
        "mean_full_success_gain_vs_no_context": mean("full_success_gain_vs_no_context"),
        "min_performance_closure_rate": min(float(r["performance_closure_rate"]) for r in records),
        "max_performance_closure_rate": max(float(r["performance_closure_rate"]) for r in records),
    }


def write_outputs(
    records: Sequence[Mapping[str, float]],
    summary: Mapping[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "performance_robustness_summary.json").write_text(
        json.dumps({"summary": summary, "records": list(records)}, indent=2),
        encoding="utf-8",
    )
    columns = list(records[0].keys()) if records else []
    with (output_dir / "performance_robustness_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_values(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Performance closure robustness sweep.")
    parser.add_argument("--mismatch_values", type=str, default="-2,0,2,4")
    parser.add_argument("--asymmetry_values", type=str, default="0,1,2,3,4")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = run_robustness(
        mismatch_values=parse_values(args.mismatch_values),
        asymmetry_values=parse_values(args.asymmetry_values),
    )
    summary = aggregate(records)
    write_outputs(records, summary, args.output_dir)
    print("metric,value")
    for key, value in summary.items():
        print(f"{key},{value:.6f}")
    print(f"\nWrote {args.output_dir / 'performance_robustness_summary.json'}")
    print(f"Wrote {args.output_dir / 'performance_robustness_grid.csv'}")


if __name__ == "__main__":
    main()
