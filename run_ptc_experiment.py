"""Run the first Potential-Trigger-Collapse controlled experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping

from ptc_gridworld import REGIMES, all_regime_names, sample_trajectories
from ptc_metrics import (
    compact_metric_row,
    potential_trigger_collapse,
    summarize_distribution,
)


COMPACT_COLUMNS = (
    "regime",
    "potential_effective_modes",
    "collapse_bits",
    "trigger_effect_js_bits",
    "trigger_specificity_js_bits",
    "sacrifice_probability_shift",
    "macro_predictability_gain",
    "team_return_gain_after_trigger",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled Potential-Trigger-Collapse emergence experiment."
    )
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=list(all_regime_names()),
        choices=list(all_regime_names()),
    )
    return parser


def run_regime(
    regime_name: str, episodes: int, seed: int, temperature: float
) -> Dict[str, object]:
    prior = sample_trajectories(
        regime_name=regime_name,
        n=episodes,
        seed=seed,
        forced_action=None,
        temperature=temperature,
    )
    trigger = sample_trajectories(
        regime_name=regime_name,
        n=episodes,
        seed=seed + 100_003,
        forced_action="trigger",
        temperature=temperature,
    )
    non_trigger = sample_trajectories(
        regime_name=regime_name,
        n=episodes,
        seed=seed + 200_003,
        forced_action="non_trigger",
        temperature=temperature,
    )
    metrics = potential_trigger_collapse(
        prior=prior,
        trigger=trigger,
        non_trigger=non_trigger,
    )
    return {
        "regime": regime_name,
        "description": REGIMES[regime_name].description,
        "prior": summarize_distribution(prior),
        "trigger": summarize_distribution(trigger),
        "non_trigger": summarize_distribution(non_trigger),
        "ptc": metrics,
    }


def write_outputs(
    results: List[Mapping[str, object]], output_dir: Path, seed: int, episodes: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "episodes_per_condition": episodes,
        "results": results,
    }
    (output_dir / "ptc_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_dir / "ptc_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COMPACT_COLUMNS)
        for item in results:
            writer.writerow(
                compact_metric_row(
                    regime_name=str(item["regime"]),
                    metrics=item["ptc"],  # type: ignore[arg-type]
                )
            )


def print_compact_summary(results: List[Mapping[str, object]]) -> None:
    print(",".join(COMPACT_COLUMNS))
    for item in results:
        print(
            ",".join(
                compact_metric_row(
                    regime_name=str(item["regime"]),
                    metrics=item["ptc"],  # type: ignore[arg-type]
                )
            )
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results = [
        run_regime(
            regime_name=regime,
            episodes=args.episodes,
            seed=args.seed + idx * 10_000,
            temperature=args.temperature,
        )
        for idx, regime in enumerate(args.regimes)
    ]
    write_outputs(
        results=results,
        output_dir=args.output_dir,
        seed=args.seed,
        episodes=args.episodes,
    )
    print_compact_summary(results)
    print(f"\nWrote {args.output_dir / 'ptc_results.json'}")
    print(f"Wrote {args.output_dir / 'ptc_summary.csv'}")


if __name__ == "__main__":
    main()
