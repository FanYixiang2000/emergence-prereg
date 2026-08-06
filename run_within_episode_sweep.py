"""Multi-seed sweep with bootstrap confidence intervals for the
within-episode possibility-collapse probe.

Reviewer objection: "the within-episode result is a single seed." This sweep
retrains the tabular policy under several seeds, repeats the probe, and
reports seed-level means with 95% bootstrap percentile intervals for the key
quantities: initial future entropy, intervention JS divergence, and the
intervention return gap (whose sign separates useful from harmful collapse).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from contextual_sacrifice_gridworld import MODES, train_policy
from within_episode_collapse_probe import probe_episode, summarize


OUTPUTS = Path(__file__).resolve().parent / "outputs"
KEY_METRICS = (
    "trigger_rate",
    "initial_future_entropy",
    "intervention_js",
    "intervention_return_gap",
    "do_trigger_p_rescue",
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: Sequence[float], rng: random.Random, resamples: int = 2000) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "lo95": 0.0, "hi95": 0.0}
    means: List[float] = []
    n = len(values)
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    return {
        "mean": mean(values),
        "lo95": means[int(0.025 * resamples)],
        "hi95": means[int(0.975 * resamples)],
    }


def run_sweep(
    regimes: Sequence[str],
    seeds: Sequence[int],
    train_episodes: int,
    probe_episodes: int,
    samples: int,
    temperature: float,
    probe_temperature: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, float | str]] = []
    for regime in regimes:
        for seed in seeds:
            q_table = train_policy(regime, train_episodes, seed)
            for mode in MODES:
                episodes = [
                    probe_episode(
                        q_table,
                        regime,
                        mode,
                        episode_idx=ep,
                        temperature=temperature,
                        probe_temperature=probe_temperature,
                        samples=samples,
                        seed=seed * 101 + ep * 7 + (0 if mode == "rescue" else 3),
                    )
                    for ep in range(probe_episodes)
                ]
                summary = summarize(regime, mode, episodes)
                per_seed_rows.append({"seed": float(seed), **summary})

    rng = random.Random(20260703)
    ci_rows: List[Dict[str, float | str]] = []
    for regime in regimes:
        for mode in MODES:
            selected = [
                row for row in per_seed_rows
                if row["regime"] == regime and row["mode"] == mode
            ]
            ci_row: Dict[str, float | str] = {
                "regime": regime,
                "mode": mode,
                "n_seeds": float(len(selected)),
            }
            for metric in KEY_METRICS:
                values = [float(row[metric]) for row in selected]
                ci = bootstrap_ci(values, rng)
                ci_row[f"{metric}_mean"] = ci["mean"]
                ci_row[f"{metric}_lo95"] = ci["lo95"]
                ci_row[f"{metric}_hi95"] = ci["hi95"]
            gaps = [float(row["intervention_return_gap"]) for row in selected]
            positive = sum(1 for gap in gaps if gap > 0)
            ci_row["sign_consistency"] = max(positive, len(gaps) - positive) / max(len(gaps), 1)
            ci_rows.append(ci_row)

    with (output_dir / "within_episode_sweep_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow(row)
    with (output_dir / "within_episode_sweep_ci.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ci_rows[0].keys()))
        writer.writeheader()
        for row in ci_rows:
            writer.writerow(row)
    (output_dir / "within_episode_sweep_ci.json").write_text(
        json.dumps({"summary": ci_rows}, indent=2),
        encoding="utf-8",
    )
    print("regime,mode,H0_mean,iv_js_mean,iv_gap_mean,iv_gap_lo95,iv_gap_hi95,sign_consistency")
    for row in ci_rows:
        print(
            f"{row['regime']},{row['mode']},"
            f"{float(row['initial_future_entropy_mean']):.4f},"
            f"{float(row['intervention_js_mean']):.4f},"
            f"{float(row['intervention_return_gap_mean']):.4f},"
            f"{float(row['intervention_return_gap_lo95']):.4f},"
            f"{float(row['intervention_return_gap_hi95']):.4f},"
            f"{float(row['sign_consistency']):.3f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-seed within-episode sweep with bootstrap CI.")
    parser.add_argument("--regimes", nargs="*", default=["pure_team", "uncertain_preference"])
    parser.add_argument("--seeds", default="1013,2027,3041,4057,5077")
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--probe_episodes", type=int, default=24)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--probe_temperature", type=float, default=0.9)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    run_sweep(
        regimes=args.regimes,
        seeds=seeds,
        train_episodes=args.train_episodes,
        probe_episodes=args.probe_episodes,
        samples=args.samples,
        temperature=args.temperature,
        probe_temperature=args.probe_temperature,
        output_dir=args.output_dir,
    )
    print(f"\nWrote {args.output_dir / 'within_episode_sweep_ci.csv'}")
    print(f"Wrote {args.output_dir / 'within_episode_sweep_per_seed.csv'}")


if __name__ == "__main__":
    main()
