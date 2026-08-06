"""Estimator robustness: do the probe conclusions depend on hyperparameters?

Reviewer objection: "your Monte Carlo estimates of P_t(B | s_t) depend on the
number of rollout samples and the probe softmax temperature; the conclusions
might be artifacts of those choices."

This sweep varies rollout samples and probe temperature over a grid and checks
whether three qualitative conclusions survive every cell:

1. sign(+) of the intervention return gap in rescue mode (uncertain pref.);
2. sign(-) of the intervention return gap in bridge mode;
3. ordering H0(uncertain_preference) > H0(pure_team) (open vs collapsed
   potential).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from contextual_sacrifice_gridworld import ContextualSacrificeEnv, train_policy
from within_episode_collapse_probe import (
    entropy,
    estimate_future,
    js,
    mean,
    probe_contexts,
)


OUTPUTS = Path(__file__).resolve().parent / "outputs"


def probe_cell(
    q_table,
    regime: str,
    mode: str,
    probe_episodes: int,
    samples: int,
    probe_temperature: float,
    seed: int,
) -> Dict[str, float]:
    contexts = probe_contexts(regime)
    h0_values: List[float] = []
    js_values: List[float] = []
    gap_values: List[float] = []
    for episode in range(probe_episodes):
        rng = random.Random(seed + episode * 23)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        dist, _ = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng
        )
        h0_values.append(entropy(dist))
        do_t_dist, do_t_ret = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_trigger",
        )
        do_n_dist, do_n_ret = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_non_trigger",
        )
        js_values.append(js(do_t_dist, do_n_dist))
        gap_values.append(do_t_ret - do_n_ret)
    return {
        "h0_bits": mean(h0_values),
        "intervention_js": mean(js_values),
        "intervention_return_gap": mean(gap_values),
    }


def run_sweep(
    train_episodes: int,
    probe_episodes: int,
    sample_grid: Sequence[int],
    temperature_grid: Sequence[float],
    seed: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = {
        "uncertain_preference": train_policy("uncertain_preference", train_episodes, seed),
        "pure_team": train_policy("pure_team", train_episodes, seed + 10_000),
    }
    rows: List[Dict[str, float | str | int]] = []
    for samples in sample_grid:
        for probe_temperature in temperature_grid:
            cell: Dict[str, float | str | int] = {
                "samples": samples,
                "probe_temperature": probe_temperature,
            }
            for regime in ("uncertain_preference", "pure_team"):
                for mode in ("rescue", "bridge"):
                    result = probe_cell(
                        policies[regime], regime, mode,
                        probe_episodes=probe_episodes,
                        samples=samples,
                        probe_temperature=probe_temperature,
                        seed=seed + samples * 101 + int(probe_temperature * 1000),
                    )
                    prefix = f"{regime}_{mode}"
                    cell[f"{prefix}_h0"] = result["h0_bits"]
                    cell[f"{prefix}_js"] = result["intervention_js"]
                    cell[f"{prefix}_gap"] = result["intervention_return_gap"]
            cell["rescue_gap_positive"] = int(float(cell["uncertain_preference_rescue_gap"]) > 0)
            cell["bridge_gap_negative"] = int(float(cell["uncertain_preference_bridge_gap"]) < 0)
            cell["h0_ordering_holds"] = int(
                float(cell["uncertain_preference_rescue_h0"]) > float(cell["pure_team_rescue_h0"])
            )
            rows.append(cell)

    checks = {
        "rescue_gap_positive_rate": mean(float(row["rescue_gap_positive"]) for row in rows),
        "bridge_gap_negative_rate": mean(float(row["bridge_gap_negative"]) for row in rows),
        "h0_ordering_rate": mean(float(row["h0_ordering_holds"]) for row in rows),
        "n_cells": float(len(rows)),
    }

    with (output_dir / "estimator_robustness_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (output_dir / "estimator_robustness_summary.json").write_text(
        json.dumps({"checks": checks, "grid": rows}, indent=2),
        encoding="utf-8",
    )
    print("samples,probe_temp,up_rescue_gap,up_bridge_gap,up_rescue_h0,team_rescue_h0")
    for row in rows:
        print(
            f"{row['samples']},{row['probe_temperature']},"
            f"{float(row['uncertain_preference_rescue_gap']):+.3f},"
            f"{float(row['uncertain_preference_bridge_gap']):+.3f},"
            f"{float(row['uncertain_preference_rescue_h0']):.3f},"
            f"{float(row['pure_team_rescue_h0']):.3f}"
        )
    print("\nchecks:", json.dumps(checks, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimator robustness sweep.")
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--probe_episodes", type=int, default=16)
    parser.add_argument("--sample_grid", default="12,24,48,96")
    parser.add_argument("--temperature_grid", default="0.6,0.9,1.2")
    parser.add_argument("--seed", type=int, default=7013)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_sweep(
        train_episodes=args.train_episodes,
        probe_episodes=args.probe_episodes,
        sample_grid=[int(x) for x in str(args.sample_grid).split(",") if x.strip()],
        temperature_grid=[float(x) for x in str(args.temperature_grid).split(",") if x.strip()],
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\nWrote {args.output_dir / 'estimator_robustness_grid.csv'}")
    print(f"Wrote {args.output_dir / 'estimator_robustness_summary.json'}")


if __name__ == "__main__":
    main()
