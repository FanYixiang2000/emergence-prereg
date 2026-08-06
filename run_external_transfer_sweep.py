"""Multi-seed replication of the pre-registered external swarm transfer.

The single-run external transfer (external_swarm_criterion_transfer.py)
passed all three registered predictions, but with one REINFORCE training
run. This sweep repeats the entire battery across independent seeds:
each seed trains its own learner, re-initializes its own untrained twin,
and re-measures every system with seed-shifted evaluation streams.

Reported per prediction: pass rate across seeds. Reported per metric:
mean and 95% bootstrap CI across seeds. The registered protocol and
thresholds (EXTERNAL_TRANSFER_PREREGISTRATION.md) are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from external_swarm_criterion_transfer import (
    COMPONENTS,
    OUTPUTS,
    audit_label,
    component_passes,
    measure_system,
    rule_controller,
    scorer_controller,
    train_marl_scorer,
    untrained_scorer,
)


def bootstrap_ci(values: Sequence[float], n_boot: int = 4000, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    values = list(values)
    if not values:
        return {"mean": float("nan"), "lo95": float("nan"), "hi95": float("nan")}
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "lo95": means[int(0.025 * n_boot)],
        "hi95": means[int(0.975 * n_boot)],
    }


def run_one_seed(seed: int, iters: int, batch: int, lr: float, n_eval: int) -> Dict[str, Any]:
    print(f"\n=== seed {seed}: training external REINFORCE learner ===")
    learned, history = train_marl_scorer(iters, batch, lr, seed)
    untrained = untrained_scorer(seed + 999)

    systems = (
        ("marl_learned", scorer_controller(learned), False),
        ("marl_untrained", scorer_controller(untrained), False),
        ("nearest_only", rule_controller(["nearest"]), True),
        ("role_oracle", rule_controller(["threat", "fragile", "non_decoy", "nearest"]), True),
        ("damage_aware", rule_controller(["damage", "nearest"]), True),
    )
    rows: List[Dict[str, Any]] = []
    for idx, (name, controller, prespec) in enumerate(systems):
        row = measure_system(name, controller, prespec, n_eval, seed + idx * 50_000)
        row["seed"] = seed
        rows.append(row)
        print(
            f"  {name:15s} H0 {row['h0_bits']:.3f} p_trig {row['natural_trigger_rate']:.2f} "
            f"(pas {row['passive_trigger_rate']:.2f}/agg {row['aggressive_trigger_rate']:.2f}) "
            f"gap {row['usefulness_gap']:+.2f}"
        )

    labels = {
        row["system"]: (audit_label(row) if row["system"] == "marl_learned" else 0)
        for row in rows
    }
    verdicts = {}
    correct = 0
    for row in rows:
        passes = component_passes(row)
        predicted = int(all(passes.values()))
        verdicts[row["system"]] = {
            "passes": passes,
            "full_criterion": predicted,
            "audited_label": labels[row["system"]],
        }
        correct += int(predicted == labels[row["system"]])

    damage_passes = verdicts["damage_aware"]["passes"]
    learned_row = next(row for row in rows if row["system"] == "marl_learned")
    checks = {
        "p1_all_verdicts_correct": correct == len(rows),
        "p2_damage_aware_endogeneity_only": (
            all(damage_passes[c] for c in COMPONENTS if c != "endogeneity")
            and not damage_passes["endogeneity"]
        ),
        "p3_gap_sign_flips": (
            float(learned_row["iv_gap_aggressive"]) > 0 > float(learned_row["iv_gap_passive"])
        ),
        "audit_learner_selective": labels["marl_learned"] == 1,
    }
    return {
        "seed": seed,
        "rows": rows,
        "verdicts": verdicts,
        "accuracy": correct / len(rows),
        "checks": checks,
        "final_train_win": history[-1]["win"] if history else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed external transfer replication.")
    parser.add_argument("--seeds", type=str, default="7031,7131,7231,7331,7431")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n_eval", type=int, default=120)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = [run_one_seed(seed, args.iters, args.batch, args.lr, args.n_eval) for seed in seeds]

    check_names = list(results[0]["checks"].keys())
    check_rates = {
        name: sum(1 for r in results if r["checks"][name]) / len(results)
        for name in check_names
    }
    metric_cis: Dict[str, Dict[str, float]] = {}
    for system in ("marl_learned", "marl_untrained", "nearest_only", "role_oracle", "damage_aware"):
        for metric in ("h0_bits", "usefulness_gap", "natural_trigger_rate",
                       "passive_trigger_rate", "aggressive_trigger_rate"):
            values = [
                float(row[metric])
                for r in results for row in r["rows"] if row["system"] == system
            ]
            metric_cis[f"{system}.{metric}"] = bootstrap_ci(values, seed=hash((system, metric)) % 10_000)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [
        {k: v for k, v in row.items() if k != "basin_distribution"}
        for r in results for row in r["rows"]
    ]
    with (args.output_dir / "external_transfer_sweep_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    summary = {
        "preregistration": "EXTERNAL_TRANSFER_PREREGISTRATION.md",
        "seeds": seeds,
        "accuracy_per_seed": {str(r["seed"]): r["accuracy"] for r in results},
        "prediction_pass_rates": check_rates,
        "metric_cis": metric_cis,
        "verdicts_per_seed": {
            str(r["seed"]): {
                system: {
                    "full_criterion": v["full_criterion"],
                    "audited_label": v["audited_label"],
                }
                for system, v in r["verdicts"].items()
            }
            for r in results
        },
    }
    (args.output_dir / "external_transfer_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nprediction,pass_rate")
    for name, rate in check_rates.items():
        print(f"{name},{rate:.3f}")
    print("\nmetric,mean,lo95,hi95")
    for key, ci in metric_cis.items():
        print(f"{key},{ci['mean']:.3f},{ci['lo95']:.3f},{ci['hi95']:.3f}")
    print(f"\nWrote {args.output_dir / 'external_transfer_sweep_summary.json'}")


if __name__ == "__main__":
    main()
