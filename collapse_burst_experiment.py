"""Information-theoretic collapse burst experiment.

This experiment formalizes the distinction discussed in the theory notes:

- ordinary learning can gradually collapse possibilities;
- strong emergence should show a burst-like, structured, useful collapse;
- reward-shaped coordination should have high guidance and lower emergence score;
- noise may change distributions but should not produce stable useful structure.

We model time-indexed basin distributions P_t(B) and measure:

    C_t = KL(P_t(B) || P_0(B))
    B_t = max(C_t - C_{t-1}, 0)

The experiment is synthetic but information-theoretic and controlled. It is
meant to validate the proposed scoring logic before applying it to real traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


BASINS = ("selfish", "direct_team", "trigger_success", "noise")
OUTPUTS = Path(__file__).resolve().parent / "outputs"


def normalize(raw: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in raw.values())
    if total <= 0:
        return {basin: 1.0 / len(BASINS) for basin in BASINS}
    return {basin: max(raw.get(basin, 0.0), 0.0) / total for basin in BASINS}


def interpolate(a: Mapping[str, float], b: Mapping[str, float], alpha: float) -> Dict[str, float]:
    return normalize({basin: (1.0 - alpha) * a[basin] + alpha * b[basin] for basin in BASINS})


def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    eps = 1e-12
    return sum(p[b] * math.log((p[b] + eps) / (q[b] + eps), 2) for b in BASINS if p[b] > 0)


def entropy(p: Mapping[str, float]) -> float:
    eps = 1e-12
    return -sum(p[b] * math.log(p[b] + eps, 2) for b in BASINS if p[b] > 0)


def trajectory(regime: str, steps: int) -> List[Dict[str, float]]:
    prior = normalize({"selfish": 0.25, "direct_team": 0.25, "trigger_success": 0.25, "noise": 0.25})
    target_trigger = normalize({"selfish": 0.03, "direct_team": 0.08, "trigger_success": 0.86, "noise": 0.03})
    target_direct = normalize({"selfish": 0.02, "direct_team": 0.90, "trigger_success": 0.04, "noise": 0.04})
    target_noise = normalize({"selfish": 0.20, "direct_team": 0.20, "trigger_success": 0.20, "noise": 0.40})
    rows: List[Dict[str, float]] = []
    for t in range(steps):
        x = t / max(1, steps - 1)
        if regime == "ordinary_gradual":
            alpha = x
            dist = interpolate(prior, target_direct, alpha)
            utility = 0.45 + 0.25 * x
            guidance = 0.75
            stability = 0.80
            nondecomposable = 0.25
        elif regime == "reward_shaped":
            alpha = min(1.0, 1.4 * x)
            dist = interpolate(prior, target_direct, alpha)
            utility = 0.55 + 0.30 * x
            guidance = 0.95
            stability = 0.95
            nondecomposable = 0.20
        elif regime == "collapse_emergence":
            # Long plateau, then rapid collapse to a useful trigger basin.
            alpha = 1.0 / (1.0 + math.exp(-35.0 * (x - 0.62)))
            dist = interpolate(prior, target_trigger, alpha)
            utility = 0.35 + 0.55 * alpha
            guidance = 0.10
            stability = 0.93
            nondecomposable = 0.85
        elif regime == "random_instability":
            alpha = 0.5 + 0.35 * math.sin(18.0 * x)
            dist = interpolate(prior, target_noise, alpha)
            utility = 0.35
            guidance = 0.05
            stability = 0.20
            nondecomposable = 0.10
        else:
            raise ValueError(f"unknown regime: {regime}")
        row = {"t": float(t), "utility": utility, "guidance": guidance, "stability": stability, "nondecomposable": nondecomposable}
        row.update({f"p_{basin}": dist[basin] for basin in BASINS})
        rows.append(row)
    return rows


def summarize_regime(regime: str, rows: Sequence[Mapping[str, float]]) -> Dict[str, float | str]:
    prior = {basin: float(rows[0][f"p_{basin}"]) for basin in BASINS}
    collapses: List[float] = []
    entropies: List[float] = []
    for row in rows:
        dist = {basin: float(row[f"p_{basin}"]) for basin in BASINS}
        collapses.append(kl(dist, prior))
        entropies.append(entropy(dist))
    bursts = [0.0] + [max(collapses[i] - collapses[i - 1], 0.0) for i in range(1, len(collapses))]
    max_burst = max(bursts)
    burst_t = float(bursts.index(max_burst))
    total_collapse = collapses[-1]
    burst_fraction = max_burst / max(total_collapse, 1e-12)
    final = rows[-1]
    final_utility = float(final["utility"])
    stability = float(final["stability"])
    nondecomposable = float(final["nondecomposable"])
    guidance = float(final["guidance"])
    emergence_score = total_collapse * burst_fraction * stability * nondecomposable * (1.0 - guidance) * max(final_utility, 0.0)
    return {
        "regime": regime,
        "total_collapse_kl": total_collapse,
        "max_burst_delta_kl": max_burst,
        "burst_time": burst_t,
        "burst_fraction": burst_fraction,
        "initial_entropy": entropies[0],
        "final_entropy": entropies[-1],
        "entropy_drop": entropies[0] - entropies[-1],
        "final_utility": final_utility,
        "stability": stability,
        "nondecomposable": nondecomposable,
        "guidance": guidance,
        "collapse_burst_emergence_score": emergence_score,
    }


def run_all(steps: int, output_dir: Path) -> None:
    regimes = ("ordinary_gradual", "reward_shaped", "collapse_emergence", "random_instability")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, float | str]] = []
    summaries: List[Dict[str, float | str]] = []
    for regime in regimes:
        rows = trajectory(regime, steps)
        summaries.append(summarize_regime(regime, rows))
        for row in rows:
            all_rows.append({"regime": regime, **row})
    with (output_dir / "collapse_burst_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    with (output_dir / "collapse_burst_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    (output_dir / "collapse_burst_summary.json").write_text(json.dumps({"summary": summaries}, indent=2), encoding="utf-8")
    print("regime,total_collapse,max_burst,burst_fraction,guidance,score")
    for row in summaries:
        print(
            f"{row['regime']},{float(row['total_collapse_kl']):.4f},"
            f"{float(row['max_burst_delta_kl']):.4f},{float(row['burst_fraction']):.4f},"
            f"{float(row['guidance']):.4f},{float(row['collapse_burst_emergence_score']):.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collapse burst experiment.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(args.steps, args.output_dir)
    print(f"\nWrote {args.output_dir / 'collapse_burst_summary.csv'}")
    print(f"Wrote {args.output_dir / 'collapse_burst_timeseries.csv'}")


if __name__ == "__main__":
    main()
