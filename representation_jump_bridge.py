"""Bridge possibility collapse to representation-jump definitions of emergence.

Some definitions of emergence use an abrupt jump in a learned or latent
representation space as the observable signal. This experiment treats that jump
as a projection of a lower-level change in future possibility distributions.

Let B be a future basin, P_t(B) the time-indexed future-basin distribution, and
phi(B) a basin-level macro embedding. We measure:

    C_t = KL(P_t(B) || P_0(B))
    burst_t = max(C_t - C_{t-1}, 0)
    R_t = E_{B ~ P_t}[phi(B)]
    J_t = ||R_t - R_{t-1}||_2

If representation jumps are consequences of possibility collapse, burst_t and
J_t should align in the strong-emergence regime, while gradual convergence,
reward-shaped convergence, and random instability should fail at least one part
of the evidence chain.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


BASINS = ("selfish", "direct_team", "trigger_success", "noise")
OUTPUTS = Path(__file__).resolve().parent / "outputs"

BASIN_EMBEDDINGS: Mapping[str, Tuple[float, float]] = {
    "selfish": (1.0, -0.15),
    "direct_team": (-0.25, 0.85),
    "trigger_success": (0.95, 1.05),
    "noise": (-0.70, -0.55),
}


def normalize(raw: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(value, 0.0) for value in raw.values())
    if total <= 0:
        return {basin: 1.0 / len(BASINS) for basin in BASINS}
    return {basin: max(raw.get(basin, 0.0), 0.0) / total for basin in BASINS}


def interpolate(a: Mapping[str, float], b: Mapping[str, float], alpha: float) -> Dict[str, float]:
    return normalize({basin: (1.0 - alpha) * a[basin] + alpha * b[basin] for basin in BASINS})


def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    eps = 1e-12
    return sum(p[basin] * math.log((p[basin] + eps) / (q[basin] + eps), 2) for basin in BASINS if p[basin] > 0)


def entropy(p: Mapping[str, float]) -> float:
    eps = 1e-12
    return -sum(p[basin] * math.log(p[basin] + eps, 2) for basin in BASINS if p[basin] > 0)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return 0.0
    return numerator / (denom_x * denom_y)


def representation(dist: Mapping[str, float]) -> Tuple[float, float]:
    x = sum(dist[basin] * BASIN_EMBEDDINGS[basin][0] for basin in BASINS)
    y = sum(dist[basin] * BASIN_EMBEDDINGS[basin][1] for basin in BASINS)
    return (x, y)


def l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def trajectory(regime: str, steps: int) -> List[Dict[str, float]]:
    prior = normalize({"selfish": 0.25, "direct_team": 0.25, "trigger_success": 0.25, "noise": 0.25})
    target_trigger = normalize({"selfish": 0.03, "direct_team": 0.08, "trigger_success": 0.86, "noise": 0.03})
    target_direct = normalize({"selfish": 0.02, "direct_team": 0.90, "trigger_success": 0.04, "noise": 0.04})
    target_noise = normalize({"selfish": 0.15, "direct_team": 0.15, "trigger_success": 0.15, "noise": 0.55})
    rows: List[Dict[str, float]] = []
    for t in range(steps):
        x = t / max(1, steps - 1)
        if regime == "ordinary_gradual":
            alpha = x
            dist = interpolate(prior, target_direct, alpha)
            utility = 0.70
            guidance = 0.75
            stability = 0.85
            nondecomposable = 0.25
        elif regime == "reward_shaped":
            alpha = min(1.0, 1.35 * x)
            dist = interpolate(prior, target_direct, alpha)
            utility = 0.85
            guidance = 0.95
            stability = 0.96
            nondecomposable = 0.20
        elif regime == "collapse_emergence":
            alpha = 1.0 / (1.0 + math.exp(-38.0 * (x - 0.62)))
            dist = interpolate(prior, target_trigger, alpha)
            utility = 0.35 + 0.55 * alpha
            guidance = 0.10
            stability = 0.94
            nondecomposable = 0.88
        elif regime == "random_instability":
            alpha = 0.50 + 0.35 * math.sin(20.0 * x)
            dist = interpolate(prior, target_noise, alpha)
            utility = 0.35
            guidance = 0.05
            stability = 0.18
            nondecomposable = 0.10
        else:
            raise ValueError(f"unknown regime: {regime}")
        rep = representation(dist)
        row = {
            "t": float(t),
            "utility": utility,
            "guidance": guidance,
            "stability": stability,
            "nondecomposable": nondecomposable,
            "rep_x": rep[0],
            "rep_y": rep[1],
        }
        row.update({f"p_{basin}": dist[basin] for basin in BASINS})
        rows.append(row)
    return rows


def summarize_regime(regime: str, rows: Sequence[Mapping[str, float]]) -> Dict[str, float | str]:
    prior = {basin: float(rows[0][f"p_{basin}"]) for basin in BASINS}
    collapses: List[float] = []
    entropies: List[float] = []
    reps: List[Tuple[float, float]] = []
    for row in rows:
        dist = {basin: float(row[f"p_{basin}"]) for basin in BASINS}
        collapses.append(kl(dist, prior))
        entropies.append(entropy(dist))
        reps.append((float(row["rep_x"]), float(row["rep_y"])))

    collapse_bursts = [0.0] + [max(collapses[i] - collapses[i - 1], 0.0) for i in range(1, len(collapses))]
    rep_jumps = [0.0] + [l2(reps[i], reps[i - 1]) for i in range(1, len(reps))]
    max_burst = max(collapse_bursts)
    max_jump = max(rep_jumps)
    peak_burst_t = float(collapse_bursts.index(max_burst))
    peak_jump_t = float(rep_jumps.index(max_jump))
    peak_alignment = 1.0 / (1.0 + abs(peak_burst_t - peak_jump_t))
    burst_jump_corr = pearson(collapse_bursts, rep_jumps)
    final = rows[-1]
    representation_bridge_score = (
        max_burst
        * max_jump
        * max(burst_jump_corr, 0.0)
        * peak_alignment
        * float(final["utility"])
        * float(final["stability"])
        * float(final["nondecomposable"])
        * (1.0 - float(final["guidance"]))
    )
    return {
        "regime": regime,
        "total_collapse_kl": collapses[-1],
        "entropy_drop": entropies[0] - entropies[-1],
        "max_collapse_burst": max_burst,
        "max_representation_jump": max_jump,
        "peak_collapse_burst_t": peak_burst_t,
        "peak_representation_jump_t": peak_jump_t,
        "peak_alignment": peak_alignment,
        "burst_jump_correlation": burst_jump_corr,
        "final_utility": float(final["utility"]),
        "guidance": float(final["guidance"]),
        "stability": float(final["stability"]),
        "nondecomposable": float(final["nondecomposable"]),
        "representation_bridge_score": representation_bridge_score,
    }


def add_dynamic_metrics(rows: Sequence[Mapping[str, float | str]], regime: str) -> List[Dict[str, float | str]]:
    selected = [row for row in rows if row["regime"] == regime]
    prior = {basin: float(selected[0][f"p_{basin}"]) for basin in BASINS}
    enriched: List[Dict[str, float | str]] = []
    previous_collapse = 0.0
    previous_rep = (float(selected[0]["rep_x"]), float(selected[0]["rep_y"]))
    for idx, row in enumerate(selected):
        dist = {basin: float(row[f"p_{basin}"]) for basin in BASINS}
        current_collapse = kl(dist, prior)
        current_rep = (float(row["rep_x"]), float(row["rep_y"]))
        enriched_row = dict(row)
        enriched_row["collapse_kl"] = current_collapse
        enriched_row["collapse_burst"] = max(current_collapse - previous_collapse, 0.0) if idx else 0.0
        enriched_row["representation_jump"] = l2(current_rep, previous_rep) if idx else 0.0
        enriched.append(enriched_row)
        previous_collapse = current_collapse
        previous_rep = current_rep
    return enriched


def run_all(steps: int, output_dir: Path) -> None:
    regimes = ("ordinary_gradual", "reward_shaped", "collapse_emergence", "random_instability")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, float | str]] = []
    summaries: List[Dict[str, float | str]] = []
    for regime in regimes:
        rows = trajectory(regime, steps)
        summaries.append(summarize_regime(regime, rows))
        raw_rows = [{"regime": regime, **row} for row in rows]
        all_rows.extend(add_dynamic_metrics(raw_rows, regime))

    with (output_dir / "representation_jump_bridge_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    with (output_dir / "representation_jump_bridge_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    (output_dir / "representation_jump_bridge_summary.json").write_text(
        json.dumps({"summary": summaries}, indent=2),
        encoding="utf-8",
    )
    print("regime,total_collapse,max_burst,max_jump,corr,align,guidance,score")
    for row in summaries:
        print(
            f"{row['regime']},{float(row['total_collapse_kl']):.4f},"
            f"{float(row['max_collapse_burst']):.4f},{float(row['max_representation_jump']):.4f},"
            f"{float(row['burst_jump_correlation']):.4f},{float(row['peak_alignment']):.4f},"
            f"{float(row['guidance']):.4f},{float(row['representation_bridge_score']):.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge possibility collapse to representation jumps.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(args.steps, args.output_dir)
    print(f"\nWrote {args.output_dir / 'representation_jump_bridge_summary.csv'}")
    print(f"Wrote {args.output_dir / 'representation_jump_bridge_timeseries.csv'}")


if __name__ == "__main__":
    main()
