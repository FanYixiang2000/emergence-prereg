"""SYM-BRIDGE: spontaneous symmetry-breaking bridge calibration.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Extends
the ant-colony bridge dynamics to explicitly test external
underdetermination vs internal episode-level selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from ant_contrast import K, RHO
from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
ALPHA = 2.0
Q_TOTAL = 0.5
N = 100
N_TRIPS = 900
GRID = tuple(range(0, 901, 10))
N_EPISODES = 120
SEED = 104_001


def h2(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def episode(seed: int, bias: float = 1.0) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    ph_a = ph_b = 1.0
    q = Q_TOTAL / N
    p_grid = {}
    for t in range(N_TRIPS + 1):
        a = bias * (K + ph_a) ** ALPHA
        b = (K + ph_b) ** ALPHA
        p = a / (a + b)
        if t in GRID:
            p_grid[t] = float(p)
        if t == N_TRIPS:
            break
        n_a = rng.binomial(N, p)
        ph_a = ph_a * (1 - RHO) + q * n_a
        ph_b = ph_b * (1 - RHO) + q * (N - n_a)
    final_p = p_grid[GRID[-1]]
    return {
        "p_grid": p_grid,
        "final_side_a": bool(final_p >= 0.5),
        "final_lock": abs(final_p - 0.5) * 2.0,
    }


def run_condition(name: str, bias: float) -> Dict[str, object]:
    episodes = [episode(SEED + i * 17 + int(1000 * bias), bias=bias)
                for i in range(N_EPISODES)]
    curves = np.array([[h2(ep["p_grid"][t]) for t in GRID] for ep in episodes])
    med = np.median(curves, axis=0)
    adj = adjudicate(GRID, med * math.log2(3))
    h = adj.get("hinge", {})
    t_star = h.get("t_star")
    final_sides = np.array([ep["final_side_a"] for ep in episodes], dtype=bool)
    final_locks = np.array([ep["final_lock"] for ep in episodes])
    if t_star is None:
        precursor_acc = None
    else:
        t_near = min(GRID, key=lambda t: abs(t - t_star))
        pred = np.array([ep["p_grid"][t_near] >= 0.5 for ep in episodes], dtype=bool)
        precursor_acc = float(np.mean(pred == final_sides))
    frac_a = float(np.mean(final_sides))
    balance = 1.0 - abs(frac_a - 0.5) * 2.0
    return {
        "bias": bias,
        "median_openness": [round(v, 5) for v in med],
        "adj": adj,
        "final_frac_a": round(frac_a, 4),
        "across_episode_balance": round(balance, 4),
        "mean_final_lock": round(float(np.mean(final_locks)), 4),
        "precursor_accuracy": None if precursor_acc is None else round(precursor_acc, 4),
    }


def main() -> None:
    rows = {
        "symmetric": run_condition("symmetric", 1.0),
        "biased": run_condition("biased", 1.08),
    }
    sym = rows["symmetric"]
    biased = rows["biased"]
    outcomes = {
        "SB1_symmetric_onset": bool(sym["adj"]["b5_onset"]),
        "SB2_spontaneous_symmetry_breaking": bool(
            0.35 <= sym["final_frac_a"] <= 0.65
            and sym["mean_final_lock"] >= 0.9
        ),
        "SB3_external_bias_contrast": bool(
            biased["across_episode_balance"] < sym["across_episode_balance"]
        ),
        "SB4_precursor_intelligibility": bool(
            sym["precursor_accuracy"] is not None
            and sym["precursor_accuracy"] > 0.65
        ),
    }
    for name, row in rows.items():
        h = row["adj"].get("hinge", {})
        print(f"{name}: B5={row['adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} t*={h.get('t_star')} "
              f"fracA={row['final_frac_a']} lock={row['mean_final_lock']} "
              f"prec={row['precursor_accuracy']}", flush=True)
    report = {
        "status": "SYM-BRIDGE spontaneous symmetry-breaking calibration; preregistered",
        "config": {"N": N, "n_trips": N_TRIPS, "grid": GRID,
                   "episodes": N_EPISODES, "q_total": Q_TOTAL,
                   "alpha": ALPHA},
        "conditions": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "sym_bridge.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
