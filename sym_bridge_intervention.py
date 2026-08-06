"""SYM-BRIDGE-INT: profile predicts controllability.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Tests
whether pre-intervention openness predicts if a counter-regime impulse
can switch the final bridge side.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from ant_contrast import K, RHO

OUTPUTS = Path(__file__).resolve().parent / "outputs"
ALPHA = 2.0
Q_TOTAL = 0.5
N = 100
HORIZON = 900
TAUS = (120, 220, 280, 340, 460, 620)
N_EP = 200
SEED = 107_001


def h2(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def prob_a(ph_a: float, ph_b: float) -> float:
    a = (K + ph_a) ** ALPHA
    b = (K + ph_b) ** ALPHA
    return float(a / (a + b))


def advance(ph_a: float, ph_b: float, rng: np.random.Generator):
    p = prob_a(ph_a, ph_b)
    n_a = rng.binomial(N, p)
    q = Q_TOTAL / N
    ph_a = ph_a * (1 - RHO) + q * n_a
    ph_b = ph_b * (1 - RHO) + q * (N - n_a)
    return ph_a, ph_b


def trial(seed: int, tau: int) -> Dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    ph_a = ph_b = 1.0
    for _ in range(tau):
        ph_a, ph_b = advance(ph_a, ph_b, rng)
    p_pre = prob_a(ph_a, ph_b)
    incipient_a = p_pre >= 0.5

    if incipient_a:
        ph_a *= 0.55
        ph_b *= 1.45
    else:
        ph_b *= 0.55
        ph_a *= 1.45

    for _ in range(tau, HORIZON):
        ph_a, ph_b = advance(ph_a, ph_b, rng)
    p_final = prob_a(ph_a, ph_b)
    final_a = p_final >= 0.5
    return {
        "p_pre": p_pre,
        "openness_pre": h2(p_pre),
        "incipient_a": incipient_a,
        "p_final": p_final,
        "final_a": final_a,
        "switch": bool(final_a != incipient_a),
    }


def rank_corr(x, y) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    rows = {}
    pooled_o, pooled_s = [], []
    for i, tau in enumerate(TAUS):
        trials = [trial(SEED + i * 10_000 + j * 17, tau) for j in range(N_EP)]
        openness = np.array([t["openness_pre"] for t in trials])
        switches = np.array([t["switch"] for t in trials], dtype=float)
        rows[str(tau)] = {
            "mean_openness": round(float(openness.mean()), 5),
            "median_openness": round(float(np.median(openness)), 5),
            "switch_rate": round(float(switches.mean()), 5),
            "rank_corr_within_tau": round(rank_corr(openness, switches), 5),
        }
        pooled_o.extend(openness.tolist())
        pooled_s.extend(switches.tolist())
        print(f"tau={tau}: O={rows[str(tau)]['mean_openness']} "
              f"switch={rows[str(tau)]['switch_rate']}", flush=True)

    mean_o = [rows[str(t)]["mean_openness"] for t in TAUS]
    switch = [rows[str(t)]["switch_rate"] for t in TAUS]
    outcomes = {
        "SBI1_openness_control_law": bool(rank_corr(mean_o, switch) > 0.5),
        "SBI2_pre_post_contrast": bool(
            np.mean([rows["120"]["switch_rate"], rows["220"]["switch_rate"]])
            > np.mean([rows["460"]["switch_rate"], rows["620"]["switch_rate"]])
        ),
        "SBI3_episode_level_rank": bool(rank_corr(pooled_o, pooled_s) > 0.1),
        "SBI4_late_robustness": bool(rows["620"]["switch_rate"] < 0.2),
        "tau_level_rank_corr": round(rank_corr(mean_o, switch), 5),
        "pooled_episode_rank_corr": round(rank_corr(pooled_o, pooled_s), 5),
    }
    report = {
        "status": "SYM-BRIDGE-INT profile predicts controllability; preregistered",
        "config": {"N": N, "horizon": HORIZON, "taus": TAUS,
                   "episodes_per_tau": N_EP,
                   "impulse": {"incipient_multiplier": 0.55,
                               "opposite_multiplier": 1.45}},
        "per_tau": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "sym_bridge_intervention.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
