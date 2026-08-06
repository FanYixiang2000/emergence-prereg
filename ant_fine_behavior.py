"""ANT-FINE-B: onset on the current-state (behavioral) object.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Object: o_t = H2(p_t), the colony's behavioral openness --
a current-state variable, not an endpoint projection. Median across
30 fresh episodes per regime; matured V3.1 detector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ant_contrast import K, RHO
from ant_fine_onset import adjudicate, DEV_SUSTAIN, REGIMES, ALPHA

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_EPISODES = 30
SEED_BASE = 59_000


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def episode_behavior(Q: float, n_trips: int, grid, seed: int):
    rng = np.random.default_rng(seed)
    phA = phB = 1.0
    o = {}
    run = 0
    t_completion = None
    wanted = set(grid)
    for t in range(n_trips + 1):
        a = (K + phA) ** ALPHA
        b = (K + phB) ** ALPHA
        p = a / (a + b)
        if t in wanted:
            o[t] = h2(p)
        if t == n_trips:
            break
        if abs(p - 0.5) * 2 >= 0.9:
            run += 1
            if run >= DEV_SUSTAIN and t_completion is None:
                t_completion = t - DEV_SUSTAIN + 1
        else:
            run = 0
        c = 0 if rng.random() < p else 1
        phA *= (1 - RHO)
        phB *= (1 - RHO)
        if c == 0:
            phA += Q
        else:
            phB += Q
    return o, t_completion


def main() -> None:
    results = {}
    for name, cfg in REGIMES.items():
        grid, Q = cfg["grid"], cfg["Q"]
        curves, completions = [], []
        for ep in range(N_EPISODES):
            o, t_comp = episode_behavior(Q, cfg["n_trips"], grid,
                                         SEED_BASE + ep)
            curves.append([o[t] for t in grid])
            completions.append(t_comp)
        med = np.median(np.array(curves), axis=0)
        # H2 is already in [0,1]; adjudicate() divides by log2(3)
        # for the basin object, so pass the un-normalized curve
        # scaled to match its gate convention: use raw H2 * log2(3)
        adj = adjudicate(grid, med * math.log2(3))
        comp = [c for c in completions if c is not None]
        adj["n_committing"] = len(comp)
        adj["median_completion"] = (float(np.median(comp))
                                    if comp else None)
        adj["median_openness"] = [round(v, 4) for v in med]
        results[name] = adj
        h = adj.get("hinge", {})
        print(f"{name} (Q={Q}): drop={adj['drop']} "
              f"verdict={adj.get('verdict', 'hinge_tested')} "
              f"b5_onset={adj['b5_onset']} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->"
              f"{h.get('slope_after')} "
              f"med_comp={adj['median_completion']}", flush=True)

    grad, lk = results["gradual"], results["large_kick"]
    afb1 = bool(grad["b5_onset"])
    afb2 = bool(lk["b5_onset"])
    afb3 = bool(afb1 and grad["median_completion"] is not None
                and 60.0 < grad["hinge"]["t_star"]
                < grad["median_completion"])

    outcomes = {"AFB1_gradual_onset": afb1,
                "AFB2_large_kick_onset_MAY_PASS": afb2,
                "AFB3_placement": afb3}
    report = {"status": ("ANT-FINE-B onset on the current-state "
                         "behavioral object; registered before run; "
                         "matured V3.1 contract"),
              "config": {"episodes": N_EPISODES,
                         "seed_base": SEED_BASE},
              "regimes": results,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "ant_fine_behavior.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
