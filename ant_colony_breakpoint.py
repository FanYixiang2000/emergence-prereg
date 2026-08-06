"""ANT-COLONY-BP: finite-size scaling of the breakpoint.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run; timescale constants from the disclosed pilot). N concurrent
ants per step; per-step relative fluctuation scales 1/sqrt(N);
object = behavioral openness H2(p_t); matured V3.1 detector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ant_contrast import K, RHO
from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
ALPHA = 2.0
Q_TOTAL = 0.5
N_TRIPS = 900
GRID = tuple(range(0, 901, 10))
N_EPISODES = 30
SEED_BASE = 61_000
SIZES = (1, 10, 100)
DEV_SUSTAIN = 20


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def episode(N: int, seed: int):
    rng = np.random.default_rng(seed)
    phA = phB = 1.0
    q = Q_TOTAL / N
    o = {}
    run = 0
    t_completion = None
    wanted = set(GRID)
    for t in range(N_TRIPS + 1):
        a = (K + phA) ** ALPHA
        b = (K + phB) ** ALPHA
        p = a / (a + b)
        if t in wanted:
            o[t] = h2(p)
        if t == N_TRIPS:
            break
        if abs(p - 0.5) * 2 >= 0.9:
            run += 1
            if run >= DEV_SUSTAIN and t_completion is None:
                t_completion = t - DEV_SUSTAIN + 1
        else:
            run = 0
        nA = rng.binomial(N, p)
        phA = phA * (1 - RHO) + q * nA
        phB = phB * (1 - RHO) + q * (N - nA)
    return o, t_completion


def main() -> None:
    results = {}
    for N in SIZES:
        curves, completions = [], []
        for ep in range(N_EPISODES):
            o, t_comp = episode(N, SEED_BASE + N * 1_000 + ep)
            curves.append([o[t] for t in GRID])
            completions.append(t_comp)
        med = np.median(np.array(curves), axis=0)
        adj = adjudicate(GRID, med * math.log2(3))
        comp = [c for c in completions if c is not None]
        adj["n_committing"] = len(comp)
        adj["median_completion"] = (float(np.median(comp))
                                    if comp else None)
        adj["median_openness"] = [round(v, 4) for v in med]
        results[str(N)] = adj
        h = adj.get("hinge", {})
        print(f"N={N}: drop={adj['drop']} "
              f"verdict={adj.get('verdict', 'hinge_tested')} "
              f"b5_onset={adj['b5_onset']} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->"
              f"{h.get('slope_after')} "
              f"med_comp={adj['median_completion']}", flush=True)

    onset_sizes = [N for N in SIZES if results[str(N)]["b5_onset"]]
    acb1 = bool(results["100"]["b5_onset"])
    post = {N: abs(results[str(N)]["hinge"]["slope_after"])
            for N in onset_sizes}
    pre = {N: abs(results[str(N)]["hinge"]["slope_before"])
           for N in onset_sizes}
    acb2 = bool((1 not in onset_sizes) and len(onset_sizes) >= 1
                and all(post[a] < post[b] for a, b in
                        zip(onset_sizes, onset_sizes[1:])))
    acb3 = bool(len(onset_sizes) >= 2
                and all(pre[a] > pre[b] for a, b in
                        zip(onset_sizes, onset_sizes[1:])))
    if len(onset_sizes) == 1:
        acb3 = None  # single size: comparison undefined, reported

    outcomes = {"ACB1_collective_onset": acb1,
                "ACB2_finite_size_sharpening": acb2,
                "ACB3_flattening_open_phase": acb3,
                "onset_sizes": onset_sizes}
    report = {"status": ("ANT-COLONY-BP finite-size scaling of the "
                         "breakpoint; registered before run; pilot "
                         "constants disclosed"),
              "config": {"sizes": SIZES, "Q_total": Q_TOTAL,
                         "n_trips": N_TRIPS, "episodes": N_EPISODES,
                         "seed_base": SEED_BASE},
              "per_size": results,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "ant_colony_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
