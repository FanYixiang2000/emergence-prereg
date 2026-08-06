"""ANT-FINE: onset resolution in the ant system, both regimes.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run; gradual-regime constants fixed by the disclosed pilot).
Regime A (large-kick, RE-2 constants Q=1) on a 1-trip grid;
regime B (gradual, Q=0.5) on a 5-trip grid over 400 trips.
Matured V3.1 detector contract throughout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ant_contrast import K, RHO
from tri_c_breakpoint import hinge_linear
from kuramoto_breakpoint_r2 import truncate_at_saturation

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_EPISODES = 30
N_CONT = 30
HORIZON = 200
BASIN_WINDOW = 40
DEV_SUSTAIN = 20
GATE = 0.1
ALPHA = 2.0
REGIMES = {
    "large_kick": {"Q": 1.0, "grid": tuple(range(0, 61, 1)),
                   "n_trips": 60},
    "gradual": {"Q": 0.5, "grid": tuple(range(0, 401, 5)),
                "n_trips": 400},
}


def choice_p(phA, phB):
    a = (K + phA) ** ALPHA
    b = (K + phB) ** ALPHA
    return a / (a + b)


def run_episode(Q: float, n_trips: int, grid, seed: int):
    rng = np.random.default_rng(seed)
    phA = phB = 1.0
    states = {}
    run = 0
    t_completion: Optional[int] = None
    wanted = set(grid)
    for t in range(n_trips + 1):
        if t in wanted:
            states[t] = (phA, phB)
        if t == n_trips:
            break
        p = choice_p(phA, phB)
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
    return states, t_completion


def continuation_entropy(state, Q: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    phA = np.full(N_CONT, state[0])
    phB = np.full(N_CONT, state[1])
    tail = np.zeros((N_CONT, BASIN_WINDOW), dtype=int)
    for t in range(HORIZON):
        p = choice_p(phA, phB)
        c = (rng.random(N_CONT) >= p).astype(int)
        tail[:, t % BASIN_WINDOW] = c
        phA = phA * (1 - RHO) + Q * (c == 0)
        phB = phB * (1 - RHO) + Q * (c == 1)
    fB = tail.mean(axis=1)
    basins = np.where(fB < 0.3, 0, np.where(fB > 0.7, 1, 2))
    h = 0.0
    for b in (0, 1, 2):
        pr = (basins == b).mean()
        if pr > 0:
            h -= pr * math.log2(pr)
    return h


def adjudicate(grid, y: np.ndarray) -> Dict:
    x = np.array(grid, dtype=float)
    yn = y / math.log2(3)
    drop = float(yn[0] - yn[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        out["b5_onset"] = False
        return out
    xw, yw, t_sat = truncate_at_saturation(x, yn)
    out["t_sat"] = t_sat
    out["window_points"] = len(yw)
    if len(yw) < 10:
        out["verdict"] = "window_too_short_no_resolvable_onset"
        out["b5_onset"] = False
        return out
    full = hinge_linear(xw, yw)
    span = xw[-1] - xw[0]
    thin_ok = True
    thin = {}
    for parity in (0, 1):
        t = hinge_linear(xw[parity::2], yw[parity::2])
        ok = (t["delta_bic"] >= 2.0 and t["onset_type"]
              and abs(t["t_star"] - full["t_star"]) <= 0.10 * span)
        t["ok"] = bool(ok)
        thin_ok = thin_ok and ok
        thin[f"parity{parity}"] = t
    out.update({
        "hinge": full, "thinning": thin,
        "b5_onset": bool(full["delta_bic"] >= 10
                         and full["onset_type"] and thin_ok),
    })
    return out


def main() -> None:
    results = {}
    for name, cfg in REGIMES.items():
        grid, Q = cfg["grid"], cfg["Q"]
        per_ep, completions = [], []
        for ep in range(N_EPISODES):
            states, t_comp = run_episode(Q, cfg["n_trips"], grid,
                                         57_000 + ep)
            completions.append(t_comp)
            vals = [continuation_entropy(states[t], Q,
                                         58_000 + ep * 10_000 + t)
                    for t in grid]
            per_ep.append(vals)
        med = np.median(np.array(per_ep), axis=0)
        adj = adjudicate(grid, med)
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
              f"commit={adj['n_committing']}/{N_EPISODES} "
              f"med_comp={adj['median_completion']}", flush=True)

    grad, lk = results["gradual"], results["large_kick"]
    af1 = bool(grad["b5_onset"])
    af2 = bool(not lk["b5_onset"])
    af3 = bool(af1 and grad["median_completion"] is not None
               and grad["hinge"]["t_star"] < grad["median_completion"])

    outcomes = {"AF1_gradual_onset": af1,
                "AF2_large_kick_no_onset": af2,
                "AF3_gradual_lead": af3}
    report = {"status": ("ANT-FINE onset resolution in both regimes; "
                         "registered before run; constants from the "
                         "disclosed pilot; matured V3.1 contract"),
              "config": {"regimes": {k: {"Q": v["Q"],
                                         "grid_step": v["grid"][1],
                                         "n_trips": v["n_trips"]}
                                     for k, v in REGIMES.items()},
                         "episodes": N_EPISODES, "n_cont": N_CONT},
              "regimes": results,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "ant_fine_onset.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
