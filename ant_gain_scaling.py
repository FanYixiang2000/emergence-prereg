"""ANT-GAIN: breakpoint scaling with feedback gain.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Sweeps the Deneubourg nonlinearity alpha; openness instrument
identical in kind to RE-2 (cloned continuations, basin entropy);
detector = matured V3.1 contract. Continuations vectorized across
clones for speed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from ant_contrast import K, Q, RHO
from tri_c_breakpoint import hinge_linear
from kuramoto_breakpoint_r2 import truncate_at_saturation

OUTPUTS = Path(__file__).resolve().parent / "outputs"
ALPHAS = (1.0, 1.5, 2.0, 3.0, 4.0)
N_TRIPS = 400
GRID = tuple(range(0, 401, 10))
N_EPISODES = 12
N_CONT = 20
HORIZON = 200
BASIN_WINDOW = 40
GATE = 0.1


def choice_p(phA, phB, alpha: float):
    a = (K + phA) ** alpha
    b = (K + phB) ** alpha
    return a / (a + b)


def run_episode_states(alpha: float, seed: int) -> Dict[int, tuple]:
    rng = np.random.default_rng(seed)
    phA, phB = 1.0, 1.0
    states = {}
    for t in range(N_TRIPS + 1):
        if t in GRID:
            states[t] = (phA, phB)
        if t == N_TRIPS:
            break
        p = choice_p(phA, phB, alpha)
        c = 0 if rng.random() < p else 1
        phA *= (1 - RHO)
        phB *= (1 - RHO)
        if c == 0:
            phA += Q
        else:
            phB += Q
    return states


def continuation_entropy(state, alpha: float, seed: int) -> float:
    """Basin entropy of N_CONT vectorized cloned continuations."""
    rng = np.random.default_rng(seed)
    phA = np.full(N_CONT, state[0])
    phB = np.full(N_CONT, state[1])
    tail = np.zeros((N_CONT, BASIN_WINDOW), dtype=int)
    for t in range(HORIZON):
        p = choice_p(phA, phB, alpha)
        c = (rng.random(N_CONT) >= p).astype(int)  # 1 = branch B
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


def median_curve(alpha: float) -> np.ndarray:
    per_ep = []
    for ep in range(N_EPISODES):
        states = run_episode_states(alpha, 55_000 + ep)
        vals = [continuation_entropy(states[t], alpha,
                                     56_000 + ep * 1_000 + t)
                for t in GRID]
        per_ep.append(vals)
    return np.median(np.array(per_ep), axis=0)


def adjudicate(y: np.ndarray) -> Dict:
    x = np.array(GRID, dtype=float)
    # normalize openness to [0,1] scale for the gate (max entropy
    # of 3 basins = log2 3)
    yn = y / math.log2(3)
    drop = float(yn[0] - yn[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        return out
    xw, yw, t_sat = truncate_at_saturation(x, yn)
    out["t_sat"] = t_sat
    if len(yw) < 10:
        # collapse completes almost immediately: no resolvable
        # pre-onset phase exists at this grid density
        out["verdict"] = "window_too_short_no_resolvable_onset"
        out["b5"] = False
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
        "b5": bool(full["delta_bic"] >= 10 and full["onset_type"]
                   and thin_ok),
    })
    return out


def main() -> None:
    rows = {}
    for alpha in ALPHAS:
        y = median_curve(alpha)
        adj = adjudicate(y)
        adj["median_openness"] = [round(v, 4) for v in y]
        rows[str(alpha)] = adj
        h = adj.get("hinge", {})
        print(f"alpha={alpha}: drop={adj['drop']} "
              f"gate={adj['gate_passed']} b5={adj.get('b5')} "
              f"t*={h.get('t_star')} dBIC={h.get('delta_bic')} "
              f"slope_after={h.get('slope_after')}", flush=True)

    high = [a for a in (2.0, 3.0, 4.0) if rows[str(a)].get("b5")]
    ag1 = bool(not rows["1.0"].get("b5", False) and len(high) == 3)
    passing = [a for a in ALPHAS if rows[str(a)].get("b5")]
    tstars = [rows[str(a)]["hinge"]["t_star"] for a in passing]
    slopes = [abs(rows[str(a)]["hinge"]["slope_after"])
              for a in passing]
    ag2 = bool(len(passing) >= 2
               and all(t1 > t2 for t1, t2 in zip(tstars, tstars[1:])))
    ag3 = bool(len(passing) >= 2
               and all(s1 < s2 for s1, s2 in zip(slopes, slopes[1:])))

    outcomes = {"AG1_existence_boundary": ag1,
                "AG2_onset_law": ag2, "AG3_sharpness_law": ag3,
                "passing_alphas": passing,
                "t_stars": tstars,
                "post_slopes": [round(s, 6) for s in slopes]}
    report = {"status": ("ANT-GAIN breakpoint scaling with feedback "
                         "gain; registered before run; matured V3.1 "
                         "detector contract"),
              "config": {"alphas": ALPHAS, "episodes": N_EPISODES,
                         "n_cont": N_CONT, "horizon": HORIZON},
              "per_alpha": rows,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "ant_gain_scaling.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
