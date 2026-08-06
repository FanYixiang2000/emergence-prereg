"""ANT-INT: episode-time commitment-window intervention.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Paired counterfactuals: control and perturbed episodes share
the SAME rng stream (one uniform draw per trip in both), so the only
difference is the choice rule inside the intervention window.

Conditions: none / early(5) / commit(30, covers RE-2 hinge t*=40) /
late(150) / random(U{0..270}); W = 30 forced-random trips.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ant_contrast import ALPHA, K, Q, RHO

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_TRIPS = 500
W = 30
N_EP = 200
DEV_SUSTAIN = 20
BASIN_WINDOW = 40
CONDITIONS = ("none", "early", "commit", "late", "random")
START = {"early": 5, "commit": 30, "late": 150}


def run_episode(seed: int, window_start: Optional[int]) -> Dict:
    rng = np.random.default_rng(seed)
    phA, phB = 1.0, 1.0
    choices: List[int] = []
    dev_run = 0
    t_completion: Optional[int] = None
    t_reentry: Optional[int] = None
    w_end = (window_start + W) if window_start is not None else None
    for t in range(N_TRIPS):
        a = (K + phA) ** ALPHA
        b = (K + phB) ** ALPHA
        p = a / (a + b)
        forced = (window_start is not None
                  and window_start <= t < w_end)
        p_eff = 0.5 if forced else p
        dev = abs(p - 0.5) * 2.0  # commitment measured on the trail
        dev_run = dev_run + 1 if (dev >= 0.9 and not forced) else 0
        if dev_run >= DEV_SUSTAIN:
            t_here = t - DEV_SUSTAIN + 1
            if t_completion is None:
                t_completion = t_here
            if (t_reentry is None and w_end is not None
                    and t_here >= w_end):
                t_reentry = t_here - w_end
        c = 0 if rng.random() < p_eff else 1
        choices.append(c)
        phA *= (1 - RHO)
        phB *= (1 - RHO)
        if c == 0:
            phA += Q
        else:
            phB += Q
    fB = sum(choices[-BASIN_WINDOW:]) / BASIN_WINDOW
    route = "A" if fB < 0.3 else ("B" if fB > 0.7 else "open")
    return {"route": route, "t_completion": t_completion,
            "t_reentry": t_reentry}


def binom_two_sided(k1: int, n1: int, k2: int, n2: int,
                    n_perm: int = 20_000, seed: int = 7) -> float:
    """Permutation test for difference in two proportions."""
    rng = np.random.default_rng(seed)
    pool = np.concatenate([np.ones(k1), np.zeros(n1 - k1),
                           np.ones(k2), np.zeros(n2 - k2)])
    obs = k1 / n1 - k2 / n2
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        d = pool[:n1].mean() - pool[n1:].mean()
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return cnt / n_perm


def main() -> None:
    rng_windows = np.random.default_rng(424242)
    random_starts = rng_windows.integers(0, 271, size=N_EP)

    rows: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}
    for ep in range(N_EP):
        seed = 700_000 + ep
        for cond in CONDITIONS:
            if cond == "none":
                ws = None
            elif cond == "random":
                ws = int(random_starts[ep])
            else:
                ws = START[cond]
            rows[cond].append(run_episode(seed, ws))

    controls = rows["none"]
    committing = [i for i, r in enumerate(controls)
                  if r["route"] in ("A", "B")]

    stats = {}
    for cond in CONDITIONS[1:]:
        flips = sum(1 for i in committing
                    if rows[cond][i]["route"] != controls[i]["route"])
        delays = [rows[cond][i]["t_completion"]
                  - controls[i]["t_completion"]
                  for i in committing
                  if rows[cond][i]["t_completion"] is not None
                  and controls[i]["t_completion"] is not None]
        reentries = [r["t_reentry"] for r in rows[cond]
                     if r["t_reentry"] is not None]
        stats[cond] = {
            "n_paired": len(committing),
            "flips": flips,
            "flip_rate": round(flips / len(committing), 4),
            "median_delay": (float(np.median(delays)) if delays
                             else None),
            "median_reentry": (float(np.median(reentries))
                               if reentries else None),
            "n_no_completion": sum(
                1 for i in committing
                if rows[cond][i]["t_completion"] is None),
        }

    pvals = {}
    for other in ("early", "late", "random"):
        pvals[f"commit_vs_{other}"] = binom_two_sided(
            stats["commit"]["flips"], len(committing),
            stats[other]["flips"], len(committing))

    ai1 = all(stats["commit"]["flips"] > stats[o]["flips"]
              and pvals[f"commit_vs_{o}"] < 0.05
              for o in ("early", "late", "random"))
    delays_ok = [c for c in ("early", "late", "random")
                 if stats[c]["median_delay"] is not None]
    ai2 = (stats["commit"]["median_delay"] is not None
           and all(stats["commit"]["median_delay"]
                   > stats[o]["median_delay"] for o in delays_ok))
    ai3 = stats["late"]["flip_rate"] < 0.05

    outcomes = {"AI1_commit_flip_maximal": bool(ai1),
                "AI2_commit_delay_maximal": bool(ai2),
                "AI3_late_robust": bool(ai3)}
    report = {
        "status": ("ANT-INT episode-time commitment-window "
                   "intervention; registered before run; paired "
                   "same-seed counterfactuals; E3C's training-time "
                   "withdrawal unaffected"),
        "config": {"W": W, "N_EP": N_EP, "starts": START,
                   "random_start_range": [0, 270],
                   "hinge_from_RE2": 40},
        "n_control_committing": len(committing),
        "stats": stats,
        "permutation_pvals": pvals,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ant_commitment_intervention.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"stats": {c: {k: v for k, v in s.items()
                                    if k in ("flip_rate",
                                             "median_delay",
                                             "median_reentry")}
                                for c, s in stats.items()},
                      "pvals": pvals, **outcomes}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
