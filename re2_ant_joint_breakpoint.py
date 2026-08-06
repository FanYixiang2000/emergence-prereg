"""RE-2: V3 re-adjudication of the ant double bridge (DIRECTIONAL).

Registered in V2_ALIGNMENT_PREREGISTRATION.md (RE battery, frozen
before this run). Same Deneubourg dynamics/constants as the frozen
v1 battery (imported from ant_contrast). New measurement object: the
FORWARD JOINT possibility space -- from saved pheromone states,
cloned continuations with fixed horizon give the macro-basin
distribution; its entropy is the colony's openness at trip t.

Declared before running: hinge model comparison uses LINEAR x = trip
index (trips are linearly scaled, unlike training steps); the
log10(t+1) variant is reported alongside but the verdict is on
linear x.

RE2-1 TRAIL median openness has a hinge breakpoint (Delta-BIC >= 2).
RE2-2 t*(hinge) < median t_completion; per-episode half-collapse
      precedes completion in >= 80% of committing episodes.
RE2-3 SOLO: no breakpoint, no commitment.
RE2-4 RE2-1 survives 2x thinning (both parities).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ant_contrast import ALPHA, K, N_ANTS, Q, RHO
from breakpoint_model_comparison import bic, fit_one_segment, \
    fit_two_segment

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_TRIPS = 500
GRID = tuple(range(0, 401, 10))          # 41 saved states per episode
N_EPISODES = 30
N_CONT = 30
HORIZON = 200
BASIN_WINDOW = 40
DEV_SUSTAIN = 20


def choice_prob(phA: float, phB: float, mode: str,
                rng: np.random.Generator) -> float:
    if mode == "SOLO":
        return 0.5
    a = (K + phA) ** ALPHA
    b = (K + phB) ** ALPHA
    return a / (a + b)


def step_state(phA: float, phB: float, c: int) -> Tuple[float, float]:
    phA *= (1 - RHO)
    phB *= (1 - RHO)
    if c == 0:
        phA += Q
    else:
        phB += Q
    return phA, phB


def run_episode(mode: str, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    phA, phB = 1.0, 1.0
    states = {}
    dev_run = 0
    t_completion: Optional[int] = None
    for t in range(N_TRIPS):
        if t in GRID:
            states[t] = (phA, phB)
        p = choice_prob(phA, phB, mode, rng)
        dev = abs(p - 0.5) * 2.0
        dev_run = dev_run + 1 if dev >= 0.9 else 0
        if dev_run >= DEV_SUSTAIN and t_completion is None:
            t_completion = t - DEV_SUSTAIN + 1
        c = 0 if rng.random() < p else 1
        phA, phB = step_state(phA, phB, c)
    return {"states": states, "t_completion": t_completion}


def continuation_basin(state: Tuple[float, float], mode: str,
                       seed: int) -> str:
    rng = np.random.default_rng(seed)
    phA, phB = state
    choices: List[int] = []
    for _t in range(HORIZON):
        p = choice_prob(phA, phB, mode, rng)
        c = 0 if rng.random() < p else 1
        choices.append(c)
        phA, phB = step_state(phA, phB, c)
    fB = sum(choices[-BASIN_WINDOW:]) / BASIN_WINDOW
    if fB < 0.3:
        return "A"
    if fB > 0.7:
        return "B"
    return "open"


def basin_entropy(basins: List[str]) -> float:
    n = len(basins)
    h = 0.0
    for b in set(basins):
        p = basins.count(b) / n
        h -= p * math.log2(p)
    return h


def hinge_test(x: np.ndarray, y: np.ndarray) -> Dict:
    n = len(y)
    rss1 = fit_one_segment(x, y)
    best = None
    for bi in range(1, n - 1):
        rss2 = fit_two_segment(x, y, bi)
        if best is None or rss2 < best[1]:
            best = (bi, rss2)
    bi, rss2 = best
    delta = bic(rss1, n, 2) - bic(rss2, n, 4)
    return {"delta_bic": round(delta, 3), "hinge_x": float(x[bi]),
            "hinge_index": bi, "verdict": bool(delta >= 2.0)}


def openness_curves(mode: str) -> Tuple[np.ndarray, List[Dict]]:
    per_episode = []
    for ep in range(N_EPISODES):
        run = run_episode(mode, 50_000 + ep)
        vals = {}
        for t, st in run["states"].items():
            basins = [continuation_basin(st, mode,
                                         60_000 + ep * 10_000
                                         + t * 100 + k)
                      for k in range(N_CONT)]
            vals[t] = basin_entropy(basins)
        per_episode.append({"episode": ep, "openness": vals,
                            "t_completion": run["t_completion"]})
    med = np.array([np.median([e["openness"][t] for e in per_episode])
                    for t in GRID])
    return med, per_episode


def main() -> None:
    x_lin = np.array(GRID, dtype=float)
    report: Dict = {"status": ("RE-2 ant joint-possibility breakpoint; "
                               "registered RE battery, frozen before "
                               "run; verdict axis = linear trips as "
                               "declared in header")}

    med_trail, eps_trail = openness_curves("TRAIL")
    med_solo, eps_solo = openness_curves("SOLO")

    ht = hinge_test(x_lin, med_trail)
    ht_log = hinge_test(np.log10(x_lin + 1), med_trail)
    hs = hinge_test(x_lin, med_solo)

    committing = [e for e in eps_trail if e["t_completion"] is not None]
    comp_times = [e["t_completion"] for e in committing]
    med_completion = float(np.median(comp_times)) if comp_times else None

    half_before = 0
    for e in committing:
        o0 = e["openness"][0]
        t_half = None
        for t in GRID:
            if o0 > 0 and e["openness"][t] <= 0.5 * o0:
                t_half = t
                break
        if t_half is not None and t_half < e["t_completion"]:
            half_before += 1
    frac_half_before = (half_before / len(committing)
                        if committing else 0.0)

    thin = {}
    for parity in (0, 1):
        xt = x_lin[parity::2]
        yt = med_trail[parity::2]
        t2 = hinge_test(xt, yt)
        step = 20.0
        t2["hinge_ok"] = bool(abs(t2["hinge_x"] - ht["hinge_x"])
                              <= step)
        thin[f"parity{parity}"] = t2

    re2_1 = bool(ht["verdict"])
    re2_2 = bool(med_completion is not None
                 and ht["hinge_x"] < med_completion
                 and frac_half_before >= 0.8)
    re2_3 = bool((not hs["verdict"])
                 and all(e["t_completion"] is None for e in eps_solo))
    re2_4 = all(t["verdict"] and t["hinge_ok"] for t in thin.values())

    outcomes = {"RE2_1_trail_breakpoint": re2_1,
                "RE2_2_collapse_before_completion": re2_2,
                "RE2_3_solo_null": re2_3,
                "RE2_4_thinning_persistence": re2_4}
    report.update({
        "grid_trips": list(GRID),
        "median_openness_trail": [round(v, 4) for v in med_trail],
        "median_openness_solo": [round(v, 4) for v in med_solo],
        "hinge_trail_linear": ht, "hinge_trail_log": ht_log,
        "hinge_solo_linear": hs,
        "n_committing_episodes": len(committing),
        "median_t_completion": med_completion,
        "frac_half_collapse_before_completion": round(frac_half_before,
                                                      3),
        "thinning": thin,
        "per_episode_trail": [{"episode": e["episode"],
                               "t_completion": e["t_completion"]}
                              for e in eps_trail],
        "registered_outcomes": outcomes,
    })
    out = OUTPUTS / "re2_ant_joint_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print("hinge (linear):", ht, "| median completion:", med_completion,
          "| frac half<completion:", round(frac_half_before, 3))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
