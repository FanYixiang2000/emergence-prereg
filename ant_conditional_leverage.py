"""ANT-INT-C: per-episode conditional openness-leverage.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Conditions outcome-flip leverage on each episode's OWN
openness o_t = H_2(p_t) at the intervention start, read from the
paired control trajectory. Fresh seeds (710000+).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ant_contrast import ALPHA, K, Q, RHO
from ant_commitment_intervention import run_episode as ref_episode

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_TRIPS = 500
W = 30
N_EP = 300
BASIN_WINDOW = 40
STARTS = tuple(range(0, 271, 10))
BINS = ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0001))


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def run_episode_traj(seed: int,
                     window_start: Optional[int]
                     ) -> Tuple[str, List[float]]:
    """Identical dynamics to ant_commitment_intervention.run_episode
    (same one-uniform-per-trip rng semantics), additionally
    returning the pre-choice p trajectory."""
    rng = np.random.default_rng(seed)
    phA, phB = 1.0, 1.0
    choices: List[int] = []
    p_traj: List[float] = []
    w_end = (window_start + W) if window_start is not None else None
    for t in range(N_TRIPS):
        a = (K + phA) ** ALPHA
        b = (K + phB) ** ALPHA
        p = a / (a + b)
        p_traj.append(p)
        forced = (window_start is not None
                  and window_start <= t < w_end)
        p_eff = 0.5 if forced else p
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
    return route, p_traj


def main() -> None:
    # sanity: trajectory variant must reproduce the frozen dynamics
    for s in (710_000, 710_001, 710_002, 710_003, 710_004,
              710_005, 710_006, 710_007, 710_008, 710_009):
        assert run_episode_traj(s, None)[0] == \
            ref_episode(s, None)["route"]
        assert run_episode_traj(s, 30)[0] == \
            ref_episode(s, 30)["route"]

    controls = {}
    for ep in range(N_EP):
        seed = 710_000 + ep
        controls[ep] = run_episode_traj(seed, None)
    committing = [ep for ep, (route, _t) in controls.items()
                  if route in ("A", "B")]

    pairs = []  # (openness_at_s, flip)
    for ep in committing:
        seed = 710_000 + ep
        c_route, c_traj = controls[ep]
        for s in STARTS:
            o = h2(c_traj[s])
            route, _ = run_episode_traj(seed, s)
            pairs.append((o, int(route != c_route)))

    o_arr = np.array([p[0] for p in pairs])
    f_arr = np.array([p[1] for p in pairs])

    bin_rows = []
    for lo, hi in BINS:
        m = (o_arr >= lo) & (o_arr < hi)
        rate = float(f_arr[m].mean()) if m.sum() else None
        bin_rows.append({"bin": [lo, min(hi, 1.0)],
                         "n": int(m.sum()),
                         "flip_rate": (round(rate, 4)
                                       if rate is not None else None)})
        print(f"bin [{lo},{min(hi, 1.0)}): n={m.sum()} "
              f"flip_rate={rate}", flush=True)

    rates = [r["flip_rate"] for r in bin_rows]
    aic1 = bool(all(r is not None for r in rates)
                and all(rates[i] < rates[i + 1] for i in range(3)))
    aic2 = bool(rates[0] is not None and rates[0] < 0.05)

    obs = float(o_arr[f_arr == 1].mean() - o_arr[f_arr == 0].mean())
    rng = np.random.default_rng(11)
    cnt = 0
    for _ in range(20_000):
        perm = rng.permutation(f_arr)
        d = float(o_arr[perm == 1].mean() - o_arr[perm == 0].mean())
        if d >= obs - 1e-12:
            cnt += 1
    pval = cnt / 20_000
    aic3 = bool(obs > 0 and pval < 0.001)

    outcomes = {"AIC1_conditional_monotone": aic1,
                "AIC2_closed_uncontrollable": aic2,
                "AIC3_separation": aic3,
                "mean_openness_flipped_minus_not": round(obs, 4),
                "permutation_p": pval}
    report = {
        "status": ("ANT-INT-C per-episode conditional "
                   "openness-leverage; registered before run; fresh "
                   "seeds 710000+; paired same-seed counterfactuals"),
        "n_committing": len(committing),
        "n_pairs": len(pairs),
        "bins": bin_rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ant_conditional_leverage.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
