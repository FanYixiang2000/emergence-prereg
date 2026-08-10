"""ANT-FSS: full finite-size scaling of commitment collapse.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running (with the
mechanistic ln-N derivation recorded up front). Constants identical to
ANT-COLONY-BP; only the size range, horizon and scaling analyses are
new.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ant_colony_breakpoint import episode as _episode_short  # noqa: F401 (constants doc)
from ant_contrast import K, RHO
from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
ALPHA = 2.0
Q_TOTAL = 0.5
N_TRIPS = 1500
GRID = list(range(0, N_TRIPS + 1, 10))
N_EPISODES = 30
SEED_BASE = 71_000
SIZES = (1, 2, 5, 10, 20, 50, 100, 200, 500)
COLLAPSE_SIZES = (50, 100, 200, 500)
N_BOOT = 200
WINDOW = 100  # +/- around t50 for the translation collapse


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def episode(N: int, seed: int):
    rng = np.random.default_rng(seed)
    phA = phB = 1.0
    q = Q_TOTAL / N
    vals = []
    wanted = set(GRID)
    for t in range(N_TRIPS + 1):
        a = (K + phA) ** ALPHA
        b = (K + phB) ** ALPHA
        p = a / (a + b)
        if t in wanted:
            vals.append(h2(p))
        if t == N_TRIPS:
            break
        nA = rng.binomial(N, p)
        phA = phA * (1 - RHO) + q * nA
        phB = phB * (1 - RHO) + q * (N - nA)
    return np.array(vals)


def crossing(grid, curve, level):
    for i, v in enumerate(curve):
        if v <= level:
            return float(grid[i])
    return None


def main() -> None:
    rng = np.random.default_rng(SEED_BASE)
    per_size = {}
    curves_by_size = {}
    for N in SIZES:
        eps = np.array([episode(N, SEED_BASE + N * 1_000 + e)
                        for e in range(N_EPISODES)])
        med = np.median(eps, axis=0)
        curves_by_size[N] = eps
        adj = adjudicate(GRID, med * math.log2(3))
        t50 = crossing(GRID, med, 0.5)
        t80 = crossing(GRID, med, 0.8)
        t20 = crossing(GRID, med, 0.2)
        width = (t20 - t80) if (t20 is not None and t80 is not None) else None
        per_size[str(N)] = {
            "b5_onset": adj["b5_onset"],
            "verdict": adj.get("verdict", "hinge_tested"),
            "t_star": adj.get("hinge", {}).get("t_star"),
            "delta_bic": adj.get("hinge", {}).get("delta_bic"),
            "t50": t50, "width": width,
            "median_openness": [round(float(v), 4) for v in med],
        }
        print(f"N={N}: onset={adj['b5_onset']} t*={per_size[str(N)]['t_star']} "
              f"t50={t50} width={width}", flush=True)

    # FSS-1
    fss1 = bool(not per_size["1"]["b5_onset"]
                and all(per_size[str(N)]["b5_onset"] for N in SIZES if N >= 10))

    # FSS-2: log law on onset sizes
    onset_sizes = [N for N in SIZES if per_size[str(N)]["b5_onset"]
                   and per_size[str(N)]["t50"] is not None]
    ln_n = np.log([float(N) for N in onset_sizes])
    t50s = np.array([per_size[str(N)]["t50"] for N in onset_sizes])
    b, a = np.polyfit(ln_n, t50s, 1)
    pred = a + b * ln_n
    ss_res = float(np.sum((t50s - pred) ** 2))
    ss_tot = float(np.sum((t50s - t50s.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    boots = []
    for _ in range(N_BOOT):
        bt = []
        for N in onset_sizes:
            idx = rng.integers(0, N_EPISODES, N_EPISODES)
            med = np.median(curves_by_size[N][idx], axis=0)
            bt.append(crossing(GRID, med, 0.5))
        if any(v is None for v in bt):
            continue
        bb, _ = np.polyfit(ln_n, np.array(bt, dtype=float), 1)
        boots.append(bb)
    b_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    fss2 = bool(b > 0 and r2 >= 0.85)

    # FSS-3: width invariance at large N
    widths = [per_size[str(N)]["width"] for N in COLLAPSE_SIZES]
    fss3 = bool(all(w is not None for w in widths)
                and max(widths) / min(widths) <= 2.0)

    # FSS-4: translation collapse
    rel_grid = np.arange(-WINDOW, WINDOW + 1, 10, dtype=float)
    aligned, unaligned = [], []
    for N in COLLAPSE_SIZES:
        med = np.array(per_size[str(N)]["median_openness"])
        t50 = per_size[str(N)]["t50"]
        aligned.append(np.interp(rel_grid + t50, GRID, med))
        unaligned.append(np.interp(rel_grid + np.mean(
            [per_size[str(M)]["t50"] for M in COLLAPSE_SIZES]), GRID, med))
    def mean_pair_rms(curves):
        vals = []
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                vals.append(float(np.sqrt(np.mean(
                    (curves[i] - curves[j]) ** 2))))
        return float(np.mean(vals))
    rms_aligned = mean_pair_rms(aligned)
    rms_unaligned = mean_pair_rms(unaligned)
    fss4 = bool(rms_aligned <= 0.05
                and rms_aligned <= 0.30 * rms_unaligned)

    outcomes = {
        "FSS1_onset_pattern": fss1,
        "FSS2_log_law": fss2,
        "FSS3_width_invariance": fss3,
        "FSS4_translation_collapse": fss4,
        "log_law": {"b": round(float(b), 3), "a": round(float(a), 3),
                    "r2": round(r2, 4), "b_ci95": [round(v, 3) for v in b_ci]},
        "widths_large_N": {str(N): per_size[str(N)]["width"]
                           for N in COLLAPSE_SIZES},
        "rms_aligned": round(rms_aligned, 4),
        "rms_unaligned": round(rms_unaligned, 4),
        "onset_sizes": onset_sizes,
        "threshold_region": {str(N): per_size[str(N)]["b5_onset"]
                             for N in (2, 5)},
    }
    report = {"status": ("ANT-FSS full finite-size scaling; constants from "
                         "ANT-COLONY-BP; ln-N derivation recorded before "
                         "run; registered before run"),
              "config": {"sizes": SIZES, "n_trips": N_TRIPS,
                         "episodes": N_EPISODES, "grid_step": 10,
                         "n_boot": N_BOOT, "seed_base": SEED_BASE},
              "per_size": {k: {kk: vv for kk, vv in v.items()
                               if kk != "median_openness"}
                           for k, v in per_size.items()},
              "median_curves": {k: v["median_openness"]
                                for k, v in per_size.items()},
              "registered_outcomes": outcomes}
    out = OUTPUTS / "ant_fss.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
