"""BP: regime-breakpoint test on stored joint-collapse curves.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (BP section, frozen
before this run, with the stored-data disclosure). Implements the
one-segment vs continuous two-segment linear model comparison with
BIC on x = log10(steps), for all four declared series.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SERIES = ("collapse_norm", "C_individual", "C_env", "C_relational")
FILES = {
    "s93001": "overcooked_joint_collapse_s93001.json",
    "s93002": "overcooked_joint_collapse_s93002.json",
    "s93003": "overcooked_joint_collapse_s93003.json",
    "dense": "overcooked_joint_collapse_dense_s93001.json",
}


def fit_one_segment(x: np.ndarray, y: np.ndarray) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(((y - A @ coef) ** 2).sum())


def fit_two_segment(x: np.ndarray, y: np.ndarray,
                    bi: int) -> float:
    """Continuous piecewise linear with breakpoint at x[bi]."""
    xb = x[bi]
    A = np.vstack([x, np.maximum(x - xb, 0.0), np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(((y - A @ coef) ** 2).sum())


def bic(rss: float, n: int, k: int) -> float:
    rss = max(rss, 1e-12)
    return n * math.log(rss / n) + k * math.log(n)


def breakpoint_test(steps: List[int], y: List[float]) -> Dict:
    x = np.log10(np.array(steps, dtype=float))
    ya = np.array(y, dtype=float)
    n = len(ya)
    rss1 = fit_one_segment(x, ya)
    best = None
    for bi in range(1, n - 1):  # interior grid points
        rss2 = fit_two_segment(x, ya, bi)
        if best is None or rss2 < best[1]:
            best = (bi, rss2)
    bi, rss2 = best
    delta = bic(rss1, n, 2) - bic(rss2, n, 4)
    return {"delta_bic": round(delta, 3),
            "breakpoint_step": steps[bi],
            "verdict": bool(delta >= 2.0),
            "rss_1seg": round(rss1, 6), "rss_2seg": round(rss2, 6)}


def run_file(name: str) -> Dict:
    r = json.loads((OUTPUTS / FILES[name]).read_text(encoding="utf-8"))
    grid = r["checkpoint_grid"]
    out = {}
    for s in SERIES:
        y = [r["curve"][str(c)][s] for c in grid]
        out[s] = breakpoint_test(grid, y)
    out["_grid"] = grid
    return out


def main() -> None:
    results = {name: run_file(name) for name in FILES}

    def in_window(step: int, lo: int, hi: int) -> bool:
        return lo <= step <= hi

    bp1_seeds = {}
    for seed in ("s93001", "s93002", "s93003"):
        t = results[seed]["C_env"]
        bp1_seeds[seed] = bool(t["verdict"] and
                               in_window(t["breakpoint_step"],
                                         640_000, 1_500_000))
    bp1 = all(bp1_seeds.values())

    td = results["dense"]["C_env"]
    bp2 = bool(td["verdict"] and in_window(td["breakpoint_step"],
                                           820_000, 1_250_000))

    # BP-3: 2x thinning of the dense grid, both parities
    r = json.loads((OUTPUTS / FILES["dense"]).read_text(encoding="utf-8"))
    grid = r["checkpoint_grid"]
    y = [r["curve"][str(c)]["C_env"] for c in grid]
    thin = {}
    dense_bp = td["breakpoint_step"]
    dense_idx = grid.index(dense_bp)
    for parity in (0, 1):
        g2 = grid[parity::2]
        y2 = y[parity::2]
        t2 = breakpoint_test(g2, y2)
        # same window +/- one grid step of the dense breakpoint
        lo = grid[max(dense_idx - 1, 0)]
        hi = grid[min(dense_idx + 1, len(grid) - 1)]
        t2["window_ok"] = bool(in_window(t2["breakpoint_step"], lo, hi))
        thin[f"parity{parity}"] = t2
    bp3 = all(t["verdict"] and t["window_ok"] for t in thin.values())

    outcomes = {"BP1_seeds_env_breakpoint": bp1,
                "BP2_dense_env_breakpoint": bp2,
                "BP3_grid_persistence": bp3,
                "BP1_by_seed": bp1_seeds}
    report = {
        "status": ("BP regime-breakpoint model comparison on STORED "
                   "collapse curves; registered with disclosure that "
                   "curves were previously inspected (limited "
                   "confirmatory value); thresholds frozen before run; "
                   "fresh-seed confirmation registered as BP-FRESH "
                   "future work"),
        "method": ("one-segment vs continuous two-segment linear on "
                   "log10(steps), BIC comparison, Delta-BIC >= 2"),
        "results": results,
        "thinning": thin,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "breakpoint_model_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    for name in FILES:
        row = {s: (results[name][s]["delta_bic"],
                   results[name][s]["breakpoint_step"],
                   results[name][s]["verdict"]) for s in SERIES}
        print(name, row)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
