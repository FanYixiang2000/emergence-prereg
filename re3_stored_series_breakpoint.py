"""RE-3: V3 re-adjudication of Pythia / MultiBERTs stored series.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (RE battery, frozen
before this run, with the stored-data disclosure). Object:
test_entropy_bits (predictive openness) per stored checkpoint.
CLOSING breakpoint = Delta-BIC >= 2 AND post-hinge slope < pre-hinge
slope AND post-hinge slope < 0, on x = log10(step + 1). Persistence
= 2x thinning both parities (plus 4x for MultiBERTs, n = 29) keep
the closing verdict with hinge within +/- one (coarse) grid step.
Negative controls: random_target and shuffled_vocab must show no
persistent closing breakpoint.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from breakpoint_model_comparison import bic, fit_one_segment, \
    fit_two_segment

OUTPUTS = Path(__file__).resolve().parent / "outputs"
PYTHIA = ("410m", "1b", "1.4b", "2.8b", "6.9b")
MB_SEEDS = (1, 2, 3, 4)


def closing_hinge(steps: List[int], y: List[float]) -> Dict:
    x = np.log10(np.array(steps, dtype=float) + 1.0)
    ya = np.array(y, dtype=float)
    n = len(ya)
    rss1 = fit_one_segment(x, ya)
    best = None
    for bi in range(1, n - 1):
        rss2 = fit_two_segment(x, ya, bi)
        if best is None or rss2 < best[1]:
            best = (bi, rss2)
    bi, rss2 = best
    delta = bic(rss1, n, 2) - bic(rss2, n, 4)
    xb = x[bi]
    A = np.vstack([x, np.maximum(x - xb, 0.0), np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, ya, rcond=None)
    slope1 = float(coef[0])
    slope2 = float(coef[0] + coef[1])
    closing = bool(delta >= 2.0 and slope2 < slope1 and slope2 < 0.0)
    return {"delta_bic": round(float(delta), 3),
            "hinge_step": int(steps[bi]), "hinge_index": bi,
            "slope_pre": round(slope1, 4),
            "slope_post": round(slope2, 4),
            "closing_verdict": closing}


def adjudicate(steps: List[int], y: List[float],
               thin_factors: List[int]) -> Dict:
    full = closing_hinge(steps, y)
    variants = {}
    persistent = full["closing_verdict"]
    for f in thin_factors:
        for parity in range(f):
            s2 = steps[parity::f]
            y2 = y[parity::f]
            if len(s2) < 5:
                continue
            v = closing_hinge(s2, y2)
            i = full["hinge_index"]
            lo = steps[max(i - f, 0)]
            hi = steps[min(i + f, len(steps) - 1)]
            v["hinge_ok"] = bool(lo <= v["hinge_step"] <= hi)
            variants[f"thin{f}_p{parity}"] = v
            persistent = persistent and v["closing_verdict"] \
                and v["hinge_ok"]
    return {"full": full, "variants": variants,
            "persistent_closing": bool(persistent)}


def load_series(path: Path) -> Dict[str, Dict[str, List]]:
    byrun = defaultdict(lambda: {"steps": [], "y": []})
    for row in csv.DictReader(open(path)):
        byrun[row["run"]]["steps"].append(int(row["epoch"]))
        byrun[row["run"]]["y"].append(float(row["test_entropy_bits"]))
    return dict(byrun)


def main() -> None:
    results: Dict = {}
    control_ok = True
    probe_persistent = []

    for size in PYTHIA:
        f = OUTPUTS / f"pythia_collapse_timeseries_{size}.csv"
        series = load_series(f)
        block = {}
        for run, d in series.items():
            adj = adjudicate(d["steps"], d["y"], [2])
            block[run] = adj
            if run == "pythia_agreement":
                probe_persistent.append((f"pythia_{size}",
                                         adj["persistent_closing"]))
            else:
                control_ok = control_ok and \
                    not adj["persistent_closing"]
        results[f"pythia_{size}"] = block

    for seed in MB_SEEDS:
        f = OUTPUTS / f"multiberts_collapse_timeseries_seed{seed}.csv"
        series = load_series(f)
        block = {}
        for run, d in series.items():
            adj = adjudicate(d["steps"], d["y"], [2, 4])
            block[run] = adj
            if run == "multiberts_agreement":
                probe_persistent.append((f"multiberts_seed{seed}",
                                         adj["persistent_closing"]))
            else:
                control_ok = control_ok and \
                    not adj["persistent_closing"]
        results[f"multiberts_seed{seed}"] = block

    outcomes = {
        "RE3_1_probe_persistent_closing": dict(probe_persistent),
        "RE3_1_n_persistent": sum(p for _, p in probe_persistent),
        "RE3_1_n_series": len(probe_persistent),
        "RE3_2_controls_null": bool(control_ok),
    }
    report = {
        "status": ("RE-3 stored-series breakpoint adjudication; "
                   "registered RE battery with stored-data "
                   "disclosure; thresholds frozen before run"),
        "results": results,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "re3_stored_series_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
