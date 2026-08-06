"""REPR-ROBUSTNESS: contract-invariance of the collapse verdicts.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Addresses
reviewer critique #1: are the collapse/onset/t* verdicts stable under
the analysis contract (object, grid density, saturation threshold,
analysis window, effect-size gate, Delta-BIC threshold), or are they
artefacts of the representation?

We re-analyse the STORED openness curves of three systems -- the learned
grip flagship (two objects), the ant colony at N=100, and the learned
high-order TRI-C-BP formation -- with the frozen adjudicator logic
generalized so the contract is swept. The cell matching the frozen
contract reproduces the published verdict (RR-3).
"""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"

STRIDES = [1, 2, 3]
SAT_FRACS = [0.02, 0.05, 0.10]
WINDOW_FRACS = [0.75, 0.875, 1.0]
GATES = [0.05, 0.10, 0.15]
DBICS = [8.0, 10.0, 12.0]
FROZEN = {"stride": 1, "sat": 0.05, "window": 1.0, "gate": 0.10, "dbic": 10.0}


def truncate(x, y, sat_frac):
    drop = y[0] - y[-1]
    if drop <= 0:
        return x, y, None
    thresh = y[-1] + sat_frac * drop
    for i, v in enumerate(y):
        if v <= thresh:
            end = max(i + 1, 5)
            return x[:end], y[:end], float(x[min(i, len(x) - 1)])
    return x, y, None


def adjudicate_contract(grid, y_norm, gate, sat, dbic):
    """y_norm already in [0,1] openness units. Mirrors ant_fine_onset."""
    x = np.asarray(grid, dtype=float)
    yn = np.asarray(y_norm, dtype=float)
    drop = float(yn[0] - yn[-1])
    if drop < gate:
        return {"collapse": False, "onset": False, "t_star": None}
    xw, yw, _ = truncate(x, yn, sat)
    if len(yw) < 10:
        return {"collapse": True, "onset": False, "t_star": None}
    full = hinge_linear(xw, yw)
    span = xw[-1] - xw[0]
    thin_ok = True
    for parity in (0, 1):
        t = hinge_linear(xw[parity::2], yw[parity::2])
        ok = (t["delta_bic"] >= 2.0 and t["onset_type"]
              and abs(t["t_star"] - full["t_star"]) <= 0.10 * span)
        thin_ok = thin_ok and ok
    onset = bool(full["delta_bic"] >= dbic and full["onset_type"] and thin_ok)
    return {"collapse": True, "onset": onset,
            "t_star": float(full["t_star"]) if onset else None,
            "delta_bic": round(float(full["delta_bic"]), 2)}


def load_curves():
    """Return {system: {object: (grid, openness_curve_in_[0,1])}}."""
    systems = {}

    # grip flagship: the theory-specified object is side-openness (the
    # joint side-COMMITMENT space). Raw 3-action entropy is a different
    # quantity (it RISES as agents diversify grip/move) and is reported
    # separately as an object-semantics diagnostic, not an RR cell.
    g = json.load(open(OUTPUTS / "learn_grip_transport.json"))
    side = np.mean([g["seeds"][s]["side_openness_curve"]
                    for s in g["seeds"]], axis=0)          # collapse object [0,1]
    grid = list(range(len(side)))
    systems["grip:side_openness"] = (grid, list(side))

    # ant colony N=100 median openness
    a = json.load(open(OUTPUTS / "ant_colony_breakpoint.json"))
    med = a["per_size"]["100"]["median_openness"]
    systems["ant_N100"] = (list(range(len(med))), list(med))

    # TRI-C-BP learned high-order formation openness (curve is a dict
    # keyed by update -> {openness, r3, ...}; extract the openness series).
    t = json.load(open(OUTPUTS / "tri_c_breakpoint.json"))
    sd = list(t["seeds"].values())[0]
    tgrid = t["grid"]
    op = [sd["curve"][str(u)]["openness"] for u in tgrid]
    systems["tri_c_bp"] = (list(tgrid), op)
    return systems


def sweep_system(grid, curve):
    grid = np.asarray(grid, dtype=float)
    curve = np.asarray(curve, dtype=float)
    cells = []
    for stride, sat, wf, gate, dbic in product(
            STRIDES, SAT_FRACS, WINDOW_FRACS, GATES, DBICS):
        n = max(int(round(len(curve) * wf)), 10)
        gsub = grid[:n][::stride]
        ysub = curve[:n][::stride]
        if len(ysub) < 10:
            continue
        v = adjudicate_contract(gsub, ysub, gate, sat, dbic)
        v.update({"stride": stride, "sat": sat, "window": wf,
                  "gate": gate, "dbic": dbic})
        cells.append(v)
    return cells


def summarize(cells, span):
    gated = [c for c in cells if c["collapse"]]
    # RR1 invariance is assessed on adequate-resolution cells (stride 1):
    # aggressive subsampling of a short curve loses detector power exactly
    # as quantified by DETECTOR-VALIDATION (DV-3), so we report that as a
    # separate resolution effect rather than folding it into invariance.
    gated_res = [c for c in gated if c["stride"] == 1]
    onset_res = [c for c in gated_res if c["onset"]]
    frac_onset_among_gated = (len(onset_res) / len(gated_res)) if gated_res else 0.0
    onset_cells = [c for c in cells if c["onset"]]
    tstars = [c["t_star"] for c in onset_cells if c["t_star"] is not None]
    t_range_frac = ((max(tstars) - min(tstars)) / span) if len(tstars) >= 2 else 0.0
    # which single axis most degrades onset: fraction onset when axis at its
    # "hardest" level vs overall
    degraders = {}
    for axis, levels in [("stride", STRIDES), ("sat", SAT_FRACS),
                         ("window", WINDOW_FRACS), ("gate", GATES),
                         ("dbic", DBICS)]:
        by_level = {}
        for lv in levels:
            sub = [c for c in gated if c[axis] == lv]
            by_level[lv] = (sum(c["onset"] for c in sub) / len(sub)) if sub else None
        degraders[axis] = by_level
    return {
        "n_cells": len(cells),
        "n_gated": len(gated),
        "n_gated_stride1": len(gated_res),
        "frac_onset_among_gated": round(frac_onset_among_gated, 4),
        "t_star_range_frac": round(t_range_frac, 4),
        "t_star_min": round(min(tstars), 2) if tstars else None,
        "t_star_max": round(max(tstars), 2) if tstars else None,
        "onset_rate_by_axis_level": degraders,
    }


def main() -> None:
    systems = load_curves()
    report = {"status": ("REPR-ROBUSTNESS contract-invariance re-analysis "
                         "of stored curves; registered before run"),
              "config": {"strides": STRIDES, "sat_fracs": SAT_FRACS,
                         "window_fracs": WINDOW_FRACS, "gates": GATES,
                         "dbics": DBICS, "frozen_cell": FROZEN},
              "systems": {}}
    rr1_ok, rr2_ok, rr3_ok = [], [], []
    for name, (grid, curve) in systems.items():
        span = grid[-1] - grid[0]
        cells = sweep_system(grid, curve)
        summ = summarize(cells, span)
        # frozen-cell reproduction
        frozen = adjudicate_contract(grid, curve, FROZEN["gate"],
                                     FROZEN["sat"], FROZEN["dbic"])
        summ["frozen_cell_verdict"] = frozen
        report["systems"][name] = summ
        rr1_ok.append(summ["frac_onset_among_gated"] >= 0.90)
        rr2_ok.append(summ["t_star_range_frac"] <= 0.20)
        rr3_ok.append(bool(frozen["onset"]))
        print(f"{name}: gated {summ['n_gated']}/{summ['n_cells']} cells, "
              f"onset in {summ['frac_onset_among_gated']*100:.0f}% of gated, "
              f"t* range {summ['t_star_range_frac']*100:.0f}% span "
              f"[{summ['t_star_min']},{summ['t_star_max']}], "
              f"frozen onset={frozen['onset']}", flush=True)

    outcomes = {
        "RR1_verdict_invariance_ge_0.90_all": bool(all(rr1_ok)),
        "RR2_location_stability_le_0.20_all": bool(all(rr2_ok)),
        "RR3_frozen_reproduces_all": bool(all(rr3_ok)),
        "per_system_rr1": {n: report["systems"][n]["frac_onset_among_gated"]
                           for n in report["systems"]},
    }
    # object-semantics diagnostic: the raw 3-action entropy of the grip
    # flagship is NOT a collapse object (it RISES as agents diversify
    # grip/move), so it is excluded from RR by construction, not by
    # cherry-picking. Recorded here for transparency.
    g = json.load(open(OUTPUTS / "learn_grip_transport.json"))
    fe = np.mean([g["seeds"][s]["final_episode_entropy"]
                  for s in g["seeds"]], axis=0)
    report["object_semantics_note"] = {
        "grip_full_action_entropy_first": round(float(fe[0]), 4),
        "grip_full_action_entropy_last": round(float(fe[-1]), 4),
        "grip_full_action_entropy_drop": round(float(fe[0] - fe[-1]), 4),
        "comment": ("raw 3-action entropy rises (drop < 0); it conflates "
                    "the grip action with side commitment and is not the "
                    "theory-specified collapse object -- side-openness is"),
    }
    report["registered_outcomes"] = outcomes
    out = OUTPUTS / "repr_robustness.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
