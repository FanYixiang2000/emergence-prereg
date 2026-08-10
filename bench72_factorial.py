"""BENCH-72: full-factorial analytic ground-truth battery.

Registered in V2_ALIGNMENT_PREREGISTRATION.md, Wave 4 (frozen
2026-07-23, pre-run shape amendment recorded before any run).

72 mechanism cells = 4 sources x 3 temporal shapes x 2 stability x
3 values, plus 5 declared pseudo-controls. Generators reuse the exact
SD-battery model (3 agents, 10 actions, binary E). The instrument half
is blind: it sees only the per-stage distributions and the declared Z
channel, never the knobs, and must recover source, M, J, t*, rho, V.

Everything is exact enumeration; there is no sampling noise.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from collapse_source_decomposition import (BASE, HIGH, KNOB_TO_COMPONENT,
                                           KNOBS, joint_with_env, ladder)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

N_STAGES = 25
FORM_END = 18          # formation completes here (x = 1)
REVERT_FROM = 19       # transient cells revert to BASE from this stage
STEP_X = 0.6           # punctuated step location on x
TRUE_TSTAR_PUNCT = 11  # first stage with min(s/18,1) >= 0.6
H_Q0 = 3 * math.log2(10)

SOURCES = tuple(KNOBS)  # lambda_ind, rho_env, kappa_pair, gamma_high
SHAPES = ("gradual", "sigmoid", "punctuated")
STABILITIES = ("persistent", "transient")
VALUES = (1, 0, -1)

COMPONENTS = ("C_individual", "C_env", "C_pair", "C_high")


def shape_f(shape: str, x: float) -> float:
    if shape == "gradual":
        return x
    if shape == "sigmoid":
        raw = lambda t: 1.0 / (1.0 + math.exp(-10.0 * (t - 0.5)))
        return (raw(x) - raw(0.0)) / (raw(1.0) - raw(0.0))
    if shape == "punctuated":
        return 0.0 if x < STEP_X else 1.0
    raise ValueError(shape)


def knob_schedule(source: str, shape: str, stability: str) -> List[Dict]:
    """Per-stage knob dict (the generative, non-blind half)."""
    stages = []
    for s in range(N_STAGES):
        knobs = {k: BASE for k in KNOBS}
        if stability == "transient" and s >= REVERT_FROM:
            f = 0.0
        else:
            f = shape_f(shape, min(s / FORM_END, 1.0))
        knobs[source] = BASE + (HIGH - BASE) * f
        stages.append(knobs)
    return stages


_LADDER_CACHE: Dict[tuple, Dict] = {}


def components_for(knobs: Dict[str, float]) -> Dict[str, float]:
    key = tuple(round(knobs[k], 9) for k in KNOBS)
    if key not in _LADDER_CACHE:
        pe = joint_with_env(knobs["lambda_ind"], knobs["rho_env"],
                            knobs["kappa_pair"], knobs["gamma_high"])
        _LADDER_CACHE[key] = ladder(pe)["components"]
    return _LADDER_CACHE[key]


# blind instrument

def measure(comp_curve: List[Dict[str, float]],
            z_curve: List[float]) -> Dict:
    """Blind recovery from the stage-wise component curves + Z only."""
    d = {c: np.array([row[c] - comp_curve[0][c] for row in comp_curve])
         for c in COMPONENTS}
    dtot = np.array([row["C_total"] - comp_curve[0]["C_total"]
                     for row in comp_curve])
    source_est = max(COMPONENTS, key=lambda c: float(d[c].max()))
    peak = int(np.argmax(dtot))
    m_est = float(dtot[peak]) / H_Q0
    deltas = np.diff(dtot[: peak + 1]) if peak >= 1 else np.array([0.0])
    pos = deltas[deltas > 0]
    j_est = float(pos.max() / pos.sum()) if pos.sum() > 0 else 0.0
    tstar_est = (int(np.argmax(deltas)) + 1) if peak >= 1 else 0
    persistent_est = bool(dtot[-1] >= 0.5 * dtot[peak]) \
        if dtot[peak] > 0 else True
    dz = z_curve[peak] - z_curve[0]
    v_est = 0 if abs(dz) < 0.05 else (1 if dz > 0 else -1)
    return {"source_est": source_est, "M_est": m_est, "J_est": j_est,
            "tstar_est": tstar_est, "persistent_est": persistent_est,
            "V_est": v_est, "peak_stage": peak,
            "dtot_final": float(dtot[-1]), "dtot_peak": float(dtot[peak])}


def run_cell(source: str, shape: str, stability: str, value: int) -> Dict:
    sched = knob_schedule(source, shape, stability)
    comp_curve = [components_for(k) for k in sched]
    z_curve = [0.5 + value * 0.4 * shape_f(shape, min(s / FORM_END, 1.0))
               for s in range(N_STAGES)]
    est = measure(comp_curve, z_curve)
    truth = {"source": source, "shape": shape, "stability": stability,
             "value": value,
             "tstar_true": TRUE_TSTAR_PUNCT if shape == "punctuated"
             else None}
    return {"truth": truth, "estimate": est}


# pseudo-controls

def truncated_pe(knobs: Dict[str, float], keep: int = 2) -> Dict:
    pe = joint_with_env(knobs["lambda_ind"], knobs["rho_env"],
                        knobs["kappa_pair"], knobs["gamma_high"])
    out = {}
    for e, p in pe.items():
        q = np.zeros_like(p)
        q[:keep, :keep, :keep] = p[:keep, :keep, :keep]
        out[e] = q / q.sum()
    return out


def point_pe() -> Dict:
    p = np.zeros((10, 10, 10))
    p[0, 0, 0] = 1.0
    return {0: p, 1: p.copy()}


def run_controls() -> Dict:
    base_knobs = {k: BASE for k in KNOBS}
    base_comp = components_for(base_knobs)
    controls = {}

    def curve_from(pe_fn_by_stage) -> List[Dict[str, float]]:
        rows = []
        for s in range(N_STAGES):
            pe = pe_fn_by_stage(s)
            rows.append(base_comp if pe is None
                        else ladder(pe)["components"])
        return rows

    # 1) external action mask at stage 15 (collapse real, B3 flag)
    cc = curve_from(lambda s: truncated_pe(base_knobs) if s >= 15
                    else None)
    controls["external_mask"] = {
        "external_by_construction": True,
        "estimate": measure(cc, [0.5] * N_STAGES)}
    # 2) external policy overwrite at stage 15
    cc = curve_from(lambda s: point_pe() if s >= 15 else None)
    controls["external_overwrite"] = {
        "external_by_construction": True,
        "estimate": measure(cc, [0.5] * N_STAGES)}
    # 3) revelation-only: flat distributions, Z jumps at 15
    cc = curve_from(lambda s: None)
    z = [0.5 if s < 15 else 0.9 for s in range(N_STAGES)]
    controls["revelation_only"] = {"estimate": measure(cc, z)}
    # 4) metric artifact: flat distributions, nonlinear metric jumps
    metric = [1.0 / (1.0 + math.exp(-20.0 * (min(s / FORM_END, 1.0)
                                             - 0.6)))
              for s in range(N_STAGES)]
    controls["metric_artifact"] = {"metric_curve": [round(m, 4)
                                                    for m in metric],
                                   "estimate": measure(cc,
                                                       [0.5] * N_STAGES)}
    # 5) transient sync: kappa spike stages 12-14 only
    def sync(s):
        if 12 <= s <= 14:
            k = dict(base_knobs)
            k["kappa_pair"] = HIGH
            pe = joint_with_env(k["lambda_ind"], k["rho_env"],
                                k["kappa_pair"], k["gamma_high"])
            return pe
        return None
    cc = curve_from(sync)
    controls["transient_sync"] = {"estimate": measure(cc,
                                                      [0.5] * N_STAGES)}
    return controls


# checks

def evaluate(cells: List[Dict], controls: Dict) -> Dict:
    n_src_ok = sum(1 for c in cells
                   if KNOB_TO_COMPONENT[c["truth"]["source"]]
                   == c["estimate"]["source_est"])
    b1 = {"accuracy": n_src_ok / len(cells), "pass": n_src_ok
          >= 0.9 * len(cells)}

    groups: Dict[tuple, Dict[str, Dict]] = {}
    for c in cells:
        t = c["truth"]
        groups.setdefault((t["source"], t["stability"], t["value"]),
                          {})[t["shape"]] = c["estimate"]
    m_ok = j_ok = 0
    group_detail = []
    for key, g in groups.items():
        ms = [g[s]["M_est"] for s in SHAPES]
        rel_range = ((max(ms) - min(ms)) / np.mean(ms)
                     if np.mean(ms) > 0 else 0.0)
        m_pass = rel_range < 0.2
        j_pass = (g["punctuated"]["J_est"] > g["sigmoid"]["J_est"]
                  > g["gradual"]["J_est"])
        m_ok += m_pass
        j_ok += j_pass
        group_detail.append({"group": list(key),
                             "M_rel_range": round(float(rel_range), 4),
                             "J": {s: round(g[s]["J_est"], 4)
                                   for s in SHAPES},
                             "M_pass": bool(m_pass),
                             "J_pass": bool(j_pass)})
    b2 = {"M_groups_ok": m_ok, "J_groups_ok": j_ok, "n_groups":
          len(groups),
          "pass": m_ok == len(groups) and j_ok >= 0.9 * len(groups)}

    punct = [c for c in cells if c["truth"]["shape"] == "punctuated"]
    errs = [abs(c["estimate"]["tstar_est"] - c["truth"]["tstar_true"])
            for c in punct]
    b3 = {"max_error_stages": max(errs), "pass": max(errs) <= 2}

    rho_ok = sum(1 for c in cells
                 if c["estimate"]["persistent_est"]
                 == (c["truth"]["stability"] == "persistent"))
    b4 = {"accuracy": rho_ok / len(cells), "pass": rho_ok == len(cells)}

    nonneutral = [c for c in cells if c["truth"]["value"] != 0]
    v_ok = sum(1 for c in nonneutral
               if c["estimate"]["V_est"] == c["truth"]["value"])
    v_zero_ok = sum(1 for c in cells if c["truth"]["value"] == 0
                    and c["estimate"]["V_est"] == 0)
    b5 = {"nonneutral_ok": v_ok, "n_nonneutral": len(nonneutral),
          "neutral_ok": v_zero_ok,
          "pass": v_ok == len(nonneutral)}

    rev = controls["revelation_only"]["estimate"]
    met = controls["metric_artifact"]["estimate"]
    syn = controls["transient_sync"]["estimate"]
    b6 = {"revelation_peak_norm": rev["dtot_peak"] / H_Q0,
          "metric_peak_norm": met["dtot_peak"] / H_Q0,
          "transient_sync_persistent_est": syn["persistent_est"],
          "pass": (rev["dtot_peak"] / H_Q0 < 0.02
                   and met["dtot_peak"] / H_Q0 < 0.02
                   and not syn["persistent_est"])}

    return {"B72_1_source": b1, "B72_2_M_vs_B": b2,
            "B72_3_tstar": b3, "B72_4_persistence": b4,
            "B72_5_value": b5, "B72_6_controls": b6,
            "group_detail": group_detail}


def main() -> None:
    cells = [run_cell(src, shp, stab, val)
             for src, shp, stab, val in itertools.product(
                 SOURCES, SHAPES, STABILITIES, VALUES)]
    controls = run_controls()
    checks = evaluate(cells, controls)
    outcomes = {k: v["pass"] for k, v in checks.items()
                if k.startswith("B72")}
    report = {
        "status": ("BENCH-72 full-factorial analytic ground truth; "
                   "registered Wave 4 in V2_ALIGNMENT_PREREGISTRATION.md "
                   "with pre-run shape amendment; blind instrument "
                   "recovery of source/M/J/t*/rho/V plus 5 declared "
                   "pseudo-controls; exact enumeration, no sampling"),
        "n_cells": len(cells),
        "registered_outcomes": outcomes,
        "checks": checks,
        "controls": controls,
        "cells": cells,
    }
    out = OUTPUTS / "bench72_factorial.json"
    def np_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))

    out.write_text(json.dumps(report, indent=2, default=np_default),
                   encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
