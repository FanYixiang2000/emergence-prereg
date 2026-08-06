"""DEF-CAL: surprise/spontaneity/regime-formation calibration.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Tests
the lottery objection: low probability alone is not emergence; a new
endogenous persistent macro-regime must reorganize future possibilities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
GRID = list(range(0, 121, 5))
ALIGN_GRID = list(range(-20, 41, 5))
N_EP = 80
N_CONT = 200
HORIZON = 200
SEED = 93_001


def h2(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def js_bernoulli(p: float, q: float) -> float:
    m = 0.5 * (p + q)
    return h2(m) - 0.5 * h2(p) - 0.5 * h2(q)


def simulate_path(kind: str, seed: int):
    rng = np.random.default_rng(seed)
    states = {}
    z = 0
    m = 0
    crossed: Optional[int] = None
    for t in range(HORIZON + 1):
        if t in GRID:
            states[t] = (t, z, m, crossed)
        if t == HORIZON:
            break
        if kind == "lottery":
            if t == 50 and rng.random() < 0.01:
                z = 1
            # z marks a past rare sample; it does not affect future draws.
        elif kind == "mask":
            if t >= 50:
                z = 1
        elif kind == "random_mask":
            if t == 50 and rng.random() < 0.01:
                z = 1
        elif kind == "nucleation":
            if z == 1:
                m = min(50, m + rng.binomial(50 - m, 0.15))
            else:
                if m >= 8:
                    z = 1
                    crossed = t if crossed is None else crossed
                elif rng.random() < 0.0005:
                    # Rare internal critical nucleus. The rare fluctuation
                    # is the seed; the autocatalytic absorbing growth after
                    # it is the candidate emergence event.
                    m = 8
                    crossed = t
                else:
                    m = 0
        elif kind == "smooth":
            if z == 1:
                m = min(50, m + rng.binomial(50 - m, 0.03))
            else:
                births = rng.binomial(50 - m, 0.01)
                deaths = rng.binomial(m, 0.004)
                m = min(50, max(0, m + births - deaths))
                if m >= 40:
                    z = 1
                    crossed = t if crossed is None else crossed
        else:
            raise ValueError(kind)
    return states


def continue_prob(kind: str, state, seed: int, perturb: bool = False) -> int:
    rng = np.random.default_rng(seed)
    start_t, z, m, crossed = state
    if perturb and z == 1 and kind not in {"mask", "random_mask"}:
        m = int(round(0.8 * m))
        z = 1 if m >= 8 else 0
    for t in range(start_t, HORIZON):
        if kind == "lottery":
            # Future rare draw remains p=0.01 at its own t=50 analog.
            if t == 50 and rng.random() < 0.01:
                z = 1
        elif kind == "mask":
            if t >= 50:
                z = 1
        elif kind == "random_mask":
            if t == 50 and rng.random() < 0.01:
                z = 1
        elif kind == "nucleation":
            if z == 1:
                m = min(50, m + rng.binomial(50 - m, 0.15))
            else:
                if m >= 8:
                    z = 1
                elif rng.random() < 0.0005:
                    m = 8
                else:
                    m = 0
        elif kind == "smooth":
            if z == 1:
                m = min(50, m + rng.binomial(50 - m, 0.03))
            else:
                births = rng.binomial(50 - m, 0.01)
                deaths = rng.binomial(m, 0.004)
                m = min(50, max(0, m + births - deaths))
                if m >= 40:
                    z = 1
        else:
            raise ValueError(kind)
    return int(z == 1)


def estimate_p(kind: str, state, seed: int, perturb: bool = False) -> float:
    vals = [
        continue_prob(kind, state, seed + i * 17, perturb=perturb)
        for i in range(N_CONT)
    ]
    return float(np.mean(vals))


def hinge_on_openness(median_o):
    x = np.array(GRID, dtype=float)
    y = np.array(median_o)
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= 0.1)}
    if not out["gate_passed"]:
        out["b5_onset"] = False
        return out
    full = hinge_linear(x, y)
    out["hinge"] = full
    out["b5_onset"] = bool(full["delta_bic"] >= 10 and full["onset_type"])
    return out


def run_kind(kind: str, g_spontaneous: int):
    per_ep = []
    for ep in range(N_EP):
        states = simulate_path(kind, SEED + ep * 101)
        probs = {}
        for t in GRID:
            probs[t] = estimate_p(kind, states[t], SEED + ep * 10_000 + t)
        per_ep.append({"states": states, "p": probs})

    med_p = [float(np.median([ep["p"][t] for ep in per_ep])) for t in GRID]
    med_o = [h2(p) for p in med_p]
    h = hinge_on_openness(med_o)

    p0 = med_p[0]
    # decisive state: max median p increase point for ordinary curves.
    diffs = np.diff(med_p)
    idx = int(np.argmax(diffs)) + 1 if len(diffs) else 0
    t_seed = GRID[idx]
    p_seed = med_p[idx]
    if kind == "nucleation":
        # For rare endogenous emergence, the decisive state is the first
        # threshold-crossed state. This distinguishes a rare seed from the
        # generated regime it creates.
        crossed = []
        for ep in per_ep:
            for t in GRID:
                state = ep["states"][t]
                if state[1] == 1:
                    crossed.append((t, ep["p"][t]))
                    break
        if crossed:
            t_seed = int(np.median([c[0] for c in crossed]))
            p_seed = float(np.median([c[1] for c in crossed]))
    s_prior = -math.log2(max(p0, 1e-12))
    d_regime = js_bernoulli(p0, p_seed)
    x_explain = math.log2(max(p_seed, 1e-12) / max(p0, 1e-12))

    if kind == "random_mask":
        event_ps = []
        event_ts = []
        for ep in per_ep:
            for t in GRID:
                state = ep["states"][t]
                if state[1] == 1:
                    event_ts.append(t)
                    event_ps.append(ep["p"][t])
                    break
        if event_ps:
            t_seed = int(np.median(event_ts))
            p_seed = float(np.median(event_ps))
            d_regime = js_bernoulli(p0, p_seed)
            x_explain = math.log2(max(p_seed, 1e-12) / max(p0, 1e-12))

    # Persistence estimated from states whose future is already committed.
    persist_states = [ep["states"][GRID[-1]] for ep in per_ep
                      if ep["p"][GRID[-1]] > 0.9]
    if persist_states:
        r_vals = [
            estimate_p(kind, st, SEED + 900_000 + i * 1000, perturb=True)
            for i, st in enumerate(persist_states[:20])
        ]
        r_persist = float(np.mean(r_vals))
    else:
        r_persist = 0.0

    aligned = None
    if kind in {"nucleation", "random_mask"}:
        aligned_ps = {dt: [] for dt in ALIGN_GRID}
        for ep in per_ep:
            event_t = None
            for t in GRID:
                if ep["states"][t][1] == 1:
                    event_t = t
                    break
            if event_t is None:
                continue
            for dt in ALIGN_GRID:
                t = event_t + dt
                if t in ep["p"]:
                    aligned_ps[dt].append(ep["p"][t])
        if any(aligned_ps[dt] for dt in ALIGN_GRID):
            aligned_p = [
                float(np.median(aligned_ps[dt])) if aligned_ps[dt] else float("nan")
                for dt in ALIGN_GRID
            ]
            valid = [(dt, p) for dt, p in zip(ALIGN_GRID, aligned_p) if not np.isnan(p)]
            aligned_o = [h2(p) for _, p in valid]
            aligned_h = {"insufficient": True}
            if len(aligned_o) >= 8:
                x = np.array([dt for dt, _ in valid], dtype=float)
                y = np.array(aligned_o)
                drop = float(np.nanmax(y) - y[-1])
                aligned_h = {"drop_from_peak": round(drop, 4),
                             "gate_passed": bool(drop >= 0.1)}
                if aligned_h["gate_passed"]:
                    fit = hinge_linear(x, y)
                    aligned_h["hinge"] = fit
                    aligned_h["b5_onset"] = bool(
                        fit["delta_bic"] >= 10 and fit["onset_type"]
                    )
                else:
                    aligned_h["b5_onset"] = False
            aligned = {
                "grid": [dt for dt, _ in valid],
                "median_p": [round(p, 5) for _, p in valid],
                "openness": [round(v, 5) for v in aligned_o],
                "hinge": aligned_h,
            }

    qualifies = bool(d_regime > 0.05 and g_spontaneous == 1
                     and r_persist > 0.8)
    return {
        "median_p_final": [round(v, 5) for v in med_p],
        "median_openness": [round(v, 5) for v in med_o],
        "hinge": h,
        "t_seed_proxy": t_seed,
        "S_prior_bits": round(s_prior, 4),
        "D_regime_js_bits": round(d_regime, 4),
        "X_explain_bits": round(x_explain, 4),
        "R_persist": round(r_persist, 4),
        "G_spontaneous": g_spontaneous,
        "qualifies_DGR": qualifies,
        "event_aligned": aligned,
    }


def main() -> None:
    configs = {
        "LOTTERY": 0,
        "MASK": 0,
        "RANDOM_MASK": 0,
        "NUCLEATION": 1,
        "SMOOTH": 1,
    }
    rows = {}
    for kind, g in configs.items():
        rows[kind] = run_kind(kind.lower(), g)
        row = rows[kind]
        print(f"{kind}: S={row['S_prior_bits']} "
              f"D={row['D_regime_js_bits']} X={row['X_explain_bits']} "
              f"R={row['R_persist']} G={g} "
              f"B5={row['hinge'].get('b5_onset')} "
              f"qualifies={row['qualifies_DGR']}", flush=True)

    outcomes = {
        "DC1_lottery_excluded": bool(
            rows["LOTTERY"]["S_prior_bits"] > 3
            and rows["LOTTERY"]["D_regime_js_bits"] < 0.05
            and rows["LOTTERY"]["R_persist"] < 0.2
            and not rows["LOTTERY"]["qualifies_DGR"]
        ),
        "DC2_mask_excluded": bool(
            rows["RANDOM_MASK"]["D_regime_js_bits"] > 0.3
            and rows["RANDOM_MASK"]["R_persist"] > 0.8
            and rows["RANDOM_MASK"]["G_spontaneous"] == 0
            and not rows["RANDOM_MASK"]["qualifies_DGR"]
        ),
        "DC3_nucleation_accepted": bool(
            rows["NUCLEATION"]["S_prior_bits"] > 3
            and rows["NUCLEATION"]["D_regime_js_bits"] > 0.3
            and rows["NUCLEATION"]["X_explain_bits"] > 3
            and rows["NUCLEATION"]["R_persist"] > 0.8
            and rows["NUCLEATION"]["G_spontaneous"] == 1
            and rows["NUCLEATION"]["event_aligned"]["hinge"].get("b5_onset")
            and rows["NUCLEATION"]["qualifies_DGR"]
        ),
        "DC4_smooth_weak_gradual": bool(
            rows["SMOOTH"]["G_spontaneous"] == 1
            and not rows["SMOOTH"]["hinge"].get("b5_onset")
        ),
    }
    report = {
        "status": "DEF-CAL surprise/spontaneity/regime calibration; preregistered",
        "grid": GRID,
        "systems": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "definition_calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
