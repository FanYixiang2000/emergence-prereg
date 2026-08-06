"""CEB-TURING-R: corrected Brusselator Turing battery.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Uses
A=1, B=1.8 so the homogeneous system is stable without diffusion
contrast and Turing-unstable for Du=1,Dv=10.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ceb_turing import (DT, DU, DV_STABLE, M, N_REP, NOISE, OUTPUTS,
                        SAVE_EVERY, SEED, T_STEPS, adjudicate, laplacian,
                        spectral_stats)

A = 1.0
B = 1.8
DV_PATTERN = 10.0
DV_STABLE_R = 1.0
GRID = list(range(0, T_STEPS + 1, SAVE_EVERY))


def simulate(dv: float, seed: int):
    rng = np.random.default_rng(seed)
    u0 = A
    v0 = B / A
    u = u0 + NOISE * rng.standard_normal((N_REP, M))
    v = v0 + NOISE * rng.standard_normal((N_REP, M))
    openness, native_power, top3 = [], [], []
    for step in range(T_STEPS + 1):
        if step % SAVE_EVERY == 0:
            o, p, t3 = spectral_stats(u)
            openness.append(o)
            native_power.append(p)
            top3.append(t3)
        if step == T_STEPS:
            break
        uvv = (u ** 2) * v
        du = A - (B + 1) * u + uvv + DU * laplacian(u)
        dvdt = B * u - uvv + dv * laplacian(v)
        u = np.clip(u + DT * du, 0.0, None)
        v = np.clip(v + DT * dvdt, 0.0, None)
    return {
        "openness": openness,
        "native_power": native_power,
        "top3_mass": top3,
        "openness_first_last": [round(openness[0], 4),
                                round(openness[-1], 4)],
        "power_first_last": [float(native_power[0]),
                             float(native_power[-1])],
        "top3_first_last": [round(top3[0], 4), round(top3[-1], 4)],
    }


def main() -> None:
    rows = {}
    for name, dv in (("pattern", DV_PATTERN), ("stable", DV_STABLE_R)):
        row = simulate(dv, SEED + 500 + len(rows) * 101)
        adj = adjudicate(row["openness"])
        row["adj"] = adj
        rows[name] = row
        h = adj.get("hinge", {})
        print(f"{name}: O {row['openness_first_last']} "
              f"power {row['power_first_last']} "
              f"top3 {row['top3_first_last']} "
              f"b5={adj.get('b5_onset')} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')}", flush=True)

    pat = rows["pattern"]
    st = rows["stable"]
    pat_growth = pat["native_power"][-1] / max(pat["native_power"][0], 1e-18)
    st_growth = st["native_power"][-1] / max(st["native_power"][0], 1e-18)
    h = pat["adj"].get("hinge", {})
    p_final = pat["native_power"][-1]
    t_power80 = next((GRID[i] for i, v in enumerate(pat["native_power"])
                      if p_final > 0 and v >= 0.8 * p_final), None)
    turr1 = bool(pat["top3_mass"][-1] >= 0.60
                 and pat["top3_mass"][0] <= 0.35
                 and pat_growth >= 100)
    turr2 = bool(st_growth < 10 and st["top3_mass"][-1] < 0.60)
    turr3 = bool(pat["adj"].get("b5_onset")
                 and t_power80 is not None
                 and h["t_star"] < t_power80)
    outcomes = {
        "TURR1_mode_selective_pattern": turr1,
        "TURR2_stable_control": turr2,
        "TURR3_temporal_B5_if_present": turr3,
        "pattern_power_growth": float(pat_growth),
        "stable_power_growth": float(st_growth),
        "pattern_t_power80": t_power80,
    }
    report = {
        "status": "CEB-TURING-R corrected Turing battery; preregistered",
        "config": {"A": A, "B": B, "Du": DU, "Dv_pattern": DV_PATTERN,
                   "Dv_stable": DV_STABLE_R, "M": M, "dt": DT,
                   "steps": T_STEPS, "n_rep": N_REP},
        "grid": GRID,
        "conditions": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_turing_r.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
