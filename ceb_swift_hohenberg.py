"""CEB-SH: Swift-Hohenberg canonical pattern-formation battery.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Tests
spectral possibility collapse in a clean pattern-selection normal form.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from kuramoto_breakpoint_r2 import truncate_at_saturation
from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
M = 256
L = 64 * np.pi
DT = 0.5
T_STEPS = 500
SAVE_EVERY = 5
GRID = list(range(0, T_STEPS + 1, SAVE_EVERY))
N_REP = 20
NOISE = 1e-3
R_PATTERN = 0.2
R_STABLE = -0.2
GATE = 0.1
SEED = 91_001


def spectral_stats(u: np.ndarray):
    fft = np.fft.rfft(u - u.mean(axis=1, keepdims=True), axis=1)
    power = np.abs(fft[:, 1:]) ** 2
    n_modes = power.shape[1]
    k = 2 * np.pi * np.fft.rfftfreq(M, d=L / M)[1:]
    band = np.abs(k - 1.0) <= 0.1
    ents, band_share, totals = [], [], []
    for row in power:
        total = row.sum()
        totals.append(float(total))
        if total <= 1e-30:
            ents.append(1.0)
            band_share.append(0.0)
        else:
            p = row / total
            q = p[p > 0]
            ents.append(float(-(q * np.log2(q)).sum()
                              / math.log2(n_modes)))
            band_share.append(float(p[band].sum()))
    return (float(np.median(ents)), float(np.median(band_share)),
            float(np.median(totals)))


def simulate(r: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    u = NOISE * rng.standard_normal((N_REP, M))
    k = 2 * np.pi * np.fft.rfftfreq(M, d=L / M)
    lin = r - (1 - k ** 2) ** 2
    denom = 1 - DT * lin
    openness, band_share, total_power = [], [], []
    for step in range(T_STEPS + 1):
        if step % SAVE_EVERY == 0:
            o, b, p = spectral_stats(u)
            openness.append(o)
            band_share.append(b)
            total_power.append(p)
        if step == T_STEPS:
            break
        uhat = np.fft.rfft(u, axis=1)
        nonlin = np.fft.rfft(-(u ** 3), axis=1)
        uhat_new = (uhat + DT * nonlin) / denom[None, :]
        u = np.fft.irfft(uhat_new, n=M, axis=1)
    return {
        "openness": openness,
        "band_share": band_share,
        "total_power": total_power,
        "openness_first_last": [round(openness[0], 4),
                                round(openness[-1], 4)],
        "band_first_last": [round(band_share[0], 4),
                            round(band_share[-1], 4)],
        "power_first_last": [float(total_power[0]),
                             float(total_power[-1])],
    }


def adjudicate(openness) -> Dict:
    x = np.array(GRID, dtype=float)
    y = np.array(openness)
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        out["b5_onset"] = False
        return out
    xw, yw, t_sat = truncate_at_saturation(x, y)
    out["t_sat"] = t_sat
    if len(yw) < 8:
        out["verdict"] = "window_too_short"
        out["b5_onset"] = False
        return out
    full = hinge_linear(xw, yw)
    span = xw[-1] - xw[0]
    thin_ok = True
    thin = {}
    for parity in (0, 1):
        if len(xw[parity::2]) < 5:
            t = {"verdict": "too_few_points", "ok": False}
            ok = False
        else:
            t = hinge_linear(xw[parity::2], yw[parity::2])
            ok = (t["delta_bic"] >= 2.0 and t["onset_type"]
                  and abs(t["t_star"] - full["t_star"]) <= 0.10 * span)
        t["ok"] = bool(ok)
        thin[f"parity{parity}"] = t
        thin_ok = thin_ok and ok
    out.update({
        "hinge": full,
        "thinning": thin,
        "b5_onset": bool(full["delta_bic"] >= 10
                         and full["onset_type"] and thin_ok),
    })
    return out


def main() -> None:
    rows = {}
    for name, r in (("pattern", R_PATTERN), ("stable", R_STABLE)):
        row = simulate(r, SEED + len(rows) * 101)
        adj = adjudicate(row["openness"])
        row["adj"] = adj
        rows[name] = row
        h = adj.get("hinge", {})
        print(f"{name}: O {row['openness_first_last']} "
              f"band {row['band_first_last']} "
              f"power {row['power_first_last']} "
              f"b5={adj.get('b5_onset')} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')}",
              flush=True)

    pat = rows["pattern"]
    h = pat["adj"].get("hinge", {})
    band_final = pat["band_share"][-1]
    t_band80 = next((GRID[i] for i, v in enumerate(pat["band_share"])
                     if band_final > 0 and v >= 0.8 * band_final), None)
    sh1 = bool(pat["adj"].get("b5_onset"))
    sh2 = bool(sh1 and t_band80 is not None and h["t_star"] < t_band80)
    sh3 = bool(not rows["stable"]["adj"].get("b5_onset"))
    sh4 = bool(pat["band_share"][-1] >= 0.7
               and pat["band_share"][0] <= 0.3)
    outcomes = {
        "SH1_pattern_onset": sh1,
        "SH2_native_alignment": sh2,
        "SH3_stable_null": sh3,
        "SH4_mode_selectivity": sh4,
        "pattern_t_band80": t_band80,
    }
    report = {
        "status": "CEB-SH Swift-Hohenberg pattern battery; preregistered",
        "config": {"M": M, "L": L, "dt": DT, "steps": T_STEPS,
                   "n_rep": N_REP, "r_pattern": R_PATTERN,
                   "r_stable": R_STABLE},
        "grid": GRID,
        "conditions": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_swift_hohenberg.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
