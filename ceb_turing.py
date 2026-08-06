"""CEB-TURING: canonical reaction-diffusion pattern battery.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Uses a
1D periodic Brusselator. Detector sees only spectral-openness of the
activator field; native nonzero Fourier power is reported separately.
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
M = 128
DT = 0.002
T_STEPS = 6000
SAVE_EVERY = 50
GRID = list(range(0, T_STEPS + 1, SAVE_EVERY))
N_REP = 20
A = 1.0
B = 3.0
DU = 1.0
DV_PATTERN = 10.0
DV_STABLE = 1.0
NOISE = 1e-3
GATE = 0.1
SEED = 89_001


def laplacian(x: np.ndarray) -> np.ndarray:
    return np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) - 2 * x


def spectral_stats(u: np.ndarray) -> tuple[float, float, float]:
    """Return normalized spectral entropy, nonzero power, top-3 mass."""
    fft = np.fft.rfft(u - u.mean(axis=1, keepdims=True), axis=1)
    power = np.abs(fft[:, 1:]) ** 2  # remove zero mode
    entropies = []
    top3 = []
    totals = power.sum(axis=1)
    for row, total in zip(power, totals):
        if total <= 1e-18:
            entropies.append(1.0)
            top3.append(0.0)
        else:
            p = row / total
            q = p[p > 0]
            entropies.append(float(-(q * np.log2(q)).sum()
                                   / math.log2(len(row))))
            top3.append(float(np.sort(p)[-3:].sum()))
    return (float(np.median(entropies)), float(np.median(totals)),
            float(np.median(top3)))


def simulate(dv: float, seed: int) -> Dict:
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
        u = u + DT * du
        v = v + DT * dvdt
        # keep the explicit scheme out of nonphysical negatives
        u = np.clip(u, 0.0, None)
        v = np.clip(v, 0.0, None)

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


def adjudicate(openness) -> Dict:
    x = np.array(GRID, dtype=float)
    y = np.array(openness, dtype=float)
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
    thin = {}
    thin_ok = True
    for parity in (0, 1):
        if len(xw[parity::2]) < 5:
            t = {"verdict": "too_few_points", "ok": False}
            ok = False
        else:
            t = hinge_linear(xw[parity::2], yw[parity::2])
            ok = (t["delta_bic"] >= 2.0 and t["onset_type"]
                  and abs(t["t_star"] - full["t_star"]) <= 0.10 * span)
        t["ok"] = bool(ok)
        thin_ok = thin_ok and ok
        thin[f"parity{parity}"] = t
    out.update({
        "hinge": full,
        "thinning": thin,
        "b5_onset": bool(full["delta_bic"] >= 10
                         and full["onset_type"] and thin_ok),
    })
    return out


def main() -> None:
    rows = {}
    for name, dv in (("pattern", DV_PATTERN), ("stable", DV_STABLE)):
        row = simulate(dv, SEED + len(rows) * 101)
        adj = adjudicate(row["openness"])
        row["adj"] = adj
        rows[name] = row
        h = adj.get("hinge", {})
        print(f"{name}: O {row['openness_first_last']} "
              f"power {row['power_first_last']} "
              f"top3 {row['top3_first_last']} "
              f"b5={adj.get('b5_onset')} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')}",
              flush=True)

    pat = rows["pattern"]
    h = pat["adj"].get("hinge", {})
    p_final = pat["native_power"][-1]
    t_power80 = next((GRID[i] for i, v in enumerate(pat["native_power"])
                      if p_final > 0 and v >= 0.8 * p_final), None)
    tur1 = bool(pat["adj"].get("b5_onset"))
    tur2 = bool(tur1 and t_power80 is not None
                and h["t_star"] < t_power80)
    tur3 = bool(not rows["stable"]["adj"].get("b5_onset"))
    tur4 = bool(pat["top3_mass"][-1] >= 0.60
                and pat["top3_mass"][0] <= 0.35)
    outcomes = {
        "TUR1_pattern_onset": tur1,
        "TUR2_native_alignment": tur2,
        "TUR3_stable_null": tur3,
        "TUR4_mode_selectivity": tur4,
        "pattern_t_power80": t_power80,
    }
    report = {
        "status": "CEB-TURING reaction-diffusion pattern battery; preregistered",
        "config": {"M": M, "dt": DT, "steps": T_STEPS, "n_rep": N_REP,
                   "A": A, "B": B, "Du": DU, "Dv_pattern": DV_PATTERN,
                   "Dv_stable": DV_STABLE},
        "grid": GRID,
        "conditions": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_turing.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
