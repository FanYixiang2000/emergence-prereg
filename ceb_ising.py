"""CEB-ISING: canonical 2D Ising spontaneous magnetization battery.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Tests
whether a control-axis possibility collapse aligns with the textbook
Ising phase transition.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
L = 40
N_REP = 8
TEMPS = (4.0, 3.2, 2.8, 2.5, 2.35, 2.25, 2.1, 1.9, 1.6, 1.3)
THERM_SWEEPS = 800
SAMPLE_SWEEPS = 400
GATE = 0.1
TC = 2.269
SEED = 90_001


def h2_from_abs_m(m_abs: float) -> float:
    p = (1 + min(max(m_abs, 0.0), 1.0)) / 2
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def checkerboard_sweep(spins: np.ndarray, temp: float,
                       rng: np.random.Generator) -> None:
    for parity in (0, 1):
        neigh = (np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
                 + np.roll(spins, 1, axis=2) + np.roll(spins, -1, axis=2))
        dE = 2 * spins * neigh
        mask = ((np.indices((L, L)).sum(axis=0) % 2) == parity)[None, :, :]
        accept = (dE <= 0) | (rng.random(spins.shape) < np.exp(-dE / temp))
        spins[mask & accept] *= -1


def run_temp(temp: float, seed: int):
    rng = np.random.default_rng(seed)
    spins = rng.choice((-1, 1), size=(N_REP, L, L)).astype(np.int8)
    for _ in range(THERM_SWEEPS):
        checkerboard_sweep(spins, temp, rng)
    mags = []
    for _ in range(SAMPLE_SWEEPS):
        checkerboard_sweep(spins, temp, rng)
        mags.append(np.abs(spins.mean(axis=(1, 2))))
    mags = np.array(mags)
    per_rep = mags.mean(axis=0)
    return float(np.median(per_rep))


def main() -> None:
    m_abs, openness = [], []
    for i, temp in enumerate(TEMPS):
        m = run_temp(temp, SEED + i * 17)
        o = h2_from_abs_m(m)
        m_abs.append(m)
        openness.append(o)
        print(f"T={temp}: |m|={m:.4f} O={o:.4f}", flush=True)

    drop = openness[0] - openness[-1]
    adj = {"drop": round(float(drop), 4),
           "gate_passed": bool(drop >= GATE)}
    if adj["gate_passed"]:
        x = np.arange(len(TEMPS), dtype=float)
        h = hinge_linear(x, np.array(openness))
        adj["hinge"] = h
        adj["b5_control"] = bool(h["delta_bic"] >= 10)
    else:
        adj["b5_control"] = False

    dm = np.diff(m_abs)
    steep_i = int(np.argmax(dm)) + 1
    hinge_i = int(adj.get("hinge", {}).get("t_star", -99))
    tc_i = min(range(len(TEMPS)), key=lambda i: abs(TEMPS[i] - TC))
    isg1 = bool(adj.get("b5_control"))
    isg2 = bool(isg1 and abs(hinge_i - steep_i) <= 1
                and abs(hinge_i - tc_i) <= 2)
    isg3 = bool(m_abs[0] < 0.2 and m_abs[-1] > 0.8)
    outcomes = {
        "ISG1_control_collapse": isg1,
        "ISG2_native_alignment": isg2,
        "ISG3_high_low_contrast": isg3,
        "hinge_index": hinge_i,
        "steepest_m_index": steep_i,
        "tc_grid_index": tc_i,
    }
    report = {
        "status": "CEB-ISING canonical Ising battery; preregistered",
        "config": {"L": L, "n_rep": N_REP, "temps": TEMPS,
                   "therm_sweeps": THERM_SWEEPS,
                   "sample_sweeps": SAMPLE_SWEEPS},
        "temperature_high_to_low": TEMPS,
        "abs_magnetization": [round(v, 5) for v in m_abs],
        "openness": [round(v, 5) for v in openness],
        "adj": adj,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_ising.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
