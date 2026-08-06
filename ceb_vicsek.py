"""CEB-VICSEK: canonical emergence battery for Vicsek flocking.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. The
detector sees only heading-bin openness; the native polarization
order parameter is reported separately for external validity.
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
N = 200
L = 10.0
SPEED = 0.03
RADIUS = 1.0
N_REP = 25
T_STEPS = 500
SAVE_EVERY = 5
GRID = list(range(0, T_STEPS + 1, SAVE_EVERY))
NBINS = 24
GATE = 0.1
SEED = 88_001


def entropy_from_counts(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def circular_mean(angles: np.ndarray) -> np.ndarray:
    z = np.exp(1j * angles)
    return np.angle(z.mean(axis=-1))


def simulate(condition: str, eta: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, size=(N_REP, N, 2))
    theta = rng.uniform(-np.pi, np.pi, size=(N_REP, N))
    openness, phi = [], []

    for step in range(T_STEPS + 1):
        if step % SAVE_EVERY == 0:
            hs = []
            phis = []
            for r in range(N_REP):
                bins = np.floor(((theta[r] + np.pi) / (2 * np.pi))
                                * NBINS).astype(int) % NBINS
                counts = np.bincount(bins, minlength=NBINS)
                hs.append(entropy_from_counts(counts) / math.log2(NBINS))
                phis.append(abs(np.exp(1j * theta[r]).mean()))
            openness.append(float(np.median(hs)))
            phi.append(float(np.median(phis)))

        if step == T_STEPS:
            break

        new_theta = np.empty_like(theta)
        if condition == "field":
            # Source-sanity control: common exogenous direction, no neighbor
            # alignment. This should reduce heading entropy but is ENV-driven.
            target = 0.0
            new_theta = target + rng.uniform(-eta / 2, eta / 2,
                                             size=theta.shape)
        else:
            for r in range(N_REP):
                dx = x[r, :, None, :] - x[r, None, :, :]
                dx = (dx + L / 2) % L - L / 2
                dist2 = (dx ** 2).sum(axis=-1)
                neigh = dist2 <= RADIUS ** 2
                z = neigh @ np.exp(1j * theta[r])
                mean = np.angle(z)
                new_theta[r] = mean + rng.uniform(-eta / 2, eta / 2,
                                                  size=N)
        theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        x = (x + SPEED * np.stack([np.cos(theta), np.sin(theta)], axis=-1)) % L

    return {
        "openness": openness,
        "phi": phi,
        "openness_first_last": [round(openness[0], 4),
                                round(openness[-1], 4)],
        "phi_first_last": [round(phi[0], 4), round(phi[-1], 4)],
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
    for name, eta in (("low_noise", 0.15), ("high_noise", 2.5),
                      ("field", 0.15)):
        row = simulate(name if name == "field" else "vicsek", eta,
                       SEED + len(rows) * 101)
        adj = adjudicate(row["openness"])
        row["adj"] = adj
        rows[name] = row
        h = adj.get("hinge", {})
        print(f"{name}: O {row['openness_first_last']} "
              f"phi {row['phi_first_last']} b5={adj.get('b5_onset')} "
              f"t*={h.get('t_star')} dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')}",
              flush=True)

    low = rows["low_noise"]
    h = low["adj"].get("hinge", {})
    phi_final = low["phi"][-1]
    t_phi80 = next((GRID[i] for i, v in enumerate(low["phi"])
                    if phi_final > 0 and v >= 0.8 * phi_final), None)
    vsk1 = bool(low["adj"].get("b5_onset"))
    vsk2 = bool(vsk1 and t_phi80 is not None and h["t_star"] < t_phi80)
    vsk3 = bool(not rows["high_noise"]["adj"].get("b5_onset"))
    vsk4 = bool(rows["field"]["adj"]["drop"] > 0.1)
    outcomes = {
        "VSK1_low_noise_onset": vsk1,
        "VSK2_native_alignment": vsk2,
        "VSK3_high_noise_null": vsk3,
        "VSK4_field_source_sanity": vsk4,
        "low_noise_t_phi80": t_phi80,
    }
    report = {
        "status": "CEB-VICSEK canonical flocking battery; preregistered",
        "config": {"N": N, "L": L, "speed": SPEED, "radius": RADIUS,
                   "n_rep": N_REP, "steps": T_STEPS, "nbins": NBINS},
        "grid": GRID,
        "conditions": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_vicsek.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
