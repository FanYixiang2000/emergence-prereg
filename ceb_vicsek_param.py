"""CEB-VICSEK-PARAM: Vicsek control-parameter transition.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Measures
final heading openness and polarization across noise eta, testing the
canonical transition along the control parameter rather than within-run
relaxation from random initial conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tri_c_breakpoint import hinge_linear
from ceb_vicsek import N, L, SPEED, RADIUS, NBINS, entropy_from_counts

OUTPUTS = Path(__file__).resolve().parent / "outputs"
ETAS = (3.0, 2.5, 2.0, 1.6, 1.2, 0.8, 0.4, 0.2, 0.1, 0.05)
N_REP = 10
T_STEPS = 500
GATE = 0.1
SEED = 88_901


def final_stats(eta: float, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, L, size=(N_REP, N, 2))
    theta = rng.uniform(-np.pi, np.pi, size=(N_REP, N))
    for _step in range(T_STEPS):
        new_theta = np.empty_like(theta)
        for r in range(N_REP):
            dx = x[r, :, None, :] - x[r, None, :, :]
            dx = (dx + L / 2) % L - L / 2
            neigh = (dx ** 2).sum(axis=-1) <= RADIUS ** 2
            z = neigh @ np.exp(1j * theta[r])
            mean = np.angle(z)
            new_theta[r] = mean + rng.uniform(-eta / 2, eta / 2, size=N)
        theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        x = (x + SPEED * np.stack([np.cos(theta), np.sin(theta)], axis=-1)) % L

    hs, phis = [], []
    for r in range(N_REP):
        bins = np.floor(((theta[r] + np.pi) / (2 * np.pi))
                        * NBINS).astype(int) % NBINS
        counts = np.bincount(bins, minlength=NBINS)
        hs.append(entropy_from_counts(counts) / np.log2(NBINS))
        phis.append(abs(np.exp(1j * theta[r]).mean()))
    return float(np.median(hs)), float(np.median(phis))


def hinge_axis(y):
    x = np.arange(len(y), dtype=float)
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["b5_control"] = False
        return out
    full = hinge_linear(x, np.array(y))
    out["hinge"] = full
    out["b5_control"] = bool(full["delta_bic"] >= 10)
    return out


def main() -> None:
    openness, phi = [], []
    for i, eta in enumerate(ETAS):
        o, p = final_stats(eta, SEED + i * 13)
        openness.append(o)
        phi.append(p)
        print(f"eta={eta}: O={o:.4f} phi={p:.4f}", flush=True)

    adj = hinge_axis(openness)
    phi_deltas = np.diff(phi)
    steep_i = int(np.argmax(phi_deltas)) + 1
    hinge_i = int(adj.get("hinge", {}).get("t_star", -99))
    vskp1 = bool(adj.get("b5_control"))
    vskp2 = bool(vskp1 and abs(hinge_i - steep_i) <= 1)
    vskp3 = bool(openness[0] - openness[-1] >= 0.3)
    outcomes = {
        "VSKP1_control_breakpoint": vskp1,
        "VSKP2_native_alignment": vskp2,
        "VSKP3_low_high_gap": vskp3,
        "steepest_phi_index": steep_i,
        "hinge_index": hinge_i,
    }
    report = {
        "status": "CEB-VICSEK-PARAM control-axis battery; preregistered",
        "etas_high_to_low": ETAS,
        "openness": [round(v, 5) for v in openness],
        "phi": [round(v, 5) for v in phi],
        "adj": adj,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_vicsek_param.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
