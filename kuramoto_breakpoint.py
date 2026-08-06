"""KUR-BP: onset breakpoint at the Kuramoto synchronization
transition.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Supercritical sync (autocatalytic) is predicted to imprint an
onset-type hinge on the joint openness of a tagged oscillator
triple measured in raw phases across replicas; subcritical coupling
is the null control. Detector mirrors RE-2 (linear axis, Delta-BIC,
2x thinning).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from breakpoint_model_comparison import bic, fit_one_segment, \
    fit_two_segment
from triad_relational_collapse import entropy, ipf_pairwise_generic

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_OSC = 200
R_REP = 20_000
DT = 0.02
N_STEPS = 600
MEASURE_EVERY = 10
SIGMA = 0.05
OMEGA_SD = 0.5
NBINS = 10
TWO_PI = 2 * np.pi
K_SUPER, K_SUB = 1.5, 0.3
SEED = 81_001


def ladder2(table: np.ndarray) -> Dict:
    p = table / table.sum()
    h_p = entropy(p)
    h_q0 = math.log2(p.size)
    m = [p.sum(axis=tuple(a for a in range(3) if a != i))
         for i in range(3)]
    h_qi = entropy(np.einsum("i,j,k->ijk", m[0], m[1], m[2]))
    h_qpair = entropy(ipf_pairwise_generic(p))
    return {"C_individual": h_q0 - h_qi, "C_pair": h_qi - h_qpair,
            "C_high": h_qpair - h_p, "C_total": h_q0 - h_p,
            "H_P": h_p}


def simulate(k_coupling: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    omegas = rng.normal(0.0, OMEGA_SD, size=N_OSC).astype(np.float32)
    tagged = np.argsort(np.abs(omegas))[:3]
    theta = rng.uniform(0, TWO_PI,
                        size=(R_REP, N_OSC)).astype(np.float32)
    grid, openness, r_curve = [], [], []
    final_table = None
    for step in range(N_STEPS + 1):
        if step % MEASURE_EVERY == 0:
            tag = theta[:, tagged]
            bins = np.floor((tag % TWO_PI)
                            / (TWO_PI / NBINS)).astype(int) % NBINS
            table = np.zeros((NBINS,) * 3)
            np.add.at(table, (bins[:, 0], bins[:, 1], bins[:, 2]), 1.0)
            h = entropy(table / table.sum())
            z = np.exp(1j * theta).mean(axis=1)
            grid.append(round(step * DT, 4))
            openness.append(h / (3 * math.log2(NBINS)))
            r_curve.append(float(np.median(np.abs(z))))
            final_table = table
        if step == N_STEPS:
            break
        z = np.exp(1j * theta).mean(axis=1)
        r = np.abs(z)[:, None]
        psi = np.angle(z)[:, None]
        drift = omegas[None, :] + k_coupling * r * np.sin(psi - theta)
        noise = SIGMA * np.sqrt(DT) * rng.standard_normal(
            theta.shape).astype(np.float32)
        theta = theta + DT * drift.astype(np.float32) + noise
    return {"grid": grid, "openness": openness, "r": r_curve,
            "final_ladder": {k: round(v, 5) for k, v in
                             ladder2(final_table).items()}}


def hinge(x: np.ndarray, y: np.ndarray) -> Dict:
    n = len(y)
    rss1 = fit_one_segment(x, y)
    best = None
    for bi in range(1, n - 1):
        rss2 = fit_two_segment(x, y, bi)
        if best is None or rss2 < best[1]:
            best = (bi, rss2)
    bi, rss2 = best
    delta = bic(rss1, n, 2) - bic(rss2, n, 4)
    xb = x[bi]
    A = np.vstack([x, np.maximum(x - xb, 0.0), np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    s_before = float(coef[0])
    s_after = float(coef[0] + coef[1])
    return {"delta_bic": round(float(delta), 3), "t_star": float(xb),
            "slope_before": round(s_before, 6),
            "slope_after": round(s_after, 6),
            "onset_type": bool(s_after < s_before)}


def main() -> None:
    results = {}
    for name, k in (("super", K_SUPER), ("sub", K_SUB)):
        res = simulate(k, SEED)
        x = np.array(res["grid"])
        y = np.array(res["openness"])
        full = hinge(x, y)
        thin = {}
        step2 = 2 * MEASURE_EVERY * DT
        for parity in (0, 1):
            t = hinge(x[parity::2], y[parity::2])
            t["hinge_ok"] = bool(abs(t["t_star"] - full["t_star"])
                                 <= step2)
            thin[f"parity{parity}"] = t
        r_final = res["r"][-1]
        t_r90 = next((g for g, r in zip(res["grid"], res["r"])
                      if r >= 0.9 * r_final), None)
        results[name] = {
            "K": k, "hinge": full, "thinning": thin,
            "r_final": round(r_final, 4), "t_r90": t_r90,
            "openness_first_last": [round(y[0], 4), round(y[-1], 4)],
            "final_ladder": res["final_ladder"],
            "grid": res["grid"],
            "openness_curve": [round(v, 5) for v in y],
            "r_curve": [round(v, 5) for v in res["r"]],
        }
        print(f"{name} K={k}: dBIC={full['delta_bic']} "
              f"t*={full['t_star']} onset={full['onset_type']} "
              f"O: {y[0]:.3f}->{y[-1]:.3f} r_fin={r_final:.3f} "
              f"t_r90={t_r90}", flush=True)
        print(f"  final ladder: {res['final_ladder']}", flush=True)

    sup, sub = results["super"], results["sub"]
    c_sup = sup["final_ladder"]["C_total"]
    c_sub = sub["final_ladder"]["C_total"]
    h1 = sup["hinge"]
    kbp1 = bool(h1["delta_bic"] >= 10 and h1["onset_type"]
                and sup["t_r90"] is not None
                and h1["t_star"] < sup["t_r90"])
    kbp2 = all(t["delta_bic"] >= 10 and t["onset_type"]
               and t["hinge_ok"]
               for t in sup["thinning"].values())
    h_sub = sub["hinge"]
    kbp3 = bool((h_sub["delta_bic"] < 10 or not h_sub["onset_type"])
                and c_sub < 0.1 * c_sup)
    lad = sup["final_ladder"]
    kbp4 = bool((lad["C_pair"] + lad["C_high"]) >= 0.8 * lad["C_total"]
                and lad["C_individual"] <= 0.1 * lad["C_total"])

    outcomes = {"KURBP1_onset": kbp1, "KURBP2_persistence": bool(kbp2),
                "KURBP3_subcritical_null": kbp3,
                "KURBP4_relational_carrier": kbp4}
    report = {
        "status": ("KUR-BP onset breakpoint at the synchronization "
                   "transition; registered before run; RE-2 hinge "
                   "contract; raw-phase joint table of the tagged "
                   "triple"),
        "config": {"N": N_OSC, "R": R_REP, "K_super": K_SUPER,
                   "K_sub": K_SUB, "sigma": SIGMA,
                   "omega_sd": OMEGA_SD, "dt": DT,
                   "steps": N_STEPS, "nbins": NBINS},
        "conditions": results,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "kuramoto_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
