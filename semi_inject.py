"""SEMI-INJ: detector validation on a real-noise substrate.

Preregistered in V2_ALIGNMENT_PREREGISTRATION.md (2026-08-05). Builds
semi-synthetic formation curves from the stored ring evaluation
machinery -- real 100-point grids, real per-checkpoint committed-episode
counts, the pipeline's own Laplace-smoothed binary-entropy estimator --
with a KNOWN injected commitment time t0. The frozen detector is applied
unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LOG2_3 = math.log2(3)

T0_FRACS = (0.3, 0.5, 0.7)
WIDTHS = (1, 3, 6)
N_PER_CELL = 25
N_NEG_PER_FAMILY = 100
P_OPEN, P_COMMIT = 0.5, 0.97


def real_substrate():
    """Real grids and committed-episode-count sequences from the 8 seeds."""
    orig = json.load(open(OUTPUTS / "overcooked_ring_convention.json"))
    ext = json.load(open(OUTPUTS / "oc_ring_ext.json"))
    recs = list(orig["systems"]["ring"].values()) + \
        list(ext["ext_seeds"].values())
    grids = [r["grid"] for r in recs]
    ncoms = [[c["n_committed_episodes"] for c in r["curves"]] for r in recs]
    popen = [[c["p_ccw"] for c in r["curves"]] for r in recs]
    return grids, ncoms, popen


def openness_from_p(rng, p_seq, ncom_seq):
    """Sample directions and apply the pipeline's own estimator."""
    out = []
    for p, n in zip(p_seq, ncom_seq):
        n_ccw = rng.binomial(n, p) if n > 0 else 0
        ph = (n_ccw + 1) / (n + 2)
        out.append(-(ph * math.log2(ph) + (1 - ph) * math.log2(1 - ph)))
    return np.array(out)


def main() -> None:
    grids, ncoms, popen = real_substrate()
    rng = np.random.default_rng(98001)

    pos_results = {}
    all_pos = []
    for w in WIDTHS:
        for f in T0_FRACS:
            ok_onset, t_errs = 0, []
            for i in range(N_PER_CELL):
                k = rng.integers(len(grids))
                grid = np.array(grids[k], dtype=float)
                ncom = ncoms[k]
                n = len(grid)
                t0_idx = int(f * n)
                idx = np.arange(n)
                # logistic approach from P_OPEN to P_COMMIT, width w points
                z = 1.0 / (1.0 + np.exp(-(idx - t0_idx) / max(w / 2, 0.5)))
                p_seq = P_OPEN + (P_COMMIT - P_OPEN) * z
                o = openness_from_p(rng, p_seq, ncom)
                adj = adjudicate(grid, o * LOG2_3)
                onset = bool(adj.get("b5_onset"))
                span = grid[-1] - grid[0]
                if onset:
                    t_errs.append(abs(adj["hinge"]["t_star"]
                                      - grid[t0_idx]) / span)
                ok_onset += onset
                all_pos.append({"w": w, "f": f, "onset": onset})
            pos_results[f"w{w}_f{f}"] = {
                "power": ok_onset / N_PER_CELL,
                "median_tstar_err_frac": (float(np.median(t_errs))
                                          if t_errs else None)}

    neg_results = {}
    n_fp_total = 0
    for family in ("constant_open", "linear_drift", "shuffled_real"):
        fp = 0
        for i in range(N_NEG_PER_FAMILY):
            k = rng.integers(len(grids))
            grid = np.array(grids[k], dtype=float)
            ncom = ncoms[k]
            n = len(grid)
            if family == "constant_open":
                p_seq = np.full(n, P_OPEN)
            elif family == "linear_drift":
                p_seq = np.linspace(P_OPEN, P_COMMIT, n)
            else:
                # shuffle a real seed's uncommitted-phase p values
                src = [p for p in popen[k] if 0.2 <= p <= 0.8]
                if len(src) < 5:
                    src = popen[k]
                p_seq = rng.choice(src, size=n, replace=True)
            o = openness_from_p(rng, p_seq, ncom)
            adj = adjudicate(grid, o * LOG2_3)
            fp += bool(adj.get("b5_onset"))
        neg_results[family] = fp / N_NEG_PER_FAMILY
        n_fp_total += fp

    pow_w_le3 = np.mean([r["onset"] for r in all_pos if r["w"] <= 3])
    fpr = n_fp_total / (3 * N_NEG_PER_FAMILY)
    errs = [v["median_tstar_err_frac"] for v in pos_results.values()
            if v["median_tstar_err_frac"] is not None]
    outcomes = {
        "SI1_power_w_le3": float(pow_w_le3),
        "SI1_pass": bool(pow_w_le3 >= 0.90),
        "SI2_fpr_pooled": float(fpr),
        "SI2_pass": bool(fpr <= 0.05),
        "SI3_median_tstar_err_frac": float(np.median(errs)),
        "SI3_pass": bool(np.median(errs) <= 0.05),
        "w6_power_descriptive": float(np.mean(
            [r["onset"] for r in all_pos if r["w"] == 6])),
    }
    (OUTPUTS / "semi_inject.json").write_text(json.dumps({
        "status": ("SEMI-INJ semi-synthetic injection on real substrate; "
                   "frozen detector unchanged; registered before run"),
        "config": {"t0_fracs": T0_FRACS, "widths": WIDTHS,
                   "n_per_cell": N_PER_CELL,
                   "n_neg_per_family": N_NEG_PER_FAMILY,
                   "p_open": P_OPEN, "p_commit": P_COMMIT},
        "positives": pos_results, "negatives": neg_results,
        "registered_outcomes": outcomes}, indent=1))
    print(json.dumps(outcomes, indent=1))
    print("negatives:", neg_results)
    print("positives:", json.dumps(pos_results, indent=1))


if __name__ == "__main__":
    main()
