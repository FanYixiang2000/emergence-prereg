"""LEARN-GRIP-CONFOUND: openness-controllability beyond time and |x|.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Addresses
reviewer critique #8 (openness may only proxy episode progress).
Training, environment, kick and seeds are byte-identical to
LEARN-GRIP-UTILITY (imported); we only retain per-episode records and
run the preregistered conditional analyses CC-1..CC-3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from learn_grip_transport import N_SEEDS, SEED
from learn_grip_utility import intervention_eval, train
from learn_transport_eq_utility import auc, rank_corr

OUTPUTS = Path(__file__).resolve().parent / "outputs"
TAUS = (18, 20, 22, 24, 26, 28)
EVAL_BATCH = 4096
N_PERM = 1000
PERM_SEED = 515_001


def within_cell_perm_corr(values, labels, cells, rng, n_perm=N_PERM):
    """Pooled within-cell rank corr with within-cell permutation null."""
    def pooled_corr(vals):
        num, wsum = 0.0, 0.0
        for c in np.unique(cells):
            m = cells == c
            if m.sum() < 10 or labels[m].std() == 0 or vals[m].std() == 0:
                continue
            r = rank_corr(vals[m], labels[m])
            num += r * m.sum()
            wsum += m.sum()
        return num / wsum if wsum else 0.0

    obs = pooled_corr(values)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        shuffled = values.copy()
        for c in np.unique(cells):
            m = np.where(cells == c)[0]
            shuffled[m] = shuffled[rng.permutation(m)]
        null[i] = pooled_corr(shuffled)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return float(obs), p


def logistic_fit(X, y, iters=50):
    """Newton logistic regression; returns coef and SEs."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p)
        H = Xb.T @ (Xb * W[:, None]) + 1e-6 * np.eye(Xb.shape[1])
        g = Xb.T @ (y - p)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    z = np.clip(Xb @ w, -30, 30)
    p = 1 / (1 + np.exp(-z))
    W = p * (1 - p)
    H = Xb.T @ (Xb * W[:, None]) + 1e-6 * np.eye(Xb.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return w, se


def main() -> None:
    recs = {k: [] for k in ("switch", "side_open", "absx", "absv",
                            "att", "tau", "seed")}
    for i in range(N_SEEDS):
        policy = train(SEED + i * 101)
        for tau in TAUS:
            row = intervention_eval(policy, tau, seed=SEED + i * 101 + tau,
                                    batch=EVAL_BATCH)
            n = len(row["switch"])
            recs["switch"].extend(row["switch"].tolist())
            recs["side_open"].extend(row["side_open"].tolist())
            recs["absx"].extend(row["absx"].tolist())
            recs["absv"].extend(row["absv"].tolist())
            recs["att"].extend(row["att"].tolist())
            recs["tau"].extend([float(tau)] * n)
            recs["seed"].extend([i] * n)
            print(f"seed={i} tau={tau}: switch={row['switch_rate']:.3f} "
                  f"open_mean={float(np.mean(row['side_open'])):.3f}",
                  flush=True)
    d = {k: np.asarray(v) for k, v in recs.items()}

    # --- CC-1: fixed time (within each tau, pooled over seeds) ---
    cc1 = {}
    aucs = []
    for tau in TAUS:
        m = d["tau"] == tau
        if d["switch"][m].std() == 0:
            cc1[str(tau)] = {"auc": None, "n": int(m.sum()),
                             "switch_rate": float(d["switch"][m].mean())}
            continue
        a = auc(d["side_open"][m], d["switch"][m])
        aucs.append(a)
        cc1[str(tau)] = {"auc": round(float(a), 5), "n": int(m.sum()),
                         "switch_rate": round(float(d["switch"][m].mean()), 4)}
    cc1_mean = float(np.mean(aucs)) if aucs else 0.0
    cc1_above = sum(a > 0.5 for a in aucs)

    # --- CC-2: fixed (tau x |x|-quintile) cells ---
    qs = np.quantile(d["absx"], [0.2, 0.4, 0.6, 0.8])
    xbin = np.digitize(d["absx"], qs)
    cells = (d["tau"].astype(int) * 10 + xbin).astype(int)
    rng = np.random.default_rng(PERM_SEED)
    cc2_corr, cc2_p = within_cell_perm_corr(d["side_open"].copy(),
                                            d["switch"], cells, rng)

    # --- CC-3: logistic partial effect ---
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-12)
    X = np.column_stack([z(d["side_open"]), z(d["absx"]), z(d["absv"]),
                         z(d["att"]), z(d["tau"])])
    w, se = logistic_fit(X, d["switch"])
    names = ["intercept", "side_open", "absx", "absv", "att", "tau"]
    cc3 = {nm: {"coef": round(float(c), 4), "se": round(float(s), 4)}
           for nm, c, s in zip(names, w, se)}

    outcomes = {
        "CC1_fixed_time_mean_auc_ge_0.60": bool(cc1_mean >= 0.60),
        "CC1_above_chance_ge_5of6": bool(cc1_above >= 5),
        "CC2_fixed_absx_positive_p_lt_0.05": bool(cc2_corr > 0
                                                  and cc2_p < 0.05),
        "CC3_openness_coef_positive": bool(cc3["side_open"]["coef"] > 0),
        "cc1_mean_auc": round(cc1_mean, 4),
        "cc1_cells_above_chance": int(cc1_above),
        "cc2_pooled_within_cell_corr": round(cc2_corr, 4),
        "cc2_permutation_p": cc2_p,
    }
    report = {
        "status": ("LEARN-GRIP-CONFOUND conditional controllability; "
                   "training/kick byte-identical to LEARN-GRIP-UTILITY; "
                   "registered before run"),
        "config": {"taus": TAUS, "eval_batch": EVAL_BATCH,
                   "n_perm": N_PERM, "perm_seed": PERM_SEED,
                   "n_records": int(len(d["switch"]))},
        "cc1_by_tau": cc1,
        "cc2": {"pooled_within_cell_corr": round(cc2_corr, 4),
                "permutation_p": cc2_p,
                "absx_quintile_edges": [round(float(q), 4) for q in qs]},
        "cc3_logistic": cc3,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_confound.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
