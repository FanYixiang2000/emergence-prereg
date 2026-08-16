"""STAT-UNIT: seed-level statistics for the grip intervention races.

Registered as an analysis addendum in
METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen before this file
was written). Training, discovery, intervention grid, kick and
predictor definitions are byte-identical to LEARN-GRIP-UTILITY and
RDC, so the episode streams are identical reruns; the only new
content is the seed-level aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from learn_grip_transport import N_SEEDS, SEED
from learn_grip_discovery_utility import intervention_eval
from learn_grip_utility import EVAL_BATCH, TAUS, train
from learn_transport_eq_utility import auc
from regime_discovery_audit import (CLUSTER_SEED, GRIP_EPISODES, KNN,
                                    choose_k, grip_traces)

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

OUTPUTS = Path(__file__).resolve().parent / "outputs"
NAMES = ("disc_open", "side_open", "pol_ent", "absx", "absv", "att", "tau")
SIGNS = {"disc_open": 1.0, "side_open": 1.0, "pol_ent": 1.0, "absx": -1.0,
         "absv": -1.0, "att": -1.0, "tau": -1.0}
N_BOOT = 10_000
BOOT_SEED = 0


def main() -> None:
    torch.set_num_threads(8)
    per_seed = []  # one dict of predictor arrays + switch per seed
    for i in range(N_SEEDS):
        seed = SEED + i * 101
        policy = train(seed)
        xs, vs, ats = grip_traces(policy, GRIP_EPISODES)
        k = choose_k(xs, kmax=8)
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=CLUSTER_SEED).fit(xs)
        data = {n: [] for n in NAMES + ("switch",)}
        fixed = {}
        for tau in TAUS:
            state_tau = np.stack([xs[:, tau], vs[:, tau], ats[:, tau]],
                                 axis=1)
            knn = KNeighborsClassifier(n_neighbors=KNN).fit(state_tau,
                                                            km.labels_)
            row = intervention_eval(policy, tau, seed + tau, knn, k)
            for n in NAMES[:-1]:
                data[n].extend(row[n].tolist())
            data["tau"].extend([float(tau)] * EVAL_BATCH)
            data["switch"].extend(row["switch"].tolist())
            sw = row["switch"]
            n_pos, n_neg = int(sw.sum()), int(len(sw) - sw.sum())
            if min(n_pos, n_neg) >= 20:
                fixed[str(tau)] = {
                    "auc_side": round(auc(row["side_open"], sw), 5),
                    "auc_disc": round(auc(row["disc_open"], sw), 5),
                }
        per_seed.append({"arrays": {n: np.array(data[n]) for n in
                                    NAMES + ("switch",)},
                         "fixed_tau": fixed, "k": k})
        print(f"seed {i}: k={k} episodes={len(data['switch'])}", flush=True)

    seed_aucs = {n: [] for n in NAMES}
    for s in per_seed:
        sw = s["arrays"]["switch"]
        for n in NAMES:
            seed_aucs[n].append(round(auc(SIGNS[n] * s["arrays"][n], sw), 5))

    rng = np.random.default_rng(BOOT_SEED)
    boot = {n: [] for n in ("side_open", "disc_open", "pol_ent")}
    for _ in range(N_BOOT):
        pick = rng.integers(0, N_SEEDS, N_SEEDS)
        sw = np.concatenate([per_seed[j]["arrays"]["switch"] for j in pick])
        for n in boot:
            vals = np.concatenate([per_seed[j]["arrays"][n] for j in pick])
            boot[n].append(auc(SIGNS[n] * vals, sw))
    boot_ci = {n: [round(float(np.percentile(v, 2.5)), 5),
                   round(float(np.percentile(v, 97.5)), 5)]
               for n, v in boot.items()}

    loo = {}
    for n in ("side_open", "disc_open"):
        vals = []
        for drop in range(N_SEEDS):
            keep = [j for j in range(N_SEEDS) if j != drop]
            sw = np.concatenate([per_seed[j]["arrays"]["switch"]
                                 for j in keep])
            v = np.concatenate([per_seed[j]["arrays"][n] for j in keep])
            vals.append(round(auc(SIGNS[n] * v, sw), 5))
        loo[n] = vals

    su1 = all(a > b for a, b in zip(seed_aucs["side_open"],
                                    seed_aucs["pol_ent"]))
    outcomes = {
        "k_discovered": [s["k"] for s in per_seed],
        "per_seed_auc": seed_aucs,
        "boot_ci_95_seed_cluster": boot_ci,
        "leave_one_seed_out": loo,
        "per_seed_fixed_tau": [s["fixed_tau"] for s in per_seed],
        "SU1_side_beats_entropy_every_seed": bool(su1),
        "SU2_side_ci_above_095": bool(boot_ci["side_open"][0] > 0.95),
    }
    out = OUTPUTS / "learn_grip_stat_unit.json"
    out.write_text(json.dumps({
        "status": ("STAT-UNIT seed-level statistics; episode streams "
                   "byte-identical to LEARN-GRIP-UTILITY/RDC; analysis "
                   "addendum frozen before run"),
        "config": {"n_seeds": N_SEEDS, "n_boot": N_BOOT,
                   "boot_seed": BOOT_SEED},
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
