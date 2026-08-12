"""REGIME-DISCOVERY v2: cross-fitted correction of the grip estimator.

The v1 audit (regime_discovery_audit.py) certified its formation-axis
results but failed on the grip realization axis for a mechanical
reason: the kNN estimate of P(cluster | state) was evaluated on its own
training points, so once per-episode states became unique the
classifier memorized cluster labels and reported spurious collapse --
including in 3/5 untrained controls. v1 is retained unchanged as a
registered record. v2 repairs only that estimator, exactly as the
earlier basin work did, by cross-fitting: episodes are split into
halves by index parity, the kNN is fit on one half and evaluated on
the other, and the two directions are averaged. Clustering, features,
detector and all thresholds are byte-identical to v1.

Registered predictions (frozen before the run):
  RD2b verdict agreement: the cross-fitted discovered-regime B5 verdict
       equals the stored declared verdict in >= 4/5 grip seeds.
  RD3b breakpoint agreement: wherever both certify onset,
       |t*_discovered - t*_declared| <= 5 steps.
  RD4b controls: 0/5 untrained controls certify onset.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from ant_fine_onset import adjudicate
import learn_grip_transport as LG
from regime_discovery_audit import (choose_k, grip_traces, GRIP_EPISODES,
                                    KNN, CLUSTER_SEED)

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def crossfit_openness(xs, vs, ats):
    k = choose_k(xs, kmax=8)
    km = KMeans(n_clusters=k, n_init=10, random_state=CLUSTER_SEED).fit(xs)
    labels = km.labels_
    n = len(labels)
    half_a = np.arange(n) % 2 == 0
    curve = []
    for t in range(LG.MAX_STEPS):
        state = np.stack([xs[:, t], vs[:, t], ats[:, t]], axis=1)
        h = np.zeros(n)
        for fit_mask in (half_a, ~half_a):
            knn = KNeighborsClassifier(n_neighbors=KNN).fit(
                state[fit_mask], labels[fit_mask])
            proba = knn.predict_proba(state[~fit_mask])
            h[~fit_mask] = -(np.where(
                proba > 0, proba * np.log2(np.clip(proba, 1e-12, 1)),
                0.0)).sum(axis=1)
        curve.append(float(h.mean() / math.log2(k)) if k > 1 else 0.0)
    return k, np.array(curve)


def main() -> None:
    torch.set_num_threads(4)
    b5 = json.loads((OUTPUTS / "learn_grip_transport_b5.json").read_text())
    stored = json.loads((OUTPUTS / "learn_grip_transport.json").read_text())
    rows = {}
    for i in range(LG.N_SEEDS):
        seed = LG.SEED + i * 101
        torch.manual_seed(seed)
        np.random.seed(seed)
        policy = LG.GripPolicy()
        opt = torch.optim.Adam(policy.parameters(), lr=LG.LR)
        baseline = 0.0
        for _ in range(LG.UPDATES):
            returns, logp, _done = LG.rollout_batch(policy, LG.BATCH,
                                                    train=True)
            adv = returns.detach() - baseline
            baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
            loss = -(logp * adv).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        ev = LG.eval_policy(policy)
        repro_err = float(np.max(np.abs(
            ev["episode_side_openness_curve"]
            - np.array(stored["seeds"][str(i)]["side_openness_curve"]))))
        xs, vs, ats = grip_traces(policy, GRIP_EPISODES)
        k, disc = crossfit_openness(xs, vs, ats)
        adj = adjudicate(range(LG.MAX_STEPS), disc * math.log2(3))
        untrained = LG.GripPolicy()
        xs0, vs0, ats0 = grip_traces(untrained, GRIP_EPISODES)
        ku, untr = crossfit_openness(xs0, vs0, ats0)
        adj_untr = adjudicate(range(LG.MAX_STEPS), untr * math.log2(3))
        rows[str(i)] = {
            "repro_max_abs_err": repro_err,
            "k_discovered": k,
            "declared_b5": b5["seeds"][str(i)]["adj"]["b5_onset"],
            "declared_t_star": b5["seeds"][str(i)]["adj"].get(
                "hinge", {}).get("t_star"),
            "discovered_adj": adj,
            "control_adj": {"b5_onset": adj_untr["b5_onset"],
                            "verdict": adj_untr.get("verdict"), "k": ku},
            "discovered_curve": [round(float(v), 5) for v in disc],
        }
        h = adj.get("hinge", {})
        print(f"[grip-v2] seed={i} repro_err={repro_err:.2e} k={k} "
              f"declared_B5={rows[str(i)]['declared_b5']} "
              f"disc_B5={adj['b5_onset']} dBIC={h.get('delta_bic')} "
              f"t*={h.get('t_star')} (declared t*="
              f"{rows[str(i)]['declared_t_star']}) "
              f"control_B5={adj_untr['b5_onset']}", flush=True)

    agree = sum(r["discovered_adj"]["b5_onset"] == r["declared_b5"]
                for r in rows.values())
    tstar_ok, tstar_n = 0, 0
    for r in rows.values():
        if r["discovered_adj"]["b5_onset"] and r["declared_b5"]:
            tstar_n += 1
            if abs(r["discovered_adj"]["hinge"]["t_star"]
                   - r["declared_t_star"]) <= 5:
                tstar_ok += 1
    controls_clean = sum(not r["control_adj"]["b5_onset"]
                         for r in rows.values())
    outcomes = {
        "RD2b_verdict_agreement": f"{agree}/5",
        "RD2b_pass": bool(agree >= 4),
        "RD3b_t_star_within_5": f"{tstar_ok}/{tstar_n}",
        "RD3b_pass": bool(tstar_n == 0 or tstar_ok == tstar_n),
        "RD4b_controls_clean": f"{controls_clean}/5",
        "RD4b_pass": bool(controls_clean == 5),
    }
    report = {
        "status": ("REGIME-DISCOVERY v2; cross-fitted kNN repairs the v1 "
                   "self-neighbour leakage on the grip realization axis; "
                   "clustering, features and thresholds identical to v1; "
                   "predictions RD2b-RD4b frozen in the docstring"),
        "config": {"grip_episodes": GRIP_EPISODES, "knn": KNN,
                   "cluster_seed": CLUSTER_SEED, "crossfit": "index parity"},
        "results": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "regime_discovery_audit2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
