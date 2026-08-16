"""RDC: discovered-regime controllability race.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Training, intervention grid and kick
are byte-identical to LEARN-GRIP-UTILITY; the discovery recipe is
byte-identical to the REGIME-DISCOVERY audit's grip arm. The question
is functional: with the analyst-declared side variable removed, does
the machine-discovered regime variable still predict whether an
intervention can switch the outcome?
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from learn_grip_transport import (ACCEL, DAMP, GOAL, GRIP_DECAY, GRIP_GAIN,
                                  GRIP_MIN, MAX_STEPS, N_AGENTS, N_SEEDS,
                                  SEED, THRESHOLD, side_openness)
from learn_grip_utility import EVAL_BATCH, KICK_V, KICK_X, TAUS, train
from learn_transport_eq_utility import auc, rank_corr
from regime_discovery_audit import CLUSTER_SEED, GRIP_EPISODES, KNN, choose_k
from regime_discovery_audit import grip_traces

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LOG2_3 = math.log2(3)


def knn_openness(knn, k, state):
    proba = knn.predict_proba(state)
    h = -(np.where(proba > 0,
                   proba * np.log2(np.clip(proba, 1e-12, 1)), 0.0)).sum(axis=1)
    return h / math.log2(k) if k > 1 else h


def intervention_eval(policy, tau, seed, knn, k):
    gen = torch.Generator().manual_seed(seed)
    batch = EVAL_BATCH
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    att = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    incipient = torch.zeros(batch)
    preds = {}
    for t in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v, att], dim=1)
        with torch.no_grad():
            probs = torch.softmax(policy(obs), dim=-1)
        if t == tau:
            state_side = torch.sign(x + 0.5 * v)
            rand_side = torch.where(
                torch.rand(batch, generator=gen) < 0.5,
                -torch.ones(batch), torch.ones(batch))
            incipient = torch.where(state_side != 0, state_side, rand_side)
            state = np.stack([x.numpy(), v.numpy(), att.numpy()], axis=1)
            preds["disc_open"] = knn_openness(knn, k, state)
            preds["side_open"] = side_openness(probs).numpy()
            ent = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=1)
            preds["pol_ent"] = (ent / LOG2_3).numpy()
            preds["absx"] = (torch.abs(x) / GOAL).numpy()
            preds["absv"] = torch.abs(v).numpy()
            preds["att"] = att.numpy().copy()
            x = torch.clamp(x - KICK_X * incipient, -GOAL, GOAL)
            v = v - KICK_V * incipient
            obs = torch.stack([x / GOAL, v, att], dim=1)
            with torch.no_grad():
                probs = torch.softmax(policy(obs), dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS,
                                               probs=probs)
        counts = dist.sample()
        grip_frac = counts[:, 2] / N_AGENTS
        att = torch.clamp(att + GRIP_GAIN * grip_frac - GRIP_DECAY, 0.0, 1.0)
        force = counts[:, 1] - counts[:, 0]
        active = (att >= GRIP_MIN) & (torch.abs(force) >= THRESHOLD)
        v = DAMP * v + active.float() * ACCEL * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        done = done | (torch.abs(x) >= GOAL - 1e-6)
    final_side = torch.sign(x)
    switch = ((final_side != 0) & (final_side != incipient)).numpy()
    preds["switch"] = switch.astype(float)
    return preds


def main() -> None:
    torch.set_num_threads(8)
    names = ("disc_open", "side_open", "pol_ent", "absx", "absv", "att")
    pool = {n: [] for n in names + ("switch", "tau")}
    per_tau_fixed = {str(t): {"disc": [], "side": [], "switch": []}
                     for t in TAUS}
    ks = []
    for i in range(N_SEEDS):
        seed = SEED + i * 101
        policy = train(seed)
        xs, vs, ats = grip_traces(policy, GRIP_EPISODES)
        k = choose_k(xs, kmax=8)
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=CLUSTER_SEED).fit(xs)
        ks.append(k)
        for tau in TAUS:
            state_tau = np.stack([xs[:, tau], vs[:, tau], ats[:, tau]],
                                 axis=1)
            knn = KNeighborsClassifier(n_neighbors=KNN).fit(state_tau,
                                                            km.labels_)
            row = intervention_eval(policy, tau, seed + tau, knn, k)
            for n in names:
                pool[n].extend(row[n].tolist())
            pool["switch"].extend(row["switch"].tolist())
            pool["tau"].extend([float(tau)] * EVAL_BATCH)
            ft = per_tau_fixed[str(tau)]
            ft["disc"].extend(row["disc_open"].tolist())
            ft["side"].extend(row["side_open"].tolist())
            ft["switch"].extend(row["switch"].tolist())
        print(f"seed {i}: k={k} pooled so far n={len(pool['switch'])}",
              flush=True)

    switch = np.array(pool["switch"])
    race = {}
    for name, sign in (("disc_open", 1.0), ("side_open", 1.0),
                       ("pol_ent", 1.0), ("absx", -1.0), ("absv", -1.0),
                       ("att", -1.0), ("tau", -1.0)):
        vals = sign * np.array(pool[name])
        race[name] = {"rank_corr": round(rank_corr(vals, switch), 5),
                      "auc": round(auc(vals, switch), 5)}

    fixed = {}
    for t in TAUS:
        ft = per_tau_fixed[str(t)]
        sw = np.array(ft["switch"])
        n_pos, n_neg = int(sw.sum()), int(len(sw) - sw.sum())
        entry = {"n_switch": n_pos, "n_hold": n_neg}
        if min(n_pos, n_neg) >= 20:
            entry["auc_disc"] = round(auc(np.array(ft["disc"]), sw), 5)
            entry["auc_side"] = round(auc(np.array(ft["side"]), sw), 5)
        fixed[str(t)] = entry

    eligible = [v for v in fixed.values() if "auc_disc" in v]
    outcomes = {
        "k_discovered": ks,
        "RDC1_pooled_auc_disc": race["disc_open"]["auc"],
        "RDC1_pass": bool(race["disc_open"]["auc"] >= 0.80),
        "RDC2_fixed_tau_aucs_disc": {t: v.get("auc_disc")
                                     for t, v in fixed.items()},
        "RDC2_pass": bool(eligible
                          and all(v["auc_disc"] >= 0.80 for v in eligible)),
        "RDC3_pass": bool(race["disc_open"]["auc"] > race["tau"]["auc"]),
        "RDC4_side_vs_entropy": (race["side_open"]["auc"],
                                 race["pol_ent"]["auc"]),
        "RDC4_pass": bool(race["side_open"]["auc"]
                          > race["pol_ent"]["auc"]),
        "race": race,
        "fixed_tau": fixed,
    }
    report = {"status": ("RDC discovered-regime controllability race; "
                         "recipe and grid byte-identical to prior "
                         "audits; registered in METHOD_BASELINE_"
                         "FIXEDTIME_PREREGISTRATION.md before run"),
              "config": {"taus": TAUS, "eval_batch": EVAL_BATCH,
                         "grip_episodes": GRIP_EPISODES, "knn": KNN,
                         "cluster_seed": CLUSTER_SEED},
              "registered_outcomes": outcomes}
    out = OUTPUTS / "learn_grip_discovery_utility.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
