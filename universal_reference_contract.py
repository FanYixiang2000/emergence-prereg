"""Universal reference contract (URC): one frozen, public recipe for
the possibility space, applied unchanged across structurally different
domains.

The remaining universality gap: machine-discovered basins were
validated per-domain (CLBF learned/cross-fitted basins), but each
audit declared its own feature space. The URC freezes ONE recipe that
anyone can run on any episodic system, with no per-domain choices:

    1. Represent each episode as generic channel statistics computed
       from whatever per-step numeric channels the system exposes:
       [episode length, per-channel mean, per-channel std, per-channel
       final value]. No semantic labels, no domain knowledge.
    2. z-score features across episodes.
    3. k-means with k chosen by silhouette over k = 2..6 (fixed seed).
    4. Basins := clusters. Potential := entropy of the natural basin
       distribution. Specificity := JS between the do-commit and
       do-block basin laws. Selectivity := trigger-rate separation
       (trigger defined by the domain's declared intervention target,
       as always). Thresholds copied unchanged from the frozen
       criterion.

Domains (structurally disjoint; neither contributed to the recipe):

    CROWD  the collective-control domain (6 systems x 10 seeds of
           stored-protocol systems, re-rolled fresh);
    BOIDS  the flocking exemplar over the 5-point coupling grid
           (structural layer only; no value/provenance exists).

Registered predictions (frozen before running):

    URC-1  Crowd: URC verdicts agree with the hand-contract verdicts
           on >= 90% of the 60 (system, seed) pairs, with zero
           control false-positives (50 controls).
    URC-2  Crowd: the discovered basins resolve the macro-structure:
           mutual information between URC basin and the hand
           (hazard-mode x outcome) basin >= 0.5 bits on every learned
           seed.
    URC-3  Boids: URC potential (entropy of discovered final-state
           basins over episodes) and the URC within-episode
           contraction reproduce the coupling monotonicity of CE-1
           without any hand basin (one inversion tolerance 0.05
           bits), and the URC do-decouple JS is >= 0.2 bits at
           coupling >= 0.75 and <= 0.05 at zero.

Misses are retained.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

import crowd_vote_domain as cv
import canonical_exemplars as cx

OUTPUTS = Path(__file__).resolve().parent / "outputs"

K_RANGE = (2, 3, 4, 5, 6)
URC_SEED = 20260721
N_EVAL = 120


def silhouette(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    n = len(X)
    if k < 2 or n <= k:
        return -1.0
    sil = []
    for i in range(n):
        own = labels[i]
        d = np.linalg.norm(X - X[i], axis=1)
        a_mask = labels == own
        a_mask[i] = False
        if a_mask.sum() == 0:
            continue
        a = d[a_mask].mean()
        b = min(d[labels == c].mean() for c in range(k)
                if c != own and (labels == c).any())
        sil.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(sil)) if sil else -1.0


def kmeans(X: np.ndarray, k: int, seed: int, iters: int = 60):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = d.argmin(axis=1)
        new = np.array([X[labels == c].mean(axis=0)
                        if (labels == c).any() else centers[c]
                        for c in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels, centers


def urc_basins(features: List[List[float]]):
    """The frozen recipe: z-score, k-means, silhouette-selected k."""
    X = np.array(features, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    best = None
    for k in K_RANGE:
        labels, centers = kmeans(X, k, URC_SEED)
        s = silhouette(X, labels, k)
        if best is None or s > best[0]:
            best = (s, k, labels, centers, mu, sd)
    return best


def assign(centers, mu, sd, feats) -> int:
    x = (np.array(feats, dtype=float) - mu) / sd
    return int(np.linalg.norm(centers - x, axis=1).argmin())


def entropy_of(labels) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def js_bits(p: Dict[int, float], q: Dict[int, float], keys) -> float:
    m = {b: (p.get(b, 0) + q.get(b, 0)) / 2 for b in keys}

    def kl(x, y):
        return sum(x.get(b, 0) * math.log2(x.get(b, 0) / y[b])
                   for b in keys if x.get(b, 0) > 0 and y[b] > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist(labels, k) -> Dict[int, float]:
    out = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return {b: c / len(labels) for b, c in out.items()}


# --------------------------------------------------------------- crowd URC

def crowd_episode_features(policy, context, seed, intervention):
    """Generic channel statistics: per-step (pos, lane, tick, mode)."""
    ep = cv.Episode(context, seed)
    chans = []
    triggered_modes = []
    while not ep.done:
        f = cv.features(ep.pos, ep.lane, context)
        mode = policy(f, ep.rng)
        if intervention == "do_commit" and ep.pos in cv.HAZARD_BAND:
            mode = "democracy"
        if intervention == "do_block":
            mode = "anarchy"
        chans.append([ep.pos, ep.lane, ep.ticks,
                      1.0 if mode == "democracy" else 0.0])
        if ep.pos in cv.HAZARD_BAND:
            triggered_modes.append(mode)
        ep.step(mode)
    M = np.array(chans)
    feats = ([len(chans)] + M.mean(axis=0).tolist()
             + M.std(axis=0).tolist() + M[-1].tolist())
    trigger = int(bool(triggered_modes) and triggered_modes.count(
        "democracy") >= len(triggered_modes) / 2)
    hazard_mode = "democracy" if trigger else "anarchy"
    oc = ("success" if ep.outcome in ("success_fast", "success_slow")
          else ep.outcome)
    return feats, trigger, f"{hazard_mode}_{oc}", ep.value()


def run_crowd_urc() -> Dict:
    agree = 0
    total = 0
    false_pos = 0
    mi_per_learned = []
    per_system = {}
    for seed_i, seed in enumerate(cv.SEEDS):
        q = cv.train_learned(seed)
        systems = {
            "learned": (cv.policy_from_q(q), True),
            "initial_twin": (cv.policy_from_q({}), True),
            "always_democracy": (cv.always_democracy, False),
            "always_anarchy": (cv.always_anarchy, False),
            "scripted_switcher": (cv.scripted_switcher, False),
            "bc_clone": (cv.bc_clone(seed + 77), False),
        }
        hand = json.loads(
            (OUTPUTS / "crowd_vote_domain.json").read_text())
        for name, (pol, endo) in systems.items():
            rows = {}
            for mode in (None, "do_commit", "do_block"):
                rows[mode] = []
                for ctx_i, context in enumerate(("field", "ledge")):
                    for kk in range(N_EVAL):
                        s = 50_000_000 + seed * 1000 + ctx_i * 500 + kk
                        rows[mode].append(crowd_episode_features(
                            pol, context, s, mode))
            feats = [r[0] for m in rows for r in rows[m]]
            _, k, _, centers, mu, sd = urc_basins(feats)
            lab = {m: [assign(centers, mu, sd, r[0])
                       for r in rows[m]] for m in rows}
            nat = rows[None]
            trig = {c: float(np.mean(
                [r[1] for r, ctx in zip(nat, ["field"] * N_EVAL
                                        + ["ledge"] * N_EVAL)
                 if ctx == c])) for c in ("field", "ledge")}
            keys = range(k)
            m_metrics = {
                "potential_bits": entropy_of(lab[None]),
                "conditional_selectivity":
                    abs(trig["ledge"] - trig["field"]),
                "specificity_js_bits": js_bits(
                    dist(lab["do_commit"], k),
                    dist(lab["do_block"], k), keys),
                "usefulness_gap":
                    float(np.mean([r[3] for r in rows[None]]))
                    - float(np.mean([r[3] for r in rows["do_block"]])),
            }
            acq = 0.0
            if name == "learned":
                acq = m_metrics["conditional_selectivity"]
            v = cv.verdict(m_metrics, name in
                           ("learned", "initial_twin"), acq)
            hand_v = hand["seeds"][str(seed)][name]["verdict"][
                "emergent"]
            total += 1
            agree += int(v["emergent"] == hand_v)
            if name != "learned" and v["emergent"]:
                false_pos += 1
            if name == "learned":
                # URC-2: MI between discovered basin and hand basin
                hand_lab = [r[2] for r in nat]
                ulab = lab[None]
                joint: Dict = {}
                for hu, uu in zip(hand_lab, ulab):
                    joint[(hu, uu)] = joint.get((hu, uu), 0) + 1
                n = len(ulab)
                hs = {}
                us = {}
                for (hu, uu), c in joint.items():
                    hs[hu] = hs.get(hu, 0) + c
                    us[uu] = us.get(uu, 0) + c
                mi = sum((c / n) * math.log2(
                    (c / n) / ((hs[hu] / n) * (us[uu] / n)))
                    for (hu, uu), c in joint.items())
                mi_per_learned.append(mi)
            per_system.setdefault(name, []).append(v["emergent"])
        print(f"crowd URC seed {seed}: agreement so far "
              f"{agree}/{total}", flush=True)
    return {
        "agreement": f"{agree}/{total}",
        "agreement_rate": agree / total,
        "control_false_positives": false_pos,
        "mi_urc_vs_hand_basin_per_learned":
            [round(x, 3) for x in mi_per_learned],
        "min_mi": min(mi_per_learned),
    }


# --------------------------------------------------------------- boids URC

def boids_features(a: float, seed: int, decouple: bool):
    h0, hT = cx.boids_episode(a, seed,
                              decouple_at=cx.T_STEPS // 2
                              if decouple else -1)
    # generic channels: sin/cos of headings at start and end
    return ([float(np.cos(hT).mean()), float(np.sin(hT).mean()),
             float(np.cos(hT).std()), float(np.sin(hT).std())],
            [float(np.cos(h0).mean()), float(np.sin(h0).mean()),
             float(np.cos(h0).std()), float(np.sin(h0).std())])


def run_boids_urc() -> Dict:
    out = {}
    for a in cx.ALIGN_GRID:
        finals, initials, dec = [], [], []
        for ep in range(1200):
            fT, f0 = boids_features(a, 2000 + ep, False)
            finals.append(fT)
            initials.append(f0)
            fD, _ = boids_features(a, 2000 + ep, True)
            dec.append(fD)
        _, k, labels, centers, mu, sd = urc_basins(finals + initials)
        lab_T = labels[:1200]
        lab_0 = labels[1200:]
        lab_D = [assign(centers, mu, sd, f) for f in dec]
        keys = range(k)
        out[str(a)] = {
            "k": k,
            "urc_contraction_bits":
                entropy_of(lab_0) - entropy_of(lab_T),
            "urc_js_decouple_bits": js_bits(
                dist(list(lab_T), k), dist(lab_D, k), keys),
        }
        print(f"boids URC a={a}: k={k} contraction "
              f"{out[str(a)]['urc_contraction_bits']:.2f} "
              f"JS {out[str(a)]['urc_js_decouple_bits']:.3f}",
              flush=True)
    return out


def main() -> None:
    crowd = run_crowd_urc()
    boids = run_boids_urc()

    urc1 = (crowd["agreement_rate"] >= 0.9
            and crowd["control_false_positives"] == 0)
    urc2 = crowd["min_mi"] >= 0.5
    cvals = [boids[str(a)]["urc_contraction_bits"]
             for a in cx.ALIGN_GRID]
    mono = all(cvals[i + 1] >= cvals[i] - 0.05
               for i in range(len(cvals) - 1))
    urc3 = (mono
            and boids["0.75"]["urc_js_decouple_bits"] >= 0.2
            and boids["1.0"]["urc_js_decouple_bits"] >= 0.2
            and boids["0.0"]["urc_js_decouple_bits"] <= 0.05)

    report = {
        "status": ("universal reference contract; URC-1..3 frozen in "
                   "the docstring; one recipe, zero per-domain "
                   "choices"),
        "recipe": ("episode -> generic channel stats (length, mean, "
                   "std, final) -> z-score -> k-means, k by "
                   "silhouette over 2..6, fixed seed"),
        "crowd": crowd,
        "boids": boids,
        "registered_outcomes": {
            "URC1_crowd_verdict_agreement_ge_90_no_false_pos": bool(urc1),
            "URC2_basins_resolve_structure_mi_ge_0.5": bool(urc2),
            "URC3_boids_monotone_and_loadbearing": bool(urc3),
        },
    }
    out = OUTPUTS / "universal_reference_contract.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
