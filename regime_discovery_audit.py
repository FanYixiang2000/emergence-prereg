"""REGIME-DISCOVERY audit: machine-discovered regime variables vs declared.

Addresses the analyst-freedom question directly: are the certified
verdicts a property of the systems, or of the analyst's choice of
regime object? For the three tabular learned systems (convention,
roles, grip) we rerun training with byte-identical seeds, record raw
episode events (no semantic labels), let k-means with silhouette-chosen
k discover a regime variable from those events, compute the openness of
the discovered variable, and adjudicate it with the frozen B5 detector.
One identical recipe across all three systems; no per-system tuning.

Recipe (fixed before running):
  formation axis (convention, roles): at every stored evaluation
    checkpoint, sample 512 raw episode records with an RNG that is
    independent of the training stream (numpy PCG64); featurize as
    one-hot event vectors (convention: meaning ++ symbol ++ guess,
    15 dims; roles: joint role vector, 36 dims); pool episodes across
    all checkpoints of a seed; k-means (seed 0, k by silhouette over
    2..10 on a 4,000-episode subsample); discovered openness at a
    checkpoint = normalized entropy of its cluster histogram.
  realization axis (grip): after the stored evaluation, roll out 2,048
    fresh episodes; featurize each episode by its raw position trace
    (80 dims); k-means as above (k over 2..8); discovered openness at
    step t = mean over episodes of the normalized entropy of a 25-NN
    estimate of P(cluster | x_t, v_t, att_t).
  adjudication: the frozen B5 detector (ant_fine_onset.adjudicate),
    applied exactly as in the primary experiments.
  controls: the identical pipeline applied to the untrained
    (update-zero) population/policy of every seed.

Registered predictions (frozen before the run):
  RD1 reproduction: the rerun reproduces every stored declared-object
      openness curve to within 1e-4 per point (5+5+5 seeds).
  RD2 verdict agreement: the discovered-regime B5 verdict equals the
      stored declared-object verdict in >= 12/15 seeds.
  RD3 breakpoint agreement: wherever both certify onset,
      |t*_discovered - t*_declared| <= 50 updates (formation) or
      <= 5 steps (realization).
  RD4 controls: 0/15 untrained controls certify onset.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from ant_fine_onset import adjudicate
import learn_convention as LC
import learn_roles as LR
import learn_grip_transport as LG

OUTPUTS = Path(__file__).resolve().parent / "outputs"
EPISODES = 512
GRIP_EPISODES = 2048
KNN = 25
CLUSTER_SEED = 0
SIL_SUBSAMPLE = 4000
EVAL_SEED = 990_101


def choose_k(feats: np.ndarray, kmax: int) -> int:
    rng = np.random.default_rng(CLUSTER_SEED)
    idx = rng.choice(len(feats), min(SIL_SUBSAMPLE, len(feats)), replace=False)
    sub = feats[idx]
    best_k, best_s = 2, -1.0
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=CLUSTER_SEED).fit(sub)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(sub, km.labels_)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def cluster_openness(feats: np.ndarray, groups: np.ndarray, kmax: int):
    """Cluster pooled episode features; entropy of the per-group histogram."""
    k = choose_k(feats, kmax)
    km = KMeans(n_clusters=k, n_init=10, random_state=CLUSTER_SEED).fit(feats)
    labels = km.labels_
    curve = []
    for g in range(groups.max() + 1):
        hist = np.bincount(labels[groups == g], minlength=k).astype(float)
        p = hist / hist.sum()
        nz = p[p > 0]
        curve.append(float(-(nz * np.log2(nz)).sum() / math.log2(k)))
    return k, np.array(curve)


def convention_episodes(speak, listen, rng):
    with torch.no_grad():
        sp = torch.softmax(speak, dim=-1).numpy()
        li = torch.softmax(listen, dim=-1).numpy()
    n, K = sp.shape[0], sp.shape[1]
    s_idx = rng.integers(0, n, EPISODES)
    l_idx = (s_idx + rng.integers(1, n, EPISODES)) % n
    m = rng.integers(0, K, EPISODES)
    feats = np.zeros((EPISODES, 3 * K), dtype=float)
    for e in range(EPISODES):
        sym = rng.choice(K, p=sp[s_idx[e], m[e]])
        guess = rng.choice(K, p=li[l_idx[e], sym])
        feats[e, m[e]] = 1.0
        feats[e, K + sym] = 1.0
        feats[e, 2 * K + guess] = 1.0
    return feats


def role_episodes(logits, rng):
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1).numpy()
    n, R = p.shape
    feats = np.zeros((EPISODES, n * R), dtype=float)
    for e in range(EPISODES):
        for i in range(n):
            feats[e, i * R + rng.choice(R, p=p[i])] = 1.0
    return feats


def formation_audit(system: str):
    stored = json.loads((OUTPUTS / f"learn_{system}.json").read_text())
    mod = LC if system == "convention" else LR
    grid = mod.GRID
    rows = {}
    for i in range(mod.N_SEEDS):
        seed = mod.SEED + i * 101
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(EVAL_SEED + seed)
        if system == "convention":
            speak = torch.zeros((mod.N_AGENTS, mod.K, mod.K), requires_grad=True)
            listen = torch.zeros((mod.N_AGENTS, mod.K, mod.K), requires_grad=True)
            params = [speak, listen]
        else:
            logits = torch.zeros((mod.N_AGENTS, mod.R), requires_grad=True)
            params = [logits]
        opt = torch.optim.Adam(params, lr=mod.LR)
        baseline = 0.0
        gen = torch.Generator().manual_seed(seed) if system == "convention" else None
        declared, feats_all, groups = [], [], []
        untr_feats, untr_groups = [], []
        rng0 = np.random.default_rng(EVAL_SEED + seed + 7)
        for u in range(mod.UPDATES + 1):
            if u % mod.EVAL_EVERY == 0:
                g = u // mod.EVAL_EVERY
                if system == "convention":
                    declared.append(LC.convention_openness(speak))
                    feats_all.append(convention_episodes(speak, listen, rng))
                    untr_feats.append(convention_episodes(
                        torch.zeros_like(speak), torch.zeros_like(listen), rng0))
                else:
                    declared.append(LR.openness(logits))
                    feats_all.append(role_episodes(logits, rng))
                    untr_feats.append(role_episodes(
                        torch.zeros_like(logits), rng0))
                groups.append(np.full(EPISODES, g))
                untr_groups.append(np.full(EPISODES, g))
            if u == mod.UPDATES:
                break
            if system == "convention":
                s_idx = torch.randint(0, mod.N_AGENTS, (mod.BATCH,), generator=gen)
                shift = torch.randint(1, mod.N_AGENTS, (mod.BATCH,), generator=gen)
                l_idx = (s_idx + shift) % mod.N_AGENTS
                m = torch.randint(0, mod.K, (mod.BATCH,), generator=gen)
                sp_dist = torch.distributions.Categorical(logits=speak[s_idx, m])
                sym = sp_dist.sample()
                li_dist = torch.distributions.Categorical(logits=listen[l_idx, sym])
                guess = li_dist.sample()
                r = (guess == m).float()
                adv = r - baseline
                baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
                loss = -(adv.detach() * (sp_dist.log_prob(sym)
                                         + li_dist.log_prob(guess))).mean()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                roles = dist.sample((mod.BATCH,))
                covered = torch.zeros((mod.BATCH, mod.R))
                covered.scatter_(1, roles, 1.0)
                r = (covered.sum(dim=1) == mod.R).float()
                adv = r - baseline
                baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
                loss = -(adv.detach() * dist.log_prob(roles).sum(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        stored_curve = np.array(stored["seeds"][str(i)]["openness_curve"])
        repro_err = float(np.max(np.abs(np.array(declared) - stored_curve)))
        k, disc = cluster_openness(np.vstack(feats_all),
                                   np.concatenate(groups), kmax=10)
        adj = adjudicate(grid, disc * math.log2(3))
        ku, untr = cluster_openness(np.vstack(untr_feats),
                                    np.concatenate(untr_groups), kmax=10)
        adj_untr = adjudicate(grid, untr * math.log2(3))
        rows[str(i)] = {
            "repro_max_abs_err": repro_err,
            "k_discovered": k,
            "declared_b5": stored["seeds"][str(i)]["adj"]["b5_onset"],
            "declared_t_star": stored["seeds"][str(i)]["adj"].get(
                "hinge", {}).get("t_star"),
            "discovered_adj": adj,
            "control_adj": {"b5_onset": adj_untr["b5_onset"],
                            "verdict": adj_untr.get("verdict"),
                            "k": ku},
            "discovered_curve": [round(float(v), 5) for v in disc],
        }
        h = adj.get("hinge", {})
        print(f"[{system}] seed={i} repro_err={repro_err:.2e} k={k} "
              f"declared_B5={rows[str(i)]['declared_b5']} "
              f"disc_B5={adj['b5_onset']} dBIC={h.get('delta_bic')} "
              f"t*={h.get('t_star')} (declared t*="
              f"{rows[str(i)]['declared_t_star']}) "
              f"control_B5={adj_untr['b5_onset']}", flush=True)
    return rows


def grip_traces(policy, episodes):
    with torch.no_grad():
        x = torch.zeros(episodes)
        v = torch.zeros(episodes)
        att = torch.zeros(episodes)
        xs, vs, ats = [], [], []
        for _ in range(LG.MAX_STEPS):
            obs = torch.stack([x / LG.GOAL, v, att], dim=1)
            probs = torch.softmax(policy(obs), dim=-1)
            counts = torch.distributions.Multinomial(
                total_count=LG.N_AGENTS, probs=probs).sample()
            grip_frac = counts[:, 2] / LG.N_AGENTS
            att = torch.clamp(att + LG.GRIP_GAIN * grip_frac - LG.GRIP_DECAY,
                              0.0, 1.0)
            force = counts[:, 1] - counts[:, 0]
            active = (att >= LG.GRIP_MIN) & (torch.abs(force) >= LG.THRESHOLD)
            v = LG.DAMP * v + active.float() * LG.ACCEL * torch.sign(force)
            x = torch.clamp(x + v, -LG.GOAL, LG.GOAL)
            xs.append(x.clone())
            vs.append(v.clone())
            ats.append(att.clone())
    return (torch.stack(xs, dim=1).numpy(), torch.stack(vs, dim=1).numpy(),
            torch.stack(ats, dim=1).numpy())


def grip_discovered_openness(xs, vs, ats):
    k = choose_k(xs, kmax=8)
    km = KMeans(n_clusters=k, n_init=10, random_state=CLUSTER_SEED).fit(xs)
    labels = km.labels_
    curve = []
    for t in range(LG.MAX_STEPS):
        state = np.stack([xs[:, t], vs[:, t], ats[:, t]], axis=1)
        knn = KNeighborsClassifier(n_neighbors=KNN).fit(state, labels)
        proba = knn.predict_proba(state)
        nz_h = -(np.where(proba > 0, proba * np.log2(np.clip(proba, 1e-12, 1)),
                          0.0)).sum(axis=1)
        curve.append(float(nz_h.mean() / math.log2(k)))
    return k, np.array(curve)


def grip_audit():
    stored = json.loads((OUTPUTS / "learn_grip_transport.json").read_text())
    b5 = json.loads((OUTPUTS / "learn_grip_transport_b5.json").read_text())
    rows = {}
    for i in range(LG.N_SEEDS):
        seed = LG.SEED + i * 101
        torch.manual_seed(seed)
        np.random.seed(seed)
        policy = LG.GripPolicy()
        opt = torch.optim.Adam(policy.parameters(), lr=LG.LR)
        baseline = 0.0
        for _ in range(LG.UPDATES):
            returns, logp, _done = LG.rollout_batch(policy, LG.BATCH, train=True)
            adv = returns.detach() - baseline
            baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
            loss = -(logp * adv).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        ev = LG.eval_policy(policy)
        stored_curve = np.array(stored["seeds"][str(i)]["side_openness_curve"])
        repro_err = float(np.max(np.abs(
            ev["episode_side_openness_curve"] - stored_curve)))
        xs, vs, ats = grip_traces(policy, GRIP_EPISODES)
        k, disc = grip_discovered_openness(xs, vs, ats)
        adj = adjudicate(range(LG.MAX_STEPS), disc * math.log2(3))
        untrained = LG.GripPolicy()  # fresh init, never trained
        xs0, vs0, ats0 = grip_traces(untrained, GRIP_EPISODES)
        ku, untr = grip_discovered_openness(xs0, vs0, ats0)
        adj_untr = adjudicate(range(LG.MAX_STEPS), untr * math.log2(3))
        rows[str(i)] = {
            "repro_max_abs_err": repro_err,
            "k_discovered": k,
            "declared_b5": b5["seeds"][str(i)]["adj"]["b5_onset"],
            "declared_t_star": b5["seeds"][str(i)]["adj"].get(
                "hinge", {}).get("t_star"),
            "discovered_adj": adj,
            "control_adj": {"b5_onset": adj_untr["b5_onset"],
                            "verdict": adj_untr.get("verdict"),
                            "k": ku},
            "discovered_curve": [round(float(v), 5) for v in disc],
        }
        h = adj.get("hinge", {})
        print(f"[grip] seed={i} repro_err={repro_err:.2e} k={k} "
              f"declared_B5={rows[str(i)]['declared_b5']} "
              f"disc_B5={adj['b5_onset']} dBIC={h.get('delta_bic')} "
              f"t*={h.get('t_star')} (declared t*="
              f"{rows[str(i)]['declared_t_star']}) "
              f"control_B5={adj_untr['b5_onset']}", flush=True)
    return rows


def main() -> None:
    torch.set_num_threads(4)
    results = {"convention": formation_audit("convention"),
               "roles": formation_audit("roles"),
               "grip": grip_audit()}

    repro_ok = all(r["repro_max_abs_err"] <= 1e-4
                   for sys_rows in results.values()
                   for r in sys_rows.values())
    cells = [(s, k, r) for s, sys_rows in results.items()
             for k, r in sys_rows.items()]
    agree = sum(r["discovered_adj"]["b5_onset"] == r["declared_b5"]
                for _s, _k, r in cells)
    tstar_ok, tstar_n = 0, 0
    for s, _k, r in cells:
        if r["discovered_adj"]["b5_onset"] and r["declared_b5"]:
            tstar_n += 1
            tol = 5 if s == "grip" else 50
            if abs(r["discovered_adj"]["hinge"]["t_star"]
                   - r["declared_t_star"]) <= tol:
                tstar_ok += 1
    controls_clean = sum(not r["control_adj"]["b5_onset"]
                         for _s, _k, r in cells)
    outcomes = {
        "RD1_reproduction": bool(repro_ok),
        "RD2_verdict_agreement": f"{agree}/15",
        "RD2_pass": bool(agree >= 12),
        "RD3_t_star_within_tol": f"{tstar_ok}/{tstar_n}",
        "RD3_pass": bool(tstar_n == 0 or tstar_ok == tstar_n),
        "RD4_controls_clean": f"{controls_clean}/15",
        "RD4_pass": bool(controls_clean == 15),
    }
    report = {
        "status": ("REGIME-DISCOVERY audit; machine-discovered regime "
                   "variables (k-means on raw episode records, k by "
                   "silhouette, one recipe across systems) adjudicated "
                   "with the frozen B5 detector; predictions RD1-RD4 "
                   "frozen in the docstring before running"),
        "config": {"episodes_per_checkpoint": EPISODES,
                   "grip_episodes": GRIP_EPISODES, "knn": KNN,
                   "cluster_seed": CLUSTER_SEED,
                   "silhouette_subsample": SIL_SUBSAMPLE,
                   "eval_seed": EVAL_SEED},
        "results": results,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "regime_discovery_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
