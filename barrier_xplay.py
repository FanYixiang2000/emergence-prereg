"""BARRIER-XPLAY: direct measurement of the joint exploration barrier.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Re-runs
LEARN-CONVENTION and LEARN-ROLES with byte-identical configs and saves
policy snapshots, then measures (1) unilateral adoption gain at the
pre-commitment checkpoint, (2) unilateral deviation cost at the final
checkpoint, (3) cross-seed cross-play compatibility. Determinism check:
final codes/assignments must match the stored positive-result JSONs.
"""
from __future__ import annotations

import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate
import learn_convention as LC
import learn_roles as LR

OUTPUTS = Path(__file__).resolve().parent / "outputs"
PERMS_K = list(permutations(range(LC.K)))       # 120 codes
PERMS_R = LR.PERMS                              # 720 role permutations


# ---------------------------------------------------------------- convention

def train_convention_with_snapshots(seed: int):
    """Byte-identical replay of learn_convention.run_seed with snapshots."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    speak = torch.zeros((LC.N_AGENTS, LC.K, LC.K), requires_grad=True)
    listen = torch.zeros((LC.N_AGENTS, LC.K, LC.K), requires_grad=True)
    opt = torch.optim.Adam([speak, listen], lr=LC.LR)
    baseline = 0.0
    open_curve, snaps = [], {}
    gen = torch.Generator().manual_seed(seed)
    for u in range(LC.UPDATES + 1):
        if u % LC.EVAL_EVERY == 0:
            open_curve.append(LC.convention_openness(speak))
            snaps[u] = (speak.detach().clone(), listen.detach().clone())
        if u == LC.UPDATES:
            break
        s_idx = torch.randint(0, LC.N_AGENTS, (LC.BATCH,), generator=gen)
        shift = torch.randint(1, LC.N_AGENTS, (LC.BATCH,), generator=gen)
        l_idx = (s_idx + shift) % LC.N_AGENTS
        m = torch.randint(0, LC.K, (LC.BATCH,), generator=gen)
        sp_dist = torch.distributions.Categorical(logits=speak[s_idx, m])
        sym = sp_dist.sample()
        li_dist = torch.distributions.Categorical(logits=listen[l_idx, sym])
        guess = li_dist.sample()
        r = (guess == m).float()
        adv = r - baseline
        baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
        loss = -(adv.detach() * (sp_dist.log_prob(sym)
                                 + li_dist.log_prob(guess))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return open_curve, snaps


def committed_tables(code) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot speak (m->s) and listen (s->m) probability tables."""
    sp = torch.zeros((LC.K, LC.K))
    li = torch.zeros((LC.K, LC.K))
    for m, s in enumerate(code):
        sp[m, s] = 1.0
        li[s, m] = 1.0
    return sp, li


def probe_payoff(sp_i, li_i, sp_pop, li_pop, i: int) -> float:
    """Symmetric expected intelligibility of agent i vs random partner.

    sp_i, li_i: probability tables (K, K) for the probe.
    sp_pop, li_pop: probability tables (N, K, K) for the population.
    """
    n = sp_pop.shape[0]
    mask = torch.ones(n)
    mask[i] = 0.0
    # i speaks, j listens: mean_m sum_s sp_i[m,s] li_j[s,m]
    a = torch.einsum("ms,jsm->j", sp_i, li_pop) / LC.K
    # j speaks, i listens
    b = torch.einsum("jms,sm->j", sp_pop, li_i) / LC.K
    return float((0.5 * (a + b) * mask).sum().item() / (n - 1))


def convention_measures(snaps, adj, open_curve):
    prob = {u: (torch.softmax(s, dim=-1), torch.softmax(l, dim=-1))
            for u, (s, l) in snaps.items()}
    grid = sorted(snaps.keys())
    final = grid[-1]
    sp_f, li_f = prob[final]

    out = {}
    hinge = adj.get("hinge") or {}
    t_star = hinge.get("t_star")
    if adj.get("b5_onset") and t_star is not None:
        pre = min(grid, key=lambda u: abs(u - 0.5 * t_star))
        sp_p, li_p = prob[pre]
        gains = []
        for i in range(LC.N_AGENTS):
            base = probe_payoff(sp_p[i], li_p[i], sp_p, li_p, i)
            best = max(probe_payoff(*committed_tables(c), sp_p, li_p, i)
                       for c in PERMS_K)
            gains.append(best - base)
        out["pre_checkpoint"] = pre
        out["t_star"] = t_star
        out["adoption_gain_pre"] = float(np.mean(gains))

    maj = tuple(int(i) for i in
                torch.softmax(snaps[final][0], -1).mean(0).argmax(-1))
    costs = []
    for i in range(LC.N_AGENTS):
        on_code = probe_payoff(*committed_tables(maj), sp_f, li_f, i)
        best_dev = max(probe_payoff(*committed_tables(c), sp_f, li_f, i)
                       for c in PERMS_K if c != maj)
        costs.append(on_code - best_dev)
    out["majority_code"] = list(maj)
    out["deviation_cost_post"] = float(np.mean(costs))
    out["final_openness"] = open_curve[-1]
    out["final_tables"] = (sp_f, li_f)
    return out


# --------------------------------------------------------------------- roles

def train_roles_with_snapshots(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    logits = torch.zeros((LR.N_AGENTS, LR.R), requires_grad=True)
    opt = torch.optim.Adam([logits], lr=LR.LR)
    baseline = 0.0
    open_curve, snaps = [], {}
    for u in range(LR.UPDATES + 1):
        if u % LR.EVAL_EVERY == 0:
            open_curve.append(LR.openness(logits))
            snaps[u] = logits.detach().clone()
        if u == LR.UPDATES:
            break
        dist = torch.distributions.Categorical(logits=logits)
        roles = dist.sample((LR.BATCH,))
        covered = torch.zeros((LR.BATCH, LR.R))
        covered.scatter_(1, roles, 1.0)
        r = (covered.sum(dim=1) == LR.R).float()
        adv = r - baseline
        baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
        logp = dist.log_prob(roles).sum(dim=1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return open_curve, snaps


def permanent(p: np.ndarray) -> float:
    total = 0.0
    for perm in PERMS_R:
        prod = 1.0
        for i, r in enumerate(perm):
            prod *= p[i, r]
        total += prod
    return float(total)


def roles_measures(snaps, adj):
    grid = sorted(snaps.keys())
    final = grid[-1]
    p_f = torch.softmax(snaps[final], dim=-1).numpy()
    succ_f = permanent(p_f)
    assign = [int(i) for i in p_f.argmax(axis=-1)]

    out = {}
    hinge = adj.get("hinge") or {}
    t_star = hinge.get("t_star")
    if adj.get("b5_onset") and t_star is not None:
        pre = min(grid, key=lambda u: abs(u - 0.5 * t_star))
        p_p = torch.softmax(snaps[pre], dim=-1).numpy()
        succ_p = permanent(p_p)
        gains = []
        for i in range(LR.N_AGENTS):
            best = -1.0
            for r in range(LR.R):
                q = p_p.copy()
                q[i] = 0.0
                q[i, r] = 1.0
                best = max(best, permanent(q))
            gains.append(best - succ_p)
        out["pre_checkpoint"] = pre
        out["t_star"] = t_star
        out["success_pre"] = succ_p
        out["adoption_gain_pre"] = float(np.mean(gains))

    costs = []
    for i in range(LR.N_AGENTS):
        best_dev = -1.0
        for r in range(LR.R):
            if r == assign[i]:
                continue
            q = p_f.copy()
            q[i] = 0.0
            q[i, r] = 1.0
            best_dev = max(best_dev, permanent(q))
        costs.append(succ_f - best_dev)
    out["assignment"] = assign
    out["success_post"] = succ_f
    out["deviation_cost_post"] = float(np.mean(costs))
    out["final_p"] = p_f
    return out


# ----------------------------------------------------------------- crossplay

def convention_crossplay(rows):
    """Intelligibility of speaker-pop A with listener-pop B, ordered pairs."""
    within, cross = [], []
    keys = sorted(rows.keys())
    for a in keys:
        sp_a, li_a = rows[a]["final_tables"]
        within.append(float(
            torch.einsum("ims,jsm->", sp_a, li_a).item()
            / (LC.K * LC.N_AGENTS ** 2)))
        for b in keys:
            if a == b or rows[a]["majority_code"] == rows[b]["majority_code"]:
                continue
            sp_b, li_b = rows[b]["final_tables"]
            v = float(torch.einsum("ims,jsm->", sp_a, li_b).item()
                      / (LC.K * LC.N_AGENTS ** 2))
            cross.append(v)
    return within, cross


def roles_crossplay(rows):
    """Hybrid team success for all proper subsets, distinct-assignment pairs."""
    keys = sorted(rows.keys())
    within = [rows[k]["success_post"] for k in keys]
    hybrid = []
    for ai in range(len(keys)):
        for bi in range(ai + 1, len(keys)):
            a, b = keys[ai], keys[bi]
            if rows[a]["assignment"] == rows[b]["assignment"]:
                continue
            pa, pb = rows[a]["final_p"], rows[b]["final_p"]
            for mask in range(1, 2 ** LR.N_AGENTS - 1):
                q = np.array([pa[i] if (mask >> i) & 1 else pb[i]
                              for i in range(LR.N_AGENTS)])
                hybrid.append(permanent(q))
    return within, hybrid


# ---------------------------------------------------------------------- main

def main() -> None:
    torch.set_num_threads(4)
    ref_c = json.loads((OUTPUTS / "learn_convention.json").read_text())
    ref_r = json.loads((OUTPUTS / "learn_roles.json").read_text())

    conv_rows = {}
    for i in range(LC.N_SEEDS):
        seed = LC.SEED + i * 101
        curve, snaps = train_convention_with_snapshots(seed)
        adj = adjudicate(LC.GRID, np.array(curve) * math.log2(3))
        row = convention_measures(snaps, adj, curve)
        ref_code = ref_c["seeds"][str(i)]["code"]
        row["determinism_ok"] = (row["majority_code"] == ref_code)
        conv_rows[str(i)] = row
        print(f"conv seed={i} det={row['determinism_ok']} "
              f"pre={row.get('pre_checkpoint')} "
              f"gain={row.get('adoption_gain_pre')} "
              f"cost={row['deviation_cost_post']:.4f}", flush=True)

    role_rows = {}
    for i in range(LR.N_SEEDS):
        seed = LR.SEED + i * 101
        curve, snaps = train_roles_with_snapshots(seed)
        adj = adjudicate(LR.GRID, np.array(curve) * math.log2(3))
        row = roles_measures(snaps, adj)
        ref_assign = ref_r["seeds"][str(i)]["assignment"]
        row["determinism_ok"] = (row["assignment"] == ref_assign)
        role_rows[str(i)] = row
        print(f"roles seed={i} det={row['determinism_ok']} "
              f"pre={row.get('pre_checkpoint')} "
              f"gain={row.get('adoption_gain_pre')} "
              f"cost={row['deviation_cost_post']:.4f}", flush=True)

    det_ok = (all(r["determinism_ok"] for r in conv_rows.values())
              and all(r["determinism_ok"] for r in role_rows.values()))

    c_within, c_cross = convention_crossplay(conv_rows)
    r_within, r_hybrid = roles_crossplay(role_rows)

    conv_gain = float(np.mean([r["adoption_gain_pre"] for r in
                               conv_rows.values() if "adoption_gain_pre" in r]))
    conv_cost = float(np.mean([r["deviation_cost_post"]
                               for r in conv_rows.values()]))
    role_gain = float(np.mean([r["adoption_gain_pre"] for r in
                               role_rows.values() if "adoption_gain_pre" in r]))
    role_cost = float(np.mean([r["deviation_cost_post"]
                               for r in role_rows.values()]))

    mc_within, mc_cross = float(np.mean(c_within)), float(np.mean(c_cross))
    mr_within, mr_hybrid = float(np.mean(r_within)), float(np.mean(r_hybrid))

    bx1 = (mc_cross <= 0.35 and mc_within >= 0.80
           and mr_hybrid <= 0.50 * mr_within)
    bx2 = (conv_gain <= 0.10 and role_gain <= 0.10)
    bx3 = (conv_cost >= 0.50 and role_cost >= 0.50)
    bx4 = (conv_cost / max(conv_gain, 0.02) >= 5
           and role_cost / max(role_gain, 0.02) >= 5)

    outcomes = {
        "determinism_ok": det_ok,
        "BX1_regime_exclusivity": bool(bx1),
        "BX2_no_unilateral_gradient_pre": bool(bx2),
        "BX3_lockin_post": bool(bx3),
        "BX4_barrier_asymmetry": bool(bx4),
        "conv_within_intel": round(mc_within, 4),
        "conv_cross_intel": round(mc_cross, 4),
        "conv_adoption_gain_pre": round(conv_gain, 4),
        "conv_deviation_cost_post": round(conv_cost, 4),
        "roles_within_success": round(mr_within, 4),
        "roles_hybrid_success": round(mr_hybrid, 4),
        "roles_adoption_gain_pre": round(role_gain, 4),
        "roles_deviation_cost_post": round(role_cost, 4),
        "n_cross_pairs_conv": len(c_cross),
        "n_hybrid_teams_roles": len(r_hybrid),
    }

    def strip(rows):
        out = {}
        for k, r in rows.items():
            out[k] = {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                      for kk, vv in r.items()
                      if kk not in ("final_tables", "final_p")}
        return out

    report = {
        "status": ("BARRIER-XPLAY direct measurement of the joint "
                   "exploration barrier; registered before run"),
        "convention_seeds": strip(conv_rows),
        "roles_seeds": strip(role_rows),
        "crossplay": {
            "conv_within": [round(v, 4) for v in c_within],
            "conv_cross": [round(v, 4) for v in c_cross],
            "roles_within": [round(v, 4) for v in r_within],
            "roles_hybrid_mean": round(mr_hybrid, 4),
            "roles_hybrid_min": round(float(np.min(r_hybrid)), 4),
            "roles_hybrid_max": round(float(np.max(r_hybrid)), 4),
        },
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "barrier_xplay.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
