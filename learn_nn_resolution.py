"""NN-RES + NN-INIT: fine-grid rerun and initialization-scale sweep.

Registered as an amendment in V2_ALIGNMENT_PREREGISTRATION.md before
running. Systems and training recipes are byte-identical to
learn_convention_nn.py / learn_roles_nn.py; the only changes are the
evaluation density (every 5 updates instead of 25) and, for NN-INIT,
the initialization scale sigma. The frozen detector is unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate
from learn_convention import (convention_openness, mutual_success, code_of,
                              BATCH as C_BATCH, K, N_AGENTS as C_N,
                              UPDATES as C_UPDATES)
from learn_roles import (openness as role_openness, success_exact,
                         converged_assignment, BATCH as R_BATCH,
                         N_AGENTS as R_N, R, UPDATES as R_UPDATES)
from learn_convention_nn import HIDDEN as C_HIDDEN, LR as C_LR
from learn_roles_nn import HIDDEN as R_HIDDEN, LR as R_LR

OUTPUTS = Path(__file__).resolve().parent / "outputs"
EVAL_EVERY = 5
SIGMA_DEFAULT = 1.0 / math.sqrt(K)   # = the LEARN-*-NN scale (0.447)
SIGMAS = (0.02, 0.1, SIGMA_DEFAULT)
LOG2_3 = math.log2(3)


def mlp_bank(n, k_in, k_out, hidden, sigma, gen):
    w1 = (torch.randn((n, k_in, hidden), generator=gen) * sigma
          ).requires_grad_(True)
    b1 = torch.zeros((n, 1, hidden), requires_grad=True)
    w2 = (torch.randn((n, hidden, k_out), generator=gen)
          / math.sqrt(hidden)).requires_grad_(True)
    b2 = torch.zeros((n, 1, k_out), requires_grad=True)
    return [w1, b1, w2, b2]


def bank_table(p):
    w1, b1, w2, b2 = p
    return torch.einsum("nkh,nho->nko", torch.tanh(w1 + b1), w2) + b2


def run_convention(seed: int, sigma: float):
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_gen = torch.Generator().manual_seed(seed + 7)
    speak = mlp_bank(C_N, K, K, C_HIDDEN, sigma, init_gen)
    listen = mlp_bank(C_N, K, K, C_HIDDEN, sigma, init_gen)
    opt = torch.optim.Adam(speak + listen, lr=C_LR)
    baseline = 0.0
    grid, open_curve, succ_curve = [], [], []
    gen = torch.Generator().manual_seed(seed)
    for u in range(C_UPDATES + 1):
        sp_table = bank_table(speak)
        li_table = bank_table(listen)
        if u % EVAL_EVERY == 0:
            grid.append(u)
            open_curve.append(convention_openness(sp_table.detach()))
            succ_curve.append(mutual_success(sp_table.detach(),
                                             li_table.detach()))
        if u == C_UPDATES:
            break
        s_idx = torch.randint(0, C_N, (C_BATCH,), generator=gen)
        shift = torch.randint(1, C_N, (C_BATCH,), generator=gen)
        l_idx = (s_idx + shift) % C_N
        m = torch.randint(0, K, (C_BATCH,), generator=gen)
        sp_dist = torch.distributions.Categorical(logits=sp_table[s_idx, m])
        sym = sp_dist.sample()
        li_dist = torch.distributions.Categorical(logits=li_table[l_idx, sym])
        guess = li_dist.sample()
        r = (guess == m).float()
        adv = r - baseline
        baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
        loss = -(adv.detach() * (sp_dist.log_prob(sym)
                                 + li_dist.log_prob(guess))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return grid, open_curve, succ_curve, code_of(bank_table(speak).detach())


def run_roles(seed: int, sigma: float):
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_gen = torch.Generator().manual_seed(seed + 7)
    w1 = (torch.randn((R_N, R_HIDDEN), generator=init_gen) * sigma
          ).requires_grad_(True)
    b1 = torch.zeros(R_HIDDEN, requires_grad=True)
    w2 = (torch.randn((R_HIDDEN, R), generator=init_gen)
          / math.sqrt(R_HIDDEN)).requires_grad_(True)
    b2 = torch.zeros(R, requires_grad=True)
    opt = torch.optim.Adam([w1, b1, w2, b2], lr=R_LR)
    baseline = 0.0
    grid, open_curve, succ_curve = [], [], []
    for u in range(R_UPDATES + 1):
        logits = torch.tanh(w1 + b1) @ w2 + b2
        if u % EVAL_EVERY == 0:
            grid.append(u)
            open_curve.append(role_openness(logits.detach()))
            succ_curve.append(success_exact(logits.detach()))
        if u == R_UPDATES:
            break
        dist = torch.distributions.Categorical(logits=logits)
        roles = dist.sample((R_BATCH,))
        covered = torch.zeros((R_BATCH, R))
        covered.scatter_(1, roles, 1.0)
        r = (covered.sum(dim=1) == R).float()
        adv = r - baseline
        baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
        logp = dist.log_prob(roles).sum(dim=1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    logits = (torch.tanh(w1 + b1) @ w2 + b2).detach()
    return grid, open_curve, succ_curve, converged_assignment(logits)


def summarize_seed(grid, open_curve, succ_curve, ident, total_updates):
    adj = adjudicate(grid, np.array(open_curve) * LOG2_3)
    s = np.array(succ_curve)
    cross = next((grid[i] for i in range(len(s)) if s[i] >= 0.9), None)
    o = np.array(open_curve)
    t50 = next((grid[i] for i in range(len(o)) if o[i] < 0.5), None)
    h = adj.get("hinge", {})
    return {
        "final_success": round(float(s[-1]), 5),
        "b5_onset": adj["b5_onset"],
        "t_star": h.get("t_star"),
        "delta_bic": h.get("delta_bic"),
        "success_090_cross": cross,
        "t50_openness": t50,
        "tstar_in_first_10pct": (h.get("t_star") is not None
                                 and h["t_star"] <= 0.1 * total_updates),
        "identity": ident,
        "openness_curve_ds": [round(float(v), 5) for v in o[::5]],
        "success_curve_ds": [round(float(v), 5) for v in s[::5]],
    }


def batch(system, runner, seeds, sigma, total_updates):
    rows = {}
    for i, seed in enumerate(seeds):
        grid, oc, sc, ident = runner(seed, sigma)
        rows[str(i)] = summarize_seed(grid, oc, sc, ident, total_updates)
        r = rows[str(i)]
        print(f"{system} sigma={sigma:.3f} seed={i}: S={r['final_success']} "
              f"B5={r['b5_onset']} dBIC={r['delta_bic']} t*={r['t_star']} "
              f"t50={r['t50_openness']} s090={r['success_090_cross']}",
              flush=True)
    return rows


def outcomes_for(rows, total_updates):
    learned = {k: r for k, r in rows.items() if r["final_success"] >= 0.8}
    onset = {k: r for k, r in learned.items() if r["b5_onset"]}
    lead = all(r["t_star"] <= r["success_090_cross"]
               for r in onset.values()
               if r["success_090_cross"] is not None) if onset else False
    idents = {tuple(r["identity"]) for r in learned.values()}
    rate = len(onset) / len(learned) if learned else 0.0
    first10 = all(r["tstar_in_first_10pct"] for r in onset.values()) \
        if onset else False
    return {
        "n_learned": len(learned), "n_onset": len(onset),
        "onset_rate_learned": round(rate, 4),
        "collapse_leads_capability": bool(lead and onset),
        "all_tstar_first_10pct": bool(first10),
        "n_distinct_identities": len(idents),
    }


def main() -> None:
    torch.set_num_threads(4)
    conv_seeds = [818_001 + i * 101 for i in range(10)]
    role_seeds = [919_001 + i * 101 for i in range(10)]

    # ---- NN-RES: fine grid, default sigma, same 10 seeds
    res = {}
    rows_c = batch("conv", run_convention, conv_seeds, SIGMA_DEFAULT,
                   C_UPDATES)
    res["convention"] = {"seeds": rows_c,
                         "outcomes": outcomes_for(rows_c, C_UPDATES)}
    rows_r = batch("roles", run_roles, role_seeds, SIGMA_DEFAULT, R_UPDATES)
    res["roles"] = {"seeds": rows_r,
                    "outcomes": outcomes_for(rows_r, R_UPDATES)}
    reg = {
        "RESa_onset_rate_ge_060_both": bool(
            res["convention"]["outcomes"]["onset_rate_learned"] >= 0.6
            and res["roles"]["outcomes"]["onset_rate_learned"] >= 0.6),
        "RESb_collapse_leads_capability_both": bool(
            res["convention"]["outcomes"]["collapse_leads_capability"]
            and res["roles"]["outcomes"]["collapse_leads_capability"]),
        "RESc_tstar_first_10pct_both": bool(
            res["convention"]["outcomes"]["all_tstar_first_10pct"]
            and res["roles"]["outcomes"]["all_tstar_first_10pct"]),
    }
    out = OUTPUTS / "learn_nn_resolution.json"
    out.write_text(json.dumps({
        "status": ("NN-RES fine-grid rerun (eval_every=5), amendment "
                   "registered before run"),
        "config": {"eval_every": EVAL_EVERY, "sigma": SIGMA_DEFAULT},
        "systems": res, "registered_outcomes": reg}, indent=2),
        encoding="utf-8")
    print(json.dumps(reg, indent=2))
    print(f"Wrote {out}", flush=True)

    # ---- NN-INIT: sigma sweep, 5 seeds per sigma per system
    sweep = {}
    for sigma in SIGMAS:
        rc = batch("conv", run_convention, conv_seeds[:5], sigma, C_UPDATES)
        rr = batch("roles", run_roles, role_seeds[:5], sigma, R_UPDATES)
        sweep[f"{sigma:.3f}"] = {
            "convention": {"seeds": rc,
                           "outcomes": outcomes_for(rc, C_UPDATES),
                           "median_t50": _med(rc)},
            "roles": {"seeds": rr, "outcomes": outcomes_for(rr, R_UPDATES),
                      "median_t50": _med(rr)},
        }
    keys = [f"{s:.3f}" for s in SIGMAS]
    mono = {}
    for sysname in ("convention", "roles"):
        t50s = [sweep[k][sysname]["median_t50"] for k in keys]
        mono[sysname] = {"median_t50_by_sigma": dict(zip(keys, t50s)),
                         "monotone_decreasing": bool(
                             all(a > b for a, b in zip(t50s, t50s[1:])
                                 if a is not None and b is not None))}
    reg2 = {"INIT_t50_monotone_decreasing_both": bool(
        mono["convention"]["monotone_decreasing"]
        and mono["roles"]["monotone_decreasing"])}
    out2 = OUTPUTS / "learn_nn_init.json"
    out2.write_text(json.dumps({
        "status": ("NN-INIT initialization-scale sweep, amendment "
                   "registered before run"),
        "config": {"eval_every": EVAL_EVERY, "sigmas": list(SIGMAS)},
        "sweep": sweep, "monotonicity": mono,
        "registered_outcomes": reg2}, indent=2), encoding="utf-8")
    print(json.dumps({"monotonicity": mono, "registered": reg2}, indent=2))
    print(f"Wrote {out2}")


def _med(rows):
    vals = [r["t50_openness"] for r in rows.values()
            if r["t50_openness"] is not None]
    return float(np.median(vals)) if vals else None


if __name__ == "__main__":
    main()
