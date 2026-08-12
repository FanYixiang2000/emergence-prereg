"""REGIME-ENSEMBLE audit: verdict stability across plausible regime objects.

The declared regime object is one admissible choice among several. This
audit enumerates, from the environment specification alone, the other
regime variables an independent analyst could plausibly have declared,
plus variables that are clearly inadmissible (they erase the symmetric
competition, or are exogenous by construction), and adjudicates every
one with the frozen B5 detector on byte-identical reruns. The claim
under test: verdicts are a property of the system, stable across the
admissible class, and inadmissible variables do not manufacture onsets.

Candidates (fixed before running, closed form, no sampling noise):
  convention: P1 speaker code (declared); P2 listener decode map;
    P3 composed speaker->listener channel; P4 single-meaning sub-regime
    (meaning 0). Controls: X1 pooled symbol marginal (erases which
    meaning maps where -- uniform under every permutation code);
    X2 speaker identity (exogenous, uniform by construction).
  roles: P1 assignment openness (declared); P2 agent 0's role;
    P3 owner of role 0. Control: X1 pooled role marginal (uniform
    under every permutation).
  grip: P1 policy side-openness (declared); P2 realized force-sign
    regime (binary entropy of the sign of the joint push force across
    episodes, ties excluded). Control: X1 |x| tertile occupancy
    (erases the left/right symmetry).

Registered predictions (frozen before the run):
  RE0 reproduction: reruns reproduce every stored declared curve to
      within 1e-4 per point.
  RE1 verdict stability: non-declared plausible cells (15 convention,
      10 roles, 5 grip) match the declared verdict in >= 24/30.
  RE2 breakpoint stability: wherever a plausible cell and the declared
      object both certify onset, |t* - t*_declared| <= 10 percent of
      the analysis span (the detector's own thinning tolerance).
  RE3 controls: 0/20 inadmissible cells certify onset.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate
import learn_convention as LC
import learn_roles as LR
import learn_grip_transport as LG

OUTPUTS = Path(__file__).resolve().parent / "outputs"
GRIP_EPISODES = 4096


def hnorm(p: torch.Tensor) -> float:
    p = p / torch.clamp(p.sum(), min=1e-12)
    h = -(p * torch.log2(torch.clamp(p, min=1e-12))).sum()
    return float(h.item() / math.log2(len(p)))


def hnorm_rows(mat: torch.Tensor) -> float:
    h = -(mat * torch.log2(torch.clamp(mat, min=1e-12))).sum(dim=-1)
    return float(h.mean().item() / math.log2(mat.shape[-1]))


def convention_candidates(speak, listen):
    with torch.no_grad():
        sp = torch.softmax(speak, dim=-1).mean(dim=0)    # (m, s)
        li = torch.softmax(listen, dim=-1).mean(dim=0)   # (s, m)
        q = sp @ li                                       # (m, m')
    return {
        "P1_speaker_code": hnorm_rows(sp),
        "P2_listener_code": hnorm_rows(li),
        "P3_composed_channel": hnorm_rows(q / q.sum(dim=-1, keepdim=True)),
        "P4_meaning0_symbol": hnorm(sp[0]),
        "X1_symbol_marginal": hnorm(sp.mean(dim=0)),
        "X2_speaker_identity": 1.0,
    }


def role_candidates(logits):
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)                # (agents, roles)
        owner0 = p[:, 0]
    return {
        "P1_assignment": hnorm_rows(p),
        "P2_agent0_role": hnorm(p[0]),
        "P3_role0_owner": hnorm(owner0),
        "X1_role_marginal": hnorm(p.mean(dim=0)),
    }


def grip_force_traces(policy, episodes):
    with torch.no_grad():
        x = torch.zeros(episodes)
        v = torch.zeros(episodes)
        att = torch.zeros(episodes)
        fsigns, xabs = [], []
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
            fsigns.append(torch.sign(force))
            xabs.append(torch.abs(x))
    return torch.stack(fsigns, dim=1).numpy(), torch.stack(xabs, dim=1).numpy()


def grip_candidate_curves(fsigns, xabs):
    T = fsigns.shape[1]
    force_curve, xabs_curve = [], []
    for t in range(T):
        s = fsigns[:, t]
        nz = s[s != 0]
        if len(nz) == 0:
            force_curve.append(1.0)
        else:
            pp = float((nz > 0).mean())
            h = 0.0
            for p in (pp, 1 - pp):
                if p > 0:
                    h -= p * math.log2(p)
            force_curve.append(h)
        xa = xabs[:, t]
        qs = np.quantile(xa, [1 / 3, 2 / 3])
        bins = np.digitize(xa, qs)
        hist = np.bincount(bins, minlength=3).astype(float) / len(xa)
        nzh = hist[hist > 0]
        xabs_curve.append(float(-(nzh * np.log2(nzh)).sum() / math.log2(3)))
    return np.array(force_curve), np.array(xabs_curve)


def formation_audit(system):
    stored = json.loads((OUTPUTS / f"learn_{system}.json").read_text())
    mod = LC if system == "convention" else LR
    grid = mod.GRID
    rows = {}
    for i in range(mod.N_SEEDS):
        seed = mod.SEED + i * 101
        torch.manual_seed(seed)
        np.random.seed(seed)
        if system == "convention":
            speak = torch.zeros((mod.N_AGENTS, mod.K, mod.K),
                                requires_grad=True)
            listen = torch.zeros((mod.N_AGENTS, mod.K, mod.K),
                                 requires_grad=True)
            params = [speak, listen]
        else:
            logits = torch.zeros((mod.N_AGENTS, mod.R), requires_grad=True)
            params = [logits]
        opt = torch.optim.Adam(params, lr=mod.LR)
        baseline = 0.0
        gen = (torch.Generator().manual_seed(seed)
               if system == "convention" else None)
        declared = []
        curves = None
        for u in range(mod.UPDATES + 1):
            if u % mod.EVAL_EVERY == 0:
                if system == "convention":
                    declared.append(LC.convention_openness(speak))
                    cand = convention_candidates(speak, listen)
                else:
                    declared.append(LR.openness(logits))
                    cand = role_candidates(logits)
                if curves is None:
                    curves = {k: [] for k in cand}
                for k, val in cand.items():
                    curves[k].append(val)
            if u == mod.UPDATES:
                break
            if system == "convention":
                s_idx = torch.randint(0, mod.N_AGENTS, (mod.BATCH,),
                                      generator=gen)
                shift = torch.randint(1, mod.N_AGENTS, (mod.BATCH,),
                                      generator=gen)
                l_idx = (s_idx + shift) % mod.N_AGENTS
                m = torch.randint(0, mod.K, (mod.BATCH,), generator=gen)
                sp_dist = torch.distributions.Categorical(
                    logits=speak[s_idx, m])
                sym = sp_dist.sample()
                li_dist = torch.distributions.Categorical(
                    logits=listen[l_idx, sym])
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
        adjs = {k: adjudicate(grid, np.array(v) * math.log2(3))
                for k, v in curves.items()}
        rows[str(i)] = {
            "repro_max_abs_err": repro_err,
            "declared_b5": stored["seeds"][str(i)]["adj"]["b5_onset"],
            "declared_t_star": stored["seeds"][str(i)]["adj"].get(
                "hinge", {}).get("t_star"),
            "candidates": {k: {
                "b5_onset": a["b5_onset"],
                "t_star": a.get("hinge", {}).get("t_star"),
                "delta_bic": a.get("hinge", {}).get("delta_bic"),
                "verdict": a.get("verdict"),
            } for k, a in adjs.items()},
        }
        summary = " ".join(
            f"{k.split('_')[0]}={'T' if a['b5_onset'] else 'F'}"
            f"@{a.get('hinge', {}).get('t_star')}"
            for k, a in adjs.items())
        print(f"[{system}] seed={i} repro={repro_err:.1e} "
              f"declared_B5={rows[str(i)]['declared_b5']}"
              f"@{rows[str(i)]['declared_t_star']} | {summary}", flush=True)
    return rows


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
        fsigns, xabs = grip_force_traces(policy, GRIP_EPISODES)
        force_curve, xabs_curve = grip_candidate_curves(fsigns, xabs)
        adj_force = adjudicate(range(LG.MAX_STEPS),
                               force_curve * math.log2(3))
        adj_xabs = adjudicate(range(LG.MAX_STEPS), xabs_curve * math.log2(3))
        rows[str(i)] = {
            "repro_max_abs_err": repro_err,
            "declared_b5": b5["seeds"][str(i)]["adj"]["b5_onset"],
            "declared_t_star": b5["seeds"][str(i)]["adj"].get(
                "hinge", {}).get("t_star"),
            "candidates": {
                "P2_force_sign": {
                    "b5_onset": adj_force["b5_onset"],
                    "t_star": adj_force.get("hinge", {}).get("t_star"),
                    "delta_bic": adj_force.get("hinge", {}).get("delta_bic"),
                    "verdict": adj_force.get("verdict")},
                "X1_xabs_tertiles": {
                    "b5_onset": adj_xabs["b5_onset"],
                    "t_star": adj_xabs.get("hinge", {}).get("t_star"),
                    "delta_bic": adj_xabs.get("hinge", {}).get("delta_bic"),
                    "verdict": adj_xabs.get("verdict")},
            },
        }
        print(f"[grip] seed={i} repro={repro_err:.1e} "
              f"declared_B5={rows[str(i)]['declared_b5']}"
              f"@{rows[str(i)]['declared_t_star']} "
              f"force={adj_force['b5_onset']}"
              f"@{adj_force.get('hinge', {}).get('t_star')} "
              f"xabs={adj_xabs['b5_onset']}", flush=True)
    return rows


def main() -> None:
    torch.set_num_threads(4)
    results = {"convention": formation_audit("convention"),
               "roles": formation_audit("roles"),
               "grip": grip_audit()}

    plaus = {"convention": ["P2_listener_code", "P3_composed_channel",
                            "P4_meaning0_symbol"],
             "roles": ["P2_agent0_role", "P3_role0_owner"],
             "grip": ["P2_force_sign"]}
    ctrl = {"convention": ["X1_symbol_marginal", "X2_speaker_identity"],
            "roles": ["X1_role_marginal"],
            "grip": ["X1_xabs_tertiles"]}
    spans = {"convention": 4000, "roles": 6000, "grip": 79}

    repro_ok = all(r["repro_max_abs_err"] <= 1e-4
                   for sys_rows in results.values()
                   for r in sys_rows.values())
    agree = n_plaus = 0
    tstar_ok = tstar_n = 0
    ctrl_clean = n_ctrl = 0
    for s, sys_rows in results.items():
        for r in sys_rows.values():
            for c in plaus[s]:
                n_plaus += 1
                cand = r["candidates"][c]
                if cand["b5_onset"] == r["declared_b5"]:
                    agree += 1
                if cand["b5_onset"] and r["declared_b5"]:
                    tstar_n += 1
                    if (abs(cand["t_star"] - r["declared_t_star"])
                            <= 0.10 * spans[s]):
                        tstar_ok += 1
            for c in ctrl[s]:
                n_ctrl += 1
                if not r["candidates"][c]["b5_onset"]:
                    ctrl_clean += 1
    outcomes = {
        "RE0_reproduction": bool(repro_ok),
        "RE1_verdict_stability": f"{agree}/{n_plaus}",
        "RE1_pass": bool(agree >= 24),
        "RE2_t_star_within_10pct_span": f"{tstar_ok}/{tstar_n}",
        "RE2_pass": bool(tstar_n == 0 or tstar_ok == tstar_n),
        "RE3_controls_clean": f"{ctrl_clean}/{n_ctrl}",
        "RE3_pass": bool(ctrl_clean == n_ctrl),
    }
    report = {
        "status": ("REGIME-ENSEMBLE audit; plausible alternative regime "
                   "objects and inadmissible controls, all closed form, "
                   "adjudicated with the frozen B5 detector on "
                   "byte-identical reruns; predictions RE0-RE3 frozen in "
                   "the docstring before running"),
        "config": {"grip_episodes": GRIP_EPISODES},
        "results": results,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "regime_ensemble_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
