"""REGIME-ENSEMBLE v2: corrected per-episode force-direction candidate.

The ensemble audit's grip candidate P2 aggregated the sign of the joint
push force across episodes, where symmetric commitments (half the
episodes commit left, half right) cancel: the variable erases the very
competition it was meant to track, and the detector correctly refused
to certify it (0/5). That mis-specification is retained in the v1
record. This run registers the corrected candidate: for each episode,
the realized force direction over the trailing five steps (fraction of
positive signs among nonzero signs); openness at step t is the mean
over episodes of the binary entropy of that fraction. Same rollouts
protocol, same frozen detector, no other changes.

Registered predictions (frozen before the run):
  RE1b verdict agreement: the corrected candidate certifies onset in
       >= 4/5 grip seeds (declared: 5/5).
  RE2b breakpoint agreement: wherever certified, |t* - t*_declared|
       <= 8 steps (10 percent of the 80-step span).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate
import learn_grip_transport as LG
from regime_ensemble_audit import grip_force_traces, GRIP_EPISODES

OUTPUTS = Path(__file__).resolve().parent / "outputs"
WINDOW = 5


def per_episode_force_openness(fsigns):
    n, T = fsigns.shape
    curve = []
    for t in range(T):
        lo = max(0, t - WINDOW + 1)
        w = fsigns[:, lo:t + 1]
        h = np.zeros(n)
        for e in range(n):
            nz = w[e][w[e] != 0]
            if len(nz) == 0:
                h[e] = 1.0
                continue
            p = float((nz > 0).mean())
            for q in (p, 1 - p):
                if q > 0:
                    h[e] -= q * math.log2(q)
        curve.append(float(h.mean()))
    return np.array(curve)


def main() -> None:
    torch.set_num_threads(4)
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
        fsigns, _xabs = grip_force_traces(policy, GRIP_EPISODES)
        curve = per_episode_force_openness(fsigns)
        adj = adjudicate(range(LG.MAX_STEPS), curve * math.log2(3))
        rows[str(i)] = {
            "repro_max_abs_err": repro_err,
            "declared_b5": b5["seeds"][str(i)]["adj"]["b5_onset"],
            "declared_t_star": b5["seeds"][str(i)]["adj"].get(
                "hinge", {}).get("t_star"),
            "candidate_adj": adj,
            "candidate_curve": [round(float(v), 5) for v in curve],
        }
        h = adj.get("hinge", {})
        print(f"[grip-e2] seed={i} repro={repro_err:.1e} "
              f"declared_B5={rows[str(i)]['declared_b5']}"
              f"@{rows[str(i)]['declared_t_star']} "
              f"cand_B5={adj['b5_onset']} dBIC={h.get('delta_bic')} "
              f"t*={h.get('t_star')}", flush=True)

    agree = sum(r["candidate_adj"]["b5_onset"] == r["declared_b5"]
                for r in rows.values())
    tstar_ok, tstar_n = 0, 0
    for r in rows.values():
        if r["candidate_adj"]["b5_onset"] and r["declared_b5"]:
            tstar_n += 1
            if abs(r["candidate_adj"]["hinge"]["t_star"]
                   - r["declared_t_star"]) <= 8:
                tstar_ok += 1
    outcomes = {
        "RE1b_verdict_agreement": f"{agree}/5",
        "RE1b_pass": bool(agree >= 4),
        "RE2b_t_star_within_8": f"{tstar_ok}/{tstar_n}",
        "RE2b_pass": bool(tstar_n == 0 or tstar_ok == tstar_n),
    }
    report = {
        "status": ("REGIME-ENSEMBLE v2; corrected per-episode trailing-"
                   "window force-direction candidate for the grip system; "
                   "predictions RE1b-RE2b frozen in the docstring"),
        "config": {"grip_episodes": GRIP_EPISODES, "window": WINDOW},
        "results": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "regime_ensemble_audit2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
