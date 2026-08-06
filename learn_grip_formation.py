"""LEARN-GRIP-FORMATION: formation-axis profile of the grip flagship.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Retrains
the grip seeds with checkpoint evaluation to profile how the transport
capability forms across training, completing the two-timescale quadrant
(formation across updates vs realization within episodes).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate
from learn_grip_transport import (
    BATCH,
    GripPolicy,
    LR,
    MAX_STEPS,
    N_SEEDS,
    SEED,
    UPDATES,
    rollout_batch,
    side_openness,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SAVE_EVERY = 25
GRID = tuple(range(0, UPDATES + 1, SAVE_EVERY))
EVAL_BATCH = 512


def outcome_openness(policy: GripPolicy, batch: int = EVAL_BATCH):
    with torch.no_grad():
        done, final_side, _ent, side_ent, _att, _side = rollout_batch(
            policy, batch, train=False)
    succ = float(done.mean().item())
    left = float(((final_side < 0) & (done > 0)).float().mean().item())
    right = float(((final_side > 0) & (done > 0)).float().mean().item())
    fail = max(0.0, 1.0 - left - right)
    p = np.array([fail, left, right])
    nz = p[p > 0]
    o_cap = float(-(nz * np.log2(nz)).sum() / math.log2(3))
    ep_side_ent = torch.median(side_ent, dim=1).values.numpy()
    return succ, o_cap, (left, right, fail), ep_side_ent


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = GripPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    baseline = 0.0
    succ_curve, ocap_curve, outcome_hist = [], [], []
    final_side_ent = None
    for u in range(UPDATES + 1):
        if u in GRID:
            succ, o_cap, dist, ep_side_ent = outcome_openness(policy)
            succ_curve.append(succ)
            ocap_curve.append(o_cap)
            outcome_hist.append([round(x, 5) for x in dist])
            if u == UPDATES:
                final_side_ent = ep_side_ent
        if u == UPDATES:
            break
        returns, logp, _done = rollout_batch(policy, BATCH, train=True)
        adv = returns.detach() - baseline
        baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    formation_adj = adjudicate(GRID, np.array(ocap_curve) * math.log2(3))
    realization_adj = adjudicate(range(MAX_STEPS), final_side_ent)
    plateau = 0
    for val in final_side_ent:
        if val >= 0.8:
            plateau += 1
        else:
            break
    return {
        "success_curve": [round(x, 5) for x in succ_curve],
        "ocap_curve": [round(x, 5) for x in ocap_curve],
        "outcome_hist": outcome_hist,
        "final_success": round(succ_curve[-1], 5),
        "formation_adj": formation_adj,
        "realization_plateau": plateau,
        "realization_adj": realization_adj,
        "final_side_openness_curve": [round(float(x), 5) for x in final_side_ent],
    }


def main() -> None:
    rows = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        rows[str(i)] = row
        fh = row["formation_adj"].get("hinge", {})
        print(f"seed={i}: succ={row['final_success']} "
              f"formB5={row['formation_adj']['b5_onset']} "
              f"form_dBIC={fh.get('delta_bic')} "
              f"realB5={row['realization_adj']['b5_onset']} "
              f"plateau={row['realization_plateau']}", flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    outcomes = {
        "LGF1_learnability": bool(len(learned) >= 4),
        "LGF2_formation_smooth": bool(
            sum(r["formation_adj"]["b5_onset"] for r in learned) <= 1),
        "LGF3_realization_stable": bool(
            len(learned) >= 4
            and sum(r["realization_adj"]["b5_onset"] for r in learned) >= 3
            and all(r["realization_plateau"] >= 5 for r in learned)),
        "n_learned": len(learned),
        "formation_b5_count": sum(
            r["formation_adj"]["b5_onset"] for r in learned),
        "realization_b5_count": sum(
            r["realization_adj"]["b5_onset"] for r in learned),
    }
    report = {
        "status": "LEARN-GRIP-FORMATION two-timescale quadrant; preregistered",
        "config": {"grid_every": SAVE_EVERY, "eval_batch": EVAL_BATCH},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_formation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
