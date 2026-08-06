"""LEARN-GRIP-FORMATION-FINE: fine-grid formation adjudication.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running.
Checkpoints every 5 updates over the first 400 updates of the grip
flagship, adjudicating whether the fast capability formation is itself
abrupt or smooth at fine temporal resolution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate
from learn_grip_formation import outcome_openness
from learn_grip_transport import (
    BATCH,
    GripPolicy,
    LR,
    N_SEEDS,
    SEED,
    rollout_batch,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
FINE_UPDATES = 400
SAVE_EVERY = 5
GRID = tuple(range(0, FINE_UPDATES + 1, SAVE_EVERY))


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = GripPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    baseline = 0.0
    succ_curve, ocap_curve = [], []
    for u in range(FINE_UPDATES + 1):
        if u in GRID:
            succ, o_cap, _dist, _ep = outcome_openness(policy)
            succ_curve.append(succ)
            ocap_curve.append(o_cap)
        if u == FINE_UPDATES:
            break
        returns, logp, _done = rollout_batch(policy, BATCH, train=True)
        adv = returns.detach() - baseline
        baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    ocap_adj = adjudicate(GRID, np.array(ocap_curve) * math.log2(3))
    # Success curve rises 0 -> 1: adjudicate on 1 - success as a
    # "collapse-shaped" object so the detector's drop gate applies.
    succ_adj = adjudicate(GRID, (1.0 - np.array(succ_curve)) * math.log2(3))
    mid = next((g for g, s in zip(GRID, succ_curve) if s >= 0.5), None)
    return {
        "success_curve": [round(x, 5) for x in succ_curve],
        "ocap_curve": [round(x, 5) for x in ocap_curve],
        "final_success": round(succ_curve[-1], 5),
        "success_midpoint_update": mid,
        "ocap_adj": ocap_adj,
        "succ_adj": succ_adj,
    }


def main() -> None:
    rows = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        rows[str(i)] = row
        oh = row["ocap_adj"].get("hinge", {})
        sh = row["succ_adj"].get("hinge", {})
        print(f"seed={i}: succ={row['final_success']} mid={row['success_midpoint_update']} "
              f"ocapB5={row['ocap_adj']['b5_onset']} ocap_dBIC={oh.get('delta_bic')} "
              f"succB5={row['succ_adj']['b5_onset']} succ_dBIC={sh.get('delta_bic')}",
              flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    outcomes = {
        "LGFF1_learnability": bool(len(learned) >= 4),
        "LGFF2_formation_smooth": bool(
            sum(r["ocap_adj"]["b5_onset"] for r in learned) <= 1),
        "LGFF3_fast_rise": bool(
            learned and all(
                r["success_midpoint_update"] is not None
                and r["success_midpoint_update"] <= 150 for r in learned)),
        "n_learned": len(learned),
        "ocap_b5_count": sum(r["ocap_adj"]["b5_onset"] for r in learned),
        "succ_b5_count": sum(r["succ_adj"]["b5_onset"] for r in learned),
        "midpoints": [r["success_midpoint_update"] for r in rows.values()],
    }
    report = {
        "status": "LEARN-GRIP-FORMATION-FINE fine-grid formation; preregistered",
        "config": {"fine_updates": FINE_UPDATES, "save_every": SAVE_EVERY},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_formation_fine.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
