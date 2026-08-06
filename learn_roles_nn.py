"""LEARN-ROLES-NN: neural replication of endogenous role lock-in.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Identical
environment and evaluation to learn_roles.py; the only change is the
policy class -- one SHARED MLP for all six agents (agent one-hot -> 32
tanh -> 6 logits, Adam 3e-3), so role differentiation must emerge
through the shared network rather than independent tables -- and 10
fresh seeds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from learn_roles import (BATCH, EVAL_EVERY, GRID, LOG2R, N_AGENTS, R,
                         UPDATES, adjudicate, converged_assignment, openness,
                         success_exact)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
HIDDEN = 32
LR = 3e-3
N_SEEDS = 10
SEED = 919_001


class SharedRolePolicy:
    """Shared MLP over agent one-hots; logits_table gives the (N, R) map."""

    def __init__(self, gen: torch.Generator) -> None:
        s = 1.0 / math.sqrt(N_AGENTS)
        self.w1 = (torch.randn((N_AGENTS, HIDDEN), generator=gen) * s
                   ).requires_grad_(True)
        self.b1 = torch.zeros(HIDDEN, requires_grad=True)
        self.w2 = (torch.randn((HIDDEN, R), generator=gen)
                   / math.sqrt(HIDDEN)).requires_grad_(True)
        self.b2 = torch.zeros(R, requires_grad=True)

    def params(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def logits_table(self) -> torch.Tensor:
        return torch.tanh(self.w1 + self.b1) @ self.w2 + self.b2


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_gen = torch.Generator().manual_seed(seed + 7)
    policy = SharedRolePolicy(init_gen)
    opt = torch.optim.Adam(policy.params(), lr=LR)
    baseline = 0.0
    open_curve, succ_curve = [], []
    for u in range(UPDATES + 1):
        logits = policy.logits_table()
        if u % EVAL_EVERY == 0:
            open_curve.append(openness(logits.detach()))
            succ_curve.append(success_exact(logits.detach()))
        if u == UPDATES:
            break
        dist = torch.distributions.Categorical(logits=logits)
        roles = dist.sample((BATCH,))
        covered = torch.zeros((BATCH, R))
        covered.scatter_(1, roles, 1.0)
        r = (covered.sum(dim=1) == R).float()
        adv = r - baseline
        baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
        logp = dist.log_prob(roles).sum(dim=1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    adj = adjudicate(GRID, np.array(open_curve) * math.log2(3))
    s = np.array(succ_curve)
    cross = next((GRID[i] for i in range(len(s)) if s[i] >= 0.9), None)
    return {
        "final_success": round(succ_curve[-1], 5),
        "final_openness": round(open_curve[-1], 5),
        "openness_curve": [round(v, 5) for v in open_curve],
        "success_curve": [round(v, 5) for v in succ_curve],
        "success_090_cross": cross,
        "assignment": converged_assignment(policy.logits_table().detach()),
        "adj": adj,
    }


def main() -> None:
    torch.set_num_threads(4)
    rows = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        rows[str(i)] = row
        h = row["adj"].get("hinge", {})
        print(f"seed={i}: S={row['final_success']} O={row['final_openness']} "
              f"B5={row['adj']['b5_onset']} dBIC={h.get('delta_bic')} "
              f"t*={h.get('t_star')} s090={row['success_090_cross']} "
              f"assign={row['assignment']}", flush=True)

    learned = {k: r for k, r in rows.items() if r["final_success"] >= 0.8}
    onset_seeds = {k: r for k, r in learned.items() if r["adj"]["b5_onset"]}
    lead = all(r["adj"]["hinge"]["t_star"] <= r["success_090_cross"]
               for r in onset_seeds.values()
               if r["success_090_cross"] is not None) if onset_seeds else False
    assigns = {tuple(r["assignment"]) for r in learned.values()}
    onset_rate = (len(onset_seeds) / len(learned)) if learned else 0.0
    outcomes = {
        "NN1_learnability": bool(len(learned) >= 8),
        "NN2_onset_rate": bool(learned and onset_rate >= 0.6),
        "NN3_collapse_leads_capability": bool(lead and len(onset_seeds) >= 1),
        "NN4_symmetry_breaking": bool(len(assigns) >= 3),
        "n_learned": len(learned),
        "n_onset": len(onset_seeds),
        "onset_rate_learned": round(onset_rate, 4),
        "n_distinct_assignments": len(assigns),
    }
    report = {
        "status": ("LEARN-ROLES-NN neural replication; one shared MLP for "
                   "all agents, 10 fresh seeds; registered before run"),
        "config": {"n_agents": N_AGENTS, "R": R, "hidden": HIDDEN,
                   "batch": BATCH, "updates": UPDATES,
                   "eval_every": EVAL_EVERY, "lr": LR,
                   "seeds": N_SEEDS, "seed0": SEED},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_roles_nn.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
