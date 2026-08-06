"""LEARN-ROLES: endogenous role lock-in (division of labour).

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Six agents
independently choose among six interchangeable roles; team reward 1 iff
all roles are covered. No gate, no threshold, no designed target: all
720 permutations are equivalent. Tests whether punctuated collapse of
the assignment possibility space arises from sparse-reward nucleation
plus self-reinforcing role avoidance.
"""
from __future__ import annotations

import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_AGENTS = 6
R = 6
BATCH = 512
UPDATES = 6000
EVAL_EVERY = 25
LR = 0.01
N_SEEDS = 5
SEED = 717_001
GRID = list(range(0, UPDATES + 1, EVAL_EVERY))
LOG2R = math.log2(R)
PERMS = list(permutations(range(R)))


def openness(logits: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)
        h = -(p * torch.log2(torch.clamp(p, min=1e-12))).sum(dim=-1)
    return float(h.mean().item() / LOG2R)


def success_exact(logits: torch.Tensor) -> float:
    """P(all roles covered) = permanent of the probability matrix."""
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1).numpy()
    total = 0.0
    for perm in PERMS:
        prod = 1.0
        for i, r in enumerate(perm):
            prod *= p[i, r]
        total += prod
    return float(total)


def converged_assignment(logits: torch.Tensor) -> list:
    with torch.no_grad():
        return [int(i) for i in torch.softmax(logits, dim=-1).argmax(dim=-1)]


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    logits = torch.zeros((N_AGENTS, R), requires_grad=True)
    opt = torch.optim.Adam([logits], lr=LR)
    baseline = 0.0
    open_curve, succ_curve = [], []
    for u in range(UPDATES + 1):
        if u % EVAL_EVERY == 0:
            open_curve.append(openness(logits))
            succ_curve.append(success_exact(logits))
        if u == UPDATES:
            break
        dist = torch.distributions.Categorical(logits=logits)
        roles = dist.sample((BATCH,))                    # (B, N)
        covered = torch.zeros((BATCH, R))
        covered.scatter_(1, roles, 1.0)
        r = (covered.sum(dim=1) == R).float()            # all roles used
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
        "assignment": converged_assignment(logits),
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
    lr3 = all(r["adj"]["hinge"]["t_star"] <= r["success_090_cross"]
              for r in onset_seeds.values()
              if r["success_090_cross"] is not None) if onset_seeds else False
    assigns = {tuple(r["assignment"]) for r in learned.values()}
    outcomes = {
        "LR1_learnability": bool(len(learned) >= 4),
        "LR2_nonconstructed_onset": bool(len(onset_seeds) >= 3),
        "LR3_collapse_leads_capability": bool(lr3 and len(onset_seeds) >= 1),
        "LR4_symmetry_breaking": bool(len(assigns) >= 2),
        "n_learned": len(learned),
        "n_onset": len(onset_seeds),
        "n_distinct_assignments": len(assigns),
    }
    report = {
        "status": ("LEARN-ROLES endogenous role lock-in; no gate, no "
                   "threshold, no designed target; registered before run"),
        "config": {"n_agents": N_AGENTS, "R": R, "batch": BATCH,
                   "updates": UPDATES, "eval_every": EVAL_EVERY, "lr": LR,
                   "seeds": N_SEEDS, "seed0": SEED},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_roles.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
