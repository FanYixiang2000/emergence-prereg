"""LEARN-CONVENTION-NN: neural replication of convention formation.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Identical
environment and evaluation to learn_convention.py; the only change is
the policy class (per-agent MLPs instead of tabular logits, Adam 3e-3)
and 10 fresh seeds. Tests that plateau-then-collapse onset does not
depend on the tabular parameterization.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate
from learn_convention import (BATCH, EVAL_EVERY, GRID, K, LOG2K, N_AGENTS,
                              UPDATES, code_of, convention_openness,
                              mutual_success)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
HIDDEN = 32
LR = 3e-3
N_SEEDS = 10
SEED = 818_001


class MLPBank:
    """Per-agent one-hidden-layer MLPs, batched as tensors.

    logits_table() returns the full (N, K, K) mapping table so the
    tabular openness / success / code functions apply unchanged.
    """

    def __init__(self, gen: torch.Generator) -> None:
        s = 1.0 / math.sqrt(K)
        self.w1 = (torch.randn((N_AGENTS, K, HIDDEN), generator=gen) * s
                   ).requires_grad_(True)
        self.b1 = torch.zeros((N_AGENTS, 1, HIDDEN), requires_grad=True)
        self.w2 = (torch.randn((N_AGENTS, HIDDEN, K), generator=gen)
                   / math.sqrt(HIDDEN)).requires_grad_(True)
        self.b2 = torch.zeros((N_AGENTS, 1, K), requires_grad=True)

    def params(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def logits_table(self) -> torch.Tensor:
        # one-hot input row m selects w1[:, m, :]
        h = torch.tanh(self.w1 + self.b1)                 # (N, K, H)
        return torch.einsum("nkh,nho->nko", h, self.w2) + self.b2


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_gen = torch.Generator().manual_seed(seed + 7)
    speak = MLPBank(init_gen)
    listen = MLPBank(init_gen)
    opt = torch.optim.Adam(speak.params() + listen.params(), lr=LR)
    baseline = 0.0
    open_curve, succ_curve = [], []
    gen = torch.Generator().manual_seed(seed)
    for u in range(UPDATES + 1):
        sp_table = speak.logits_table()
        li_table = listen.logits_table()
        if u % EVAL_EVERY == 0:
            open_curve.append(convention_openness(sp_table.detach()))
            succ_curve.append(mutual_success(sp_table.detach(),
                                             li_table.detach()))
        if u == UPDATES:
            break
        s_idx = torch.randint(0, N_AGENTS, (BATCH,), generator=gen)
        shift = torch.randint(1, N_AGENTS, (BATCH,), generator=gen)
        l_idx = (s_idx + shift) % N_AGENTS
        m = torch.randint(0, K, (BATCH,), generator=gen)
        sp_logits = sp_table[s_idx, m]
        sp_dist = torch.distributions.Categorical(logits=sp_logits)
        sym = sp_dist.sample()
        li_logits = li_table[l_idx, sym]
        li_dist = torch.distributions.Categorical(logits=li_logits)
        guess = li_dist.sample()
        r = (guess == m).float()
        adv = r - baseline
        baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
        loss = -(adv.detach() * (sp_dist.log_prob(sym)
                                 + li_dist.log_prob(guess))).mean()
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
        "code": code_of(speak.logits_table().detach()),
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
              f"code={row['code']}", flush=True)

    learned = {k: r for k, r in rows.items() if r["final_success"] >= 0.8}
    onset_seeds = {k: r for k, r in learned.items() if r["adj"]["b5_onset"]}
    lead = all(r["adj"]["hinge"]["t_star"] <= r["success_090_cross"]
               for r in onset_seeds.values()
               if r["success_090_cross"] is not None) if onset_seeds else False
    codes = {tuple(r["code"]) for r in learned.values()}
    onset_rate = (len(onset_seeds) / len(learned)) if learned else 0.0
    outcomes = {
        "NN1_learnability": bool(len(learned) >= 8),
        "NN2_onset_rate": bool(learned and onset_rate >= 0.6),
        "NN3_collapse_leads_capability": bool(lead and len(onset_seeds) >= 1),
        "NN4_symmetry_breaking": bool(len(codes) >= 3),
        "n_learned": len(learned),
        "n_onset": len(onset_seeds),
        "onset_rate_learned": round(onset_rate, 4),
        "n_distinct_codes": len(codes),
    }
    report = {
        "status": ("LEARN-CONVENTION-NN neural replication; per-agent MLP "
                   "policies, 10 fresh seeds; registered before run"),
        "config": {"n_agents": N_AGENTS, "K": K, "hidden": HIDDEN,
                   "batch": BATCH, "updates": UPDATES,
                   "eval_every": EVAL_EVERY, "lr": LR,
                   "seeds": N_SEEDS, "seed0": SEED},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_convention_nn.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
