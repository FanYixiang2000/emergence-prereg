"""LEARN-CONVENTION: non-constructed convention formation.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Population
Lewis signalling game with no gate, no threshold, no blocked channel:
any code is available from update zero and all K! codes are equivalent.
Tests whether plateau-then-collapse onset arises endogenously from the
joint exploration barrier of convention formation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_AGENTS = 10
K = 5
BATCH = 512
UPDATES = 4000
EVAL_EVERY = 25
LR = 0.01
N_SEEDS = 5
SEED = 616_001
GRID = list(range(0, UPDATES + 1, EVAL_EVERY))
LOG2K = math.log2(K)


def convention_openness(speak_logits: torch.Tensor) -> float:
    """Mean over meanings of H(population-mean symbol dist)/log2 K."""
    with torch.no_grad():
        p = torch.softmax(speak_logits, dim=-1)      # (N, K, K)
        pbar = p.mean(dim=0)                          # (K meanings, K symbols)
        h = -(pbar * torch.log2(torch.clamp(pbar, min=1e-12))).sum(dim=-1)
    return float(h.mean().item() / LOG2K)


def mutual_success(speak_logits, listen_logits) -> float:
    """Exact expected intelligibility over ordered pairs and meanings."""
    with torch.no_grad():
        sp = torch.softmax(speak_logits, dim=-1)      # (N, m, s)
        li = torch.softmax(listen_logits, dim=-1)     # (N, s, m)
        # P(pair i->j decodes m) = sum_s sp[i,m,s] li[j,s,m]
        # accuracy matrix acc[i,j] = mean_m sum_s sp[i,m,s] li[j,s,m]
        acc = torch.einsum("ims,jsm->ij", sp, li) / K
        n = acc.shape[0]
        mask = 1.0 - torch.eye(n)
    return float((acc * mask).sum().item() / (n * (n - 1)))


def code_of(speak_logits) -> list:
    """Population majority code: argmax of population-mean mapping."""
    with torch.no_grad():
        pbar = torch.softmax(speak_logits, dim=-1).mean(dim=0)
    return [int(i) for i in pbar.argmax(dim=-1)]


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    speak = torch.zeros((N_AGENTS, K, K), requires_grad=True)
    listen = torch.zeros((N_AGENTS, K, K), requires_grad=True)
    opt = torch.optim.Adam([speak, listen], lr=LR)
    baseline = 0.0
    open_curve, succ_curve = [], []
    gen = torch.Generator().manual_seed(seed)
    for u in range(UPDATES + 1):
        if u % EVAL_EVERY == 0:
            open_curve.append(convention_openness(speak))
            succ_curve.append(mutual_success(speak, listen))
        if u == UPDATES:
            break
        s_idx = torch.randint(0, N_AGENTS, (BATCH,), generator=gen)
        shift = torch.randint(1, N_AGENTS, (BATCH,), generator=gen)
        l_idx = (s_idx + shift) % N_AGENTS            # listener != speaker
        m = torch.randint(0, K, (BATCH,), generator=gen)
        sp_logits = speak[s_idx, m]                    # (B, K symbols)
        sp_dist = torch.distributions.Categorical(logits=sp_logits)
        sym = sp_dist.sample()
        li_logits = listen[l_idx, sym]                 # (B, K meanings)
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
        "code": code_of(speak),
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
    lc3 = all(r["adj"]["hinge"]["t_star"] <= r["success_090_cross"]
              for r in onset_seeds.values()
              if r["success_090_cross"] is not None) if onset_seeds else False
    codes = {tuple(r["code"]) for r in learned.values()}
    outcomes = {
        "LC1_learnability": bool(len(learned) >= 4),
        "LC2_nonconstructed_onset": bool(len(onset_seeds) >= 3),
        "LC3_collapse_leads_capability": bool(lc3 and len(onset_seeds) >= 1),
        "LC4_symmetry_breaking": bool(len(codes) >= 2),
        "n_learned": len(learned),
        "n_onset": len(onset_seeds),
        "n_distinct_codes": len(codes),
    }
    report = {
        "status": ("LEARN-CONVENTION non-constructed convention formation; "
                   "no gate, no threshold, no blocked channel; registered "
                   "before run"),
        "config": {"n_agents": N_AGENTS, "K": K, "batch": BATCH,
                   "updates": UPDATES, "eval_every": EVAL_EVERY, "lr": LR,
                   "seeds": N_SEEDS, "seed0": SEED},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_convention.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
