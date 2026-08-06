"""TRI-C: learned high-order carrier with blocked low-order
compilation.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Agents 1,2 follow private iid cues (bit marginals exogenously
mixed); agent 3 sees only partner ACTIONS and must complete parity.
Ladders reuse the generic implementation from
triad_relational_collapse (E-hidden) plus a declared-E variant
conditioning on (c1, c2).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from triad_relational_collapse import entropy, ipf_pairwise_generic, \
    ladder

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_ACT = 10
OBS_DIM = 20
BATCH = 256
N_UPDATES = 2000
CHECKPOINTS = (0, 100, 200, 400, 800, 1200, 1600, 2000)
SEEDS = (95_301, 95_302, 95_303)
EVAL_ROUNDS = 8192


class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU())
        self.pi = nn.Linear(64, N_ACT)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def obs_cue(c: torch.Tensor) -> torch.Tensor:
    B = c.shape[0]
    o = torch.zeros(B, OBS_DIM)
    o[torch.arange(B), c] = 1.0
    return o


def obs_actions(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    B = a1.shape[0]
    o = torch.zeros(B, OBS_DIM)
    o[torch.arange(B), a1] = 1.0
    o[torch.arange(B), N_ACT + a2] = 1.0
    return o


def play_batch(nets, B: int, gen: torch.Generator):
    c1 = torch.randint(0, 2, (B,), generator=gen)
    c2 = torch.randint(0, 2, (B,), generator=gen)
    o1, o2 = obs_cue(c1), obs_cue(c2)
    l1, v1 = nets[0](o1)
    l2, v2 = nets[1](o2)
    a1 = torch.distributions.Categorical(logits=l1).sample()
    a2 = torch.distributions.Categorical(logits=l2).sample()
    o3 = obs_actions(a1, a2)
    l3, v3 = nets[2](o3)
    a3 = torch.distributions.Categorical(logits=l3).sample()
    b1, b2, b3 = a1 % 2, a2 % 2, a3 % 2
    r1 = (b1 == c1).float()
    r2 = (b2 == c2).float()
    r3 = ((b1 ^ b2 ^ b3) == 0).float()
    return {"obs": (o1, o2, o3), "acts": (a1, a2, a3),
            "rews": (r1, r2, r3), "cues": (c1, c2)}


def update(nets, opts, batch):
    for i in range(3):
        obs, act, rew = batch["obs"][i], batch["acts"][i], \
            batch["rews"][i]
        logits, v = nets[i](obs)
        dist = torch.distributions.Categorical(logits=logits)
        adv = (rew - v).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = (-(dist.log_prob(act) * adv).mean()
                + 0.5 * ((v - rew) ** 2).mean()
                - 0.01 * dist.entropy().mean())
        opts[i].zero_grad()
        loss.backward()
        opts[i].step()


def ladder_declared_e(bit_tables_by_e: Dict[int, np.ndarray]) -> Dict:
    """Full ladder on 2x2x2 with E = (c1,c2) declared (4 states)."""
    pe = {e: t / t.sum() for e, t in bit_tables_by_e.items()
          if t.sum() > 0}
    w = {e: bit_tables_by_e[e].sum() for e in pe}
    tot = sum(w.values())
    p_mix = sum((w[e] / tot) * pe[e] for e in pe)
    h_p = entropy(p_mix)
    h_q0 = math.log2(p_mix.size)
    m = [p_mix.sum(axis=tuple(a for a in range(3) if a != i))
         for i in range(3)]
    h_qi = entropy(np.einsum("i,j,k->ijk", m[0], m[1], m[2]))

    def prod_marg(p):
        mm = [p.sum(axis=tuple(a for a in range(3) if a != i))
              for i in range(3)]
        return np.einsum("i,j,k->ijk", mm[0], mm[1], mm[2])

    qe = sum((w[e] / tot) * prod_marg(pe[e]) for e in pe)
    h_qe = entropy(qe)
    qpair = sum((w[e] / tot) * ipf_pairwise_generic(pe[e]) for e in pe)
    h_qpair = entropy(qpair)
    return {"C_individual": h_q0 - h_qi, "C_env": h_qi - h_qe,
            "C_pair": h_qe - h_qpair, "C_high": h_qpair - h_p,
            "C_total": h_q0 - h_p}


def evaluate(nets, seed: int, tag: int) -> Dict:
    gen = torch.Generator().manual_seed(seed * 1_000 + tag)
    with torch.no_grad():
        batch = play_batch(nets, EVAL_ROUNDS, gen)
    a1, a2, a3 = (x.numpy() for x in batch["acts"])
    c1, c2 = (x.numpy() for x in batch["cues"])
    r_tot = float(sum(r.mean().item() for r in batch["rews"]))
    t10 = np.zeros((N_ACT,) * 3)
    np.add.at(t10, (a1, a2, a3), 1)
    b1, b2, b3 = a1 % 2, a2 % 2, a3 % 2
    t2 = np.zeros((2, 2, 2))
    np.add.at(t2, (b1, b2, b3), 1)
    by_e = {}
    for e in range(4):
        mask = (c1 * 2 + c2) == e
        te = np.zeros((2, 2, 2))
        np.add.at(te, (b1[mask], b2[mask], b3[mask]), 1)
        by_e[e] = te

    def bit_entropy(b):
        p = b.mean()
        if p <= 0 or p >= 1:
            return 0.0
        return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))

    return {
        "reward_total": round(r_tot, 4),
        "bit_entropy_1": round(bit_entropy(b1), 4),
        "bit_entropy_2": round(bit_entropy(b2), 4),
        "ladder2_hidden": {k: round(v, 5) for k, v in
                           ladder(t2).items()},
        "ladder2_declared_e": {k: round(v, 5) for k, v in
                               ladder_declared_e(by_e).items()},
        "ladder10_hidden": {k: round(v, 5) for k, v in
                            ladder(t10).items()},
    }


def run_seed(seed: int) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    nets = [AgentNet() for _ in range(3)]
    opts = [torch.optim.Adam(n.parameters(), lr=3e-4) for n in nets]
    gen = torch.Generator().manual_seed(seed)
    curve = {}
    for u in range(N_UPDATES + 1):
        if u in CHECKPOINTS:
            row = evaluate(nets, seed, u)
            curve[str(u)] = row
            print(f"seed {seed} ckpt {u}: r={row['reward_total']:.3f} "
                  f"Chigh2={row['ladder2_hidden']['C_high']:.4f} "
                  f"Hb1={row['bit_entropy_1']:.3f}", flush=True)
        if u == N_UPDATES:
            break
        update(nets, opts, play_batch(nets, BATCH, gen))
    return curve


def main() -> None:
    torch.set_num_threads(4)
    seeds_out = {}
    for seed in SEEDS:
        seeds_out[str(seed)] = run_seed(seed)

    finals = {s: c[str(N_UPDATES)] for s, c in seeds_out.items()}
    learning = [s for s, f in finals.items()
                if f["reward_total"] >= 2.7]
    tric1 = len(learning) >= 2
    tric2 = all(finals[s]["bit_entropy_1"] >= 0.9
                and finals[s]["bit_entropy_2"] >= 0.9
                for s in learning) if learning else False
    tric3 = all(finals[s]["ladder2_hidden"]["C_high"] >= 0.5
                for s in learning) if learning else False
    tric4 = all(finals[s]["ladder2_declared_e"]["C_high"] < 0.05
                for s in learning) if learning else False

    outcomes = {"TRIC1_formation": bool(tric1),
                "TRIC2_mixed_marginals": bool(tric2),
                "TRIC3_learned_high_order": bool(tric3),
                "TRIC4_contract_relativity": bool(tric4),
                "learning_seeds": learning}
    report = {
        "status": ("TRI-C high-order carrier with blocked low-order "
                   "compilation; registered before run"),
        "seeds": seeds_out,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "triad_highorder_cue.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
