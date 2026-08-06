"""TRI (E4): three-agent learned relational collapse.

Registered in V2_ALIGNMENT_PREREGISTRATION.md wave 3 before running.
Three independent policy-gradient learners, 10 actions each, T=32
rounds; team reward per round is +1 if (a1%2 + a2%2 + a3%2) is odd,
minus 0.2 per agent that repeats its own previous action. Each agent
observes everyone's previous action (one-hot) and its own id. The
joint-action collapse ladder (10x10x10 and the 2x2x2 parity
projection) is computed at declared training checkpoints. This is
the first learned system in the project where C_high is not
degenerate by construction.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch
import torch.nn as nn

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_AGENTS = 3
N_ACT = 10
T_ROUNDS = 32
BATCH = 64
UPDATES = 3000
CHECKPOINTS = (0, 50, 100, 200, 400, 800, 1600, 3000)
SEEDS = (95101, 95102, 95103)
EVAL_EPISODES = 2000
OBS_DIM = N_AGENTS * N_ACT + N_AGENTS
EPS = 1e-15
# TRI-B: agent 3 observes agents 1,2's CURRENT actions (sequential
# interaction); registered separately, run with --variant sequential.
SEQUENTIAL = False
SEEDS_B = (95201, 95202, 95203)


class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU())
        self.pi = nn.Linear(64, N_ACT)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def make_obs(prev: torch.Tensor) -> torch.Tensor:
    """prev: (B, 3) long, -1 at round 0. Returns (B, 3, OBS_DIM)."""
    B = prev.shape[0]
    onehot = torch.zeros(B, N_AGENTS, N_ACT)
    mask = prev >= 0
    idx = prev.clamp(min=0)
    onehot.scatter_(2, idx.unsqueeze(-1), mask.float().unsqueeze(-1))
    flat = onehot.reshape(B, N_AGENTS * N_ACT)
    obs = torch.zeros(B, N_AGENTS, OBS_DIM)
    for i in range(N_AGENTS):
        obs[:, i, :N_AGENTS * N_ACT] = flat
        obs[:, i, N_AGENTS * N_ACT + i] = 1.0
    return obs


def team_reward(acts: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """acts, prev: (B, 3). Returns (B,)."""
    parity = (acts % 2).sum(dim=1) % 2
    r = parity.float()
    rep = ((acts == prev) & (prev >= 0)).float().sum(dim=1)
    return r - 0.2 * rep


def run_episodes(nets, n_episodes: int, gen: torch.Generator,
                 collect_tables: bool = False):
    """Runs a batch of episodes. Returns mean reward and (optionally)
    joint count tables for rounds 1..T-1."""
    prev = torch.full((n_episodes, N_AGENTS), -1, dtype=torch.long)
    total_r = torch.zeros(n_episodes)
    table10 = np.zeros((N_ACT, N_ACT, N_ACT))
    table2 = np.zeros((2, 2, 2))
    traj = []  # (obs, acts, logp, val, rew) per round
    for t in range(T_ROUNDS):
        obs = make_obs(prev)
        logps, vals, acts = [], [], []
        with torch.no_grad():
            for i, net in enumerate(nets):
                obs_i = obs[:, i]
                if SEQUENTIAL and i == 2:
                    # agent 3 sees agents 1,2's CURRENT actions and
                    # its own previous action (TRI-B)
                    seq_prev = torch.stack(
                        [acts[0], acts[1], prev[:, 2]], dim=1)
                    obs_i = make_obs(seq_prev)[:, i]
                logits, v = net(obs_i)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample() if gen is None else torch.multinomial(
                    torch.softmax(logits, dim=-1), 1, generator=gen
                ).squeeze(-1)
                logps.append(dist.log_prob(a))
                vals.append(v)
                acts.append(a)
                if SEQUENTIAL and i == 2:
                    obs = obs.clone()
                    obs[:, 2] = obs_i  # store what agent 3 actually saw
        acts_t = torch.stack(acts, dim=1)
        r = team_reward(acts_t, prev)
        total_r += r
        traj.append((obs, acts_t, torch.stack(logps, dim=1),
                     torch.stack(vals, dim=1), r))
        if collect_tables and t >= 1:
            a = acts_t.numpy()
            np.add.at(table10, (a[:, 0], a[:, 1], a[:, 2]), 1.0)
            b = a % 2
            np.add.at(table2, (b[:, 0], b[:, 1], b[:, 2]), 1.0)
        prev = acts_t
    mean_r = float(total_r.mean() / T_ROUNDS)
    return mean_r, traj, table10, table2


RETURN_MODE = "to_go"  # TRI-B amendment: "immediate" (stage-game credit)


def update(nets, opts, traj):
    T = len(traj)
    rews = torch.stack([row[4] for row in traj])          # (T, B)
    if RETURN_MODE == "immediate":
        rets = rews
    else:
        rets = torch.flip(torch.cumsum(torch.flip(rews, [0]), 0), [0])
    for i, (net, opt) in enumerate(zip(nets, opts)):
        obs = torch.cat([row[0][:, i] for row in traj])
        acts = torch.cat([row[1][:, i] for row in traj])
        ret_i = rets.reshape(-1)
        logits, v = net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        adv = (ret_i - v).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = (-(dist.log_prob(acts) * adv).mean()
                + 0.5 * ((v - ret_i) ** 2).mean()
                - 0.01 * dist.entropy().mean())
        opt.zero_grad()
        loss.backward()
        opt.step()


def entropy(p: np.ndarray) -> float:
    q = p[p > EPS]
    return float(-(q * np.log2(q)).sum())


def ipf_pairwise_generic(p: np.ndarray, iters: int = 300) -> np.ndarray:
    targets = {(0, 1): p.sum(axis=2), (0, 2): p.sum(axis=1),
               (1, 2): p.sum(axis=0)}
    q = np.full_like(p, 1.0 / p.size)
    for _ in range(iters):
        for (i, j), tgt in targets.items():
            axis = ({0, 1, 2} - {i, j}).pop()
            cur = q.sum(axis=axis)
            ratio = np.where(cur > EPS, tgt / np.maximum(cur, EPS), 0.0)
            shape = [1, 1, 1]
            shape[i], shape[j] = p.shape[i], p.shape[j]
            q = q * ratio.reshape(shape)
        s = q.sum()
        if s > 0:
            q = q / s
    return q


def ladder(table: np.ndarray) -> dict:
    p = table / table.sum()
    h_p = entropy(p)
    h_q0 = math.log2(p.size)
    m = [p.sum(axis=tuple(a for a in range(3) if a != i))
         for i in range(3)]
    qi = np.einsum("i,j,k->ijk", m[0], m[1], m[2])
    h_qi = entropy(qi)
    h_qpair = entropy(ipf_pairwise_generic(p))
    return {
        "C_individual": h_q0 - h_qi,
        "C_env": 0.0,  # declared trivial: no exogenous cue exists
        "C_pair": h_qi - h_qpair,
        "C_high": h_qpair - h_p,
        "C_total": h_q0 - h_p,
        "H_P": h_p,
    }


def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    nets = [AgentNet() for _ in range(N_AGENTS)]
    opts = [torch.optim.Adam(n.parameters(), lr=3e-4) for n in nets]
    curve = {}
    reward_curve = {}

    def evaluate(tag: int):
        gen = torch.Generator().manual_seed(seed * 100 + tag)
        with torch.no_grad():
            mean_r, _t, t10, t2 = run_episodes(
                nets, EVAL_EPISODES, gen, collect_tables=True)
        row = {"reward_per_round": mean_r,
               "ladder10": {k: round(v, 5) for k, v in
                            ladder(t10).items()},
               "ladder2_parity": {k: round(v, 5) for k, v in
                                  ladder(t2).items()}}
        curve[str(tag)] = row
        reward_curve[str(tag)] = mean_r
        l10, l2 = row["ladder10"], row["ladder2_parity"]
        print(f"  seed {seed} ckpt {tag}: r={mean_r:.3f} "
              f"Ctot={l10['C_total']:.3f} Cpair={l10['C_pair']:.3f} "
              f"Chigh={l10['C_high']:.3f} "
              f"Chigh2={l2['C_high']:.3f}", flush=True)

    evaluate(0)
    pending = [c for c in CHECKPOINTS if c > 0]
    for u in range(1, UPDATES + 1):
        _r, traj, _a, _b = run_episodes(nets, BATCH, None)
        update(nets, opts, traj)
        if pending and u == pending[0]:
            pending.pop(0)
            evaluate(u)
    return {"seed": seed, "curve": curve, "reward_curve": reward_curve}


def main() -> None:
    import argparse
    global SEQUENTIAL, RETURN_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("simultaneous", "sequential"),
                    default="simultaneous")
    args = ap.parse_args()
    SEQUENTIAL = args.variant == "sequential"
    if SEQUENTIAL:
        RETURN_MODE = "immediate"  # TRI-B amendment, declared in prereg
    seeds = SEEDS_B if SEQUENTIAL else SEEDS

    torch.set_num_threads(2)
    t0 = time.time()
    seeds_out = {}
    for seed in seeds:
        seeds_out[str(seed)] = run_seed(seed)

    def final(seed, key, sub):
        return seeds_out[str(seed)]["curve"][str(UPDATES)][key][sub]

    def first(seed, key, sub):
        return seeds_out[str(seed)]["curve"]["0"][key][sub]

    tri1 = sum(
        1 for s in seeds
        if seeds_out[str(s)]["curve"][str(UPDATES)]["reward_per_round"]
        > 0.8
        and (final(s, "ladder10", "C_total")
             - first(s, "ladder10", "C_total")) > 0.5) >= 2
    tri2 = sum(
        1 for s in seeds
        if (final(s, "ladder10", "C_pair")
            + final(s, "ladder10", "C_high")) > 0.2) >= 2
    tri3 = sum(
        1 for s in seeds
        if final(s, "ladder2_parity", "C_high") > 0.2) >= 2
    failures = [s for s in seeds
                if seeds_out[str(s)]["curve"][str(UPDATES)]
                ["reward_per_round"] < 0.5]

    prefix = "TRIB" if SEQUENTIAL else "TRI"
    report = {
        "status": (f"{prefix} three-agent learned relational collapse "
                   f"({args.variant}); registered in "
                   "V2_ALIGNMENT_PREREGISTRATION.md; first learned "
                   "system with non-degenerate C_high"),
        "variant": args.variant,
        "minutes": round((time.time() - t0) / 60, 2),
        "seeds": seeds_out,
        "registered_outcomes": {
            f"{prefix}1_formation": bool(tri1),
            f"{prefix}2_relational_carrier": bool(tri2),
            f"{prefix}3_higher_order": bool(tri3),
            "training_failure_seeds": failures,
        },
    }
    suffix = "_sequential" if SEQUENTIAL else ""
    out = OUTPUTS / f"triad_relational_collapse{suffix}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
