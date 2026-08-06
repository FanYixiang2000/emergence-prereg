"""SPREAD-REALIZATION: within-episode commitment in a standard benchmark.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running.
Environment: PettingZoo/MPE2 simple_spread_v3 (Lowe et al. 2017),
unmodified, N=3, discrete actions, max_cycles=50 (documented
constructor parameter). Shared-parameter actor-critic, 3 seeds.

The agent-to-landmark assignment is decided WITHIN each episode; in
"conflict episodes" (initial nearest-landmark map is not a permutation)
the agents must break the symmetry endogenously. The frozen detector
adjudicates the median per-step soft-assignment openness over conflict
episodes.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate
from mpe2 import simple_spread_v3

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N = 3
MAX_CYCLES = 50
TEMP = 0.3
COVER_R = 0.25
UPDATES = 1000
EPISODES_PER_UPDATE = 32
GAMMA = 0.95
LR = 1e-3
EVAL_EPISODES = 500
FORM_CKPT_EVERY = 50
FORM_EPISODES = 100
SEEDS = (97_101, 97_202, 97_303)
LOG2_3 = math.log2(3)
GRID = list(range(MAX_CYCLES))


class SharedPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(18, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, 5)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h)[:, 0]


def make_env():
    return simple_spread_v3.parallel_env(
        N=N, local_ratio=0.5, max_cycles=MAX_CYCLES,
        continuous_actions=False)


def positions(env):
    w = env.unwrapped.world
    ag = np.array([a.state.p_pos for a in w.agents])
    lm = np.array([l.state.p_pos for l in w.landmarks])
    return ag, lm


def dist_matrix(ag, lm):
    return np.linalg.norm(ag[:, None, :] - lm[None, :, :], axis=-1)


def assignment_openness(d):
    """Mean over agents of H(softmax(-d/TEMP)) / log2(N)."""
    z = -d / TEMP
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    h = -(p * np.log2(np.clip(p, 1e-12, None))).sum(axis=1)
    return float(h.mean() / LOG2_3)


def run_episode(env, net, seed, collect):
    obs, _ = env.reset(seed=seed)
    names = list(env.agents)
    rows = {"open": [], "covered_t": None}
    ag0, lm0 = positions(env)
    rows["initial_nearest"] = tuple(np.argmin(dist_matrix(ag0, lm0), axis=1))
    traj = {"obs": [], "act": [], "rew": []}
    for t in range(MAX_CYCLES):
        ag, lm = positions(env)
        d = dist_matrix(ag, lm)
        rows["open"].append(assignment_openness(d))
        if rows["covered_t"] is None and all(
                (d[:, j] < COVER_R).any() for j in range(N)):
            rows["covered_t"] = t
        x = torch.tensor(np.stack([obs[a] for a in names]),
                         dtype=torch.float32)
        with torch.no_grad():
            logits, _ = net(x)
            acts = torch.distributions.Categorical(logits=logits).sample()
        actions = {a: int(acts[i]) for i, a in enumerate(names)}
        nobs, rews, term, trunc, _ = env.step(actions)
        if collect:
            traj["obs"].append(x.numpy())
            traj["act"].append(acts.numpy())
            traj["rew"].append(np.array([rews[a] for a in names],
                                        dtype=np.float32))
        obs = nobs
        if all(term.values()) or all(trunc.values()):
            break
    ag, lm = positions(env)
    rows["final_nearest"] = tuple(np.argmin(dist_matrix(ag, lm), axis=1))
    return rows, traj


def update_net(net, opt, trajs):
    obs = torch.tensor(np.concatenate(
        [np.stack(t["obs"]).reshape(-1, 18) for t in trajs]))
    act = torch.tensor(np.concatenate(
        [np.stack(t["act"]).reshape(-1) for t in trajs]))
    rets = []
    for t in trajs:
        rew = np.stack(t["rew"])
        g = np.zeros_like(rew)
        run = np.zeros(rew.shape[1], dtype=np.float32)
        for i in reversed(range(len(rew))):
            run = rew[i] + GAMMA * run
            g[i] = run
        rets.append(g.reshape(-1))
    ret = torch.tensor(np.concatenate(rets))
    logits, v = net(obs)
    dist = torch.distributions.Categorical(logits=logits)
    adv = ret - v.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss = (-(dist.log_prob(act) * adv).mean()
            + 0.5 * ((v - ret) ** 2).mean()
            - 0.01 * dist.entropy().mean())
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()


def realization_eval(net, seed_base):
    env = make_env()
    conflict_curves, all_curves = [], []
    cover_times, perms = [], []
    for e in range(EVAL_EPISODES):
        rows, _ = run_episode(env, net, seed_base + e, collect=False)
        curve = rows["open"]
        if len(curve) < MAX_CYCLES:
            curve = curve + [curve[-1]] * (MAX_CYCLES - len(curve))
        all_curves.append(curve)
        is_conflict = len(set(rows["initial_nearest"])) < N
        if is_conflict:
            conflict_curves.append(curve)
            if rows["covered_t"] is not None:
                cover_times.append(rows["covered_t"])
                perms.append(rows["final_nearest"])
    med = np.median(np.array(conflict_curves), axis=0)
    adj = adjudicate(GRID, med * LOG2_3)
    h = adj.get("hinge", {})
    return {
        "n_conflict": len(conflict_curves),
        "n_covered_conflict": len(cover_times),
        "median_curve_conflict": [round(float(v), 5) for v in med],
        "median_curve_all": [round(float(v), 5)
                             for v in np.median(np.array(all_curves), axis=0)],
        "b5_onset": adj["b5_onset"],
        "t_star": h.get("t_star"),
        "delta_bic": h.get("delta_bic"),
        "median_cover_time": (float(np.median(cover_times))
                              if cover_times else None),
        "n_distinct_permutations": len({p for p in perms
                                        if len(set(p)) == N}),
        "coverage_rate_conflict": (len(cover_times)
                                   / max(len(conflict_curves), 1)),
    }


def formation_eval(net, seed_base):
    env = make_env()
    finals = []
    for e in range(FORM_EPISODES):
        rows, _ = run_episode(env, net, seed_base + e, collect=False)
        finals.append(rows["final_nearest"])
    cnt = Counter(finals)
    p = np.array([c / len(finals) for c in cnt.values()])
    h = -(p * np.log2(p)).sum()
    return float(h / math.log2(N ** N))


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = SharedPolicy()
    untrained = realization_eval(net, seed + 900_000)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    env = make_env()
    form_grid, form_open, mean_rews = [], [], []
    ep_seed = seed
    for u in range(UPDATES):
        trajs = []
        rew_sum = 0.0
        for _ in range(EPISODES_PER_UPDATE):
            ep_seed += 1
            _rows, traj = run_episode(env, net, ep_seed, collect=True)
            trajs.append(traj)
            rew_sum += float(np.stack(traj["rew"]).sum())
        update_net(net, opt, trajs)
        if (u + 1) % FORM_CKPT_EVERY == 0:
            form_grid.append(u + 1)
            form_open.append(formation_eval(net, seed + 500_000 + u))
            mean_rews.append(rew_sum / EPISODES_PER_UPDATE)
            print(f"  seed {seed} upd {u+1}: R={mean_rews[-1]:.1f} "
                  f"form_open={form_open[-1]:.4f}", flush=True)
    trained = realization_eval(net, seed + 700_000)
    form_adj = adjudicate(form_grid, np.array(form_open) * LOG2_3)
    return {
        "trained": trained,
        "untrained": untrained,
        "formation_grid": form_grid,
        "formation_openness": [round(v, 5) for v in form_open],
        "formation_b5": form_adj["b5_onset"],
        "mean_reward_final": mean_rews[-1],
    }


def main() -> None:
    torch.set_num_threads(4)
    rows = {}
    for seed in SEEDS:
        print(f"=== seed {seed}", flush=True)
        rows[str(seed)] = run_seed(seed)
        t = rows[str(seed)]["trained"]
        print(f"seed {seed}: B5={t['b5_onset']} t*={t['t_star']} "
              f"dBIC={t['delta_bic']} cover_t={t['median_cover_time']} "
              f"perms={t['n_distinct_permutations']} "
              f"cov_rate={t['coverage_rate_conflict']:.2f}", flush=True)

    onset = [r for r in rows.values() if r["trained"]["b5_onset"]]
    sr2 = all(r["trained"]["t_star"] < r["trained"]["median_cover_time"]
              for r in onset
              if r["trained"]["median_cover_time"] is not None)
    outcomes = {
        "SR1_onset_ge_2of3": bool(len(onset) >= 2),
        "SR2_collapse_before_coverage": bool(sr2 and onset),
        "SR3_ge3_permutations_all_seeds": bool(all(
            r["trained"]["n_distinct_permutations"] >= 3
            for r in rows.values())),
        "SR4_untrained_no_onset": bool(all(
            not r["untrained"]["b5_onset"] for r in rows.values())),
        "SR5_formation_no_onset": bool(all(
            not r["formation_b5"] for r in rows.values())),
        "n_onset": len(onset),
    }
    out = OUTPUTS / "mpe_spread_realization.json"
    out.write_text(json.dumps({
        "status": ("SPREAD-REALIZATION on unmodified PettingZoo/MPE2 "
                   "simple_spread_v3; registered before run"),
        "config": {"N": N, "max_cycles": MAX_CYCLES, "temp": TEMP,
                   "cover_r": COVER_R, "updates": UPDATES,
                   "episodes_per_update": EPISODES_PER_UPDATE,
                   "eval_episodes": EVAL_EPISODES, "seeds": SEEDS},
        "seeds": rows,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
