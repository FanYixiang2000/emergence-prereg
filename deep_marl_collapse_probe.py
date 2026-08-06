"""Within-episode possibility collapse in a deep MARL benchmark.

Registered in DEEP_MARL_PREREGISTRATION.md BEFORE any trained policy was
probed. Environment: PettingZoo/MPE simple_spread (3 agents, 3 landmarks,
cooperative coverage), an external benchmark we did not design. Policies:
parameter-shared PPO (MLP 64-64, categorical actions), trained here with
standard hyperparameters; also untrained, scripted-greedy, and noise
controls.

The measured objects are the same as within_episode_collapse_probe.py:
P_t(B | s_t) estimated by Monte-Carlo rollouts of the learned stochastic
policy from snapshotted world states; basin = the coverage assignment
(nearest landmark per agent at the final step, 27 outcomes); useful macro
structure = the assignment is a bijection. Do-operators at the maximal-
collapse (commit) step follow the registered minimal design.

Pilot notes (allowed scope: training + estimator params only; logs kept):

1. The first pilot (300-episode training, tag=pilot) exposed an
   ESTIMATOR BUG, not a parameter choice: restoring raw world state into
   a parallel-wrapper env whose previous rollout had truncated left the
   wrapper's agent list empty, so every subsequent step was a silent
   no-op and all future distributions collapsed to the current state
   (potential 0.0 everywhere). Fixed by resetting the rollout env before
   each restore; log in outputs/deep_marl_pilot_log.txt.
2. Training pilots (tags train_pilot2, seed11/22/33, seed11_long):
   decentralized-critic PPO plateaued at mean return ~-21 with final
   bijection rate 0.20-0.28 -- too weak a policy for the D2 threshold.
   Per the registration's training-tunable scope, the critic was
   upgraded to a MAPPO-style centralized critic over concatenated
   observations (training only; execution and all probe rollouts remain
   decentralized on the actor head), width 64->128, plus LR/entropy
   annealing. Pilot (tag mappo_pilot): return -19.5, bijection 0.475.
   Logs in outputs/deep_marl_seed*_log.txt, deep_marl_mappo_pilot_log.txt.
3. Non-registered diagnostics added for interpretation only (assignment
   JS and agent-0-on-target mass under the do-operators); registered
   quantities and thresholds untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from mpe2 import simple_spread_v3

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

N_AGENTS = 3
MAX_CYCLES = 25
OBS_DIM = 18
N_ACTIONS = 5
BASIN_COUNT = 27  # 3 agents x 3 landmark indices

# Estimator parameters (pilot-tunable per registration).
PROBE_ROLLOUTS = 48
EVAL_EPISODES = 40

# Training hyperparameters (pilot-tunable per registration).
TRAIN_EPISODES = 24000
PPO_LR = 7e-4
PPO_LR_FINAL = 1e-4
PPO_CLIP = 0.2
PPO_EPOCHS = 8
MINIBATCH = 512
ENT_COEF = 0.02
ENT_COEF_FINAL = 0.003
GAMMA = 0.95
GAE_LAMBDA = 0.95
BATCH_EPISODES = 32


class PolicyNet(nn.Module):
    """Actor on local observations; MAPPO-style centralized critic.

    The critic sees the concatenated observations of all agents (training
    only); execution and all probe rollouts use the actor head alone, so
    the measured system remains decentralized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.pi = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM * N_AGENTS, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi(self.body(obs))

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(global_obs).squeeze(-1)


def make_env(seed: int):
    env = simple_spread_v3.parallel_env(
        N=N_AGENTS, local_ratio=0.5, max_cycles=MAX_CYCLES,
        continuous_actions=False)
    env.reset(seed=seed)
    return env


def world_snapshot(env):
    w = env.unwrapped.world
    return (
        [(a.state.p_pos.copy(), a.state.p_vel.copy(), a.state.c.copy())
         for a in w.agents],
        [l.state.p_pos.copy() for l in w.landmarks],
        env.unwrapped.steps,
    )


def world_restore(env, snap) -> None:
    agents, landmarks, steps = snap
    w = env.unwrapped.world
    for agent, (pos, vel, c) in zip(w.agents, agents):
        agent.state.p_pos = pos.copy()
        agent.state.p_vel = vel.copy()
        agent.state.c = c.copy()
    for landmark, pos in zip(w.landmarks, landmarks):
        landmark.state.p_pos = pos.copy()
    env.unwrapped.steps = steps


def observe_all(env) -> Dict[str, np.ndarray]:
    return {a: env.unwrapped.observe(a).astype(np.float32)
            for a in env.possible_agents}


def assignment_basin(env) -> Tuple[int, ...]:
    w = env.unwrapped.world
    basin = []
    for agent in w.agents:
        dists = [float(np.linalg.norm(agent.state.p_pos - l.state.p_pos))
                 for l in w.landmarks]
        basin.append(int(np.argmin(dists)))
    return tuple(basin)


def is_bijection(basin: Tuple[int, ...]) -> bool:
    return len(set(basin)) == N_AGENTS


GREEDY_EPS = 0.05


def greedy_action_toward(env, agent_idx: int, target: np.ndarray) -> int:
    """Move along the axis with the largest gap to the target point."""
    pos = env.unwrapped.world.agents[agent_idx].state.p_pos
    delta = target - pos
    if float(np.linalg.norm(delta)) < GREEDY_EPS:
        return 0
    if abs(delta[0]) >= abs(delta[1]):
        return 2 if delta[0] > 0 else 1
    return 4 if delta[1] > 0 else 3


class Controller:
    """Uniform interface: policy network, scripted greedy, or noise."""

    def __init__(self, kind: str, net: Optional[PolicyNet] = None) -> None:
        self.kind = kind
        self.net = net

    def act(self, env, obs: Dict[str, np.ndarray], rng: random.Random,
            interventions: Optional[Dict[int, Dict]] = None) -> Dict[str, int]:
        acts: Dict[str, int] = {}
        names = env.possible_agents
        if self.kind == "policy":
            batch = torch.tensor(np.stack([obs[a] for a in names]))
            with torch.no_grad():
                logits = self.net(batch)
                probs = torch.softmax(logits, dim=1).numpy()
        for i, name in enumerate(names):
            iv = (interventions or {}).get(i)
            if iv and iv["type"] == "do_commit":
                acts[name] = greedy_action_toward(env, i, iv["target"])
                continue
            if self.kind == "noise":
                acts[name] = rng.randrange(N_ACTIONS)
            elif self.kind == "greedy_nearest":
                w = env.unwrapped.world
                agent = w.agents[i]
                dists = [float(np.linalg.norm(agent.state.p_pos - l.state.p_pos))
                         for l in w.landmarks]
                target = w.landmarks[int(np.argmin(dists))].state.p_pos
                acts[name] = greedy_action_toward(env, i, target)
            else:
                p = probs[i].copy()
                if iv and iv["type"] == "do_block":
                    # Minimal restriction: renormalize over actions that do
                    # not reduce distance to the blocked landmark.
                    pos = env.unwrapped.world.agents[i].state.p_pos
                    target = iv["target"]
                    allowed = np.ones(N_ACTIONS)
                    step_vec = {1: (-0.1, 0), 2: (0.1, 0), 3: (0, -0.1), 4: (0, 0.1)}
                    d0 = float(np.linalg.norm(pos - target))
                    for action, (dx, dy) in step_vec.items():
                        nd = float(np.linalg.norm(pos + np.array([dx, dy]) - target))
                        if nd < d0:
                            allowed[action] = 0.0
                    p = p * allowed
                    if p.sum() <= 0:
                        p = allowed / max(allowed.sum(), 1.0)
                    else:
                        p = p / p.sum()
                acts[name] = int(np.searchsorted(np.cumsum(p), rng.random()))
                acts[name] = min(acts[name], N_ACTIONS - 1)
        return acts


def train_ppo(seed: int) -> PolicyNet:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR)
    env = make_env(seed)
    ep_returns: List[float] = []

    obs_buf: List[np.ndarray] = []
    gobs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    logp_buf: List[float] = []
    adv_buf: List[float] = []
    ret_buf: List[float] = []
    episodes_in_batch = 0

    for episode in range(TRAIN_EPISODES):
        frac = episode / max(TRAIN_EPISODES - 1, 1)
        lr_now = PPO_LR + (PPO_LR_FINAL - PPO_LR) * frac
        ent_now = ENT_COEF + (ENT_COEF_FINAL - ENT_COEF) * frac
        for group in opt.param_groups:
            group["lr"] = lr_now
        obs, _ = env.reset(seed=rng.randrange(10 ** 9))
        names = env.possible_agents
        traj_obs, traj_gobs, traj_act, traj_logp, traj_rew, traj_val = \
            [], [], [], [], [], []
        total = 0.0
        for _t in range(MAX_CYCLES):
            batch = torch.tensor(np.stack([obs[a] for a in names]))
            gobs = batch.reshape(1, -1)
            with torch.no_grad():
                logits = net(batch)
                value = net.value(gobs)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logps = dist.log_prob(actions)
            act_dict = {a: int(actions[i]) for i, a in enumerate(names)}
            nobs, rewards, terms, truncs, _ = env.step(act_dict)
            traj_obs.append(batch.numpy())
            traj_gobs.append(gobs.numpy()[0])
            traj_act.append(actions.numpy())
            traj_logp.append(logps.numpy())
            traj_val.append(float(value))
            # MAPPO-style shared signal: mean reward over agents.
            traj_rew.append(float(np.mean([rewards[a] for a in names])))
            total += float(np.mean([rewards[a] for a in names]))
            obs = nobs
            if all(terms.values()) or all(truncs.values()):
                break
        ep_returns.append(total)

        # Shared-critic GAE; episodes end by truncation, so bootstrap the
        # tail with V(s_T) instead of zero.
        with torch.no_grad():
            tail_v = float(net.value(torch.tensor(
                np.stack([obs[a] for a in names])).reshape(1, -1)))
        T = len(traj_rew)
        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in reversed(range(T)):
            next_v = traj_val[t + 1] if t + 1 < T else tail_v
            delta = traj_rew[t] + GAMMA * next_v - traj_val[t]
            last = delta + GAMMA * GAE_LAMBDA * last
            adv[t] = last
        ret = adv + np.array(traj_val, dtype=np.float32)
        for t in range(T):
            for i in range(N_AGENTS):
                obs_buf.append(traj_obs[t][i])
                act_buf.append(int(traj_act[t][i]))
                logp_buf.append(float(traj_logp[t][i]))
                adv_buf.append(float(adv[t]))
            gobs_buf.append(traj_gobs[t])
            ret_buf.append(float(ret[t]))
        episodes_in_batch += 1

        if episodes_in_batch >= BATCH_EPISODES:
            obs_t = torch.tensor(np.stack(obs_buf))
            act_t = torch.tensor(act_buf)
            logp_t = torch.tensor(logp_buf)
            adv_t = torch.tensor(adv_buf)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            gobs_t = torch.tensor(np.stack(gobs_buf))
            ret_t = torch.tensor(ret_buf)
            n = len(obs_t)
            m = len(gobs_t)
            for _epoch in range(PPO_EPOCHS):
                perm = torch.randperm(n)
                for start in range(0, n, MINIBATCH):
                    idx = perm[start:start + MINIBATCH]
                    logits = net(obs_t[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    ratio = torch.exp(dist.log_prob(act_t[idx]) - logp_t[idx])
                    s1 = ratio * adv_t[idx]
                    s2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_t[idx]
                    loss = (-torch.min(s1, s2).mean()
                            - ent_now * dist.entropy().mean())
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                vperm = torch.randperm(m)
                for start in range(0, m, MINIBATCH):
                    idx = vperm[start:start + MINIBATCH]
                    values = net.value(gobs_t[idx])
                    vloss = 0.5 * ((values - ret_t[idx]) ** 2).mean()
                    opt.zero_grad()
                    vloss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
            obs_buf, gobs_buf, act_buf, logp_buf, adv_buf, ret_buf = \
                [], [], [], [], [], []
            episodes_in_batch = 0

        if (episode + 1) % 500 == 0:
            recent = sum(ep_returns[-100:]) / 100
            print(f"  seed {seed} episode {episode + 1}: "
                  f"mean return (last 100) {recent:.2f}", flush=True)
    return net


def entropy_bits(counts: Dict[Tuple[int, ...], int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total, 2)
                for c in counts.values() if c > 0)


def kl_bits(p: Dict, q: Dict, eps: float = 1e-9) -> float:
    keys = set(p) | set(q)
    tp = sum(p.values()) or 1
    tq = sum(q.values()) or 1
    out = 0.0
    for k in keys:
        pk = p.get(k, 0) / tp
        qk = q.get(k, 0) / tq
        if pk > 0:
            out += pk * math.log((pk + eps) / (qk + eps), 2)
    return out


def future_distribution(env, snap, controller: Controller, rng: random.Random,
                        interventions: Optional[Dict[int, Dict]] = None,
                        rollouts: int = PROBE_ROLLOUTS) -> Dict[Tuple[int, ...], int]:
    counts: Dict[Tuple[int, ...], int] = {}
    for _ in range(rollouts):
        # Reset first: a truncated previous rollout empties the wrapper's
        # agent bookkeeping, and restoring raw world state alone would make
        # every subsequent env.step a silent no-op (pilot bug, fixed).
        env.reset(seed=0)
        world_restore(env, snap)
        for _t in range(MAX_CYCLES - snap[2]):
            obs = observe_all(env)
            acts = controller.act(env, obs, rng, interventions)
            env.step(acts)
        basin = assignment_basin(env)
        counts[basin] = counts.get(basin, 0) + 1
    return counts


def win_mass(counts: Dict[Tuple[int, ...], int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(c for basin, c in counts.items() if is_bijection(basin)) / total


def target_mass(counts: Dict[Tuple[int, ...], int], agent: int, lm: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(c for basin, c in counts.items() if basin[agent] == lm) / total


def js_bits(p: Dict[Tuple[int, ...], int], q: Dict[Tuple[int, ...], int]) -> float:
    tp = sum(p.values()) or 1
    tq = sum(q.values()) or 1
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pk = p.get(k, 0) / tp
        qk = q.get(k, 0) / tq
        mk = 0.5 * (pk + qk)
        if pk > 0:
            out += 0.5 * pk * math.log(pk / mk, 2)
        if qk > 0:
            out += 0.5 * qk * math.log(qk / mk, 2)
    return out


def probe_condition(name: str, controller: Controller, seed: int) -> Dict:
    rng = random.Random(seed * 7919 + 13)
    env = make_env(seed)
    sim = make_env(seed + 10 ** 6)  # separate env instance for rollouts

    episodes: List[Dict] = []
    for ep in range(EVAL_EPISODES):
        env.reset(seed=1_000_000 + 997 * seed + ep)
        snaps = []
        step_stats: List[Dict] = []
        prev_dist: Optional[Dict] = None
        for t in range(MAX_CYCLES):
            snap = world_snapshot(env)
            dist = future_distribution(sim, snap, controller, rng)
            stat = {
                "t": t,
                "potential_bits": entropy_bits(dist),
                "p_win": win_mass(dist),
                "collapse_bits": kl_bits(dist, prev_dist) if prev_dist else 0.0,
            }
            step_stats.append(stat)
            snaps.append(snap)
            prev_dist = dist
            obs = observe_all(env)
            env.step(controller.act(env, obs, rng))
        final_basin = assignment_basin(env)

        commit = max(range(1, len(step_stats)),
                     key=lambda i: step_stats[i]["collapse_bits"])
        record = {
            "episode": ep,
            "early_potential_bits": float(np.mean(
                [s["potential_bits"] for s in step_stats[:3]])),
            "p_win_start": step_stats[0]["p_win"],
            "p_win_end": step_stats[-1]["p_win"],
            "final_bijection": int(is_bijection(final_basin)),
            "commit_step": commit,
            "commit_collapse_bits": step_stats[commit]["collapse_bits"],
        }
        if controller.kind == "policy":
            # Do-operators at the pre-commit snapshot for agent 0, aimed at
            # its realized final landmark.
            w = sim.unwrapped.world
            world_restore(sim, snaps[commit - 1])
            target_idx = final_basin[0]
            target = w.landmarks[target_idx].state.p_pos.copy()
            d_commit = future_distribution(
                sim, snaps[commit - 1], controller, rng,
                {0: {"type": "do_commit", "target": target}})
            d_block = future_distribution(
                sim, snaps[commit - 1], controller, rng,
                {0: {"type": "do_block", "target": target}})
            record["p_win_do_commit"] = win_mass(d_commit)
            record["p_win_do_block"] = win_mass(d_block)
            # Reported diagnostic (not a registered quantity): does the
            # intervention redirect WHICH assignment the future reaches,
            # even if the win-mass is preserved by re-coordination?
            record["do_assignment_js_bits"] = js_bits(d_commit, d_block)
            record["p_agent0_on_target_do_commit"] = target_mass(d_commit, 0,
                                                                 target_idx)
            record["p_agent0_on_target_do_block"] = target_mass(d_block, 0,
                                                                target_idx)
        episodes.append(record)

    def mean(key: str) -> float:
        vals = [e[key] for e in episodes if key in e]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "condition": name,
        "n_episodes": len(episodes),
        "early_potential_bits": mean("early_potential_bits"),
        "p_win_start": mean("p_win_start"),
        "p_win_end": mean("p_win_end"),
        "final_bijection_rate": mean("final_bijection"),
        "mean_commit_step": mean("commit_step"),
        "episodes": episodes,
    }
    if controller.kind == "policy":
        gaps = [e["p_win_do_commit"] - e["p_win_do_block"] for e in episodes]
        wins = sum(1 for g in gaps if g > 0)
        losses = sum(1 for g in gaps if g < 0)
        summary["do_gap_median"] = float(np.median(gaps))
        summary["do_gap_mean"] = float(np.mean(gaps))
        summary["do_sign_wins"] = wins
        summary["do_sign_losses"] = losses
        summary["do_sign_p"] = sign_test_p(wins, losses)
    return summary


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def main() -> None:
    global TRAIN_EPISODES
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    parser.add_argument("--train_episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    TRAIN_EPISODES = args.train_episodes

    torch.set_num_threads(16)
    results: Dict[str, Dict] = {}

    for seed in args.seeds:
        print(f"training PPO seed {seed} ...", flush=True)
        net = train_ppo(seed)
        results[f"trained_seed{seed}"] = probe_condition(
            f"trained_seed{seed}", Controller("policy", net), seed)
        print(json.dumps({k: v for k, v in results[f'trained_seed{seed}'].items()
                          if k != "episodes"}, indent=2), flush=True)

    torch.manual_seed(999)
    results["untrained"] = probe_condition("untrained",
                                           Controller("policy", PolicyNet()), 44)
    results["greedy_nearest"] = probe_condition("greedy_nearest",
                                                Controller("greedy_nearest"), 55)
    results["noise"] = probe_condition("noise", Controller("noise"), 66)

    # Registered predictions.
    trained = [v for k, v in results.items() if k.startswith("trained_")]
    d1 = {
        "trained_early_potential": [t["early_potential_bits"] for t in trained],
        "pass": all(t["early_potential_bits"] >= 1.0 for t in trained),
    }
    d2 = {
        "trained_win_shift": [t["p_win_end"] - t["p_win_start"] for t in trained],
        "trained_bijection_rate": [t["final_bijection_rate"] for t in trained],
        "untrained_bijection_rate": results["untrained"]["final_bijection_rate"],
        "noise_bijection_rate": results["noise"]["final_bijection_rate"],
        "pass": (all(t["p_win_end"] - t["p_win_start"] > 0 for t in trained)
                 and all(t["final_bijection_rate"] >= 0.5 for t in trained)
                 and results["untrained"]["final_bijection_rate"] < 0.35
                 and results["noise"]["final_bijection_rate"] < 0.35),
    }
    pooled_wins = sum(t["do_sign_wins"] for t in trained)
    pooled_losses = sum(t["do_sign_losses"] for t in trained)
    pooled_gaps: List[float] = []
    for t in trained:
        pooled_gaps.extend(e["p_win_do_commit"] - e["p_win_do_block"]
                           for e in t["episodes"])
    d3 = {
        "pooled_do_gap_median": float(np.median(pooled_gaps)),
        "pooled_sign_wins": pooled_wins,
        "pooled_sign_losses": pooled_losses,
        "pooled_sign_p": sign_test_p(pooled_wins, pooled_losses),
        "pass": (float(np.median(pooled_gaps)) > 0
                 and sign_test_p(pooled_wins, pooled_losses) < 0.05),
    }
    greedy = results["greedy_nearest"]
    d4 = {
        "greedy_bijection_rate": greedy["final_bijection_rate"],
        "greedy_early_potential": greedy["early_potential_bits"],
        "trained_min_bijection": min(t["final_bijection_rate"] for t in trained),
        "trained_min_potential": min(t["early_potential_bits"] for t in trained),
        "pass": (greedy["final_bijection_rate"]
                 < min(t["final_bijection_rate"] for t in trained)
                 and greedy["early_potential_bits"]
                 < min(t["early_potential_bits"] for t in trained)),
    }
    verdicts = {"D1_potential": d1, "D2_useful_collapse": d2,
                "D3_counterfactual": d3, "D4_greedy_contrast": d4,
                "all_pass": all(d["pass"] for d in (d1, d2, d3, d4))}

    OUTPUTS.mkdir(exist_ok=True)
    out = {"conditions": results, "verdicts": verdicts}
    (OUTPUTS / f"deep_marl_collapse_{args.tag}.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(verdicts, indent=2))
    print(f"Wrote {OUTPUTS / f'deep_marl_collapse_{args.tag}.json'}")


if __name__ == "__main__":
    main()
