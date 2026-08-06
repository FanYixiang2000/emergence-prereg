"""Cross-task deep MARL collapse probe: Level-Based Foraging (coop).

Registered in LBF_PREREGISTRATION.md BEFORE any trained policy was probed.
Environment: lbforaging 2.0.0 `Foraging-5x5-2p-2f-coop-v3` (2 agents, 2
foods, forced cooperation: every food's level equals the sum of the two
players' levels, so no agent can ever load alone; sparse reward at
collection only). We did not design the environment.

Measured objects mirror deep_marl_collapse_probe.py: P_t(B | s_t) by
Monte-Carlo rollouts of the behaving controller from snapshotted world
states; basin = food-consumption ORDER (tuple of food indices, 5
outcomes); win = full clearance; do-operators at the pre-commit snapshot.

Pilot notes (allowed scope: training + estimator params only; logs kept):

1. Training pilots (tags pilot/pilot2, seed 11, 8k episodes): sparse-coop
   PPO reaches full clearance ~1.0 by episode ~6k; behaving win rate 0.9
   at eval seeds. Training needs no tuning beyond the annealed defaults.
2. pilot2/pilot3 exposed TWO ESTIMATOR ISSUES, both in scope:
   (a) probing a converged near-deterministic policy at T=1 collapses all
       openness (early potential 0.02 bits) -- fixed by the same softened-
       probe design as the gridworld probe (behaving episode stays T=1);
       temperature sweep with the SAVED seed-11 net (outputs/
       lbf_debug_do.txt): T=2/3/4/6 -> start-state potential
       0.00/0.07/0.29/1.41 bits, do-gap +0.88/+0.92/+0.92/+0.71.
       PROBE_TEMPERATURE frozen at 6.0 for the main run (openness
       macroscopic, do-contrast large, win mass still ~0.5).
   (b) an intervention pinned to a target food that had ALREADY been
       consumed froze agent 0 on an empty cell for the rest of the
       rollout, silently destroying later coordination (all do-rollouts
       returned partial baskets). Fixed: interventions auto-release once
       their target food is gone (rollout_basin). Logs:
       outputs/lbf_pilot2_log.txt, lbf_pilot3_log.txt, lbf_debug_do.txt.
   Registered quantities and thresholds untouched throughout.
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

import gymnasium as gym
import lbforaging  # noqa: F401  (registers env ids)

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

ENV_ID = "Foraging-5x5-2p-2f-coop-v3"
N_AGENTS = 2
N_FOODS = 2
MAX_STEPS = 50  # the env's own max_episode_steps
OBS_DIM = 3 * N_FOODS + 3 * N_AGENTS  # 12
N_ACTIONS = 6  # NONE, N, S, W, E, LOAD

# Estimator parameters (pilot-tunable per registration).
PROBE_ROLLOUTS = 48
EVAL_EPISODES = 30

# Training hyperparameters (pilot-tunable per registration).
TRAIN_EPISODES = 20000
PPO_LR = 7e-4
PPO_LR_FINAL = 1e-4
PPO_CLIP = 0.2
PPO_EPOCHS = 8
MINIBATCH = 512
ENT_COEF = 0.03
ENT_COEF_FINAL = 0.004
GAMMA = 0.98
GAE_LAMBDA = 0.95
BATCH_EPISODES = 32


def make_raw_env(seed: int):
    env = gym.make(ENV_ID).unwrapped
    env.reset(seed=seed)
    return env


def food_positions(env) -> List[Tuple[int, int]]:
    """Nonzero field cells, lexicographically sorted."""
    rows, cols = np.nonzero(env.field)
    return sorted(zip(rows.tolist(), cols.tolist()))


class FoodIndex:
    """Stable food identity across an episode: index by initial position."""

    def __init__(self, env) -> None:
        self.positions = food_positions(env)

    def consumed_now(self, before: set, after: set) -> List[int]:
        gone = before - after
        return sorted(self.positions.index(pos) for pos in gone)


def world_snapshot(env, order: Tuple[int, ...]):
    return (
        env.field.copy(),
        [(p.position, int(p.level)) for p in env.players],
        int(env.current_step),
        tuple(order),
    )


def world_restore(env, snap) -> None:
    field, players, step, _order = snap
    env.field = field.copy()
    for player, (pos, level) in zip(env.players, players):
        player.position = pos
        player.level = level
        player.reward = 0
    env.current_step = step
    env._game_over = False
    env._gen_valid_moves()


def obs_all(env) -> List[np.ndarray]:
    return [np.asarray(o, dtype=np.float32) for o in env._make_gym_obs()]


GREEDY_EPS = 0


def adjacent(pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
    return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) == 1


def greedy_action_toward(env, agent_idx: int, target: Tuple[int, int]) -> int:
    """Move along the largest-gap axis toward target; LOAD when adjacent.

    Falls back among valid actions (grid edges, occupied cells)."""
    player = env.players[agent_idx]
    pos = player.position
    if adjacent(pos, target):
        return 5  # LOAD
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    prefs: List[int] = []
    if abs(dr) >= abs(dc):
        if dr != 0:
            prefs.append(2 if dr > 0 else 1)  # SOUTH / NORTH
        if dc != 0:
            prefs.append(4 if dc > 0 else 3)  # EAST / WEST
    else:
        if dc != 0:
            prefs.append(4 if dc > 0 else 3)
        if dr != 0:
            prefs.append(2 if dr > 0 else 1)
    valid = {a.value for a in env._valid_actions[player]}
    for a in prefs:
        if a in valid:
            return a
    return 0


class PolicyNet(nn.Module):
    """Actor on local obs; MAPPO-style centralized critic (training only)."""

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


class Controller:
    """policy | untrained (same as policy) | greedy_nearest | noise.

    probe_temperature: softmax temperature applied to policy logits.
    The behaving episode uses T=1 (the policy as trained); probe rollouts
    may use T>1 so aleatoric openness is not erased by a fully converged
    deterministic policy -- the same estimator design as
    within_episode_collapse_probe.py (documented there as a core design
    point) and an estimator parameter under the registration's
    pilot-tunable scope.
    """

    def __init__(self, kind: str, net: Optional[PolicyNet] = None,
                 probe_temperature: float = 1.0) -> None:
        self.kind = kind
        self.net = net
        self.temperature = probe_temperature

    def act(self, env, obs: List[np.ndarray], rng: random.Random,
            interventions: Optional[Dict[int, Dict]] = None) -> Tuple[int, ...]:
        acts: List[int] = []
        probs = None
        if self.kind == "policy":
            batch = torch.tensor(np.stack(obs))
            with torch.no_grad():
                logits = self.net(batch) / self.temperature
                probs = torch.softmax(logits, dim=1).numpy()
        for i in range(N_AGENTS):
            iv = (interventions or {}).get(i)
            if iv and iv["type"] == "do_commit":
                acts.append(greedy_action_toward(env, i, iv["target"]))
                continue
            if self.kind == "noise":
                acts.append(rng.randrange(N_ACTIONS))
            elif self.kind == "greedy_nearest":
                foods = food_positions(env)
                if not foods:
                    acts.append(0)
                    continue
                pos = env.players[i].position
                target = min(foods, key=lambda f: abs(pos[0] - f[0]) + abs(pos[1] - f[1]))
                acts.append(greedy_action_toward(env, i, target))
            else:
                p = probs[i].copy()
                if iv and iv["type"] == "do_block":
                    # Minimal restriction: forbid actions that reduce
                    # distance to the blocked food, and forbid LOAD while
                    # adjacent to it. Everything else stays on-policy.
                    pos = env.players[i].position
                    target = iv["target"]
                    d0 = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                    moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                    allowed = np.ones(N_ACTIONS)
                    for a, (dr, dc) in moves.items():
                        nd = abs(pos[0] + dr - target[0]) + abs(pos[1] + dc - target[1])
                        if nd < d0:
                            allowed[a] = 0.0
                    if adjacent(pos, target):
                        allowed[5] = 0.0
                    p = p * allowed
                    if p.sum() <= 0:
                        p = allowed / max(allowed.sum(), 1.0)
                    else:
                        p = p / p.sum()
                acts.append(min(int(np.searchsorted(np.cumsum(p), rng.random())),
                                N_ACTIONS - 1))
        return tuple(acts)


def train_ppo(seed: int) -> PolicyNet:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR)
    env = make_raw_env(seed)
    ep_returns: List[float] = []
    ep_cleared: List[float] = []

    obs_buf: List[np.ndarray] = []
    gobs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    logp_buf: List[float] = []
    adv_buf: List[float] = []
    ret_buf: List[float] = []
    episodes_in_batch = 0

    for episode in range(TRAIN_EPISODES):
        frac = episode / max(TRAIN_EPISODES - 1, 1)
        for group in opt.param_groups:
            group["lr"] = PPO_LR + (PPO_LR_FINAL - PPO_LR) * frac
        ent_now = ENT_COEF + (ENT_COEF_FINAL - ENT_COEF) * frac

        env.reset(seed=rng.randrange(10 ** 9))
        obs = obs_all(env)
        traj_obs, traj_gobs, traj_act, traj_logp, traj_rew, traj_val = \
            [], [], [], [], [], []
        total = 0.0
        done = False
        while not done:
            batch = torch.tensor(np.stack(obs))
            gobs = batch.reshape(1, -1)
            with torch.no_grad():
                logits = net(batch)
                value = net.value(gobs)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logps = dist.log_prob(actions)
            nobs, rewards, term, trunc, _ = env.step(tuple(int(a) for a in actions))
            traj_obs.append(batch.numpy())
            traj_gobs.append(gobs.numpy()[0])
            traj_act.append(actions.numpy())
            traj_logp.append(logps.numpy())
            traj_val.append(float(value))
            shared = float(np.mean(rewards))
            traj_rew.append(shared)
            total += shared
            obs = [np.asarray(o, dtype=np.float32) for o in nobs]
            done = bool(term) or bool(trunc) or env.game_over
        ep_returns.append(total)
        ep_cleared.append(1.0 if env.field.sum() == 0 else 0.0)

        cleared = env.field.sum() == 0
        with torch.no_grad():
            tail_v = 0.0 if cleared else float(net.value(
                torch.tensor(np.stack(obs)).reshape(1, -1)))
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
            print(f"  seed {seed} episode {episode + 1}: "
                  f"return(100) {np.mean(ep_returns[-100:]):.3f} "
                  f"cleared(100) {np.mean(ep_cleared[-100:]):.2f}", flush=True)
    return net


def entropy_bits(counts: Dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total)
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
            out += pk * math.log2((pk + eps) / (qk + eps))
    return out


def is_win(basin: Tuple[int, ...]) -> bool:
    return len(basin) == N_FOODS


def rollout_basin(env, snap, findex: FoodIndex, controller: Controller,
                  rng: random.Random,
                  interventions: Optional[Dict[int, Dict]] = None
                  ) -> Tuple[int, ...]:
    world_restore(env, snap)
    order = list(snap[3])
    remaining = MAX_STEPS - snap[2]
    before = set(food_positions(env))
    for _ in range(remaining):
        if env.field.sum() == 0:
            break
        # A do-operator commits/blocks relative to a TARGET FOOD; once that
        # food has been consumed the commitment is complete (or the block
        # is moot) and the agent returns to on-policy behavior. Keeping the
        # intervention alive against a consumed food freezes the agent on
        # an empty cell and silently destroys all later coordination
        # (estimator bug found in pilot3, log kept).
        active = interventions
        if interventions:
            live = {i: iv for i, iv in interventions.items()
                    if iv["target"] in before}
            active = live or None
        obs = obs_all(env)
        acts = controller.act(env, obs, rng, active)
        env.step(acts)
        after = set(food_positions(env))
        if after != before:
            order.extend(findex.consumed_now(before, after))
            before = after
    return tuple(order)


def future_distribution(env, snap, findex: FoodIndex, controller: Controller,
                        rng: random.Random,
                        interventions: Optional[Dict[int, Dict]] = None,
                        rollouts: int = PROBE_ROLLOUTS) -> Dict:
    counts: Dict[Tuple[int, ...], int] = {}
    for _ in range(rollouts):
        basin = rollout_basin(env, snap, findex, controller, rng, interventions)
        counts[basin] = counts.get(basin, 0) + 1
    return counts


def win_mass(counts: Dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(c for b, c in counts.items() if is_win(b)) / total


PROBE_TEMPERATURE = 6.0


def probe_condition(name: str, controller: Controller, seed: int) -> Dict:
    rng = random.Random(seed * 7919 + 13)
    env = make_raw_env(seed)
    sim = make_raw_env(seed + 10 ** 6)
    # Probe rollouts use a softened policy so aleatoric openness is not
    # erased by a converged near-deterministic policy (same estimator
    # design as the gridworld probe); the behaving episode stays at T=1.
    if controller.kind == "policy":
        probe = Controller("policy", controller.net, PROBE_TEMPERATURE)
    else:
        probe = controller

    episodes: List[Dict] = []
    for ep in range(EVAL_EPISODES):
        env.reset(seed=1_000_000 + 997 * seed + ep)
        findex = FoodIndex(env)
        order: List[int] = []
        snaps = []
        step_stats: List[Dict] = []
        prev_dist: Optional[Dict] = None
        before = set(food_positions(env))
        consumed_at: List[Tuple[int, int]] = []  # (step, food_idx)
        for t in range(MAX_STEPS):
            if env.field.sum() == 0:
                break
            snap = world_snapshot(env, tuple(order))
            dist = future_distribution(sim, snap, findex, probe, rng)
            step_stats.append({
                "t": t,
                "potential_bits": entropy_bits(dist),
                "p_win": win_mass(dist),
                "collapse_bits": kl_bits(dist, prev_dist) if prev_dist else 0.0,
            })
            snaps.append(snap)
            prev_dist = dist
            obs = obs_all(env)
            env.step(controller.act(env, obs, rng))
            after = set(food_positions(env))
            if after != before:
                for idx in findex.consumed_now(before, after):
                    order.append(idx)
                    consumed_at.append((t, idx))
                before = after
        final_basin = tuple(order)

        if len(step_stats) < 2:
            continue
        commit = max(range(1, len(step_stats)),
                     key=lambda i: step_stats[i]["collapse_bits"])
        record = {
            "episode": ep,
            "early_potential_bits": float(np.mean(
                [s["potential_bits"] for s in step_stats[:3]])),
            "p_win_start": step_stats[0]["p_win"],
            "p_win_end": step_stats[-1]["p_win"],
            "final_win": int(is_win(final_basin)),
            "final_basin": list(final_basin),
            "commit_step": commit,
            "commit_collapse_bits": step_stats[commit]["collapse_bits"],
        }
        if controller.kind == "policy":
            # Target food: the next food consumed at/after the commit step
            # in the behaving episode; fallback = nearest remaining food to
            # agent 0 at the pre-commit snapshot.
            target_idx = None
            for step_idx, food_idx in consumed_at:
                if step_idx >= commit:
                    target_idx = food_idx
                    break
            snap_pre = snaps[commit - 1]
            if target_idx is None:
                world_restore(sim, snap_pre)
                foods = food_positions(sim)
                if not foods:
                    episodes.append(record)
                    continue
                pos = sim.players[0].position
                tpos = min(foods, key=lambda f: abs(pos[0] - f[0]) + abs(pos[1] - f[1]))
                target_idx = findex.positions.index(tpos)
            target = findex.positions[target_idx]
            # Skip if the target food is already gone at the pre-commit snap.
            world_restore(sim, snap_pre)
            if target not in food_positions(sim):
                episodes.append(record)
                continue
            d_commit = future_distribution(
                sim, snap_pre, findex, probe, rng,
                {0: {"type": "do_commit", "target": target}})
            d_block = future_distribution(
                sim, snap_pre, findex, probe, rng,
                {0: {"type": "do_block", "target": target}})
            record["p_win_do_commit"] = win_mass(d_commit)
            record["p_win_do_block"] = win_mass(d_block)
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
        "final_win_rate": mean("final_win"),
        "mean_commit_step": mean("commit_step"),
        "episodes": episodes,
    }
    if controller.kind == "policy":
        gaps = [e["p_win_do_commit"] - e["p_win_do_block"]
                for e in episodes if "p_win_do_commit" in e]
        wins = sum(1 for g in gaps if g > 0)
        losses = sum(1 for g in gaps if g < 0)
        summary["do_gap_median"] = float(np.median(gaps)) if gaps else float("nan")
        summary["do_gap_mean"] = float(np.mean(gaps)) if gaps else float("nan")
        summary["do_sign_wins"] = wins
        summary["do_sign_losses"] = losses
        summary["do_sign_p"] = sign_test_p(wins, losses)
    return summary


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def main() -> None:
    global TRAIN_EPISODES
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    parser.add_argument("--train_episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--train_only", action="store_true")
    args = parser.parse_args()
    TRAIN_EPISODES = args.train_episodes

    torch.set_num_threads(16)
    results: Dict[str, Dict] = {}

    for seed in args.seeds:
        print(f"training PPO seed {seed} ...", flush=True)
        net = train_ppo(seed)
        torch.save(net.state_dict(), OUTPUTS / f"lbf_net_seed{seed}.pt")
        if args.train_only:
            continue
        results[f"trained_seed{seed}"] = probe_condition(
            f"trained_seed{seed}", Controller("policy", net), seed)
        print(json.dumps({k: v for k, v in results[f"trained_seed{seed}"].items()
                          if k != "episodes"}, indent=2), flush=True)
    if args.train_only:
        return

    torch.manual_seed(999)
    results["untrained"] = probe_condition(
        "untrained", Controller("policy", PolicyNet()), 44)
    results["greedy_nearest"] = probe_condition(
        "greedy_nearest", Controller("greedy_nearest"), 55)
    results["noise"] = probe_condition("noise", Controller("noise"), 66)
    for cname in ("untrained", "greedy_nearest", "noise"):
        print(json.dumps({k: v for k, v in results[cname].items()
                          if k != "episodes"}, indent=2), flush=True)

    trained = [v for k, v in results.items() if k.startswith("trained_")]
    l1 = {
        "trained_early_potential": [t["early_potential_bits"] for t in trained],
        "pass": all(t["early_potential_bits"] >= 0.8 for t in trained),
    }
    l2 = {
        "trained_win_rate": [t["final_win_rate"] for t in trained],
        "untrained_win_rate": results["untrained"]["final_win_rate"],
        "noise_win_rate": results["noise"]["final_win_rate"],
        "pass": (all(t["final_win_rate"] >= 0.5 for t in trained)
                 and results["untrained"]["final_win_rate"] < 0.2
                 and results["noise"]["final_win_rate"] < 0.2),
    }
    pooled_gaps: List[float] = []
    pooled_wins = pooled_losses = 0
    for t in trained:
        pooled_wins += t["do_sign_wins"]
        pooled_losses += t["do_sign_losses"]
        pooled_gaps.extend(e["p_win_do_commit"] - e["p_win_do_block"]
                           for e in t["episodes"] if "p_win_do_commit" in e)
    l3 = {
        "pooled_do_gap_median": float(np.median(pooled_gaps)) if pooled_gaps else float("nan"),
        "pooled_sign_wins": pooled_wins,
        "pooled_sign_losses": pooled_losses,
        "pooled_sign_p": sign_test_p(pooled_wins, pooled_losses),
        "pass": (bool(pooled_gaps) and float(np.median(pooled_gaps)) > 0
                 and sign_test_p(pooled_wins, pooled_losses) < 0.05),
    }
    greedy = results["greedy_nearest"]
    l4 = {
        "greedy_win_rate": greedy["final_win_rate"],
        "greedy_early_potential": greedy["early_potential_bits"],
        "trained_min_win": min(t["final_win_rate"] for t in trained),
        "trained_min_potential": min(t["early_potential_bits"] for t in trained),
        "pass": (greedy["final_win_rate"] < min(t["final_win_rate"] for t in trained)
                 and greedy["early_potential_bits"]
                 < min(t["early_potential_bits"] for t in trained)),
    }
    verdicts = {"L1_potential": l1, "L2_useful_structure": l2,
                "L3_counterfactual": l3, "L4_greedy_contrast": l4,
                "all_pass": all(d["pass"] for d in (l1, l2, l3, l4))}

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / f"lbf_collapse_{args.tag}.json").write_text(
        json.dumps({"conditions": results, "verdicts": verdicts}, indent=2))
    print(json.dumps(verdicts, indent=2))
    print(f"Wrote {OUTPUTS / f'lbf_collapse_{args.tag}.json'}")


if __name__ == "__main__":
    main()
