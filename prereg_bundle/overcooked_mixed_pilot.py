"""Overcooked mixed-context design pilot (DISCLOSED; pre-freeze).

Question this pilot answers, before the preregistration freezes: which
pair of ORIGINAL layouts gives a single shared policy a
context-dependent role allocation (different first-potter identity by
layout), so that conditional selectivity is a property of the learned
policy rather than a coin-flip convention?

Candidate context pairs (unmodified layouts):

    pair_A  cramped_room + asymmetric_advantages
    pair_B  coordination_ring + asymmetric_advantages

One seed per pair, mixed training (layout drawn per episode). Outputs
per-layout sparse reward and first-potter statistics. No confirmatory
claim; the chosen pair and the trigger sign are frozen in the
preregistration BEFORE confirmatory seeds.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

from overcooked_pilot import PolicyNet, HORIZON, OBS_DIM

OUTPUTS = Path(__file__).resolve().parent / "outputs"

PAIRS = {
    "pair_A": ("cramped_room", "asymmetric_advantages"),
    "pair_B": ("coordination_ring", "asymmetric_advantages"),
    "pair_C": ("forced_coordination", "asymmetric_advantages"),
}


def make_env(layout: str) -> OvercookedEnv:
    mdp = OvercookedGridworld.from_layout_name(layout)
    return OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)


def featurize(env: OvercookedEnv) -> List[np.ndarray]:
    return [o.astype(np.float32) for o in env.featurize_state_mdp(env.state)]


def train_mixed(pair: str, layouts, seed: int, total_steps: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    envs = {name: make_env(name) for name in layouts}
    layout_rng = random.Random(seed + 5)
    current = layouts[0]
    env = envs[current]
    env.reset()
    obs = featurize(env)

    step_count = 0
    episode_sparse = 0.0
    stats = {name: {"sparse": [], "potter0": []} for name in layouts}
    curve = []
    t0 = time.time()

    while step_count < total_steps:
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        for _ in range(2048):
            x = torch.tensor(np.stack(obs))
            with torch.no_grad():
                logits, vals = net(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            actions = [Action.ALL_ACTIONS[a] for a in acts.tolist()]
            _s, sparse_r, done, info = env.step(actions)
            shaped = info.get("shaped_r_by_agent", [0, 0])
            anneal = max(0.0, 1.0 - step_count / (0.6 * total_steps))
            rewards = [sparse_r + anneal * shaped[i] for i in range(2)]
            episode_sparse += sparse_r
            buf["obs"].append(np.stack(obs))
            buf["act"].append(acts.numpy())
            buf["logp"].append(logps.numpy())
            buf["val"].append(vals.numpy())
            buf["rew"].append(np.array(rewards, dtype=np.float32))
            buf["done"].append(done)
            step_count += 1
            if done:
                gs = env.game_stats
                pot_times = []
                for agent in (0, 1):
                    times = gs.get("potting_onion", [[], []])[agent]
                    if len(times):
                        pot_times.append((min(times), agent))
                stats[current]["sparse"].append(episode_sparse)
                if pot_times:
                    stats[current]["potter0"].append(
                        int(min(pot_times)[1] == 0))
                episode_sparse = 0.0
                current = layouts[layout_rng.random() > 0.5]
                env = envs[current]
                env.reset()
            obs = featurize(env)

        rews = np.stack(buf["rew"])
        vals = np.stack(buf["val"])
        dones = np.array(buf["done"], dtype=np.float32)
        T = len(dones)
        adv = np.zeros((T, 2), dtype=np.float32)
        last = np.zeros(2, dtype=np.float32)
        with torch.no_grad():
            _, boot = net(torch.tensor(np.stack(obs)))
        next_val = boot.numpy()
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rews[t] + 0.99 * next_val * mask - vals[t]
            last = delta + 0.99 * 0.95 * mask * last
            adv[t] = last
            next_val = vals[t]
        ret = adv + vals

        obs_b = torch.tensor(np.concatenate(
            [np.stack(buf["obs"])[:, i] for i in (0, 1)]))
        act_b = torch.tensor(np.concatenate(
            [np.stack(buf["act"])[:, i] for i in (0, 1)]))
        logp_b = torch.tensor(np.concatenate(
            [np.stack(buf["logp"])[:, i] for i in (0, 1)]))
        adv_b = torch.tensor(np.concatenate([adv[:, i] for i in (0, 1)]))
        ret_b = torch.tensor(np.concatenate([ret[:, i] for i in (0, 1)]))
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        idx = np.arange(len(obs_b))
        for _ in range(6):
            np.random.shuffle(idx)
            for start in range(0, len(idx), 512):
                mb = idx[start:start + 512]
                logits, v = net(obs_b[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act_b[mb]) - logp_b[mb])
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 0.8, 1.2) * adv_b[mb]
                loss = (-torch.min(s1, s2).mean()
                        + 0.5 * ((v - ret_b[mb]) ** 2).mean()
                        - 0.02 * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        snap = {"steps": step_count,
                "elapsed_s": round(time.time() - t0, 1)}
        for name in layouts:
            snap[f"{name}_sparse20"] = (
                float(np.mean(stats[name]["sparse"][-20:]))
                if stats[name]["sparse"] else 0.0)
            snap[f"{name}_potter0_20"] = (
                float(np.mean(stats[name]["potter0"][-20:]))
                if stats[name]["potter0"] else None)
        curve.append(snap)
        print(f"[{pair}] {json.dumps(snap)}", flush=True)

    torch.save(net.state_dict(), OUTPUTS / f"overcooked_mixed_{pair}.pt")
    return {"pair": pair, "layouts": list(layouts), "curve": curve}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=8901)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = {"status": "disclosed mixed-context design pilot", "runs": []}
    import os
    selected = os.environ.get("PAIRS")
    items = [(k, v) for k, v in PAIRS.items()
             if not selected or k in selected.split(",")]
    for pair, layouts in items:
        report["runs"].append(
            train_mixed(pair, layouts, args.seed, args.steps))
        out = OUTPUTS / "overcooked_mixed_pilot.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"checkpointed {out}", flush=True)


if __name__ == "__main__":
    main()
