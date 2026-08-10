"""Overcooked-AI design pilot (DISCLOSED; no confirmatory claim).

Purpose: establish, before any preregistration, (a) that self-play PPO
with the standard annealed shaping curriculum learns to deliver soups on
candidate layouts, and (b) what role-allocation structure trained
policies exhibit, so that the trigger, contexts and thresholds of the
public-environment full-criterion protocol can be frozen honestly.

Environment: unmodified overcooked_ai_py MDP, dynamics and sparse
delivery reward untouched. Training uses the community-standard shaped
auxiliary rewards (pot progress etc.) annealed to zero -- a disclosed
curriculum. Crucially, no reward term references agent identity, so the
ROLE ALLOCATION (which agent takes which role) is never prespecified;
that allocation is the candidate macro-structure.

This is a pilot: one seed per candidate layout, reduced steps. Outputs
learning curves and role statistics only.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf  # numpy-2 compat shim for overcooked_ai
import torch
import torch.nn as nn

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

OUTPUTS = Path(__file__).resolve().parent / "outputs"

HORIZON = 400
N_ACTIONS = len(Action.ALL_ACTIONS)
OBS_DIM = 96


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.pi = nn.Linear(128, N_ACTIONS)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def make_env(layout: str) -> OvercookedEnv:
    mdp = OvercookedGridworld.from_layout_name(layout)
    return OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)


def featurize(env: OvercookedEnv) -> List[np.ndarray]:
    return [o.astype(np.float32)
            for o in env.featurize_state_mdp(env.state)]


def first_pot_interactor(events: List[Dict]) -> int | None:
    for e in events:
        if e["type"] == "potting":
            return e["agent"]
    return None


def run_training(layout: str, seed: int, total_steps: int,
                 shaping_anneal_frac: float = 0.6) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    env = make_env(layout)
    env.reset()
    obs = featurize(env)

    step_count = 0
    curve = []
    episode_sparse = 0.0
    episode_events: List[Dict] = []
    recent_sparse: List[float] = []
    recent_first_potter: List[int] = []
    t0 = time.time()

    while step_count < total_steps:
        # collect rollout
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
            anneal = max(0.0, 1.0 - step_count
                         / (shaping_anneal_frac * total_steps))
            rewards = [sparse_r + anneal * shaped[i] for i in range(2)]
            episode_sparse += sparse_r
            for e in info.get("episode", {}).get("ep_game_stats", {}) or []:
                pass
            # potting events from mdp game stats
            ev = env.game_stats.get("potting_onion", [[], []])
            buf["obs"].append(np.stack(obs))
            buf["act"].append(acts.numpy())
            buf["logp"].append(logps.numpy())
            buf["val"].append(vals.numpy())
            buf["rew"].append(np.array(rewards, dtype=np.float32))
            buf["done"].append(done)
            step_count += 1
            if done:
                stats = env.game_stats
                potter = None
                pot_times = []
                for agent in (0, 1):
                    times = stats.get("potting_onion", [[], []])[agent]
                    if len(times):
                        pot_times.append((min(times), agent))
                if pot_times:
                    potter = min(pot_times)[1]
                recent_sparse.append(episode_sparse)
                if potter is not None:
                    recent_first_potter.append(potter)
                episode_sparse = 0.0
                env.reset()
            obs = featurize(env)

        # GAE + PPO update (shared params, both agents as batch)
        rews = np.stack(buf["rew"])            # T x 2
        vals = np.stack(buf["val"])            # T x 2
        dones = np.array(buf["done"], dtype=np.float32)  # T
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
                        - 0.01 * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        window = recent_sparse[-20:]
        potters = recent_first_potter[-20:]
        curve.append({
            "steps": step_count,
            "mean_sparse_20ep": float(np.mean(window)) if window else 0.0,
            "first_potter_agent0_rate": (
                float(np.mean([p == 0 for p in potters]))
                if potters else None),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[{layout} s{seed}] steps {step_count} "
              f"sparse20 {curve[-1]['mean_sparse_20ep']:.1f} "
              f"potter0 {curve[-1]['first_potter_agent0_rate']} "
              f"({curve[-1]['elapsed_s']}s)", flush=True)

    torch.save(net.state_dict(),
               OUTPUTS / f"overcooked_pilot_{layout}_s{seed}.pt")
    return {"layout": layout, "seed": seed, "curve": curve}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", nargs="*",
                        default=["cramped_room", "asymmetric_advantages"])
    parser.add_argument("--seed", type=int, default=8801)
    parser.add_argument("--steps", type=int, default=800_000)
    args = parser.parse_args()
    torch.set_num_threads(8)
    report = {"status": "disclosed design pilot (no confirmatory claim)",
              "runs": []}
    for layout in args.layouts:
        report["runs"].append(run_training(layout, args.seed, args.steps))
        out = OUTPUTS / "overcooked_pilot.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"checkpointed {out}", flush=True)


if __name__ == "__main__":
    main()
