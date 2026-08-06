"""Commitment-window intervention on Overcooked self-play training.

Registered in COMMITMENT_INTERVENTION_PREREGISTRATION.md (frozen before
run). Trains four conditions with identical seed and mechanics:

    none / early(80k-440k) / commit(640k-1M) / late(1.5M-1.86M)

During a cut window agent 1 acts from a frozen ghost copy of the
network and only agent 0's stream enters the PPO update; outside the
window training is exactly the train_mixed mechanics.

Evaluates the final 2M checkpoint with the real-vs-ghost transition
certificate and the joint-collapse ladder.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_genesis_curve import evaluate_checkpoint
from overcooked_joint_collapse_curve import (ladder_from_tables,
                                             rollout_joint_counts)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUTS = ("cramped_room", "asymmetric_advantages")
WINDOWS: Dict[str, Optional[Tuple[int, int]]] = {
    "none": None,
    "early": (80_000, 440_000),
    "commit": (640_000, 1_000_000),
    "late": (1_500_000, 1_860_000),
}


def train_with_cut(layouts, seed: int, total_steps: int,
                   window: Optional[Tuple[int, int]]) -> PolicyNet:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    envs = {name: oc.make_env(name) for name in layouts}
    layout_rng = random.Random(seed + 5)
    current = layouts[0]
    env = envs[current]
    env.reset()
    obs = oc.featurize(env)
    step_count = 0
    frozen: Optional[PolicyNet] = None

    def in_window(s: int) -> bool:
        return window is not None and window[0] <= s < window[1]

    while step_count < total_steps:
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew",
                               "done", "cut")}
        for _ in range(2048):
            cut_now = in_window(step_count)
            if cut_now and frozen is None:
                frozen = PolicyNet()
                frozen.load_state_dict(net.state_dict())
                frozen.eval()
            elif not cut_now:
                frozen = None
            x = torch.tensor(np.stack(obs))
            with torch.no_grad():
                logits, vals = net(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            acts_list = acts.tolist()
            if cut_now and frozen is not None:
                with torch.no_grad():
                    g_logits, _ = frozen(x[1:2])
                    g_dist = torch.distributions.Categorical(
                        logits=g_logits)
                    acts_list[1] = int(g_dist.sample().item())
            actions = [Action.ALL_ACTIONS[a] for a in acts_list]
            _s, sparse_r, done, info = env.step(actions)
            shaped = info.get("shaped_r_by_agent", [0, 0])
            anneal = max(0.0, 1.0 - step_count / (0.6 * total_steps))
            rewards = [sparse_r + anneal * shaped[i] for i in range(2)]
            buf["obs"].append(np.stack(obs))
            buf["act"].append(np.array(acts_list))
            buf["logp"].append(logps.numpy())
            buf["val"].append(vals.numpy())
            buf["rew"].append(np.array(rewards, dtype=np.float32))
            buf["done"].append(done)
            buf["cut"].append(cut_now)
            step_count += 1
            if done:
                current = layouts[layout_rng.random() > 0.5]
                env = envs[current]
                env.reset()
            obs = oc.featurize(env)
        rews = np.stack(buf["rew"])
        vals = np.stack(buf["val"])
        dones = np.array(buf["done"], dtype=np.float32)
        cut_mask = np.array(buf["cut"], dtype=bool)
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
        obs_arr = np.stack(buf["obs"])
        act_arr = np.stack(buf["act"])
        logp_arr = np.stack(buf["logp"])
        keep1 = ~cut_mask  # agent 1 samples only outside the window
        obs_b = torch.tensor(np.concatenate(
            [obs_arr[:, 0], obs_arr[keep1][:, 1]]))
        act_b = torch.tensor(np.concatenate(
            [act_arr[:, 0], act_arr[keep1][:, 1]]))
        logp_b = torch.tensor(np.concatenate(
            [logp_arr[:, 0], logp_arr[keep1][:, 1]]))
        adv_b = torch.tensor(np.concatenate(
            [adv[:, 0], adv[keep1][:, 1]]))
        ret_b = torch.tensor(np.concatenate(
            [ret[:, 0], ret[keep1][:, 1]]))
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
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=sorted(WINDOWS), required=True)
    ap.add_argument("--seed", type=int, default=93101)
    ap.add_argument("--train-steps", type=int, default=2_000_000)
    args = ap.parse_args()

    torch.set_num_threads(4)
    window = WINDOWS[args.condition]
    t0 = time.time()
    print(f"condition={args.condition} window={window} "
          f"seed={args.seed}", flush=True)
    net = train_with_cut(LAYOUTS, args.seed, args.train_steps, window)
    train_min = round((time.time() - t0) / 60, 2)
    ckpt = OUTPUTS / (f"overcooked_intervention_{args.condition}"
                      f"_s{args.seed}.pt")
    torch.save(net.state_dict(), ckpt)
    print(f"trained in {train_min} min; evaluating", flush=True)

    cond_index = sorted(WINDOWS).index(args.condition)
    eval_seed = 96001 + cond_index
    cert = evaluate_checkpoint(ckpt, eval_seed)
    tables, basins = rollout_joint_counts(net, eval_seed + 10_000)
    pooled = {layout: sum(t) for layout, t in tables.items()}
    ladder = ladder_from_tables(pooled)

    report = {
        "status": ("commitment-window intervention run; registered in "
                   "COMMITMENT_INTERVENTION_PREREGISTRATION.md; pilot "
                   "(one seed per condition)"),
        "condition": args.condition,
        "window": window,
        "seed": args.seed,
        "train_minutes": train_min,
        "certificate_2M": cert,
        "joint_ladder_2M": {k: ladder[k] for k in
                            ("C_individual", "C_env", "C_relational",
                             "C_total", "collapse_norm")},
    }
    out = OUTPUTS / (f"overcooked_intervention_{args.condition}"
                     f"_s{args.seed}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "condition": args.condition,
        "G": cert["G_js_bits"], "M": cert["M_score_gain"],
        "score": cert["real_score"],
        "C_rel": ladder["C_relational"],
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
