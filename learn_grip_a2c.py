"""LEARN-GRIP-A2C: algorithm-robustness check of the grip flagship.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Identical
environment (all constants imported from learn_grip_transport); the
only change is the learning algorithm: advantage actor-critic with a
learned state-value baseline and per-step returns, replacing REINFORCE
with a scalar moving-average baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate
from learn_grip_transport import (
    ACCEL,
    BATCH,
    DAMP,
    GOAL,
    GRIP_DECAY,
    GRIP_GAIN,
    GRIP_MIN,
    GripPolicy,
    LR,
    MAX_STEPS,
    N_AGENTS,
    THRESHOLD,
    eval_policy,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_SEEDS = 5
UPDATES = 1200
GAMMA = 0.99
VALUE_COEF = 0.5
SEED = 119_001


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 32), nn.Tanh(),
                               nn.Linear(32, 32), nn.Tanh(),
                               nn.Linear(32, 1))

    def forward(self, obs):
        return self.f(obs).squeeze(-1)


def rollout_a2c(policy: GripPolicy, value: ValueNet, batch: int):
    """Collect per-step logps, values and rewards for A2C."""
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    att = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    logps, values, rewards, masks = [], [], [], []
    for _ in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v, att], dim=1)
        logits = policy(obs)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS,
                                               probs=torch.softmax(logits, dim=-1))
        counts = dist.sample()
        logp = dist.log_prob(counts)
        val = value(obs)
        grip_frac = counts[:, 2] / N_AGENTS
        att = torch.clamp(att + GRIP_GAIN * grip_frac - GRIP_DECAY, 0.0, 1.0)
        force = counts[:, 1] - counts[:, 0]
        old_abs = torch.abs(x)
        active = (att >= GRIP_MIN) & (torch.abs(force) >= THRESHOLD)
        v = DAMP * v + active.float() * ACCEL * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        r = (torch.abs(x) - old_abs) - 0.004
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        r = r + newly.float() * 5.0
        mask = (~done).float()
        logps.append(logp)
        values.append(val)
        rewards.append(r)
        masks.append(mask)
        done = done | newly
    return logps, values, rewards, masks


def train(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = GripPolicy()
    value = ValueNet()
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=LR)
    for _ in range(UPDATES):
        logps, values, rewards, masks = rollout_a2c(policy, value, BATCH)
        T = len(rewards)
        returns = [torch.zeros_like(rewards[0])] * T
        run = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            run = rewards[t] + GAMMA * run * masks[t]
            returns[t] = run
        pg_loss = torch.zeros(())
        v_loss = torch.zeros(())
        for t in range(T):
            adv = (returns[t] - values[t]).detach()
            pg_loss = pg_loss - (logps[t] * adv * masks[t]).mean()
            v_loss = v_loss + ((returns[t] - values[t]) ** 2 * masks[t]).mean()
        loss = pg_loss / T + VALUE_COEF * v_loss / T
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(value.parameters()), 1.0)
        opt.step()
    return policy


def main() -> None:
    rows = {}
    for i in range(N_SEEDS):
        policy = train(SEED + i * 101)
        ev = eval_policy(policy)
        curve = np.array([float(u) for u in
                          np.array(ev["episode_side_openness_curve"])])
        adj = adjudicate(range(MAX_STEPS), curve)
        plateau = 0
        for val in curve:
            if val >= 0.8:
                plateau += 1
            else:
                break
        rows[str(i)] = {
            "final_success": round(ev["success"], 5),
            "final_side_mean": round(ev["side_mean"], 5),
            "plateau_len": plateau,
            "side_openness_final": round(float(curve[-1]), 5),
            "side_openness_curve": [round(float(u), 5) for u in curve],
            "adj": adj,
        }
        h = adj.get("hinge", {})
        print(f"seed={i}: succ={rows[str(i)]['final_success']} "
              f"plateau={plateau} B5={adj['b5_onset']} "
              f"dBIC={h.get('delta_bic')} t*={h.get('t_star')}", flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    t_stars = [r["adj"].get("hinge", {}).get("t_star")
               for r in learned if r["adj"].get("hinge")]
    outcomes = {
        "LGA1_learnability": bool(len(learned) >= 4),
        "LGA2_algorithm_independence": bool(
            len(learned) >= 4
            and sum(r["adj"]["b5_onset"] for r in learned) >= 4),
        "LGA3_breakpoint_stability": bool(
            t_stars and all(t is not None and 10 <= t <= 30
                            for t in t_stars)),
        "n_learned": len(learned),
        "b5_count": sum(r["adj"]["b5_onset"] for r in learned),
        "t_stars": t_stars,
    }
    report = {
        "status": "LEARN-GRIP-A2C algorithm-robustness check; preregistered",
        "config": {"seeds": N_SEEDS, "updates": UPDATES, "batch": BATCH,
                   "gamma": GAMMA, "value_coef": VALUE_COEF,
                   "environment": "identical to learn_grip_transport"},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_a2c.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
