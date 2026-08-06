"""LEARN-TRANSPORT-PILOT: symmetric learned collective transport.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. This is
a minimal learned spatial pilot, not the final flagship.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_AGENTS = 16
THRESHOLD = 6
GOAL = 5.0
MAX_STEPS = 45
N_SEEDS = 5
UPDATES = 1000
BATCH_EP = 32
SAVE_EVERY = 50
GRID = tuple(range(0, UPDATES + 1, SAVE_EVERY))
LR = 3e-4
GAMMA = 0.98
CLIP = 0.2
PPO_EPOCHS = 4
ENT_COEF = 0.02
SEED = 109_001


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh())
        self.pi = nn.Linear(64, 3)
        self.v = nn.Linear(64, 1)

    def forward(self, obs):
        z = self.body(obs)
        return self.pi(z), self.v(z).squeeze(-1)


def entropy_norm(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(3))


def rollout(net: Net, rng: np.random.Generator):
    x = 0.0
    v = 0.0
    obs_l, act_l, logp_l, rew_l, val_l, ent_l = [], [], [], [], [], []
    done = False
    final_side = 0
    for _ in range(MAX_STEPS):
        obs = torch.tensor([[x / GOAL, v]], dtype=torch.float32)
        logits, val = net(obs)
        dist = torch.distributions.Categorical(logits=logits.repeat(N_AGENTS, 1))
        acts = dist.sample()
        logp = dist.log_prob(acts)
        ent = dist.entropy()
        left = int((acts == 0).sum().item())
        right = int((acts == 1).sum().item())
        force = right - left
        old_abs = abs(x)
        if abs(force) >= THRESHOLD:
            v = 0.85 * v + 0.09 * np.sign(force)
        else:
            v = 0.85 * v
        x = float(np.clip(x + v, -GOAL, GOAL))
        reward = (abs(x) - old_abs) - 0.005
        if abs(x) >= GOAL - 1e-6:
            reward += 5.0
            done = True
        obs_l.append([x / GOAL, v])
        act_l.append(acts.numpy())
        logp_l.append(logp.detach().numpy())
        val_l.append(float(val.item()))
        rew_l.append(float(reward))
        ent_l.append(float(ent.mean().item()))
        if done:
            final_side = 1 if x > 0 else -1
            break
    if not done:
        final_side = 1 if x > 0 else (-1 if x < 0 else 0)
    returns = []
    g = 0.0
    for r in reversed(rew_l):
        g = r + GAMMA * g
        returns.append(g)
    returns.reverse()
    return {
        "obs": np.array(obs_l, dtype=np.float32),
        "acts": np.array(act_l, dtype=np.int64),
        "old_logp": np.array(logp_l, dtype=np.float32),
        "returns": np.array(returns, dtype=np.float32),
        "values": np.array(val_l, dtype=np.float32),
        "reward_sum": float(sum(rew_l)),
        "success": bool(done),
        "final_side": final_side,
        "mean_entropy": float(np.mean(ent_l)) if ent_l else 0.0,
    }


def symmetric_policy_stats(net: Net, mc: int = 32):
    obs = torch.zeros((1, 2), dtype=torch.float32)
    with torch.no_grad():
        logits, _ = net(obs)
        p = torch.softmax(logits, dim=-1).numpy()[0]
    # Monte Carlo exact enough for threshold success from symmetric first step.
    rng = np.random.default_rng(12345)
    succ = []
    sides = []
    for _ in range(mc):
        ro = rollout(net, rng)
        succ.append(ro["success"])
        sides.append(ro["final_side"])
    return {
        "p": p,
        "entropy": entropy_norm(p),
        "success": float(np.mean(succ)),
        "mean_side": float(np.mean(sides)),
    }


def train_seed(seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    openness, success, side, p_hist = [], [], [], []
    for update in range(UPDATES + 1):
        if update in GRID:
            st = symmetric_policy_stats(net, mc=32)
            openness.append(st["entropy"])
            success.append(st["success"])
            side.append(st["mean_side"])
            p_hist.append([float(v) for v in st["p"]])
        if update == UPDATES:
            break
        batch = [rollout(net, rng) for _ in range(BATCH_EP)]
        obs = np.concatenate([b["obs"] for b in batch])
        acts = np.concatenate([b["acts"] for b in batch])
        old_logp = np.concatenate([b["old_logp"] for b in batch])
        returns = np.concatenate([b["returns"] for b in batch])
        values = np.concatenate([np.repeat(b["values"][:, None], N_AGENTS, axis=1)
                                 for b in batch]).reshape(-1)
        obs_rep = np.repeat(obs, N_AGENTS, axis=0)
        adv = returns.repeat(N_AGENTS) - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs_t = torch.tensor(obs_rep, dtype=torch.float32)
        acts_t = torch.tensor(acts.reshape(-1), dtype=torch.int64)
        old_t = torch.tensor(old_logp.reshape(-1), dtype=torch.float32)
        ret_t = torch.tensor(returns.repeat(N_AGENTS), dtype=torch.float32)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        for _ in range(PPO_EPOCHS):
            logits, val = net(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts_t)
            ratio = torch.exp(logp - old_t)
            pg = torch.min(ratio * adv_t,
                           torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_t)
            loss = -pg.mean() + 0.5 * ((val - ret_t) ** 2).mean() - ENT_COEF * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    adj = adjudicate(GRID, np.array(openness) * math.log2(3))
    return {
        "openness": [round(float(v), 5) for v in openness],
        "success": [round(float(v), 5) for v in success],
        "side_mean": [round(float(v), 5) for v in side],
        "p_hist": [[round(float(x), 5) for x in p] for p in p_hist],
        "final_success": round(float(success[-1]), 5),
        "final_entropy": round(float(openness[-1]), 5),
        "final_p": [round(float(x), 5) for x in p_hist[-1]],
        "final_side_pref": int(np.sign(side[-1])),
        "adj": adj,
    }


def main() -> None:
    rows = {}
    for i in range(N_SEEDS):
        row = train_seed(SEED + i * 101)
        rows[str(i)] = row
        h = row["adj"].get("hinge", {})
        print(f"seed={i}: succ={row['final_success']} H={row['final_entropy']} "
              f"p={row['final_p']} B5={row['adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')}",
              flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    low_entropy = [r["final_entropy"] <= 0.35 for r in learned]
    sides = [r["final_side_pref"] for r in learned if r["final_side_pref"] != 0]
    frac_right = None if not sides else float(np.mean([s > 0 for s in sides]))
    outcomes = {
        "LTP1_learnability": bool(len(learned) >= 6),
        "LTP2_symmetry_breaking": bool(
            len(learned) >= 6 and np.mean(low_entropy) >= 0.8
            and frac_right is not None and 0.2 <= frac_right <= 0.8
        ),
        "LTP3_onset_count": sum(1 for r in learned if r["adj"]["b5_onset"]),
        "n_learned": len(learned),
        "learned_frac_right": None if frac_right is None else round(frac_right, 4),
    }
    report = {
        "status": "LEARN-TRANSPORT-PILOT-FAST feasibility run; original full bar not satisfied",
        "config": {"N_agents": N_AGENTS, "threshold": THRESHOLD, "goal": GOAL,
                   "max_steps": MAX_STEPS, "seeds": N_SEEDS,
                   "updates": UPDATES, "batch_ep": BATCH_EP,
                   "save_every": SAVE_EVERY},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_transport_pilot.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
