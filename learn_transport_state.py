"""LEARN-TRANSPORT-STATE: state-dependent vectorized transport.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Uses a
small neural policy conditioned on object position/velocity and
vectorized REINFORCE over multinomial action counts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_AGENTS = 16
THRESHOLD = 6
GOAL = 5.0
MAX_STEPS = 45
N_SEEDS = 10
UPDATES = 3000
BATCH = 512
LR = 2e-3
SAVE_EVERY = 50
GRID = tuple(range(0, UPDATES + 1, SAVE_EVERY))
SEED = 111_001


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 3))

    def forward(self, obs):
        return self.net(obs)


def entropy_norm(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(3))


def rollout_batch(policy: Policy, batch: int, train: bool = True):
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    total_r = torch.zeros(batch)
    sum_logp = torch.zeros(batch)
    side_trace = []
    entropy_trace = []
    for _ in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v], dim=1)
        logits = policy(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS, probs=probs)
        counts = dist.sample()
        logp = dist.log_prob(counts)
        left = counts[:, 0]
        right = counts[:, 1]
        force = right - left
        old_abs = torch.abs(x)
        active = torch.abs(force) >= THRESHOLD
        v = 0.85 * v + active.float() * 0.09 * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        r = (torch.abs(x) - old_abs) - 0.005
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        r = r + newly.float() * 5.0
        total_r = total_r + torch.where(done, torch.zeros_like(r), r)
        sum_logp = sum_logp + torch.where(done, torch.zeros_like(logp), logp)
        done = done | newly
        side_trace.append(torch.sign(x).detach().clone())
        ent = -(probs * torch.log2(torch.clamp(probs, min=1e-12))).sum(dim=1) / math.log2(3)
        entropy_trace.append(ent.detach().clone())
    final_side = torch.sign(x)
    if train:
        return total_r, sum_logp, done.float(), final_side
    return total_r.detach(), done.float().detach(), final_side.detach(), torch.stack(side_trace), torch.stack(entropy_trace)


def eval_policy(policy: Policy, batch: int = 2048):
    with torch.no_grad():
        obs0 = torch.zeros((1, 2))
        p0 = torch.softmax(policy(obs0), dim=-1).numpy()[0]
        _r, done, side, side_trace, ent_trace = rollout_batch(policy, batch, train=False)
    success = float(done.mean().item())
    side_mean = float(side.mean().item())
    # Realization openness proxy: policy entropy over time, median across eval episodes.
    ent_med = torch.median(ent_trace, dim=1).values.numpy()
    return {
        "p0": p0,
        "entropy0": entropy_norm(p0),
        "success": success,
        "side_mean": side_mean,
        "episode_entropy_curve": ent_med,
    }


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    baseline = 0.0
    openness, success, side, ent_curves, p_hist = [], [], [], [], []
    for u in range(UPDATES + 1):
        if u in GRID:
            ev = eval_policy(policy, batch=1024)
            openness.append(ev["entropy0"])
            success.append(ev["success"])
            side.append(ev["side_mean"])
            ent_curves.append([float(x) for x in ev["episode_entropy_curve"]])
            p_hist.append(ev["p0"].tolist())
        if u == UPDATES:
            break
        returns, logp, _done, _side = rollout_batch(policy, BATCH, train=True)
        adv = returns.detach() - baseline
        baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    adj = adjudicate(GRID, np.array(openness) * math.log2(3))
    # Episode realization collapse at the final checkpoint.
    ep_ent = np.array(ent_curves[-1])
    ep_drop = float(ep_ent[0] - ep_ent[-1])
    return {
        "openness0": [round(float(v), 5) for v in openness],
        "success": [round(float(v), 5) for v in success],
        "side_mean": [round(float(v), 5) for v in side],
        "final_episode_entropy": [round(float(v), 5) for v in ep_ent],
        "episode_entropy_drop": round(ep_drop, 5),
        "p_hist": [[round(float(x), 5) for x in p] for p in p_hist],
        "final_success": round(float(success[-1]), 5),
        "final_entropy0": round(float(openness[-1]), 5),
        "final_p0": [round(float(x), 5) for x in p_hist[-1]],
        "final_side_pref": int(np.sign(side[-1])),
        "outer_adj": adj,
    }


def main() -> None:
    rows = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        rows[str(i)] = row
        h = row["outer_adj"].get("hinge", {})
        print(f"seed={i}: succ={row['final_success']} H0={row['final_entropy0']} "
              f"epdrop={row['episode_entropy_drop']} p0={row['final_p0']} "
              f"B5={row['outer_adj']['b5_onset']} dBIC={h.get('delta_bic')}",
              flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    sides = [r["final_side_pref"] for r in learned if r["final_side_pref"] != 0]
    frac_right = None if not sides else float(np.mean([s > 0 for s in sides]))
    outcomes = {
        "LTS1_learnability": bool(len(learned) >= 6),
        "LTS2_realization_collapse": bool(
            learned and np.mean([r["episode_entropy_drop"] > 0.15 for r in learned]) >= 0.6
        ),
        "LTS3_symmetry": bool(
            len(learned) >= 6 and frac_right is not None and 0.2 <= frac_right <= 0.8
        ),
        "n_learned": len(learned),
        "learned_frac_right": None if frac_right is None else round(frac_right, 4),
        "outer_b5_count_learned": sum(1 for r in learned if r["outer_adj"]["b5_onset"]),
    }
    report = {
        "status": "LEARN-TRANSPORT-STATE state-dependent vectorized transport; preregistered",
        "config": {"N_agents": N_AGENTS, "threshold": THRESHOLD, "goal": GOAL,
                   "max_steps": MAX_STEPS, "seeds": N_SEEDS,
                   "updates": UPDATES, "batch": BATCH, "lr": LR},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_transport_state.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
