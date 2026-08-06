"""LEARN-TRANSPORT-EQUIVARIANT-SLOW: resolvable learned realization.

This variant slows the one-dimensional transport dynamics and bounds the
direction logit so the within-episode commitment has a longer temporal
window for B5 testing.
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
GOAL = 8.0
MAX_STEPS = 90
N_SEEDS = 5
UPDATES = 1200
BATCH = 512
LR = 2e-3
DAMP = 0.92
ACCEL = 0.035
DIR_BOUND = 2.2
SEED = 114_001


class EquivariantSlowPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dir = nn.Sequential(
            nn.Linear(2, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 32, bias=False),
            nn.Tanh(),
            nn.Linear(32, 1, bias=False),
        )
        self.idle = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        a = DIR_BOUND * torch.tanh(self.dir(obs).squeeze(-1))
        idle = self.idle.expand_as(a)
        return torch.stack([-a, a, idle], dim=1)


def entropy_curve_norm(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log2(torch.clamp(probs, min=1e-12))).sum(dim=1) / math.log2(3)


def entropy_norm(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(3))


def rollout_batch(policy: EquivariantSlowPolicy, batch: int, train: bool = True):
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    total_r = torch.zeros(batch)
    sum_logp = torch.zeros(batch)
    ent_trace = []
    side_trace = []
    for _ in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v], dim=1)
        logits = policy(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS, probs=probs)
        counts = dist.sample()
        logp = dist.log_prob(counts)
        force = counts[:, 1] - counts[:, 0]
        old_abs = torch.abs(x)
        active = torch.abs(force) >= THRESHOLD
        v = DAMP * v + active.float() * ACCEL * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        r = (torch.abs(x) - old_abs) - 0.004
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        r = r + newly.float() * 5.0
        total_r = total_r + torch.where(done, torch.zeros_like(r), r)
        sum_logp = sum_logp + torch.where(done, torch.zeros_like(logp), logp)
        done = done | newly
        ent_trace.append(entropy_curve_norm(probs).detach())
        side_trace.append(torch.sign(x).detach())
    if train:
        return total_r, sum_logp, done.float()
    return done.float().detach(), torch.sign(x).detach(), torch.stack(ent_trace), torch.stack(side_trace)


def eval_policy(policy: EquivariantSlowPolicy, batch: int = 4096):
    with torch.no_grad():
        p0 = torch.softmax(policy(torch.zeros((1, 2))), dim=-1).numpy()[0]
        done, final_side, ent_trace, side_trace = rollout_batch(policy, batch, train=False)
    ep_ent = torch.median(ent_trace, dim=1).values.numpy()
    side_abs = torch.mean(torch.abs(side_trace), dim=1).numpy()
    adj = adjudicate(range(MAX_STEPS), ep_ent * math.log2(3))
    return {
        "p0": p0,
        "entropy0": entropy_norm(p0),
        "success": float(done.mean().item()),
        "side_mean": float(final_side.mean().item()),
        "episode_entropy_curve": ep_ent,
        "episode_entropy_drop": float(ep_ent[0] - ep_ent[-1]),
        "side_abs_curve": side_abs,
        "episode_adj": adj,
    }


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = EquivariantSlowPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    baseline = 0.0
    for _ in range(UPDATES):
        returns, logp, _done = rollout_batch(policy, BATCH, train=True)
        adv = returns.detach() - baseline
        baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    ev = eval_policy(policy)
    return {
        "final_success": round(ev["success"], 5),
        "final_entropy0": round(ev["entropy0"], 5),
        "final_p0": [round(float(x), 5) for x in ev["p0"]],
        "final_side_mean": round(ev["side_mean"], 5),
        "episode_entropy_drop": round(ev["episode_entropy_drop"], 5),
        "final_episode_entropy": [round(float(x), 5) for x in ev["episode_entropy_curve"]],
        "side_abs_curve": [round(float(x), 5) for x in ev["side_abs_curve"]],
        "episode_adj": ev["episode_adj"],
    }


def main() -> None:
    rows = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        rows[str(i)] = row
        h = row["episode_adj"].get("hinge", {})
        print(f"seed={i}: success={row['final_success']} H0={row['final_entropy0']} "
              f"side_mean={row['final_side_mean']} epdrop={row['episode_entropy_drop']} "
              f"B5={row['episode_adj']['b5_onset']} dBIC={h.get('delta_bic')}",
              flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    outcomes = {
        "LTES1_learnability": bool(len(learned) >= 4),
        "LTES2_initial_symmetry_openness": bool(
            len(learned) >= 4
            and np.mean([r["final_entropy0"] >= 0.5 and abs(r["final_side_mean"]) <= 0.4
                         for r in learned]) >= 0.8
        ),
        "LTES3_realization_collapse": bool(
            len(learned) >= 4
            and sum(r["episode_entropy_drop"] >= 0.15 for r in learned) >= 3
        ),
        "LTES4_resolvable_onset": bool(
            len(learned) >= 4
            and sum(r["episode_adj"]["b5_onset"] for r in learned) >= 2
        ),
        "n_learned": len(learned),
        "mean_entropy0_learned": round(float(np.mean([r["final_entropy0"] for r in learned])), 5)
        if learned else None,
        "mean_epdrop_learned": round(float(np.mean([r["episode_entropy_drop"] for r in learned])), 5)
        if learned else None,
        "episode_b5_count_learned": sum(r["episode_adj"]["b5_onset"] for r in learned),
    }
    report = {
        "status": "LEARN-TRANSPORT-EQUIVARIANT-SLOW resolvable learned realization; preregistered",
        "config": {"N_agents": N_AGENTS, "threshold": THRESHOLD, "goal": GOAL,
                   "max_steps": MAX_STEPS, "seeds": N_SEEDS,
                   "updates": UPDATES, "batch": BATCH, "lr": LR,
                   "damp": DAMP, "accel": ACCEL, "dir_bound": DIR_BOUND},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_transport_equivariant_slow.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
