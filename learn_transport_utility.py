"""LEARN-TRANSPORT-UTILITY: learned transport controllability pilot.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Trains
state-dependent transport policies, then tests whether policy entropy
at intervention time predicts side-switchability after a bounded
counter-transport impulse.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from learn_transport_state import (
    BATCH,
    GOAL,
    LR,
    MAX_STEPS,
    N_AGENTS,
    Policy,
    THRESHOLD,
    eval_policy,
    rollout_batch,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_SEEDS = 5
UPDATES = 800
TAUS = (0, 5, 10, 15, 20)
EVAL_BATCH = 2048
SEED = 112_001


def rank_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def train(seed: int) -> Policy:
    torch.manual_seed(seed)
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    baseline = 0.0
    for _ in range(UPDATES):
        returns, logp, _done, _side = rollout_batch(policy, BATCH, train=True)
        adv = returns.detach() - baseline
        baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log2(torch.clamp(probs, min=1e-12))).sum(dim=1) / math.log2(3)


def intervention_eval(policy: Policy, tau: int, batch: int = EVAL_BATCH):
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    incipient = torch.zeros(batch)
    ent_pre = torch.zeros(batch)
    for t in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v], dim=1)
        with torch.no_grad():
            logits = policy(obs)
            probs = torch.softmax(logits, dim=-1)
        if t == tau:
            pref = torch.sign(probs[:, 1] - probs[:, 0])
            state_side = torch.sign(x + 0.5 * v)
            incipient = torch.where(state_side != 0, state_side, pref)
            incipient = torch.where(incipient == 0, torch.ones_like(incipient), incipient)
            ent_pre = entropy_from_probs(probs)
            x = torch.clamp(x - 0.45 * incipient, -GOAL, GOAL)
            v = v - 0.30 * incipient
            obs = torch.stack([x / GOAL, v], dim=1)
            with torch.no_grad():
                probs = torch.softmax(policy(obs), dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS, probs=probs)
        counts = dist.sample()
        left = counts[:, 0]
        right = counts[:, 1]
        force = right - left
        active = torch.abs(force) >= THRESHOLD
        v = 0.85 * v + active.float() * 0.09 * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        done = done | newly
    final_side = torch.sign(x)
    switch = (final_side != 0) & (final_side != incipient)
    return {
        "mean_entropy": float(ent_pre.mean().item()),
        "switch_rate": float(switch.float().mean().item()),
        "success": float(done.float().mean().item()),
        "entropy": ent_pre.numpy().tolist(),
        "switch": switch.numpy().astype(float).tolist(),
    }


def main() -> None:
    seeds = {}
    pooled_e, pooled_s = [], []
    for i in range(N_SEEDS):
        policy = train(SEED + i * 101)
        ev = eval_policy(policy, batch=2048)
        per_tau = {}
        for tau in TAUS:
            row = intervention_eval(policy, tau)
            pooled_e.extend(row["entropy"])
            pooled_s.extend(row["switch"])
            per_tau[str(tau)] = {
                "mean_entropy": round(row["mean_entropy"], 5),
                "switch_rate": round(row["switch_rate"], 5),
                "success": round(row["success"], 5),
            }
        seeds[str(i)] = {
            "final_success": round(ev["success"], 5),
            "final_entropy0": round(ev["entropy0"], 5),
            "final_side_mean": round(ev["side_mean"], 5),
            "per_tau": per_tau,
        }
        print(f"seed={i}: success={seeds[str(i)]['final_success']} "
              f"H0={seeds[str(i)]['final_entropy0']} "
              f"switch0={per_tau['0']['switch_rate']} "
              f"switch20={per_tau['20']['switch_rate']}", flush=True)

    learned = [s for s in seeds.values() if s["final_success"] >= 0.8]
    mean_switch_by_tau = {
        str(t): float(np.mean([seeds[str(i)]["per_tau"][str(t)]["switch_rate"]
                               for i in range(N_SEEDS)]))
        for t in TAUS
    }
    outcomes = {
        "LTU1_learnability": bool(len(learned) >= 4),
        "LTU2_openness_utility": bool(rank_corr(pooled_e, pooled_s) > 0.1),
        "LTU3_timing_utility": bool(mean_switch_by_tau["0"] > mean_switch_by_tau["20"]),
        "LTU4_all_locked_boundary": bool(max(mean_switch_by_tau.values()) < 0.1),
        "n_learned": len(learned),
        "pooled_entropy_switch_rank": round(rank_corr(pooled_e, pooled_s), 5),
        "mean_switch_by_tau": {k: round(v, 5) for k, v in mean_switch_by_tau.items()},
    }
    report = {
        "status": "LEARN-TRANSPORT-UTILITY learned controllability pilot; preregistered",
        "config": {"seeds": N_SEEDS, "updates": UPDATES, "taus": TAUS,
                   "eval_batch": EVAL_BATCH},
        "seeds": seeds,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_transport_utility.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
