"""LEARN-TRANSPORT-EQ-UTILITY: learned controllability + baseline race.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Trains
equivariant-slow transport policies (which preserve initial episode
openness), then tests whether pre-intervention policy openness predicts
side-switchability under a bounded counter-regime impulse, racing it
against generic baselines (|x|, |v|, tau).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from learn_transport_equivariant_slow import (
    ACCEL,
    BATCH,
    DAMP,
    EquivariantSlowPolicy,
    GOAL,
    LR,
    MAX_STEPS,
    N_AGENTS,
    THRESHOLD,
    entropy_curve_norm,
    eval_policy,
    rollout_batch,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_SEEDS = 5
UPDATES = 700
TAUS = (0, 2, 4, 6, 8, 12, 20)
EVAL_BATCH = 2048
KICK_X = 1.0
KICK_V = 0.35
SEED = 115_001


def rank_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(pred, label) -> float:
    """Mann-Whitney AUC of pred for label==1."""
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label, dtype=float)
    pos = pred[label == 1]
    neg = pred[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))).astype(float) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(seed: int) -> EquivariantSlowPolicy:
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
    return policy


def intervention_eval(policy: EquivariantSlowPolicy, tau: int, seed: int,
                      batch: int = EVAL_BATCH):
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    incipient = torch.zeros(batch)
    pred_ent = torch.zeros(batch)
    pred_absx = torch.zeros(batch)
    pred_absv = torch.zeros(batch)
    for t in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v], dim=1)
        with torch.no_grad():
            probs = torch.softmax(policy(obs), dim=-1)
        if t == tau:
            state_side = torch.sign(x + 0.5 * v)
            rand_side = torch.where(
                torch.rand(batch, generator=gen) < 0.5,
                -torch.ones(batch), torch.ones(batch))
            incipient = torch.where(state_side != 0, state_side, rand_side)
            pred_ent = entropy_curve_norm(probs)
            pred_absx = torch.abs(x) / GOAL
            pred_absv = torch.abs(v)
            x = torch.clamp(x - KICK_X * incipient, -GOAL, GOAL)
            v = v - KICK_V * incipient
            obs = torch.stack([x / GOAL, v], dim=1)
            with torch.no_grad():
                probs = torch.softmax(policy(obs), dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS, probs=probs)
        counts = dist.sample()
        force = counts[:, 1] - counts[:, 0]
        active = torch.abs(force) >= THRESHOLD
        v = DAMP * v + active.float() * ACCEL * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        done = done | newly
    final_side = torch.sign(x)
    switch = (final_side != 0) & (final_side != incipient)
    return {
        "switch_rate": float(switch.float().mean().item()),
        "success": float(done.float().mean().item()),
        "mean_entropy": float(pred_ent.mean().item()),
        "switch": switch.numpy().astype(float),
        "entropy": pred_ent.numpy(),
        "absx": pred_absx.numpy(),
        "absv": pred_absv.numpy(),
        "tau": np.full(batch, float(tau)),
    }


def main() -> None:
    seeds = {}
    pool = {k: [] for k in ("switch", "entropy", "absx", "absv", "tau")}
    for i in range(N_SEEDS):
        policy = train(SEED + i * 101)
        ev = eval_policy(policy, batch=2048)
        per_tau = {}
        for tau in TAUS:
            row = intervention_eval(policy, tau, seed=SEED + i * 101 + tau)
            for k in pool:
                pool[k].extend(row[k].tolist())
            per_tau[str(tau)] = {
                "switch_rate": round(row["switch_rate"], 5),
                "success": round(row["success"], 5),
                "mean_entropy": round(row["mean_entropy"], 5),
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
        str(t): round(float(np.mean(
            [seeds[str(i)]["per_tau"][str(t)]["switch_rate"]
             for i in range(N_SEEDS)])), 5)
        for t in TAUS
    }
    switch = np.array(pool["switch"])
    race = {}
    for name, sign in (("entropy", 1.0), ("absx", -1.0), ("absv", -1.0),
                       ("tau", -1.0)):
        vals = sign * np.array(pool[name])
        race[name] = {
            "rank_corr": round(rank_corr(vals, switch), 5),
            "auc": round(auc(vals, switch), 5),
            "oriented": "high predicts switch" if sign > 0 else
                        "low predicts switch (sign-flipped for AUC)",
        }
    outcomes = {
        "LTEQU1_learnability": bool(len(learned) >= 4),
        "LTEQU2_timing_law": bool(
            mean_switch_by_tau[str(TAUS[0])] - mean_switch_by_tau[str(TAUS[-1])] >= 0.3),
        "LTEQU3_openness_utility": bool(
            race["entropy"]["rank_corr"] >= 0.3 and race["entropy"]["auc"] >= 0.65),
        "LTEQU4_beats_tau_baseline": bool(
            race["entropy"]["auc"] > race["tau"]["auc"]),
        "n_learned": len(learned),
        "mean_switch_by_tau": mean_switch_by_tau,
        "baseline_race": race,
    }
    report = {
        "status": "LEARN-TRANSPORT-EQ-UTILITY learned controllability + baseline race; preregistered",
        "config": {"seeds": N_SEEDS, "updates": UPDATES, "taus": TAUS,
                   "eval_batch": EVAL_BATCH, "kick_x": KICK_X, "kick_v": KICK_V},
        "seeds": seeds,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_transport_eq_utility.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
