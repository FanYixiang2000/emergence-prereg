"""LEARN-GRIP-UTILITY: breakpoint = controllability window closing.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Retrains
the LEARN-GRIP-TRANSPORT policies (deterministic given seed) and tests
whether the LGT-B side-openness breakpoint t* marks the closing of the
counter-regime intervention window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    N_SEEDS,
    SEED,
    THRESHOLD,
    UPDATES,
    rollout_batch,
    side_openness,
)
from learn_transport_eq_utility import auc, rank_corr

OUTPUTS = Path(__file__).resolve().parent / "outputs"
TAUS = (5, 10, 14, 16, 18, 20, 24, 30)
EVAL_BATCH = 2048
KICK_X = 1.0
KICK_V = 0.35


def train(seed: int) -> GripPolicy:
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = GripPolicy()
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


def intervention_eval(policy: GripPolicy, tau: int, seed: int,
                      batch: int = EVAL_BATCH):
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    att = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    incipient = torch.zeros(batch)
    preds = {k: torch.zeros(batch) for k in
             ("side_open", "absx", "absv", "att")}
    for t in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v, att], dim=1)
        with torch.no_grad():
            probs = torch.softmax(policy(obs), dim=-1)
        if t == tau:
            state_side = torch.sign(x + 0.5 * v)
            rand_side = torch.where(
                torch.rand(batch, generator=gen) < 0.5,
                -torch.ones(batch), torch.ones(batch))
            incipient = torch.where(state_side != 0, state_side, rand_side)
            preds["side_open"] = side_openness(probs)
            preds["absx"] = torch.abs(x) / GOAL
            preds["absv"] = torch.abs(v)
            preds["att"] = att.clone()
            x = torch.clamp(x - KICK_X * incipient, -GOAL, GOAL)
            v = v - KICK_V * incipient
            obs = torch.stack([x / GOAL, v, att], dim=1)
            with torch.no_grad():
                probs = torch.softmax(policy(obs), dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS, probs=probs)
        counts = dist.sample()
        grip_frac = counts[:, 2] / N_AGENTS
        att = torch.clamp(att + GRIP_GAIN * grip_frac - GRIP_DECAY, 0.0, 1.0)
        force = counts[:, 1] - counts[:, 0]
        active = (att >= GRIP_MIN) & (torch.abs(force) >= THRESHOLD)
        v = DAMP * v + active.float() * ACCEL * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        done = done | newly
    final_side = torch.sign(x)
    switch = (final_side != 0) & (final_side != incipient)
    row = {
        "switch_rate": float(switch.float().mean().item()),
        "success": float(done.float().mean().item()),
        "switch": switch.numpy().astype(float),
        "tau_arr": np.full(batch, float(tau)),
    }
    for k, vals in preds.items():
        row[k] = vals.numpy()
    return row


def main() -> None:
    lgtb = json.loads((OUTPUTS / "learn_grip_transport_b5.json").read_text())
    seeds = {}
    pool = {k: [] for k in ("switch", "side_open", "absx", "absv", "att", "tau")}
    align_hits = []
    for i in range(N_SEEDS):
        policy = train(SEED + i * 101)
        per_tau = {}
        for tau in TAUS:
            row = intervention_eval(policy, tau, seed=SEED + i * 101 + tau)
            for k in ("switch", "side_open", "absx", "absv", "att"):
                pool[k].extend(row[k].tolist())
            pool["tau"].extend(row["tau_arr"].tolist())
            per_tau[str(tau)] = {
                "switch_rate": round(row["switch_rate"], 5),
                "success": round(row["success"], 5),
                "mean_side_open": round(float(np.mean(row["side_open"])), 5),
            }
        rates = [per_tau[str(t)]["switch_rate"] for t in TAUS]
        drops = [rates[j] - rates[j + 1] for j in range(len(rates) - 1)]
        jmax = int(np.argmax(drops))
        drop_mid = 0.5 * (TAUS[jmax] + TAUS[jmax + 1])
        t_star = lgtb["seeds"][str(i)]["adj"]["hinge"]["t_star"]
        aligned = abs(drop_mid - t_star) <= 3.0
        align_hits.append(aligned)
        seeds[str(i)] = {
            "per_tau": per_tau,
            "largest_drop_between": [TAUS[jmax], TAUS[jmax + 1]],
            "largest_drop_mid": drop_mid,
            "lgtb_t_star": t_star,
            "breakpoint_aligned": bool(aligned),
        }
        print(f"seed={i}: switch5={rates[0]} switch30={rates[-1]} "
              f"drop_mid={drop_mid} t*={t_star} aligned={aligned}", flush=True)

    switch = np.array(pool["switch"])
    race = {}
    for name, sign in (("side_open", 1.0), ("absx", -1.0), ("absv", -1.0),
                       ("att", -1.0), ("tau", -1.0)):
        vals = sign * np.array(pool[name])
        race[name] = {
            "rank_corr": round(rank_corr(vals, switch), 5),
            "auc": round(auc(vals, switch), 5),
        }
    mean_switch_by_tau = {
        str(t): round(float(np.mean(
            [seeds[str(i)]["per_tau"][str(t)]["switch_rate"]
             for i in range(N_SEEDS)])), 5)
        for t in TAUS
    }
    outcomes = {
        "LGU1_window_exists": bool(
            mean_switch_by_tau["5"] >= 0.8 and mean_switch_by_tau["30"] <= 0.3),
        "LGU2_breakpoint_alignment": bool(sum(align_hits) >= 3),
        "LGU3_openness_utility": bool(race["side_open"]["auc"] >= 0.8),
        "LGU4_beats_tau_baseline": bool(
            race["side_open"]["auc"] > race["tau"]["auc"]),
        "align_hits": int(sum(align_hits)),
        "mean_switch_by_tau": mean_switch_by_tau,
        "baseline_race": race,
    }
    report = {
        "status": "LEARN-GRIP-UTILITY breakpoint vs controllability window; preregistered",
        "config": {"taus": TAUS, "eval_batch": EVAL_BATCH,
                   "kick_x": KICK_X, "kick_v": KICK_V},
        "seeds": seeds,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_utility.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
