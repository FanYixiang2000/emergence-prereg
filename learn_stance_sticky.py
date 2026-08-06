"""LEARN-STANCE-STICKY: inertial-consensus separability flagship.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Same
hidden-coordination transport as LEARN-STANCE-TRANSPORT, but stance
changes are sticky (lean flips stance with probability 0.25), so the
consensus consolidation phase is temporally extended: the physical
order parameter stays at zero while the stance openness varies, giving
the separability test a real window.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate
from learn_stance_transport import (
    ACCEL,
    DAMP,
    FORCE_MIN,
    GOAL,
    LR,
    N_AGENTS,
    StancePolicy,
    exchangeable_ladder,
    local_field,
    stance_balance_entropy,
)
from learn_transport_eq_utility import auc, rank_corr

OUTPUTS = Path(__file__).resolve().parent / "outputs"
MAX_STEPS = 70
N_SEEDS = 5
UPDATES = 1000
BATCH = 256
STICK_P = 0.25
TAUS = (2, 4, 6, 8, 10, 14, 20, 30)
EVAL_BATCH = 2048
FLIP_COUNT = 3
LADDER_TIMES = (2, 5, 10, 15, 20, 30, 50)
SEED = 118_001


def step_env(s, x, v, actions, gen: torch.Generator | None = None):
    if gen is None:
        flip = torch.rand(s.shape) < STICK_P
    else:
        flip = torch.rand(s.shape, generator=gen) < STICK_P
    s = torch.where((actions == 0) & flip, -torch.ones_like(s), s)
    s = torch.where((actions == 1) & flip, torch.ones_like(s), s)
    push_force = (s * (actions == 2).float()).sum(dim=1)
    active = torch.abs(push_force) >= FORCE_MIN
    v = DAMP * v + active.float() * ACCEL * torch.sign(push_force)
    x = torch.clamp(x + v, -GOAL, GOAL)
    return s, x, v


def rollout_batch(policy: StancePolicy, batch: int, train: bool = True):
    s = torch.zeros(batch, N_AGENTS)
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    total_r = torch.zeros(batch)
    sum_logp = torch.zeros(batch)
    stance_hist = []
    for _ in range(MAX_STEPS):
        f = local_field(s)
        obs = torch.stack([s, f,
                           (x / GOAL).unsqueeze(1).expand(-1, N_AGENTS),
                           v.unsqueeze(1).expand(-1, N_AGENTS)], dim=-1)
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        logp = dist.log_prob(actions).sum(dim=1)
        old_abs = torch.abs(x)
        s, x, v = step_env(s, x, v, actions)
        r = (torch.abs(x) - old_abs) - 0.004
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        r = r + newly.float() * 5.0
        total_r = total_r + torch.where(done, torch.zeros_like(r), r)
        sum_logp = sum_logp + torch.where(done, torch.zeros_like(logp), logp)
        done = done | newly
        stance_hist.append(s.detach().clone())
    if train:
        return total_r, sum_logp, done.float()
    return done.float().detach(), torch.sign(x).detach(), torch.stack(stance_hist)


def eval_policy(policy: StancePolicy, batch: int = EVAL_BATCH):
    with torch.no_grad():
        done, final_side, stance_hist = rollout_batch(policy, batch, train=False)
    curve = torch.stack(
        [torch.median(stance_balance_entropy(stance_hist[t]))
         for t in range(MAX_STEPS)]).numpy()
    adj = adjudicate(range(MAX_STEPS), curve)
    sides = final_side.numpy()
    nz = sides[sides != 0]
    frac_right = float((nz > 0).mean()) if len(nz) else None
    ladder = {str(t): exchangeable_ladder(stance_hist[t].numpy())
              for t in LADDER_TIMES}
    return {
        "success": float(done.mean().item()),
        "frac_right": frac_right,
        "stance_entropy_curve": [round(float(u), 5) for u in curve],
        "episode_adj": adj,
        "ladder": ladder,
    }


def intervention_eval(policy: StancePolicy, tau: int, seed: int,
                      batch: int = EVAL_BATCH):
    gen = torch.Generator().manual_seed(seed)
    s = torch.zeros(batch, N_AGENTS)
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    incipient = torch.zeros(batch)
    preds = {k: torch.zeros(batch) for k in ("open", "absx", "absv")}
    for t in range(MAX_STEPS):
        if t == tau:
            ssum = s.sum(dim=1)
            state_side = torch.sign(ssum)
            rand_side = torch.where(
                torch.rand(batch, generator=gen) < 0.5,
                -torch.ones(batch), torch.ones(batch))
            incipient = torch.where(state_side != 0, state_side, rand_side)
            preds["open"] = stance_balance_entropy(s)
            preds["absx"] = torch.abs(x) / GOAL
            preds["absv"] = torch.abs(v)
            perm = torch.rand(batch, N_AGENTS, generator=gen).argsort(dim=1)
            flip = perm < FLIP_COUNT
            s = torch.where(flip, -incipient.unsqueeze(1).expand_as(s), s)
        f = local_field(s, gen)
        obs = torch.stack([s, f,
                           (x / GOAL).unsqueeze(1).expand(-1, N_AGENTS),
                           v.unsqueeze(1).expand(-1, N_AGENTS)], dim=-1)
        with torch.no_grad():
            logits = policy(obs)
        actions = torch.distributions.Categorical(logits=logits).sample()
        s, x, v = step_env(s, x, v, actions, gen)
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        done = done | newly
    final_side = torch.sign(x)
    switch = (final_side != 0) & (final_side != incipient)
    row = {"switch_rate": float(switch.float().mean().item()),
           "success": float(done.float().mean().item()),
           "switch": switch.numpy().astype(float),
           "tau_arr": np.full(batch, float(tau))}
    for k, vals in preds.items():
        row[k] = vals.numpy()
    return row


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = StancePolicy()
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


def main() -> None:
    rows = {}
    pool = {k: [] for k in ("switch", "open", "absx", "absv", "tau")}
    for i in range(N_SEEDS):
        policy = run_seed(SEED + i * 101)
        ev = eval_policy(policy)
        per_tau = {}
        for tau in TAUS:
            row = intervention_eval(policy, tau, seed=SEED + i * 101 + tau)
            for k in ("switch", "open", "absx", "absv"):
                pool[k].extend(row[k].tolist())
            pool["tau"].extend(row["tau_arr"].tolist())
            per_tau[str(tau)] = {"switch_rate": round(row["switch_rate"], 5),
                                 "success": round(row["success"], 5)}
        rows[str(i)] = {
            "final_success": round(ev["success"], 5),
            "frac_right": ev["frac_right"],
            "episode_adj": ev["episode_adj"],
            "stance_entropy_curve": ev["stance_entropy_curve"],
            "ladder": ev["ladder"],
            "per_tau": per_tau,
        }
        h = ev["episode_adj"].get("hinge", {})
        lad_last = ev["ladder"][str(LADDER_TIMES[-1])]
        print(f"seed={i}: succ={rows[str(i)]['final_success']} "
              f"fracR={ev['frac_right']} B5={ev['episode_adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} H1_end={lad_last['H1']} "
              f"TC_end={lad_last['TC']} "
              f"switch@2={per_tau['2']['switch_rate']} "
              f"switch@30={per_tau['30']['switch_rate']}", flush=True)

    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    switch = np.array(pool["switch"])
    race = {}
    for name, sign in (("open", 1.0), ("absx", -1.0), ("absv", -1.0),
                       ("tau", -1.0)):
        vals = sign * np.array(pool[name])
        race[name] = {"rank_corr": round(rank_corr(vals, switch), 5),
                      "auc": round(auc(vals, switch), 5)}
    lad_ok = [
        r for r in learned
        if (r["ladder"][str(LADDER_TIMES[-1])]["H1"] >= 0.7
            and r["ladder"][str(LADDER_TIMES[-1])]["TC"] >= 3.0)
    ]
    outcomes = {
        "LSS1_learnability": bool(len(learned) >= 4),
        "LSS2_separability": bool(race["open"]["auc"] > race["absx"]["auc"]),
        "LSS3_relational_collapse": bool(
            learned and len(lad_ok) >= max(3, len(learned) - 1)),
        "LSS4_realization_B5": bool(
            len(learned) >= 4
            and sum(r["episode_adj"]["b5_onset"] for r in learned) >= 2),
        "LSS5_symmetry": bool(
            learned and all(
                r["frac_right"] is not None and 0.2 <= r["frac_right"] <= 0.8
                for r in learned)),
        "n_learned": len(learned),
        "baseline_race": race,
        "b5_count_learned": sum(
            r["episode_adj"]["b5_onset"] for r in learned),
    }
    report = {
        "status": "LEARN-STANCE-STICKY inertial-consensus flagship; preregistered",
        "config": {"N_agents": N_AGENTS, "force_min": FORCE_MIN, "goal": GOAL,
                   "max_steps": MAX_STEPS, "seeds": N_SEEDS,
                   "updates": UPDATES, "batch": BATCH, "stick_p": STICK_P,
                   "taus": TAUS, "flip_count": FLIP_COUNT,
                   "ladder_times": LADDER_TIMES},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_stance_sticky.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
