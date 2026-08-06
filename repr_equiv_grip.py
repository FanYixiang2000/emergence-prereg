"""REPR-EQUIV (grip): true representation battery on the grip flagship.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Retrains
the five published seeds with the byte-identical recipe (same seed
values, same RNG consumption as learn_grip_transport.run_seed), then
measures the within-episode side-commitment openness under five
REPRESENTATIONS of the same underlying process. Only the measurement
changes; the acting policy always sees the true observation.

  G1 policy-probability side openness (published object);
  G2 observer reads the policy through states quantized to 1 decimal;
  G3 observer reads the policy through states quantized to steps of 0.5;
  G4 behavioural: empirical side entropy of the realized 16-agent
     action counts (samples, not probabilities);
  G5 probability truncation epsilon = 0.01 with renormalization.

Every representation curve (per-step batch median, then seed mean) is
adjudicated by the frozen detector on the same scale used throughout
(curve in [0,1] times log2 3).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate
from learn_grip_transport import (ACCEL, BATCH, DAMP, GOAL, GRIP_DECAY,
                                  GRIP_GAIN, GRIP_MIN, LR, MAX_STEPS,
                                  N_AGENTS, N_SEEDS, SEED, THRESHOLD,
                                  UPDATES, GripPolicy, rollout_batch,
                                  side_openness)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
EVAL_BATCH = 4096
EPS_TRUNC = 0.01
LOG2_3 = math.log2(3)
REPS = ("G1_policy_probs", "G2_state_coarse_0.1", "G3_state_coarse_0.5",
        "G4_behavioural_counts", "G5_prob_truncation")


def side_open_trunc(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clone()
    p[p < EPS_TRUNC] = 0.0
    p = p / torch.clamp(p.sum(dim=1, keepdim=True), min=1e-12)
    return side_openness(p)


def behavioural_side_entropy(counts: torch.Tensor) -> torch.Tensor:
    lr = counts[:, :2]
    tot = lr.sum(dim=1)
    frac = lr[:, 1] / torch.clamp(tot, min=1.0)
    h = -(frac * torch.log2(torch.clamp(frac, min=1e-12))
          + (1 - frac) * torch.log2(torch.clamp(1 - frac, min=1e-12)))
    return torch.where(tot > 0, h, torch.ones_like(h))


def eval_representations(policy: GripPolicy, batch: int = EVAL_BATCH):
    """One shared rollout; five observer readouts per step."""
    traces = {r: [] for r in REPS}
    with torch.no_grad():
        x = torch.zeros(batch)
        v = torch.zeros(batch)
        att = torch.zeros(batch)
        for _ in range(MAX_STEPS):
            obs = torch.stack([x / GOAL, v, att], dim=1)
            logits = policy(obs)
            probs = torch.softmax(logits, dim=-1)
            # observer readouts (measurement only, never used for acting)
            p_c1 = torch.softmax(policy(torch.round(obs * 10) / 10), dim=-1)
            p_c5 = torch.softmax(policy(torch.round(obs * 2) / 2), dim=-1)
            dist = torch.distributions.Multinomial(total_count=N_AGENTS,
                                                   probs=probs)
            counts = dist.sample()
            traces["G1_policy_probs"].append(side_openness(probs))
            traces["G2_state_coarse_0.1"].append(side_openness(p_c1))
            traces["G3_state_coarse_0.5"].append(side_openness(p_c5))
            traces["G4_behavioural_counts"].append(
                behavioural_side_entropy(counts))
            traces["G5_prob_truncation"].append(side_open_trunc(probs))
            # dynamics identical to learn_grip_transport.rollout_batch
            grip_frac = counts[:, 2] / N_AGENTS
            att = torch.clamp(att + GRIP_GAIN * grip_frac - GRIP_DECAY,
                              0.0, 1.0)
            force = counts[:, 1] - counts[:, 0]
            active = (att >= GRIP_MIN) & (torch.abs(force) >= THRESHOLD)
            v = DAMP * v + active.float() * ACCEL * torch.sign(force)
            x = torch.clamp(x + v, -GOAL, GOAL)
    return {r: torch.median(torch.stack(t), dim=1).values.numpy()
            for r, t in traces.items()}


def train_seed(seed: int) -> GripPolicy:
    """Byte-identical replication of learn_grip_transport.run_seed training."""
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


def main() -> None:
    torch.set_num_threads(4)
    per_seed = {}
    curves_sum = {r: np.zeros(MAX_STEPS) for r in REPS}
    for i in range(N_SEEDS):
        seed = SEED + i * 101
        policy = train_seed(seed)
        curves = eval_representations(policy)
        row = {}
        for r in REPS:
            curves_sum[r] += curves[r]
            adj = adjudicate(range(MAX_STEPS), curves[r] * LOG2_3)
            h = adj.get("hinge", {})
            row[r] = {"b5_onset": adj["b5_onset"],
                      "t_star": h.get("t_star"),
                      "delta_bic": h.get("delta_bic"),
                      "curve": [round(float(y), 5) for y in curves[r]]}
        per_seed[str(i)] = row
        print(f"seed={i}: " + " ".join(
            f"{r.split('_')[0]}:B5={row[r]['b5_onset']},t*={row[r]['t_star']}"
            for r in REPS), flush=True)

    rep_cells = {}
    for r in REPS:
        mean_curve = curves_sum[r] / N_SEEDS
        adj = adjudicate(range(MAX_STEPS), mean_curve * LOG2_3)
        h = adj.get("hinge", {})
        rep_cells[r] = {"b5_onset": adj["b5_onset"],
                        "t_star": h.get("t_star"),
                        "delta_bic": h.get("delta_bic"),
                        "mean_curve": [round(float(y), 5)
                                       for y in mean_curve]}
        print(f"seed-mean {r}: B5={adj['b5_onset']} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')}", flush=True)

    onsets = [r for r in REPS if rep_cells[r]["b5_onset"]]
    tstars = [rep_cells[r]["t_star"] for r in onsets
              if rep_cells[r]["t_star"] is not None]
    span = float(MAX_STEPS - 1)
    trange = (max(tstars) - min(tstars)) / span if len(tstars) >= 2 else 0.0
    outcomes = {
        "n_representations": len(REPS),
        "n_onset_preserved": len(onsets),
        "onset_preservation_rate": round(len(onsets) / len(REPS), 4),
        "t_star_values": tstars,
        "t_star_range_frac_of_span": round(trange, 4),
        "RE2_grip_tstar_range_le_15pct": bool(trange <= 0.15),
        "breaking_representations": [r for r in REPS
                                     if not rep_cells[r]["b5_onset"]],
    }
    report = {
        "status": ("REPR-EQUIV grip battery; measurement-only representation "
                   "changes on byte-identical retrained published seeds; "
                   "registered before run"),
        "config": {"seeds": N_SEEDS, "seed0": SEED, "eval_batch": EVAL_BATCH,
                   "eps_trunc": EPS_TRUNC, "representations": list(REPS)},
        "representation_cells": rep_cells,
        "per_seed": per_seed,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "repr_equiv_grip.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
