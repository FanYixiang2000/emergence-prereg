"""LEARN-GRIP-POLICY: openness as a prospective control policy.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Calibrates
the openness trigger and the fixed-time baseline ONLY on records from
the original five seeds (LEARN-GRIP-CONFOUND), then tests on five fresh
seeds trained with the byte-identical recipe. The openness trigger uses
only the policy's own action probabilities -- no environment internals,
no per-test-seed tuning.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from learn_grip_transport import (GOAL, MAX_STEPS, N_AGENTS, THRESHOLD,
                                  ACCEL, DAMP, GRIP_DECAY, GRIP_GAIN,
                                  GRIP_MIN, side_openness)
from learn_grip_utility import KICK_V, KICK_X, train

OUTPUTS = Path(__file__).resolve().parent / "outputs"
TEST_SEED0 = 816_001
N_TEST = 5
EVAL_BATCH = 4096
SWITCH_BAR = 0.95


def calibrate():
    """theta* and tau* from the original-seed confound records only."""
    cc = json.loads((OUTPUTS / "learn_grip_confound.json").read_text())
    by_tau = {int(t): v for t, v in cc["cc1_by_tau"].items()}
    taus_ok = [t for t, v in by_tau.items() if v["switch_rate"] >= SWITCH_BAR]
    tau_star = max(taus_ok)
    # mean openness at tau* per calibration seed, from the run log records
    log = (OUTPUTS / "learn_grip_confound.log").read_text()
    opens = [float(m) for m in re.findall(
        rf"seed=\d+ tau={tau_star}: switch=[0-9.]+ open_mean=([0-9.]+)", log)]
    theta_star = float(np.mean(opens))
    return tau_star, theta_star, opens


def policy_eval(policy, mode, theta, tau_fixed, seed, batch=EVAL_BATCH):
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    att = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    triggered = torch.zeros(batch, dtype=torch.bool)
    incipient = torch.zeros(batch)
    trig_step = torch.full((batch,), -1.0)
    tau_r = torch.randint(10, 31, (batch,), generator=gen).float()
    for t in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v, att], dim=1)
        with torch.no_grad():
            probs = torch.softmax(policy(obs), dim=-1)
        if mode == "open":
            fire = (~triggered) & (side_openness(probs) <= theta)
        elif mode == "fixed":
            fire = (~triggered) & torch.full((batch,), t == tau_fixed)
        else:
            fire = (~triggered) & (tau_r == t)
        if fire.any():
            state_side = torch.sign(x + 0.5 * v)
            rand_side = torch.where(
                torch.rand(batch, generator=gen) < 0.5,
                -torch.ones(batch), torch.ones(batch))
            inc = torch.where(state_side != 0, state_side, rand_side)
            incipient = torch.where(fire, inc, incipient)
            x = torch.where(fire, torch.clamp(x - KICK_X * inc, -GOAL, GOAL), x)
            v = torch.where(fire, v - KICK_V * inc, v)
            trig_step = torch.where(fire, torch.full_like(trig_step, float(t)),
                                    trig_step)
            triggered = triggered | fire
            obs = torch.stack([x / GOAL, v, att], dim=1)
            with torch.no_grad():
                probs = torch.softmax(policy(obs), dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS,
                                               probs=probs)
        counts = dist.sample()
        grip_frac = counts[:, 2] / N_AGENTS
        att = torch.clamp(att + GRIP_GAIN * grip_frac - GRIP_DECAY, 0.0, 1.0)
        force = counts[:, 1] - counts[:, 0]
        active = (att >= GRIP_MIN) & (torch.abs(force) >= THRESHOLD)
        v = DAMP * v + active.float() * ACCEL * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        done = done | (torch.abs(x) >= GOAL - 1e-6)
    final_side = torch.sign(x)
    flip = triggered & (final_side != 0) & (final_side != incipient)
    return {
        "flip_rate": float(flip.float().mean().item()),
        "trigger_rate": float(triggered.float().mean().item()),
        "mean_trigger_step": float(trig_step[triggered].mean().item())
        if triggered.any() else None,
    }


def main() -> None:
    tau_star, theta_star, cal_opens = calibrate()
    print(f"calibration: tau*={tau_star} theta*={theta_star:.4f} "
          f"(per-seed opens at tau*: {cal_opens})", flush=True)

    seeds_out = {}
    pooled = {m: {"flip": [], "step": []} for m in ("open", "fixed", "random")}
    for i in range(N_TEST):
        seed = TEST_SEED0 + i * 101
        policy = train(seed)
        row = {}
        for mode in ("open", "fixed", "random"):
            r = policy_eval(policy, mode, theta_star, tau_star,
                            seed=seed + 7_000 + hash(mode) % 1000)
            row[mode] = {k: (round(val, 5) if isinstance(val, float) else val)
                         for k, val in r.items()}
            pooled[mode]["flip"].append(r["flip_rate"])
            pooled[mode]["step"].append(r["mean_trigger_step"])
        seeds_out[str(i)] = row
        print(f"test seed {i}: open={row['open']} fixed={row['fixed']} "
              f"random={row['random']}", flush=True)

    mean_flip = {m: float(np.mean(pooled[m]["flip"])) for m in pooled}
    mean_step = {m: float(np.mean([s for s in pooled[m]["step"]
                                   if s is not None])) for m in pooled}
    open_seed_ok = sum(f >= 0.90 for f in pooled["open"]["flip"])
    outcomes = {
        "GP1_transfer_ge_0.90_in_4of5": bool(open_seed_ok >= 4),
        "GP2_adaptive_dominance": bool(
            mean_flip["open"] >= mean_flip["fixed"] - 0.02
            and mean_step["open"] >= tau_star + 1.0),
        "GP3_beats_random_by_0.15": bool(
            mean_flip["open"] - mean_flip["random"] >= 0.15),
        "pooled_flip": {m: round(v, 5) for m, v in mean_flip.items()},
        "pooled_mean_step": {m: round(v, 3) for m, v in mean_step.items()},
        "open_seeds_ge_090": int(open_seed_ok),
        "tau_star": tau_star,
        "theta_star": round(theta_star, 5),
    }
    report = {
        "status": ("LEARN-GRIP-POLICY openness-triggered intervention on "
                   "unseen seeds; calibration from original-seed records "
                   "only; registered before run"),
        "config": {"test_seed0": TEST_SEED0, "n_test": N_TEST,
                   "eval_batch": EVAL_BATCH, "switch_bar": SWITCH_BAR},
        "seeds": seeds_out,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_policy.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
