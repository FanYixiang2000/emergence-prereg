"""LEARN-TRANSPORT-VEC: vectorized multi-step transport feasibility.

This is a feasibility variant declared after the slow PPO pilot
produced no seed output. It is not the final flagship: the policy is
state-independent, but the environment has multi-step threshold object
dynamics and side-neutral transport reward.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_AGENTS = 16
THRESHOLD = 6
GOAL = 5.0
MAX_STEPS = 45
N_SEEDS = 20
UPDATES = 5000
BATCH = 1024
LR = 0.08
BASELINE_ALPHA = 0.03
SAVE_EVERY = 50
GRID = tuple(range(0, UPDATES + 1, SAVE_EVERY))
SEED = 110_001


def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_norm(p):
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(3))


def simulate_batch(p, rng, batch=BATCH):
    x = np.zeros(batch)
    v = np.zeros(batch)
    rewards = np.zeros(batch)
    grad_counts = np.zeros((batch, 3))
    done = np.zeros(batch, dtype=bool)
    final_side = np.zeros(batch)
    for _ in range(MAX_STEPS):
        counts = rng.multinomial(N_AGENTS, p, size=batch)
        left = counts[:, 0]
        right = counts[:, 1]
        force = right - left
        old_abs = np.abs(x)
        active = np.abs(force) >= THRESHOLD
        v = 0.85 * v + active * 0.09 * np.sign(force)
        x = np.clip(x + v, -GOAL, GOAL)
        r = (np.abs(x) - old_abs) - 0.005
        newly = (~done) & (np.abs(x) >= GOAL - 1e-9)
        r[newly] += 5.0
        rewards += np.where(done, 0.0, r)
        grad_counts += np.where(done[:, None], 0, counts)
        done |= newly
    final_side[x > 0] = 1
    final_side[x < 0] = -1
    return rewards, grad_counts, done.astype(float), final_side


def eval_policy(p, rng, n=4096):
    rewards, _counts, done, side = simulate_batch(p, rng, batch=n)
    return float(done.mean()), float(side.mean()), float(rewards.mean())


def run_seed(seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 0.01, size=3)
    baseline = 0.0
    openness, success, side, p_hist = [], [], [], []
    for u in range(UPDATES + 1):
        p = softmax(logits)
        if u in GRID:
            s, sd, _ret = eval_policy(p, rng, n=2048)
            openness.append(entropy_norm(p))
            success.append(s)
            side.append(sd)
            p_hist.append(p.copy())
        if u == UPDATES:
            break
        rewards, counts, _done, _side = simulate_batch(p, rng)
        adv = rewards - baseline
        baseline = (1 - BASELINE_ALPHA) * baseline + BASELINE_ALPHA * float(rewards.mean())
        grad = ((counts - MAX_STEPS * N_AGENTS * p) * adv[:, None]).mean(axis=0)
        logits += LR * grad / (MAX_STEPS * N_AGENTS)
        logits -= logits.mean()
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


def main():
    rows = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        rows[str(i)] = row
        h = row["adj"].get("hinge", {})
        print(f"seed={i}: succ={row['final_success']} H={row['final_entropy']} "
              f"p={row['final_p']} B5={row['adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')}",
              flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    sides = [r["final_side_pref"] for r in learned if r["final_side_pref"] != 0]
    frac_right = None if not sides else float(np.mean([s > 0 for s in sides]))
    outcomes = {
        "LTV_learnability": bool(len(learned) >= 12),
        "LTV_low_entropy_learned": bool(learned and np.mean([r["final_entropy"] < 0.35 for r in learned]) >= 0.8),
        "LTV_onset_count": sum(1 for r in learned if r["adj"]["b5_onset"]),
        "n_learned": len(learned),
        "learned_frac_right": None if frac_right is None else round(frac_right, 4),
    }
    report = {
        "status": "LEARN-TRANSPORT-VEC feasibility; not final flagship",
        "config": {"N_agents": N_AGENTS, "threshold": THRESHOLD, "goal": GOAL,
                   "max_steps": MAX_STEPS, "seeds": N_SEEDS,
                   "updates": UPDATES, "batch": BATCH, "lr": LR},
        "seeds": rows,
        "outcomes": outcomes,
    }
    out = OUTPUTS / "learn_transport_vec.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
