"""LEARN-SYMBRIDGE: learned symmetric quorum-bridge pilot.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. This is
a lightweight ML pilot: can a shared policy self-select a bridge side
under sparse quorum reward, and does the policy possibility space show
onset or smooth convergence?
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N = 32
K = 20
N_SEEDS = 20
UPDATES = 6000
BATCH = 96
LR = 0.04
SAVE_EVERY = 50
GRID = tuple(range(0, UPDATES + 1, SAVE_EVERY))
SEED = 105_001


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def entropy_norm(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(3))


def exact_success(p: np.ndarray) -> float:
    # Multinomial probability that A or B reaches quorum K.
    prob = 0.0
    fact = math.factorial
    for a in range(N + 1):
        for b in range(N - a + 1):
            if a >= K or b >= K:
                c = N - a - b
                prob += (
                    fact(N) / (fact(a) * fact(b) * fact(c))
                    * (p[0] ** a) * (p[1] ** b) * (p[2] ** c)
                )
    return float(prob)


def run_seed(seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 0.01, size=3)
    baseline = 0.0
    openness, success, p_hist = [], [], []
    early_sign = None
    for u in range(UPDATES + 1):
        p = softmax(logits)
        if u in GRID:
            openness.append(entropy_norm(p))
            success.append(exact_success(p))
            p_hist.append(p.tolist())
            if u == 500:
                early_sign = int(p[0] >= p[1])
        if u == UPDATES:
            break

        acts = rng.choice(3, size=(BATCH, N), p=p)
        counts_a = (acts == 0).sum(axis=1)
        counts_b = (acts == 1).sum(axis=1)
        rewards = ((counts_a >= K) | (counts_b >= K)).astype(float)
        adv = rewards - baseline
        baseline = 0.98 * baseline + 0.02 * float(rewards.mean())
        grad = np.zeros(3)
        for i in range(BATCH):
            onehot_counts = np.bincount(acts[i], minlength=3)
            grad += adv[i] * (onehot_counts - N * p)
        grad /= BATCH
        logits += LR * grad / N
        logits -= logits.mean()

    final_p = softmax(logits)
    adj = adjudicate(GRID, np.array(openness) * math.log2(3))
    return {
        "openness": [round(v, 5) for v in openness],
        "success": [round(v, 5) for v in success],
        "final_p": [round(float(v), 5) for v in final_p],
        "final_success": round(float(success[-1]), 5),
        "final_entropy": round(float(openness[-1]), 5),
        "final_side_a": bool(final_p[0] >= final_p[1]),
        "early_sign_a": early_sign,
        "adj": adj,
    }


def main() -> None:
    seeds = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        seeds[str(i)] = row
        h = row["adj"].get("hinge", {})
        print(f"seed={i}: succ={row['final_success']} H={row['final_entropy']} "
              f"p={row['final_p']} B5={row['adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')}",
              flush=True)

    learned = [r for r in seeds.values() if r["final_success"] >= 0.8]
    b5_count = sum(1 for r in seeds.values() if r["adj"]["b5_onset"])
    learned_b5 = sum(1 for r in learned if r["adj"]["b5_onset"])
    final_a = [r["final_side_a"] for r in learned]
    frac_a = float(np.mean(final_a)) if final_a else None
    low_entropy = [r["final_entropy"] <= 0.35 for r in learned]
    early_ok = [
        r["early_sign_a"] is not None and bool(r["early_sign_a"]) == r["final_side_a"]
        for r in learned
    ]
    outcomes = {
        "LSB1_learning": bool(len(learned) >= 12),
        "LSB2_onset": bool(learned_b5 >= 6),
        "LSB3_symmetry": bool(
            learned and frac_a is not None and 0.25 <= frac_a <= 0.75
            and np.mean(low_entropy) >= 0.8
        ),
        "n_learned": len(learned),
        "b5_count_all": b5_count,
        "b5_count_learned": learned_b5,
        "learned_frac_a": None if frac_a is None else round(frac_a, 4),
        "early_sign_accuracy_learned": None if not early_ok else round(float(np.mean(early_ok)), 4),
    }
    report = {
        "status": "LEARN-SYMBRIDGE learned symmetric quorum bridge; preregistered",
        "config": {"N": N, "K": K, "seeds": N_SEEDS, "updates": UPDATES,
                   "batch": BATCH, "lr": LR, "grid": GRID},
        "seeds": seeds,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_symbridge.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
