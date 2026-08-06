"""EEC-LADDER-B: corrected plateau -> seed -> lock ladder.

Registered after the first ladder miss and before this run. The goal
is to test whether an actual open plateau followed by endogenous
seed/quorum lock produces the predicted onset profile.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N = 32
N_EP = 300
T = 140
SAVE_EVERY = 2
GRID = list(range(0, T + 1, SAVE_EVERY))
GATE = 0.1
SEED = 102_001


def entropy3(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(3))


def draw(rng: np.random.Generator, probs: Tuple[float, float, float]) -> np.ndarray:
    p = np.array(probs, dtype=float)
    p /= p.sum()
    return rng.choice(3, size=N, p=p)


def episode(level: str, rng: np.random.Generator):
    target_left = int(rng.integers(0, 2))
    committed = False
    commit_t = None
    left_commit = None
    support = 0.0
    curve = []
    final_left_share = 0.5

    for t in range(T + 1):
        if level == "smooth":
            # Fast early convergence, then saturation: a deceleration control.
            u = 1.0 - math.exp(-t / 45.0)
            probs = (0.18 + 0.62 * u, 0.18 * (1 - 0.7 * u), 0.64 - 0.50 * u)
        elif committed:
            probs = (0.91, 0.025, 0.065)
        else:
            probs = (0.18, 0.18, 0.64)

        actions = draw(rng, probs)
        correct = int((actions == 0).sum())
        wrong = int((actions == 1).sum())
        idle = N - correct - wrong

        if not committed and level != "smooth":
            if level == "threshold":
                if t >= 45 and correct >= 11:
                    support += 4.0
                else:
                    support = max(0.0, support - 0.5)
                hazard = 0.0
            elif level == "feedback":
                local_signal = max(0, correct - 7)
                support = max(0.0, 0.88 * support + local_signal)
                hazard = 0.003 * support if t >= 30 else 0.0
            elif level == "anti_shortcut":
                # No stable individual shortcut: the exact target side is
                # randomized, and a same-episode micro-quorum must appear.
                local_signal = max(0, correct - wrong - 3)
                support = max(0.0, 0.82 * support + local_signal)
                hazard = 0.0025 * support if t >= 35 else 0.0
            else:
                raise ValueError(level)

            if support >= 18.0 or rng.random() < min(hazard, 0.35):
                committed = True
                commit_t = t
                left_commit = target_left

        if committed:
            final_left_share = 1.0 if left_commit == 1 else 0.0
        else:
            final_left_share = (correct if target_left == 1 else wrong) / N

        if t % SAVE_EVERY == 0:
            curve.append(entropy3(np.array([correct, wrong, idle])))

    return curve, commit_t, final_left_share


def adjudicate(openness: List[float]) -> Dict[str, object]:
    y = np.array(openness)
    drop = float(y[0] - y[-1])
    out: Dict[str, object] = {
        "drop": round(drop, 4),
        "max_adjacent_drop": round(float(np.max(-np.diff(y))), 4),
        "gate_passed": bool(drop >= GATE),
    }
    if out["gate_passed"]:
        h = hinge_linear(np.array(GRID, dtype=float), y)
        out["hinge"] = h
        out["b5_onset"] = bool(h["delta_bic"] >= 10 and h["onset_type"])
    else:
        out["b5_onset"] = False
    return out


def run_level(level: str, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    curves, locks, left = [], [], []
    for _ in range(N_EP):
        c, t_lock, l = episode(level, rng)
        curves.append(c)
        locks.append(T + 1 if t_lock is None else t_lock)
        left.append(l)
    curves = np.array(curves)
    med = np.median(curves, axis=0)
    left = np.array(left)
    locks = np.array(locks)
    balance = 1.0 - abs(float(left.mean()) - 0.5) * 2.0
    within = float(np.mean((left < 0.15) | (left > 0.85)))
    return {
        "openness_median": [round(v, 5) for v in med],
        "adj": adjudicate(med.tolist()),
        "lock_rate": round(float(np.mean(locks <= T)), 4),
        "median_lock_time": None if np.all(locks > T) else round(float(np.median(locks[locks <= T])), 3),
        "final_left_mean": round(float(left.mean()), 4),
        "within_episode_lock": round(within, 4),
        "across_episode_balance": round(balance, 4),
        "symmetry_breaking": round(balance * within, 4),
    }


def main() -> None:
    levels = ["smooth", "threshold", "feedback", "anti_shortcut"]
    rows = {}
    for i, level in enumerate(levels):
        rows[level] = run_level(level, SEED + 101 * i)
        h = rows[level]["adj"].get("hinge", {})
        print(f"{level}: drop={rows[level]['adj']['drop']} "
              f"B5={rows[level]['adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')} "
              f"lock={rows[level]['median_lock_time']} "
              f"sym={rows[level]['symmetry_breaking']}",
              flush=True)

    threshold_lock = rows["threshold"]["median_lock_time"] or 999
    feedback_lock = rows["feedback"]["median_lock_time"] or 999
    outcomes = {
        "EECB1_smooth_deceleration": bool(not rows["smooth"]["adj"]["b5_onset"]),
        "EECB2_threshold_onset": bool(rows["threshold"]["adj"]["b5_onset"]),
        "EECB3_feedback_stronger_or_earlier": bool(
            rows["feedback"]["adj"].get("hinge", {}).get("delta_bic", -999)
            >= rows["threshold"]["adj"].get("hinge", {}).get("delta_bic", -999)
            or feedback_lock <= threshold_lock
        ),
        "EECB4_anti_shortcut_symmetry": bool(
            rows["anti_shortcut"]["within_episode_lock"] > 0.8
            and rows["anti_shortcut"]["across_episode_balance"] > 0.8
        ),
    }
    report = {
        "status": "EEC-LADDER-B corrected mechanism ladder; preregistered",
        "config": {"N": N, "n_ep": N_EP, "T": T, "grid": GRID},
        "levels": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "eec_ladder_b.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
