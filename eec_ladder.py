"""EEC-LADDER: mechanism ladder for emergence-enabling conditions.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. This is
a synthetic pre-RL calibration for the proposed learned spatial
flagship: smooth -> threshold -> feedback -> anti-shortcut.
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
N_EP = 200
T = 120
SAVE_EVERY = 2
GRID = list(range(0, T + 1, SAVE_EVERY))
K = 10
GATE = 0.1
SEED = 101_001


def entropy3(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(3))


def sample_actions(rng: np.random.Generator, p_correct: float,
                   p_wrong: float, n: int = N) -> np.ndarray:
    p_idle = max(0.0, 1.0 - p_correct - p_wrong)
    probs = np.array([p_correct, p_wrong, p_idle], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(3, size=n, p=probs)


def episode(level: str, rng: np.random.Generator) -> Tuple[List[float], int, float]:
    # target_side is hidden from the profile except for correct/wrong alignment.
    target_side = int(rng.integers(0, 2))
    committed = False
    commit_side = None
    open_curve = []
    final_left_votes = 0

    for t in range(T + 1):
        if level == "smooth":
            ramp = t / T
            p_correct = 0.18 + 0.62 * ramp
            p_wrong = 0.18 * (1 - 0.5 * ramp)
        elif level == "threshold":
            if committed:
                p_correct, p_wrong = 0.86, 0.04
            else:
                p_correct, p_wrong = 0.22, 0.18
        elif level == "feedback":
            if committed:
                p_correct, p_wrong = 0.90, 0.03
            else:
                # Positive feedback is carried by the previous quorum proxy:
                # early correct attempts make additional correct action more likely.
                p_correct = 0.18 + 0.035 * (final_left_votes if target_side == 0
                                            else N - final_left_votes)
                p_correct = min(p_correct, 0.72)
                p_wrong = 0.18
        elif level == "anti_shortcut":
            if committed:
                p_correct, p_wrong = 0.92, 0.02
            else:
                # No fixed low-order shortcut: before quorum, aligned actions
                # are near-open and left/right-balanced across episodes.
                p_correct, p_wrong = 0.17, 0.17
        else:
            raise ValueError(level)

        actions = sample_actions(rng, p_correct, p_wrong)
        correct = int((actions == 0).sum())
        wrong = int((actions == 1).sum())

        if not committed:
            if level in {"threshold", "feedback", "anti_shortcut"} and correct >= K:
                committed = True
                commit_side = target_side
            elif level == "feedback" and correct >= max(6, K - 3) and rng.random() < 0.25:
                committed = True
                commit_side = target_side

        # Convert correct/wrong categories back to left votes only for the
        # symmetry-breaking diagnostic.
        if target_side == 0:
            final_left_votes = correct
        else:
            final_left_votes = wrong

        if t % SAVE_EVERY == 0:
            open_curve.append(entropy3(np.array([correct, wrong, N - correct - wrong])))

    if committed:
        final_left_share = 1.0 if commit_side == 0 else 0.0
    else:
        final_left_share = final_left_votes / N
    return open_curve, int(committed), final_left_share


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
    curves, committed, left = [], [], []
    for _ in range(N_EP):
        c, q, l = episode(level, rng)
        curves.append(c)
        committed.append(q)
        left.append(l)
    curves = np.array(curves)
    med = np.median(curves, axis=0)
    adj = adjudicate(med.tolist())
    left = np.array(left)
    # Balanced across episodes, sharp within episodes -> symmetry breaking.
    across_balance = 1.0 - abs(float(left.mean()) - 0.5) * 2.0
    within_lock = float(np.mean((left < 0.15) | (left > 0.85)))
    symmetry_breaking = across_balance * within_lock
    return {
        "openness_median": [round(v, 5) for v in med],
        "adj": adj,
        "commit_rate": round(float(np.mean(committed)), 4),
        "final_left_mean": round(float(left.mean()), 4),
        "within_episode_lock": round(within_lock, 4),
        "across_episode_balance": round(across_balance, 4),
        "symmetry_breaking": round(symmetry_breaking, 4),
    }


def main() -> None:
    levels = ["smooth", "threshold", "feedback", "anti_shortcut"]
    rows = {}
    for i, level in enumerate(levels):
        rows[level] = run_level(level, SEED + i * 101)
        h = rows[level]["adj"].get("hinge", {})
        print(f"{level}: drop={rows[level]['adj']['drop']} "
              f"B5={rows[level]['adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')} "
              f"sym={rows[level]['symmetry_breaking']}",
              flush=True)

    dbics = [rows[level]["adj"].get("hinge", {}).get("delta_bic", -999)
             for level in levels]
    post_slopes = [abs(rows[level]["adj"].get("hinge", {}).get("slope_after", 0.0))
                   for level in levels]
    sym = [rows[level]["symmetry_breaking"] for level in levels]
    outcomes = {
        "EEC1_all_collapse": all(rows[level]["adj"]["drop"] > 0.05 for level in levels),
        "EEC2_smooth_weaker": bool(
            not rows["smooth"]["adj"]["b5_onset"]
            or dbics[0] < max(dbics[1:])
        ),
        "EEC3_profile_strengthens": bool(
            max(dbics[1:]) > dbics[0]
            and max(post_slopes[1:]) > post_slopes[0]
        ),
        "EEC4_anti_shortcut_symmetry": bool(sym[-1] == max(sym)),
        "delta_bics": dbics,
        "post_slope_abs": post_slopes,
        "symmetry_breaking": sym,
    }
    report = {
        "status": "EEC-LADDER synthetic mechanism ladder; preregistered",
        "config": {"N": N, "n_ep": N_EP, "T": T, "k": K, "grid": GRID},
        "levels": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "eec_ladder.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
