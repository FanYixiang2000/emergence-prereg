"""Aggregate the per-seed deep MARL probe runs into the registered verdicts.

Reads outputs/deep_marl_collapse_seed{11,22,33}.json (each contains one
trained seed plus its own controls; controls are taken from the seed11
file, they are policy-independent) and evaluates the registered D1-D4
predictions from DEEP_MARL_PREREGISTRATION.md over the pooled data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SEEDS = (11, 22, 33)


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def main() -> None:
    data = {s: json.loads(
        (OUTPUTS / f"deep_marl_collapse_mappo_seed{s}.json").read_text())
        for s in SEEDS}
    trained = [data[s]["conditions"][f"trained_seed{s}"] for s in SEEDS]
    controls = data[SEEDS[0]]["conditions"]
    untrained, greedy, noise = (controls["untrained"], controls["greedy_nearest"],
                                controls["noise"])

    d1 = {
        "trained_early_potential": [t["early_potential_bits"] for t in trained],
        "untrained_early_potential": untrained["early_potential_bits"],
        "noise_early_potential": noise["early_potential_bits"],
        "pass": all(t["early_potential_bits"] >= 1.0 for t in trained),
    }
    d2 = {
        "trained_win_shift": [t["p_win_end"] - t["p_win_start"] for t in trained],
        "trained_bijection_rate": [t["final_bijection_rate"] for t in trained],
        "untrained_bijection_rate": untrained["final_bijection_rate"],
        "noise_bijection_rate": noise["final_bijection_rate"],
        "pass": (all(t["p_win_end"] - t["p_win_start"] > 0 for t in trained)
                 and all(t["final_bijection_rate"] >= 0.5 for t in trained)
                 and untrained["final_bijection_rate"] < 0.35
                 and noise["final_bijection_rate"] < 0.35),
    }
    gaps: List[float] = []
    wins = losses = 0
    for t in trained:
        for e in t["episodes"]:
            g = e["p_win_do_commit"] - e["p_win_do_block"]
            gaps.append(g)
            wins += g > 0
            losses += g < 0
    d3 = {
        "pooled_do_gap_median": float(np.median(gaps)),
        "pooled_do_gap_mean": float(np.mean(gaps)),
        "per_seed_do_gap_median": [t["do_gap_median"] for t in trained],
        "pooled_sign_wins": wins,
        "pooled_sign_losses": losses,
        "pooled_sign_p": sign_test_p(wins, losses),
        "pass": float(np.median(gaps)) > 0 and sign_test_p(wins, losses) < 0.05,
    }
    d4 = {
        "greedy_bijection_rate": greedy["final_bijection_rate"],
        "greedy_early_potential": greedy["early_potential_bits"],
        "trained_min_bijection": min(t["final_bijection_rate"] for t in trained),
        "trained_min_potential": min(t["early_potential_bits"] for t in trained),
        "pass": (greedy["final_bijection_rate"]
                 < min(t["final_bijection_rate"] for t in trained)
                 and greedy["early_potential_bits"]
                 < min(t["early_potential_bits"] for t in trained)),
    }
    verdicts = {"D1_potential": d1, "D2_useful_collapse": d2,
                "D3_counterfactual": d3, "D4_greedy_contrast": d4,
                "all_pass": all(d["pass"] for d in (d1, d2, d3, d4))}
    (OUTPUTS / "deep_marl_collapse_aggregate.json").write_text(
        json.dumps(verdicts, indent=2))
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
