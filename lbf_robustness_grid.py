"""LBF estimator-robustness grid: is the cross-task result temperature-tuned?

The main LBF run froze PROBE_TEMPERATURE at 6.0 after a documented pilot
sweep (lbf_collapse_probe.py docstring). A reviewer will ask whether the
4/4 registered outcome depends on that choice. This grid re-probes the
SAVED main-run networks (outputs/lbf_net_seed{11,22,33}.pt -- no
retraining, no new behaving policies) across probe temperatures
T in {2.0, 3.0, 4.5, 8.0} (main value 6.0 is the already-recorded run;
its cells are copied from lbf_collapse_main.json for reference).

SUCCESS CRITERIA -- FROZEN BEFORE RUNNING (same practice as
chess_robustness_grid.py):

G1 (counterfactual contrast is temperature-robust): in EVERY grid cell
    (temperature x pooled-over-seeds), the pooled do-gap
    median(P(win|do_commit) - P(win|do_block)) > 0 and the pooled sign
    test p < 0.05.
G2 (greedy double-dissociation direction is temperature-robust): in
    every cell, min over seeds of trained early potential > greedy
    early potential, and min over seeds of trained win rate > greedy
    win rate. Direction only -- NO absolute potential threshold: the
    pilot sweep already showed (and the chess grid C4 result predicts)
    that ABSOLUTE openness scales with observer temperature
    (T=2/3/4/6 -> 0.00/0.07/0.29/1.41 bits at the start state), so the
    registered L1 threshold (0.8 bits) is expected to fail at low T;
    that is the documented observer-scale dependence, not a failure of
    the mechanism claims, and it is reported as such either way.

Expected failure mode we commit to reporting: if G1 fails in any cell,
the LBF counterfactual result is estimator-fragile and the manuscript
must say so.

Greedy and noise controls involve no policy network, hence no
temperature; they are probed once and shared across cells.

Output: outputs/lbf_robustness_grid.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import lbf_collapse_probe as lbf

OUTPUTS = Path(__file__).resolve().parent / "outputs"

SEEDS = (11, 22, 33)
TEMPERATURES = (2.0, 3.0, 4.5, 8.0)
MAIN_TEMPERATURE = 6.0


def load_net(seed: int) -> lbf.PolicyNet:
    net = lbf.PolicyNet()
    net.load_state_dict(torch.load(OUTPUTS / f"lbf_net_seed{seed}.pt",
                                   weights_only=True))
    net.eval()
    return net


def pooled_stats(runs: List[Dict]) -> Dict:
    gaps: List[float] = []
    wins = losses = 0
    for r in runs:
        wins += r["do_sign_wins"]
        losses += r["do_sign_losses"]
        gaps.extend(e["p_win_do_commit"] - e["p_win_do_block"]
                    for e in r["episodes"] if "p_win_do_commit" in e)
    return {
        "pooled_do_gap_median": float(np.median(gaps)) if gaps else float("nan"),
        "pooled_do_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "pooled_sign_wins": wins,
        "pooled_sign_losses": losses,
        "pooled_sign_p": lbf.sign_test_p(wins, losses),
        "n_gaps": len(gaps),
    }


def main() -> None:
    torch.set_num_threads(16)
    nets = {seed: load_net(seed) for seed in SEEDS}
    main_run = json.loads((OUTPUTS / "lbf_collapse_main.json").read_text())

    print("probing temperature-independent controls once ...", flush=True)
    greedy = lbf.probe_condition("greedy_nearest",
                                 lbf.Controller("greedy_nearest"), 55)
    greedy_summary = {k: v for k, v in greedy.items() if k != "episodes"}
    print(json.dumps(greedy_summary, indent=2), flush=True)

    cells: Dict[str, Dict] = {}
    for temp in TEMPERATURES:
        lbf.PROBE_TEMPERATURE = temp
        runs: List[Dict] = []
        per_seed: Dict[str, Dict] = {}
        for seed in SEEDS:
            r = lbf.probe_condition(f"trained_seed{seed}_T{temp}",
                                    lbf.Controller("policy", nets[seed]), seed)
            runs.append(r)
            per_seed[f"seed{seed}"] = {
                "early_potential_bits": r["early_potential_bits"],
                "final_win_rate": r["final_win_rate"],
                "do_gap_median": r["do_gap_median"],
                "do_sign_wins": r["do_sign_wins"],
                "do_sign_losses": r["do_sign_losses"],
            }
            print(f"T={temp} seed {seed}: potential "
                  f"{r['early_potential_bits']:.3f} win {r['final_win_rate']:.2f} "
                  f"do_gap_med {r['do_gap_median']:+.3f} "
                  f"({r['do_sign_wins']}W/{r['do_sign_losses']}L)", flush=True)
        pooled = pooled_stats(runs)
        g1 = (pooled["pooled_do_gap_median"] > 0
              and pooled["pooled_sign_p"] < 0.05)
        min_pot = min(r["early_potential_bits"] for r in runs)
        min_win = min(r["final_win_rate"] for r in runs)
        g2 = (min_pot > greedy["early_potential_bits"]
              and min_win > greedy["final_win_rate"])
        cells[f"T{temp}"] = {
            "temperature": temp,
            "per_seed": per_seed,
            "pooled": pooled,
            "trained_min_potential": min_pot,
            "trained_min_win": min_win,
            "G1_do_contrast_pass": g1,
            "G2_greedy_direction_pass": g2,
            "L1_absolute_potential_0.8_pass": min_pot >= 0.8,
        }
        print(f"T={temp}: G1 {'PASS' if g1 else 'FAIL'} "
              f"(pooled med {pooled['pooled_do_gap_median']:+.3f}, "
              f"p {pooled['pooled_sign_p']:.2e}); "
              f"G2 {'PASS' if g2 else 'FAIL'}; "
              f"L1-absolute {'pass' if min_pot >= 0.8 else 'fails (expected at low T)'}",
              flush=True)

    # Reference cell: the recorded main run at T=6.0.
    trained_main = [main_run["conditions"][f"trained_seed{s}"] for s in SEEDS]
    pooled_main = pooled_stats(trained_main)
    cells[f"T{MAIN_TEMPERATURE}_main_run"] = {
        "temperature": MAIN_TEMPERATURE,
        "source": "lbf_collapse_main.json (recorded main run)",
        "pooled": pooled_main,
        "trained_min_potential": min(t["early_potential_bits"] for t in trained_main),
        "trained_min_win": min(t["final_win_rate"] for t in trained_main),
        "G1_do_contrast_pass": (pooled_main["pooled_do_gap_median"] > 0
                                and pooled_main["pooled_sign_p"] < 0.05),
        "G2_greedy_direction_pass": (
            min(t["early_potential_bits"] for t in trained_main)
            > greedy["early_potential_bits"]
            and min(t["final_win_rate"] for t in trained_main)
            > greedy["final_win_rate"]),
    }

    grid_cells = [c for k, c in cells.items() if not k.endswith("_main_run")]
    verdict = {
        "G1_all_cells": all(c["G1_do_contrast_pass"] for c in grid_cells),
        "G2_all_cells": all(c["G2_greedy_direction_pass"] for c in grid_cells),
        "L1_absolute_by_cell": {k: c["L1_absolute_potential_0.8_pass"]
                                for k, c in cells.items()
                                if not k.endswith("_main_run")},
    }
    out = {
        "note": ("Saved main-run nets re-probed across probe temperatures; "
                 "success criteria G1/G2 frozen in the module docstring "
                 "before running. Greedy control probed once "
                 "(temperature-independent)."),
        "greedy_control": greedy_summary,
        "cells": cells,
        "verdict": verdict,
    }
    (OUTPUTS / "lbf_robustness_grid.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"Wrote {OUTPUTS / 'lbf_robustness_grid.json'}")


if __name__ == "__main__":
    main()
