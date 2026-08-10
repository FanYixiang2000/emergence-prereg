"""Adversarial observer (basin-partition) audit on the Contextual LBF domain.

Reviewer question addressed: could an author always manufacture Potential and
Specificity by choosing a favourable basin partition? Declared design (before
running, in this docstring):

1. Bijective relabelling check: permuting the four basin labels must leave
   every partition-dependent quantity exactly unchanged (code correctness).
2. Random-observer null: micro-outcomes (basin x episode-length bucket) are
   randomly grouped into four pseudo-basins, 1000 draws with a fixed seed.
   For each draw, Potential and Specificity are recomputed from the SAME
   recorded episodes; Selectivity, Usefulness, Endogeneity and Acquisition
   are partition-independent by construction and stay at their frozen
   confirmation values. Reported: pass rates of the partition-dependent
   components and of the full conjunction under random observers, and the
   declared partition's percentile in the random specificity distribution.

Whatever the direction of the outcome, it is reported: a low random full-pass
rate shows nonsense observers cannot manufacture emergence; a high specificity
pass rate under random observers would instead show the do-contrast is so
strong that no partition choice hides it (also a defensible finding). Both
components are audited separately so the answer is attributable.

Uses the saved confirmation policies; evaluation identical to the frozen
protocol (80 episodes per context and condition, confirmation seed block).
"""

from __future__ import annotations

import json
import math
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import contextual_lbf_transfer as clbf
import lbf_collapse_probe as base

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

SEEDS = [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110]
N_EVAL = 80                       # frozen confirmation setting
EVAL_OFFSET = 8_000_000           # frozen confirmation seed block
N_RANDOM_OBSERVERS = 1000
RNG_SEED = 20260716
STEP_BUCKETS = ((1, 5), (6, 9), (10, 12), (13, 15))


def load_net(seed: int) -> base.PolicyNet:
    net = base.PolicyNet()
    net.load_state_dict(torch.load(
        OUTPUTS / f"contextual_lbf_net_seed{seed}.pt", weights_only=True))
    net.eval()
    return net


def micro_cell(row: Dict[str, Any]) -> Tuple[str, int]:
    steps = int(row["steps"])
    for b, (lo, hi) in enumerate(STEP_BUCKETS):
        if lo <= steps <= hi:
            return (row["basin"], b)
    return (row["basin"], len(STEP_BUCKETS) - 1)


def collect_rows(seed: int) -> List[Dict[str, Any]]:
    controller = clbf.TeamController("policy", load_net(seed))
    measured = clbf.evaluate(controller, N_EVAL,
                             EVAL_OFFSET + seed * 100_000)
    return measured["rows"]


def dist_over(groups: Dict[Tuple[str, int], int], rows, n_groups=4
              ) -> List[float]:
    counts = [0] * n_groups
    for row in rows:
        counts[groups[micro_cell(row)]] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def entropy(p: List[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def js(p: List[float], q: List[float]) -> float:
    out = 0.0
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def main() -> None:
    torch.set_num_threads(16)
    all_rows: Dict[int, Dict[str, list]] = {}
    cells: set = set()
    for seed in SEEDS:
        rows = collect_rows(seed)
        by_mode = {
            "natural": [r for r in rows if r["mode"] == "natural"],
            "do_trigger": [r for r in rows if r["mode"] == "do_trigger"],
            "do_non_trigger": [r for r in rows
                               if r["mode"] == "do_non_trigger"],
        }
        all_rows[seed] = by_mode
        for r in rows:
            cells.add(micro_cell(r))
        print(f"seed {seed}: {len(rows)} episodes recorded", flush=True)
    cells = sorted(cells)
    print(f"{len(cells)} occupied micro-cells", flush=True)

    # declared partition = group by basin, ignoring the step dimension
    basin_names = sorted({c[0] for c in cells})
    declared = {c: basin_names.index(c[0]) for c in cells}

    def components(groups) -> Dict[int, Dict[str, float]]:
        out = {}
        for seed in SEEDS:
            bm = all_rows[seed]
            p_nat = dist_over(groups, bm["natural"])
            p_do = dist_over(groups, bm["do_trigger"])
            p_non = dist_over(groups, bm["do_non_trigger"])
            out[seed] = {"potential": entropy(p_nat),
                         "specificity": js(p_do, p_non)}
        return out

    declared_vals = components(declared)

    # 1. bijective relabelling: all 4! permutations must be exact
    max_dev = 0.0
    for perm in permutations(range(4)):
        permuted = {c: perm[g] for c, g in declared.items()}
        vals = components(permuted)
        for seed in SEEDS:
            for k in ("potential", "specificity"):
                max_dev = max(max_dev, abs(vals[seed][k]
                                           - declared_vals[seed][k]))
    print(f"bijection max deviation: {max_dev:.2e}", flush=True)

    # 2. random observers
    conf = json.loads(
        (OUTPUTS / "contextual_lbf_confirmation.json").read_text())
    partition_free_pass = {
        seed: all(
            conf["seeds"][str(seed)]["systems"]["learned"]["verdict"]
            ["passes"][k]
            for k in ("conditional_selectivity", "usefulness",
                      "endogeneity", "acquisition"))
        for seed in SEEDS
    }

    rng = np.random.default_rng(RNG_SEED)
    pot_pass = np.zeros(N_RANDOM_OBSERVERS)
    spec_pass = np.zeros(N_RANDOM_OBSERVERS)
    full_pass_frac = np.zeros(N_RANDOM_OBSERVERS)
    spec_means = np.zeros(N_RANDOM_OBSERVERS)
    for i in range(N_RANDOM_OBSERVERS):
        while True:
            assignment = rng.integers(0, 4, size=len(cells))
            if len(set(assignment.tolist())) == 4:
                break
        groups = {c: int(g) for c, g in zip(cells, assignment)}
        vals = components(groups)
        pots = [vals[s]["potential"] >= clbf.THRESHOLDS["potential_bits"]
                for s in SEEDS]
        specs = [vals[s]["specificity"]
                 >= clbf.THRESHOLDS["specificity_js_bits"] for s in SEEDS]
        fulls = [p and q and partition_free_pass[s]
                 for p, q, s in zip(pots, specs, SEEDS)]
        pot_pass[i] = np.mean(pots)
        spec_pass[i] = np.mean(specs)
        full_pass_frac[i] = np.mean(fulls)
        spec_means[i] = np.mean([vals[s]["specificity"] for s in SEEDS])

    declared_spec_mean = float(np.mean(
        [declared_vals[s]["specificity"] for s in SEEDS]))
    percentile = float(np.mean(spec_means < declared_spec_mean))

    summary = {
        "status": "adversarial observer audit (recorded episodes; "
                  "partitions post-hoc)",
        "n_random_observers": N_RANDOM_OBSERVERS,
        "n_micro_cells": len(cells),
        "bijection_max_deviation": max_dev,
        "declared_partition": {
            "mean_potential": float(np.mean(
                [declared_vals[s]["potential"] for s in SEEDS])),
            "mean_specificity": declared_spec_mean,
            "specificity_percentile_vs_random": percentile,
        },
        "random_observers": {
            "mean_potential_pass_rate": float(pot_pass.mean()),
            "mean_specificity_pass_rate": float(spec_pass.mean()),
            "mean_full_pass_rate": float(full_pass_frac.mean()),
            "p_full_pass_ge_9_of_10": float(np.mean(full_pass_frac >= 0.9)),
            "mean_specificity_bits": float(spec_means.mean()),
        },
        "reading_rule": (
            "declared before running: low random full-pass -> nonsense "
            "observers cannot manufacture emergence; high random "
            "specificity-pass -> the do-contrast survives any partition "
            "(reported either way)"
        ),
    }
    out = OUTPUTS / "adversarial_observer_audit.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "status"},
                     indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
