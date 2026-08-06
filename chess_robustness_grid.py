"""Robustness grid for the chess within-state collapse probe.

Reviewer-facing question (same role as estimator_robustness_check.py for
the gridworld benchmarks): do the chess conclusions depend on the
pilot-tuned estimator, the basin thresholds, or the engine version?

Design: the SAME 240 sacrifice + 120 quiet positions as the registered
main run (selection is deterministic and uses the default engine at the
default depth, so the position set is identical across cells). Each grid
cell re-measures every position under one perturbation:

- observer temperature x playout depth around the frozen point
  (200/300/450 cp x depth 3/4/6, multipv fixed at 6);
- basin thresholds (win/adv cp): 300/100 (registered), 400/150, 500/200;
- engine family: Stockfish 14.1 NNUE (registered) vs Stockfish 11
  (classical handcrafted evaluation, pre-NNUE era) for playouts,
  classification, and counterfactual move generation.

Success criterion (stated before running, same spirit as the estimator
robustness check): in every cell, (a) the C2 sign-test conclusion
(key > greedy, key > random) holds at p < 1e-3, (b) the C3 do-contrast
gap P(win|key) - P(win|deep_alt) stays >= 0.15 with sign-test p < 1e-3,
(c) the C1 cost medians keep their order (key < 0 <= greedy shift), and
(d) median sacrifice potential stays >= 1.0 bits except possibly at the
strongest observer (depth 6), where the registered pilot already showed
potential compression -- reported either way.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, List

import chess_collapse_probe as probe

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SF11_PATH = HERE / "external_chess" / "stockfish_11_x" / "usr" / "games" / "stockfish"


def build_cells() -> List[Dict]:
    cells: List[Dict] = []
    for temp in (200.0, 300.0, 450.0):
        for depth in (3, 4, 6):
            name = f"temp{int(temp)}_depth{depth}"
            if temp == 300.0 and depth == 4:
                name += "_registered"
            cells.append({"name": name,
                          "overrides": {"TEMPERATURE_CP": temp,
                                        "PLAYOUT_DEPTH": depth}})
    for win_cp, adv_cp in ((400, 150), (500, 200)):
        cells.append({"name": f"basin{win_cp}_{adv_cp}",
                      "overrides": {"WIN_CP": win_cp, "ADV_CP": adv_cp}})
    cells.append({"name": "engine_sf11_classical",
                  "overrides": {"ENGINE_PATH": str(SF11_PATH)}})
    return cells


def cell_checks(summary: Dict) -> Dict[str, bool]:
    c1 = summary["C1_local_cost"]
    c2 = summary["C2_useful_collapse"]
    c3 = summary["C3_selectivity"]
    c4 = summary["C4_potential"]
    return {
        "c1_cost_order": (c1["median_local_cost_key"] < 0
                          and c1["median_local_cost_greedy"]
                          > c1["median_local_cost_key"]),
        "c2_sign_tests": (c2["sign_vs_greedy"]["p_one_sided"] < 1e-3
                          and c2["sign_vs_random"]["p_one_sided"] < 1e-3),
        "c3_gap_and_sign": (c3["gap"] >= 0.15
                            and c3["sign_vs_deep_alt"]["p_one_sided"] < 1e-3),
        "c4_potential": c4["median_potential_bits"] >= 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--n_sacrifice", type=int, default=240)
    parser.add_argument("--n_quiet", type=int, default=120)
    args = parser.parse_args()

    print("selecting positions once (default engine/params) ...", flush=True)
    sacrifice, quiet = probe.select_positions(args.n_sacrifice, args.n_quiet)
    probe.close_engine()
    tasks = sacrifice + quiet
    print(f"selected {len(sacrifice)} sacrifice + {len(quiet)} quiet", flush=True)

    ctx = mp.get_context("spawn")
    grid: Dict[str, Dict] = {}
    for cell in build_cells():
        t0 = time.time()
        with ctx.Pool(processes=args.workers, initializer=probe.worker_init,
                      initargs=(cell["overrides"],)) as pool:
            records = list(pool.imap_unordered(probe.measure_position, tasks))
        summary = probe.analyze(records)
        checks = cell_checks(summary)
        grid[cell["name"]] = {
            "overrides": {k: str(v) for k, v in cell["overrides"].items()},
            "summary": summary,
            "checks": checks,
            "elapsed_s": round(time.time() - t0, 1),
        }
        flags = " ".join(f"{k}={'PASS' if ok else 'FAIL'}"
                         for k, ok in checks.items())
        c3 = summary["C3_selectivity"]
        print(f"{cell['name']}: gap {c3['gap']:.3f} "
              f"potential {summary['C4_potential']['median_potential_bits']:.2f} "
              f"| {flags} [{grid[cell['name']]['elapsed_s']}s]", flush=True)

    n_cells = len(grid)
    core_ok = sum(1 for g in grid.values()
                  if g["checks"]["c1_cost_order"] and g["checks"]["c2_sign_tests"]
                  and g["checks"]["c3_gap_and_sign"])
    out = {
        "n_cells": n_cells,
        "core_conclusions_hold": f"{core_ok}/{n_cells}",
        "grid": grid,
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "chess_robustness_grid.json").write_text(json.dumps(out, indent=2))
    print(f"core conclusions (C1+C2+C3) hold in {core_ok}/{n_cells} cells")
    print(f"Wrote {OUTPUTS / 'chess_robustness_grid.json'}")


if __name__ == "__main__":
    main()
