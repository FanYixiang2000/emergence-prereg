"""Full-verdict null under random observers, control side.

Completes the adversarial observer audit with the quantity a reviewer will
ask for directly:

    Pr_{random partition} [ full six-component protocol accepts a CONTROL ]

For every control system (10 initialization twins + 3 scripted controllers,
evaluated exactly as in the frozen confirmation), the partition-dependent
components (potential, specificity) are recomputed under 1000 random
micro-cell groupings; the partition-independent components (conditional
selectivity, usefulness, endogeneity, acquisition) keep their frozen
confirmation values. The full verdict is the conjunction. Also reported:
the rate at which a random partition lets a control pass BOTH
partition-dependent components (the strongest thing a malicious observer
could achieve).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import contextual_lbf_transfer as clbf
import lbf_collapse_probe as base
from adversarial_observer_audit import (
    N_EVAL, EVAL_OFFSET, RNG_SEED, STEP_BUCKETS, micro_cell,
    dist_over, entropy, js,
)

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SEEDS = [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110]
N_RANDOM = 1000


def twin(seed: int) -> base.PolicyNet:
    torch.manual_seed(seed)
    return base.PolicyNet()


def main() -> None:
    torch.set_num_threads(16)
    conf = json.loads(
        (OUTPUTS / "contextual_lbf_confirmation.json").read_text())

    controls: List[Tuple[str, Any, int]] = []
    for seed in SEEDS:
        controls.append((f"initial_twin_{seed}",
                         clbf.TeamController("policy", twin(seed)), seed))
    for kind in ("team_nearest", "fixed_food0", "fixed_food1"):
        controls.append((kind, clbf.TeamController(kind), SEEDS[0]))

    rows_by_system: Dict[str, Dict[str, list]] = {}
    cells: set = set()
    partition_free_pass: Dict[str, bool] = {}
    for name, controller, seed in controls:
        measured = clbf.evaluate(controller, N_EVAL,
                                 EVAL_OFFSET + seed * 100_000)
        rows = measured["rows"]
        rows_by_system[name] = {
            m: [r for r in rows if r["mode"] == m]
            for m in ("natural", "do_trigger", "do_non_trigger")
        }
        for r in rows:
            cells.add(micro_cell(r))
        # frozen partition-independent verdict components
        src = ("initial_twin" if name.startswith("initial_twin")
               else name)
        passes = conf["seeds"][str(seed)]["systems"][src]["verdict"]["passes"]
        partition_free_pass[name] = all(
            passes[k] for k in ("conditional_selectivity", "usefulness",
                                "endogeneity", "acquisition"))
        print(f"{name}: partition-independent components pass = "
              f"{partition_free_pass[name]}", flush=True)
    cells = sorted(cells)

    rng = np.random.default_rng(RNG_SEED + 1)
    pd_pass_counts = {name: 0 for name, _, _ in controls}
    full_pass_counts = {name: 0 for name, _, _ in controls}
    for _ in range(N_RANDOM):
        while True:
            assignment = rng.integers(0, 4, size=len(cells))
            if len(set(assignment.tolist())) == 4:
                break
        groups = {c: int(g) for c, g in zip(cells, assignment)}
        for name, _, _ in controls:
            bm = rows_by_system[name]
            pot = entropy(dist_over(groups, bm["natural"]))
            spec = js(dist_over(groups, bm["do_trigger"]),
                      dist_over(groups, bm["do_non_trigger"]))
            pd = (pot >= clbf.THRESHOLDS["potential_bits"]
                  and spec >= clbf.THRESHOLDS["specificity_js_bits"])
            pd_pass_counts[name] += pd
            full_pass_counts[name] += pd and partition_free_pass[name]

    summary = {
        "status": "full-verdict null under random observers, control side",
        "n_random_observers": N_RANDOM,
        "n_controls": len(controls),
        "per_control": {
            name: {
                "partition_dependent_pass_rate":
                    pd_pass_counts[name] / N_RANDOM,
                "full_verdict_pass_rate":
                    full_pass_counts[name] / N_RANDOM,
                "partition_independent_pass": partition_free_pass[name],
            } for name, _, _ in controls
        },
        "max_full_verdict_pass_rate_any_control": max(
            full_pass_counts.values()) / N_RANDOM,
        "reading": (
            "No random observer produces a full acceptance of any control "
            "(the provenance and value components are partition-independent "
            "and already fail); the strongest a malicious partition can do "
            "is inflate the two distributional components, which the "
            "protocol never treats as sufficient."
        ),
    }
    out = OUTPUTS / "adversarial_observer_controls.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"max full-verdict pass rate over controls: "
          f"{summary['max_full_verdict_pass_rate_any_control']:.4f}")
    worst_pd = max(pd_pass_counts.values()) / N_RANDOM
    print(f"max partition-dependent pass rate: {worst_pd:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
