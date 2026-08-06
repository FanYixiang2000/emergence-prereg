"""OC-RING-EXT: five fresh ring seeds, byte-identical protocol.

Registered as an amendment in V2_ALIGNMENT_PREREGISTRATION.md before
running. Pools with the original three ring seeds for the OCE-1..4
outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ring_convention import run_system

OUTPUTS = Path(__file__).resolve().parent / "outputs"
EXT_SEEDS = (95_606, 95_707, 95_808, 95_909, 96_010)


def committed(p_ccw: float) -> bool:
    return abs(p_ccw - 0.5) >= 0.3


def main() -> None:
    torch.set_num_threads(4)
    rows = {}
    for seed in EXT_SEEDS:
        rows[str(seed)] = run_system("coordination_ring", seed,
                                     f"ringx{seed}")
        r = rows[str(seed)]
        print(f"EXT seed {seed}: circB5={r['circ_adj']['b5_onset']} "
              f"t*={r['circ_adj']['t_star']} dBIC={r['circ_adj']['delta_bic']} "
              f"p_ccw={r['final_p_ccw']} soups={r['final_soups']:.2f} "
              f"cross={r['capability_crossing']}", flush=True)

    orig = json.load(open(OUTPUTS / "overcooked_ring_convention.json"))
    pooled = dict(orig["systems"]["ring"])
    pooled.update(rows)
    n_committed = sum(committed(r["final_p_ccw"]) for r in pooled.values())
    dirs = {("ccw" if r["final_p_ccw"] > 0.5 else "cw")
            for r in pooled.values() if committed(r["final_p_ccw"])}
    onset = [r for r in pooled.values() if r["circ_adj"]["b5_onset"]]
    oce3 = (len(onset) >= 3 and all(
        r["circ_adj"]["t_star"] <= r["capability_crossing"]
        for r in onset if r["capability_crossing"] is not None))
    outcomes = {
        "OCE1_committed_ge_7of8": bool(n_committed >= 7),
        "OCE2_both_directions": bool(len(dirs) == 2),
        "OCE3_onset_ge_3of8_and_leads": bool(oce3),
        "n_pooled": len(pooled),
        "n_committed": n_committed,
        "n_onset": len(onset),
        "directions": sorted(dirs),
    }
    out = OUTPUTS / "oc_ring_ext.json"
    out.write_text(json.dumps({
        "status": ("OC-RING-EXT five fresh ring seeds, byte-identical "
                   "protocol; pooled outcomes with the original three; "
                   "registered before run"),
        "ext_seeds": rows,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
