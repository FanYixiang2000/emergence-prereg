"""OC-CC competence pilot, round 2: budget/recipe arms.

Round 1 (outputs/oc_cc_pilot.json) failed the competence check: with
the recipe's 0.6-horizon shaping anneal, the counter_circuit policy
never delivers a soup in 4M steps, and the early circulation habit
dissolves once shaping expires (~3.0M). Diagnosis: on this layout the
shaped signal ends before the sparse chain is discovered. Round 2
tests two disclosed recipe arms; whichever reaches competence sets
the confirmatory protocol, which is frozen before the confirmatory
seeds run. Pilot seeds are excluded from the confirmatory set.

Arm A: 8M steps, shaping annealed over 0.9 of the horizon.
Arm B: 6M steps, shaping never annealed (anneal_frac = 2.0).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

np.Inf = np.inf

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUT = "counter_circuit"
CKPT_EVERY = 200_000
ARMS = {
    "A_anneal09_8M": {"seed": 97_002, "total_steps": 8_000_000,
                      "anneal_frac": 0.9},
    "B_noanneal_6M": {"seed": 97_003, "total_steps": 6_000_000,
                      "anneal_frac": 2.0},
}


def run_arm(item):
    name, cfg = item
    import torch

    torch.set_num_threads(8)
    import overcooked_ring_convention as orc
    from overcooked_genesis_curve import train_with_checkpoints

    orc.RING_CENTER = (4.0, 2.0)
    ckpts = tuple(range(CKPT_EVERY, cfg["total_steps"] + 1, CKPT_EVERY))
    saved = train_with_checkpoints((LAYOUT, LAYOUT), cfg["seed"],
                                   cfg["total_steps"], ckpts,
                                   f"ccp2{cfg['seed']}",
                                   anneal_frac=cfg["anneal_frac"])
    rows = {}
    for ck in sorted(saved):
        r = orc.eval_checkpoint(saved[ck], LAYOUT, cfg["seed"])
        rows[ck] = r
        print(f"{name} ck={ck}: soups={r['mean_soups']} "
              f"open={r['circulation_openness']} p_ccw={r['p_ccw']} "
              f"n_com={r['n_committed_episodes']}", flush=True)
    return name, {"config": cfg, "curves": rows}


def main() -> None:
    report = {}
    with ProcessPoolExecutor(max_workers=len(ARMS)) as ex:
        for name, res in ex.map(run_arm, ARMS.items()):
            report[name] = res
    out = OUTPUTS / "oc_cc_pilot2.json"
    out.write_text(json.dumps({
        "status": ("OC-CC competence pilot round 2; two recipe arms; "
                   "decides the confirmatory recipe, frozen before the "
                   "confirmatory run; pilot seeds excluded"),
        "arms": report}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
