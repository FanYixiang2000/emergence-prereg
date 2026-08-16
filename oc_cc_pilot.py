"""OC-CC competence pilot: one seed on the official counter_circuit
layout to set the training budget before freezing the OC-CC protocol.

Same role as the MPE competence precondition: this pilot decides
whether the confirmatory experiment is run at all and at what budget;
its seed is excluded from the confirmatory seed set and the pilot is
reported as a pilot.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

import overcooked_criterion as oc
from overcooked_genesis_curve import train_with_checkpoints
from overcooked_ring_convention import eval_checkpoint

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUT = "counter_circuit"
SEED = 97_001
TOTAL_STEPS = 4_000_000
CKPT_EVERY = 100_000
CHECKPOINTS = tuple(range(CKPT_EVERY, TOTAL_STEPS + 1, CKPT_EVERY))

# counter_circuit central counter block spans x=2..6 at y=2
import overcooked_ring_convention as orc


def main() -> None:
    torch.set_num_threads(8)
    orc.RING_CENTER = (4.0, 2.0)
    saved = train_with_checkpoints((LAYOUT, LAYOUT), SEED, TOTAL_STEPS,
                                   CHECKPOINTS, "ccpilot")
    rows = {}
    for ck in sorted(saved):
        r = eval_checkpoint(saved[ck], LAYOUT, SEED)
        rows[ck] = r
        print(f"ck={ck}: soups={r['mean_soups']} "
              f"open={r['circulation_openness']} p_ccw={r['p_ccw']} "
              f"n_com={r['n_committed_episodes']}", flush=True)
    out = OUTPUTS / "oc_cc_pilot.json"
    out.write_text(json.dumps({
        "status": ("OC-CC competence pilot, one seed, decides budget for "
                   "the confirmatory protocol; pilot seed excluded from "
                   "confirmatory seeds"),
        "config": {"layout": LAYOUT, "seed": SEED,
                   "total_steps": TOTAL_STEPS, "ckpt_every": CKPT_EVERY,
                   "ring_center": [4.0, 2.0]},
        "curves": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
