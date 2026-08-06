"""BP-FRESH: dense-grid formation curve on a fresh seed, ladder-only.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (BP-FRESH execution
contract, frozen before any run). Trains with the dense 14-point
checkpoint grid, then evaluates ONLY the joint-action ladder per
checkpoint (no transition certificate). Output feeds the frozen
breakpoint detector.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

from overcooked_pilot import PolicyNet
from overcooked_genesis_curve import LAYOUTS, train_with_checkpoints
from overcooked_joint_collapse_curve import (ladder_from_tables,
                                             rollout_joint_counts)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
DENSE_GRID = (40_000, 80_000, 120_000, 160_000, 240_000, 320_000,
              480_000, 640_000, 820_000, 1_000_000, 1_250_000,
              1_500_000, 1_750_000, 2_000_000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OC_THREADS", "2")))

    tag = "bpfresh"
    t0 = time.time()
    train_with_checkpoints(LAYOUTS, args.seed, 2_000_000, DENSE_GRID,
                           tag)
    train_min = round((time.time() - t0) / 60, 1)
    print(f"training done in {train_min} min", flush=True)

    curve = {}
    for idx, ck in enumerate(DENSE_GRID):
        path = (OUTPUTS / f"overcooked_genesis_{tag}"
                f"_s{args.seed}_{ck}.pt")
        net = PolicyNet()
        net.load_state_dict(torch.load(path, weights_only=True,
                                       map_location="cpu"))
        net.eval()
        tables, _basins = rollout_joint_counts(
            net, args.seed + 400 + idx * 100)
        pooled = {layout: sum(t) for layout, t in tables.items()}
        curve[str(ck)] = ladder_from_tables(pooled)
        print(f"ckpt {ck}: Cenv={curve[str(ck)]['C_env']:.4f} "
              f"Cbar={curve[str(ck)]['collapse_norm']:.4f}", flush=True)

    report = {
        "status": ("BP-FRESH dense-grid ladder-only formation curve; "
                   "registered execution contract frozen before run"),
        "checkpoint_grid": list(DENSE_GRID),
        "curve": curve,
        "train_minutes": train_min,
    }
    out = OUTPUTS / f"overcooked_joint_collapse_bpfresh_s{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
