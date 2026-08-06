"""Confirmatory commitment-window intervention (E3C).

Registered in V2_ALIGNMENT_PREREGISTRATION.md, section E3C, frozen
before any run. Five conditions x five seeds, equal 360k-step cut
budget. Training mechanics imported unchanged from the pilot script.
The random-window start per seed comes from the RNG declared in the
preregistration. Each run writes a NEW output file; nothing stored is
touched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

np.Inf = np.inf
import torch

from overcooked_commitment_intervention import LAYOUTS, train_with_cut
from overcooked_genesis_curve import evaluate_checkpoint
from overcooked_joint_collapse_curve import (ladder_from_tables,
                                             rollout_joint_counts)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
CONDITIONS = ("none", "early", "commit", "late", "random")
CUT_LEN = 360_000
SEEDS = (93201, 93202, 93203, 93204, 93205)


def window_for(condition: str, seed: int) -> Optional[Tuple[int, int]]:
    if condition == "none":
        return None
    if condition == "early":
        return (80_000, 440_000)
    if condition == "commit":
        return (640_000, 1_000_000)
    if condition == "late":
        return (1_500_000, 1_860_000)
    start = random.Random(seed * 7 + 13).randrange(0, 1_640_000, 20_000)
    return (start, start + CUT_LEN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-steps", type=int, default=2_000_000)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("OC_THREADS", "2")))
    window = window_for(args.condition, args.seed)
    t0 = time.time()
    print(f"E3C condition={args.condition} seed={args.seed} "
          f"window={window}", flush=True)
    net = train_with_cut(LAYOUTS, args.seed, args.train_steps, window)
    train_min = round((time.time() - t0) / 60, 2)
    ckpt = OUTPUTS / (f"overcooked_e3c_{args.condition}"
                      f"_s{args.seed}.pt")
    torch.save(net.state_dict(), ckpt)
    print(f"trained in {train_min} min; evaluating", flush=True)

    eval_seed = (97_000 + 10 * CONDITIONS.index(args.condition)
                 + SEEDS.index(args.seed))
    cert = evaluate_checkpoint(ckpt, eval_seed)
    tables, _basins = rollout_joint_counts(net, eval_seed + 10_000)
    pooled = {layout: sum(t) for layout, t in tables.items()}
    ladder = ladder_from_tables(pooled)

    report = {
        "status": ("E3C confirmatory intervention run; registered in "
                   "V2_ALIGNMENT_PREREGISTRATION.md before launch"),
        "condition": args.condition,
        "window": window,
        "seed": args.seed,
        "train_minutes": train_min,
        "certificate_2M": cert,
        "joint_ladder_2M": {k: ladder[k] for k in
                            ("C_individual", "C_env", "C_relational",
                             "C_total", "collapse_norm")},
    }
    out = OUTPUTS / f"overcooked_e3c_{args.condition}_s{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "condition": args.condition, "seed": args.seed,
        "G": cert["G_js_bits"], "M": cert["M_score_gain"],
        "score": cert["real_score"],
        "C_rel": ladder["C_relational"],
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
