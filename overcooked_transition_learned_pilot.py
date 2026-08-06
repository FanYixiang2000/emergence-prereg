"""Small learned-policy pilot for the Overcooked transition certificate.

This is deliberately a PILOT, not a confirmatory run. It trains one or a few
small self-play PPO policies with the same mechanics as the round-1
Overcooked code, saves checkpoints, and records the exact command needed to
evaluate each checkpoint with `overcooked_transition_certificate.py`.

Purpose:
  move from scaffold-only real-vs-ghost cuts to a first learned local-feedback
  smoke test, without relabelling it as the NMI flagship experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

from overcooked_confirmation import train_mixed

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUTS = ("cramped_room", "asymmetric_advantages")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[92001])
    ap.add_argument("--train-steps", type=int, default=40_000)
    ap.add_argument("--layouts", nargs=2, default=list(LAYOUTS))
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    torch.set_num_threads(4)
    report = {
        "status": ("learned transition pilot; trains checkpoint(s) for "
                   "state-level real-vs-ghost certificate; not a "
                   "confirmatory flagship result"),
        "layouts": args.layouts,
        "train_steps": args.train_steps,
        "seeds": {},
    }
    for seed in args.seeds:
        t0 = time.time()
        net = train_mixed(tuple(args.layouts), seed, args.train_steps)
        ckpt = OUTPUTS / f"overcooked_transition_{args.tag}_s{seed}.pt"
        torch.save(net.state_dict(), ckpt)
        eval_cmd = (
            "python overcooked_transition_certificate.py "
            f"--policy checkpoint --checkpoint {ckpt} "
            f"--seed {seed} --tag {args.tag}_s{seed}"
        )
        report["seeds"][str(seed)] = {
            "checkpoint": str(ckpt),
            "train_minutes": round((time.time() - t0) / 60, 3),
            "eval_command": eval_cmd,
        }
        print(f"seed {seed}: saved {ckpt}", flush=True)
        print(f"  eval: {eval_cmd}", flush=True)

    out = OUTPUTS / f"overcooked_transition_learned_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
