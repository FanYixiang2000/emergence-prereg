"""RE-1: V3 re-adjudication of the ordinary learner (ADJUDICATION).

Registered in V2_ALIGNMENT_PREREGISTRATION.md (RE battery, frozen
before this run). Reuses the frozen v1 trainer (train_run from
ordinary_learner_control: y=(a+b)//40, GrokNet, AdamW) with the
dense eval grid (every 5 epochs), and asks the V3 question: does the
predictive-openness curve (mean test entropy, the individual channel
of a single learner) contain a persistent hinge breakpoint?

Verdict mapping (frozen, no directional bet):
- persistent closing breakpoint -> individual-channel emergence
  event under V3 typology (the v1 "false positive" dissolves);
- none -> smooth convergence, excluded by B5; the v1 burst-gate pass
  is recorded as a detector artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ordinary_learner_control import SEEDS, train_run
from re3_stored_series_breakpoint import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    torch.set_num_threads(8)
    per_seed = {}
    for seed in SEEDS:
        rows = train_run(seed, eval_every=5)
        steps = [r["epoch"] for r in rows]
        y = [r["test_entropy_bits"] for r in rows]
        adj = adjudicate(steps, y, [2])
        per_seed[str(seed)] = {
            "n_points": len(steps),
            "entropy_first_last": [round(y[0], 4), round(y[-1], 4)],
            "adjudication": adj,
        }
        print(f"seed {seed}: closing={adj['full']['closing_verdict']} "
              f"persistent={adj['persistent_closing']} "
              f"hinge={adj['full']['hinge_step']} "
              f"dBIC={adj['full']['delta_bic']}", flush=True)

    n_persistent = sum(per_seed[s]["adjudication"]["persistent_closing"]
                       for s in per_seed)
    verdict = ("individual_channel_emergence_event"
               if n_persistent == len(per_seed) else
               ("mixed" if n_persistent > 0 else
                "smooth_convergence_B5_excluded"))
    report = {
        "status": ("RE-1 ordinary-learner breakpoint adjudication; "
                   "registered RE battery, frozen verdict mapping, "
                   "no directional bet"),
        "per_seed": per_seed,
        "n_persistent": int(n_persistent),
        "verdict": verdict,
    }
    out = OUTPUTS / "re1_ordinary_learner_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n_persistent": int(n_persistent),
                      "verdict": verdict}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
