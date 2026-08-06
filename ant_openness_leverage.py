"""ANT-INT-B: the openness-leverage law.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Sweeps the intervention window over 21 start positions and
tests whether outcome-flip leverage tracks the remaining openness of
the joint possibility space (RE-2's median TRAIL openness curve).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ant_commitment_intervention import N_EP, run_episode

OUTPUTS = Path(__file__).resolve().parent / "outputs"
STARTS = tuple(range(0, 201, 10))


def main() -> None:
    re2 = json.loads((OUTPUTS / "re2_ant_joint_breakpoint.json")
                     .read_text(encoding="utf-8"))
    grid = re2["grid_trips"]
    med = re2["median_openness_trail"]
    openness_at = dict(zip(grid, med))

    controls = [run_episode(700_000 + ep, None) for ep in range(N_EP)]
    committing = [i for i, r in enumerate(controls)
                  if r["route"] in ("A", "B")]

    rows = []
    for s in STARTS:
        flips = 0
        for i in committing:
            r = run_episode(700_000 + i, s)
            if r["route"] != controls[i]["route"]:
                flips += 1
        mid = s + 15
        gpt = min(grid, key=lambda g: abs(g - mid))
        rows.append({"start": s, "midpoint": mid,
                     "openness_mid": openness_at[gpt],
                     "flips": flips,
                     "flip_rate": round(flips / len(committing), 4)})
        print(f"start {s:3d}: openness={openness_at[gpt]:.3f} "
              f"flip_rate={flips / len(committing):.3f}", flush=True)

    fr = [r["flip_rate"] for r in rows]
    op = [r["openness_mid"] for r in rows]
    rho, pval = spearmanr(fr, op)
    aib1 = bool(rho >= 0.8)
    low = [r for r in rows if r["openness_mid"] < 0.1]
    aib2 = bool(all(r["flip_rate"] < 0.05 for r in low))

    outcomes = {"AIB1_openness_leverage": aib1,
                "AIB2_closed_means_uncontrollable": aib2,
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(pval)}
    report = {
        "status": ("ANT-INT-B openness-leverage law; registered "
                   "before run; paired same-seed counterfactuals; "
                   "openness reference frozen from RE-2"),
        "n_paired": len(committing),
        "sweep": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ant_openness_leverage.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
