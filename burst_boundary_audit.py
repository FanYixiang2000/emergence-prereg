"""Boundary audit for the original burst-collapse emergence criterion.

The early framework treated burst-like possibility collapse as the sharp
signature separating emergence from ordinary gradual learning. Later evidence
showed a stronger but narrower role:

  * burst is an informative acquisition-shape signal;
  * burst alone is not sufficient for emergence;
  * burst is not necessary for collective emergence;
  * burst verdicts are partly checkpoint-grid relative.

This audit uses only stored, already-frozen outputs. It does not retrain or
rescore any system. Its purpose is to make the definitional update explicit
instead of hiding the old intuition.

REGISTERED PREDICTIONS (frozen before running):
  BB-1  Burst is not sufficient: the ordinary learner passes the old process
        proxy 6/6 with all burstiness ratios >= 5, but the separately frozen
        lower-order novelty test rejects it.
  BB-2  Burst is not necessary: the ant TRAIL condition is accepted as weak
        collective emergence (D >= 0.5, R >= 0.6) while its 10-90 commitment
        span is gradual (>= 0.10 of the horizon, max single-step share < 0.50).
  BB-3  Burst verdicts are grid-relative: 2.8B agreement has usefulness under
        the full grid but fails bounded_burst, and flips under all 9 thinning
        cells in the held-out robustness audit.
  BB-4  Burst remains useful only as an evidence channel: at least one stored
        control has bounded_burst true but fails usefulness, showing the burst
        substrate cannot decide the verdict without product/utility checks.
Misses are retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def load(name: str) -> Any:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def main() -> None:
    ordinary = load("ordinary_learner_control.json")
    novelty = load("capability_novelty_boundary.json")
    ant = load("ant_contrast.json")
    scaling = load("held_out_scaling_robustness.json")

    ordinary_burst = [
        run["stats"]["burstiness_ratio"]
        for run in ordinary["runs"].values()
    ]
    old_proxy_accepts = ordinary["finding"]["runs_accepted_by_proxy"] == "6/6"
    novelty_rejects = (
        novelty["novelty_qualified_verdict"]["ordinary"] is False
    )
    bb1 = min(ordinary_burst) >= 5.0 and old_proxy_accepts and novelty_rejects

    trail = ant["TRAIL"]
    trail_grad = trail["gradualism"]
    bb2 = (
        trail["D"] >= ant["thresholds"]["D"]
        and trail["R"] >= ant["thresholds"]["R"]
        and trail_grad["span_frac"] >= 0.10
        and trail_grad["max_step_frac"] < 0.50
    )

    p28_agreement = scaling["scales"]["2.8b"]["pythia_agreement"]
    bb3 = (
        p28_agreement["full_grid_emergent"] == 0
        and p28_agreement["full_grid_passes"]["usefulness"] is True
        and p28_agreement["full_grid_passes"]["bounded_burst"] is False
        and p28_agreement["thinning_agreement"] == 0.0
        and p28_agreement["n_thinning_cells"] == 9
    )

    bursty_failed_controls = []
    for scale, rows in scaling["scales"].items():
        for name, row in rows.items():
            if (
                name != "pythia_agreement"
                and row["full_grid_passes"].get("bounded_burst") is True
                and row["full_grid_passes"].get("usefulness") is False
                and row["full_grid_emergent"] == 0
            ):
                bursty_failed_controls.append({
                    "scale": scale,
                    "condition": name,
                })
    bb4 = len(bursty_failed_controls) >= 1

    report = {
        "status": (
            "stored-output audit: burst collapse is an acquisition-shape "
            "evidence channel, not a sufficient or necessary definition"
        ),
        "old_position_retained_as_historical_hypothesis": (
            "strong emergence should show burst-like structured useful "
            "collapse"
        ),
        "updated_position": (
            "burst is predictive when aligned with usefulness and controls; "
            "the emergence verdict requires endogenous formation, "
            "irreducibility/novelty, persistence and value"
        ),
        "values": {
            "ordinary_burst_not_sufficient": {
                "old_proxy_accepts": ordinary["finding"]
                ["runs_accepted_by_proxy"],
                "min_burstiness_ratio": min(ordinary_burst),
                "max_burstiness_ratio": max(ordinary_burst),
                "novelty_verdict": novelty["novelty_qualified_verdict"]
                ["ordinary"],
                "novelty_gap": novelty["values"]["ordinary"]
                ["novelty_gap"],
            },
            "ant_gradual_but_collective": {
                "D": trail["D"],
                "R": trail["R"],
                "span_frac": trail_grad["span_frac"],
                "max_step_frac": trail_grad["max_step_frac"],
                "dev_final": trail_grad["dev_final"],
            },
            "pythia_grid_relative": {
                "full_grid_emergent": p28_agreement["full_grid_emergent"],
                "full_grid_passes": p28_agreement["full_grid_passes"],
                "thinning_agreement": p28_agreement["thinning_agreement"],
                "n_thinning_cells": p28_agreement["n_thinning_cells"],
                "radius_verdicts": p28_agreement["radius_verdicts"],
            },
            "bursty_failed_controls": bursty_failed_controls,
        },
        "registered_outcomes": {
            "BB1_burst_not_sufficient": bool(bb1),
            "BB2_burst_not_necessary": bool(bb2),
            "BB3_burst_grid_relative": bool(bb3),
            "BB4_burst_needs_utility_controls": bool(bb4),
        },
    }

    out = OUTPUTS / "burst_boundary_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["values"], indent=2))
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
