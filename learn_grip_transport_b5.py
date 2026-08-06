"""LGT-B: side-openness B5 adjudication for LEARN-GRIP-TRANSPORT.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen after seed 0's
console line only). Applies the frozen B5 adjudicator to the median
within-episode side-openness curve, the correct current-state object
for the side-commitment regime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    src = json.loads((OUTPUTS / "learn_grip_transport.json").read_text())
    rows = {}
    for key, seed in src["seeds"].items():
        curve = np.array(seed["side_openness_curve"], dtype=float)
        adj = adjudicate(range(len(curve)), curve)
        plateau = 0
        for val in curve:
            if val >= 0.8:
                plateau += 1
            else:
                break
        rows[key] = {
            "final_success": seed["final_success"],
            "final_side_mean": seed["final_side_mean"],
            "plateau_len": plateau,
            "side_openness_first": round(float(curve[0]), 5),
            "side_openness_final": round(float(curve[-1]), 5),
            "adj": adj,
        }
        h = adj.get("hinge", {})
        print(f"seed={key}: plateau={plateau} open0={curve[0]:.3f} "
              f"openT={curve[-1]:.3f} B5={adj['b5_onset']} "
              f"dBIC={h.get('delta_bic')} t*={h.get('t_star')}", flush=True)

    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    outcomes = {
        "LGTB1_plateau_then_collapse": bool(
            learned and all(
                r["plateau_len"] >= 5 and r["side_openness_final"] <= 0.3
                for r in learned)),
        "LGTB2_resolvable_onset": bool(
            len(learned) >= 4
            and sum(r["adj"]["b5_onset"] for r in learned) >= 2),
        "LGTB3_symmetry": bool(
            learned and all(abs(r["final_side_mean"]) <= 0.4 for r in learned)),
        "n_learned": len(learned),
        "b5_count_learned": sum(r["adj"]["b5_onset"] for r in learned),
        "median_plateau_len": None if not learned else float(np.median(
            [r["plateau_len"] for r in learned])),
    }
    report = {
        "status": "LGT-B side-openness B5 adjudication; preregistered follow-up",
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_transport_b5.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
