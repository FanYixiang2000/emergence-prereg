"""Threshold-sensitivity rescoring for the Contextual LBF runs.

Pure re-analysis of stored evaluation JSONs: no retraining, no new episodes.
For each component threshold, sweep a grid around the frozen value while
holding the other thresholds fixed, and count how many learned policies pass
the full six-component rule and how many controls are rejected at each grid
point. This addresses the reviewer question of whether the frozen thresholds
sit on a knife edge.

The frozen registered verdicts are never modified; this writes a separate
sensitivity output file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

FROZEN = {
    "potential_bits": 0.5,
    "conditional_selectivity": 0.5,
    "specificity_js_bits": 0.2,
    "usefulness_gap": 0.0,
    "acquisition": 0.3,
}

GRIDS = {
    "potential_bits": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "conditional_selectivity": [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8],
    "specificity_js_bits": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "usefulness_gap": [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
    "acquisition": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
}


def verdict(metrics: Dict[str, float], endogenous: bool, acquisition: float,
            thresholds: Dict[str, float]) -> bool:
    return (
        metrics["potential_bits"] >= thresholds["potential_bits"]
        and metrics["conditional_selectivity"]
        >= thresholds["conditional_selectivity"]
        and metrics["specificity_js_bits"] >= thresholds["specificity_js_bits"]
        and metrics["usefulness_gap"] > thresholds["usefulness_gap"]
        and endogenous
        and acquisition >= thresholds["acquisition"]
    )


def collect(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for seed, entry in data["seeds"].items():
        for name, system in entry["systems"].items():
            passes = system["verdict"]["passes"]
            rows.append({
                "seed": seed,
                "system": name,
                "metrics": system["metrics"],
                "endogenous": bool(passes["endogeneity"]),
                "acquisition": float(system.get("acquisition", 0.0)),
                "is_learned": name == "learned",
                "frozen_emergent": int(system["verdict"]["emergent"]),
            })
    return rows


def sweep(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_learned = sum(row["is_learned"] for row in rows)
    n_controls = sum(not row["is_learned"] for row in rows)
    result: Dict[str, Any] = {}
    for component, grid in GRIDS.items():
        cells = []
        for value in grid:
            thresholds = dict(FROZEN)
            thresholds[component] = value
            learned_pass = sum(
                verdict(row["metrics"], row["endogenous"],
                        row["acquisition"], thresholds)
                for row in rows if row["is_learned"]
            )
            controls_rejected = sum(
                not verdict(row["metrics"], row["endogenous"],
                            row["acquisition"], thresholds)
                for row in rows if not row["is_learned"]
            )
            cells.append({
                "threshold": value,
                "is_frozen": value == FROZEN[component],
                "learned_full_passes": learned_pass,
                "controls_rejected": controls_rejected,
                "verdict_pattern_unchanged": (
                    learned_pass == sum(
                        row["frozen_emergent"] for row in rows
                        if row["is_learned"])
                    and controls_rejected == n_controls
                ),
            })
        stable = [cell["threshold"] for cell in cells
                  if cell["verdict_pattern_unchanged"]]
        result[component] = {
            "frozen_value": FROZEN[component],
            "grid": cells,
            "stable_range": [min(stable), max(stable)] if stable else None,
        }
    result["_counts"] = {"n_learned": n_learned, "n_controls": n_controls}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=OUTPUTS / "contextual_lbf_confirmation.json")
    parser.add_argument("--output", type=Path,
                        default=OUTPUTS / "contextual_lbf_threshold_sensitivity.json")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = collect(data)
    result = {
        "status": "post-hoc threshold-sensitivity rescoring (no retraining)",
        "source": str(args.input),
        "frozen_thresholds": FROZEN,
        "note": (
            "One threshold varied at a time; all others frozen. "
            "'verdict_pattern_unchanged' means the learned pass count equals "
            "the frozen-count and every control is still rejected."
        ),
        "sweep": sweep(rows),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for component, item in result["sweep"].items():
        if component.startswith("_"):
            continue
        print(f"{component}: frozen={item['frozen_value']}, "
              f"stable_range={item['stable_range']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
