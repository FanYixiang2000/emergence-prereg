"""Threshold sensitivity: do the battery verdicts survive threshold changes?

The criterion uses four numeric thresholds (potential 0.5 bits, separation
0.5, specificity 0.2 bits, usefulness 0.0) plus design flags. A reviewer
will ask whether the perfect battery accuracies are an artifact of tuning
these numbers. This analysis rescoreS existing measurements (no retraining,
no new data) under multiplicative sweeps of each threshold:

    multiplier m in {0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6}
    (usefulness threshold is 0, so it is swept additively over
     {-1.0, -0.5, 0.0, +0.5, +1.0} return units instead)

For each threshold, swept alone with the others fixed, we recompute:

    - internal battery accuracy (10 systems incl. anti_selector,
      conditional-selectivity scoring, from
      criterion_battery_measurements.csv + refined summary)
    - external fresh-seed accuracy (25 system-seed cells, refined
      six-component scoring, from refined_confirmation_external.csv)

Output: accuracy as a function of each threshold. The claim to check is a
PLATEAU: accuracy stays at 1.0 across a wide band around the registered
values, so the registered numbers are not knife-edge.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

OUTPUTS = Path(__file__).resolve().parent / "outputs"

BASE = {
    "potential_bits": 0.5,
    "separation": 0.5,
    "specificity_js": 0.2,
    "usefulness_gap": 0.0,
    "acquisition": 0.3,
}

MULTIPLIERS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
USEFULNESS_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def load_internal() -> List[Dict[str, Any]]:
    """Internal battery measurements + anti_selector from the refined summary."""
    with (OUTPUTS / "criterion_battery_measurements.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    refined = json.loads((OUTPUTS / "refined_selectivity_summary.json").read_text())
    anti = refined["anti_selector_measurement"]
    rows.append({k: str(v) for k, v in anti.items()})
    out = []
    for row in rows:
        separation = abs(
            float(row["rescue_trigger_rate"]) - float(row["bridge_trigger_rate"])
        )
        if row["system"] in ("harmful_decoy", "useful_habit"):
            separation = 0.0
        out.append({
            "system": row["system"],
            "truth": int(row["ground_truth_emergent"]),
            "h0": float(row["h0_bits"]),
            "separation": separation,
            "js": float(row["specificity_js"]),
            "gap": float(row["usefulness_gap"]),
            "prespecified": int(row["prespecified"]),
            "acquisition": None,
        })
    return out


def load_external() -> List[Dict[str, Any]]:
    with (OUTPUTS / "refined_confirmation_external.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        prespecified = int(row["system"] in ("nearest_only", "role_oracle", "damage_aware"))
        out.append({
            "system": f"{row['system']}@{row['seed']}",
            "truth": int(row["truth"]),
            "h0": float(row["h0_bits"]),
            "separation": float(row["separation"]),
            "js": float(row["specificity_js"]),
            "gap": float(row["usefulness_gap"]),
            "prespecified": prespecified,
            "acquisition": float(row["acquisition"]),
        })
    return out


def verdict(row: Dict[str, Any], th: Dict[str, float]) -> int:
    ok = (
        row["h0"] >= th["potential_bits"]
        and row["separation"] >= th["separation"]
        and row["js"] >= th["specificity_js"]
        and row["gap"] > th["usefulness_gap"]
        and row["prespecified"] == 0
    )
    if ok and row["acquisition"] is not None:
        ok = row["acquisition"] >= th["acquisition"]
    return int(ok)


def accuracy(rows: List[Dict[str, Any]], th: Dict[str, float]) -> float:
    return sum(verdict(r, th) == r["truth"] for r in rows) / len(rows)


def sweep(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for key in ("potential_bits", "separation", "specificity_js", "acquisition"):
        curve = []
        for m in MULTIPLIERS:
            th = dict(BASE)
            th[key] = BASE[key] * m
            curve.append({"multiplier": m, "value": th[key], "accuracy": accuracy(rows, th)})
        results[key] = curve
    curve = []
    for offset in USEFULNESS_OFFSETS:
        th = dict(BASE)
        th["usefulness_gap"] = offset
        curve.append({"offset": offset, "accuracy": accuracy(rows, th)})
    results["usefulness_gap"] = curve
    stable = all(
        point["accuracy"] == 1.0
        for key in results
        for point in results[key]
        if point.get("multiplier", 1.0) in (0.8, 1.0, 1.2) or point.get("offset") == 0.0
    )
    print(f"\n[{label}] plateau within +-20%: {'YES' if stable else 'NO'}")
    for key, curve in results.items():
        accs = [f"{point['accuracy']:.2f}" for point in curve]
        print(f"  {key:18s} {accs}")
    return {"curves": results, "plateau_pm20": stable}


def main() -> None:
    internal = load_internal()
    external = load_external()
    summary = {
        "base_thresholds": BASE,
        "multipliers": MULTIPLIERS,
        "usefulness_offsets": USEFULNESS_OFFSETS,
        "internal": sweep(internal, "internal battery (10 systems)"),
        "external": sweep(external, "external fresh seeds (25 cells)"),
    }
    (OUTPUTS / "threshold_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUTPUTS / 'threshold_sensitivity_summary.json'}")


if __name__ == "__main__":
    main()
