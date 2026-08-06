"""Score registered prediction S7 and radius sensitivity on the held-out
Pythia scales (1B, 1.4B, 2.8B).

S7 (PYTHIA_SCALING_PREREGISTRATION.md, frozen 2026-07-11): across
checkpoint-thinning factors 2--4 and offsets, at least 90% of condition-level
verdicts agree with the full-grid verdict.

Pure re-analysis of the stored scaling time series using the identical
analyse/verdict/thinning code from process_proxy_robustness.py (bounded burst
threshold 5/6 = registered ratio 5). Writes a separate output file; no stored
run is modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from process_proxy_robustness import analyse, load_rows, thin, verdict, RunSpec

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


@dataclass(frozen=True)
class Cond:
    label: str
    csv: str
    run: str
    endogenous: bool = True


SCALES = {
    "1b": [
        Cond("1b agreement", "pythia_collapse_timeseries_1b.csv", "pythia_agreement"),
        Cond("1b random target", "pythia_collapse_timeseries_1b.csv", "pythia_random_target"),
        Cond("1b shuffled vocab", "pythia_collapse_timeseries_1b.csv", "shuffled_vocab"),
        Cond("1b head facts", "pythia_tail_timeseries_1b.csv", "head_facts"),
        Cond("1b tail facts", "pythia_tail_timeseries_1b.csv", "tail_facts"),
        Cond("1b tail words", "pythia_tail_timeseries_1b.csv", "tail_words"),
    ],
    "1.4b": [
        Cond("1.4b agreement", "pythia_collapse_timeseries_1.4b.csv", "pythia_agreement"),
        Cond("1.4b random target", "pythia_collapse_timeseries_1.4b.csv", "pythia_random_target"),
        Cond("1.4b shuffled vocab", "pythia_collapse_timeseries_1.4b.csv", "shuffled_vocab"),
        Cond("1.4b head facts", "pythia_tail_timeseries_1.4b.csv", "head_facts"),
        Cond("1.4b tail facts", "pythia_tail_timeseries_1.4b.csv", "tail_facts"),
        Cond("1.4b tail words", "pythia_tail_timeseries_1.4b.csv", "tail_words"),
    ],
    "2.8b": [
        Cond("2.8b agreement", "pythia_collapse_timeseries_2.8b.csv", "pythia_agreement"),
        Cond("2.8b random target", "pythia_collapse_timeseries_2.8b.csv", "pythia_random_target"),
        Cond("2.8b shuffled vocab", "pythia_collapse_timeseries_2.8b.csv", "shuffled_vocab"),
        Cond("2.8b head facts", "pythia_tail_timeseries_2.8b.csv", "head_facts"),
        Cond("2.8b tail facts", "pythia_tail_timeseries_2.8b.csv", "tail_facts"),
        Cond("2.8b tail words", "pythia_tail_timeseries_2.8b.csv", "tail_words"),
    ],
}


def main() -> None:
    report: Dict[str, Any] = {
        "status": "registered S7 scoring plus radius sensitivity, "
                  "held-out scales, re-analysis of stored series",
        "scales": {},
    }
    agree_cells = 0
    total_cells = 0
    for scale, conds in SCALES.items():
        scale_entry: Dict[str, Any] = {}
        for cond in conds:
            spec = RunSpec(cond.label, cond.csv, cond.run, expected=0,
                           endogenous=cond.endogenous)
            rows = load_rows(spec)
            full = verdict(analyse(rows, radius=1), cond.endogenous)
            radius_verdicts = {
                str(r): verdict(analyse(rows, radius=r), cond.endogenous)["emergent"]
                for r in (0, 1, 2)
            }
            thin_cells = []
            for factor in (2, 3, 4):
                for offset in range(factor):
                    sampled = thin(rows, factor, offset)
                    if len(sampled) < 6:
                        continue
                    v = verdict(analyse(sampled, radius=1), cond.endogenous)
                    match = v["emergent"] == full["emergent"]
                    thin_cells.append({
                        "factor": factor, "offset": offset,
                        "emergent": v["emergent"], "match": match,
                    })
                    agree_cells += match
                    total_cells += 1
            scale_entry[cond.run] = {
                "full_grid_emergent": full["emergent"],
                "full_grid_passes": full["passes"],
                "radius_verdicts": radius_verdicts,
                "radius_stable": len(set(radius_verdicts.values())) == 1,
                "thinning_agreement": (
                    sum(c["match"] for c in thin_cells) / len(thin_cells)
                    if thin_cells else None
                ),
                "n_thinning_cells": len(thin_cells),
            }
        report["scales"][scale] = scale_entry

    fraction = agree_cells / total_cells if total_cells else 0.0
    report["S7"] = {
        "registered_criterion": ">= 0.90 condition-level thinning agreement",
        "cells_agreeing": agree_cells,
        "cells_total": total_cells,
        "fraction": fraction,
        "pass": fraction >= 0.90,
    }
    radius_stable = [
        f"{scale}:{run}"
        for scale, entry in report["scales"].items()
        for run, item in entry.items() if not item["radius_stable"]
    ]
    report["radius_unstable_conditions"] = radius_stable

    out = OUTPUTS / "held_out_scaling_robustness.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"S7: {agree_cells}/{total_cells} = {fraction:.3f} "
          f"({'PASS' if report['S7']['pass'] else 'FAIL'})")
    print(f"radius-unstable conditions: {radius_stable or 'none'}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
