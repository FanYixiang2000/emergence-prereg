"""Permutation test: does the collapse burst align with the ability jump?

Every process-level result so far claims that the largest collapse burst
COINCIDES with the largest ability jump. This script quantifies that claim
with an exchangeability null: if burst position carried no information
about ability-jump position, the burst mass observed inside the registered
anchored window (accuracy-jump anchor +- 1 interval, exactly the window
analyze_run uses) would be exchangeable with any other window position.

    p = (# window positions whose window burst >= observed) / (# positions)

computed over all interval positions of the same window width. Small p
means the ability jump sits inside one of the largest collapse-burst
windows -- the coincidence is not an artifact of picking a window.

Applied to every stored process-level emergent run:
- MultiBERTs agreement, seeds 0-4 (public system)
- grokking (MLP bridge) and transformer grokking
- induction_2layer

Also reported for the registered NON-emergent comparisons (memorizer,
no_structure, one-layer, random-target, shuffled-vocab): for these the
test is not expected to be small, because their "ability jumps" are noise.
Caveat recorded up front: the alignment p is a COINCIDENCE statistic, not
an emergence verdict. A control that shares its collapse series with an
emergent condition (multiberts random_target shares the model's
distributions with agreement) can show a small p while its accuracy jump
is pure noise (max jump ~0.05, far below the usefulness threshold); the
criterion's usefulness component, not this test, is what excludes it.
The test quantifies exactly one claim: where a REAL ability jump exists,
it lands inside one of the largest collapse-burst windows.

The runs are not mutually independent: MultiBERTs abilities share checkpoints,
its five agreement runs share data construction, and three abilities were an
exploratory addition. We therefore report per-run empirical ranks only and do
not manufacture an omnibus p value by multiplying them. These values quantify
window rank, not a family-wise significance claim.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def load_series(csv_name: str, run: str) -> Tuple[List[float], List[float]]:
    rows = [row for row in csv.DictReader((OUTPUTS / csv_name).open(encoding="utf-8"))
            if row["run"] == run]
    rows.sort(key=lambda row: int(row["epoch"]))
    collapse = [float(row["collapse_bits"]) for row in rows]
    acc = [float(row["test_acc"]) for row in rows]
    return collapse, acc


def alignment_p(collapse: List[float], acc: List[float]) -> Dict[str, float]:
    bursts = [max(collapse[i] - collapse[i - 1], 0.0) for i in range(1, len(collapse))]
    acc_jumps = [acc[i + 1] - acc[i] for i in range(len(acc) - 1)]
    anchor = int(np.argmax(acc_jumps))
    lo = max(0, anchor - 1)
    hi = min(len(acc) - 1, anchor + 2)  # same window as analyze_run
    observed = max(bursts[lo:hi], default=0.0)
    width = hi - lo
    window_maxima = [
        max(bursts[start : start + width])
        for start in range(0, len(bursts) - width + 1)
    ]
    n_ge = sum(1 for value in window_maxima if value >= observed - 1e-12)
    return {
        "anchor_index": anchor,
        "observed_window_burst": observed,
        "n_positions": len(window_maxima),
        "p_value": n_ge / len(window_maxima),
    }


def main() -> None:
    emergent_runs = [
        ("multiberts seed0 agreement", "multiberts_collapse_timeseries.csv",
         "multiberts_agreement"),
        ("multiberts seed1 agreement", "multiberts_collapse_timeseries_seed1.csv",
         "multiberts_agreement"),
        ("multiberts seed2 agreement", "multiberts_collapse_timeseries_seed2.csv",
         "multiberts_agreement"),
        ("multiberts seed3 agreement", "multiberts_collapse_timeseries_seed3.csv",
         "multiberts_agreement"),
        ("multiberts seed4 agreement", "multiberts_collapse_timeseries_seed4.csv",
         "multiberts_agreement"),
        ("grokking (MLP)", "grokking_collapse_timeseries.csv", "grokking"),
        ("grokking (transformer)", "transformer_grokking_timeseries.csv",
         "transformer_grokking"),
        ("induction 2layer", "induction_head_timeseries.csv", "induction_2layer"),
        # phenomena battery (exploratory addition after its run; the three
        # families were classified emergent, so R4 applies to all three)
        ("multiberts reflexive", "multiberts_phenomena_timeseries.csv", "reflexive"),
        ("multiberts determiner", "multiberts_phenomena_timeseries.csv", "determiner"),
        ("multiberts facts", "multiberts_phenomena_timeseries.csv", "facts"),
    ]
    control_runs = [
        ("memorizer (MLP)", "grokking_collapse_timeseries.csv", "memorizer"),
        ("no_structure (grok)", "grokking_collapse_timeseries.csv", "no_structure"),
        ("induction 1layer", "induction_head_timeseries.csv", "induction_1layer"),
        ("induction memorizer", "induction_head_timeseries.csv", "memorizer"),
        ("multiberts random_target", "multiberts_collapse_timeseries.csv",
         "multiberts_random_target"),
        ("multiberts shuffled_vocab", "multiberts_collapse_timeseries.csv",
         "shuffled_vocab"),
    ]

    results: Dict[str, Dict] = {"emergent": {}, "controls": {}}
    print(f"{'run':34s} {'p':>7s}  {'burst@anchor':>12s}  positions")
    for label, csv_name, run in emergent_runs:
        r = alignment_p(*load_series(csv_name, run))
        results["emergent"][label] = r
        print(f"{label:34s} {r['p_value']:7.4f}  {r['observed_window_burst']:12.3f}  "
              f"{r['n_positions']}")
    print("--- controls (no small p expected) ---")
    for label, csv_name, run in control_runs:
        r = alignment_p(*load_series(csv_name, run))
        results["controls"][label] = r
        print(f"{label:34s} {r['p_value']:7.4f}  {r['observed_window_burst']:12.3f}  "
              f"{r['n_positions']}")

    results["inference_note"] = (
        "Per-run empirical window ranks only; no omnibus p value because runs "
        "are dependent and the phenomena additions include exploratory tests."
    )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "burst_alignment_test.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUTPUTS / 'burst_alignment_test.json'}")


if __name__ == "__main__":
    main()
