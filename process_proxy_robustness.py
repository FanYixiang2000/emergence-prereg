"""Robustness audit for the checkpoint/process-level emergence proxy.

This is an exploratory re-analysis of stored time series. It does not create
new confirmatory evidence. It addresses three reviewer-facing questions:

1. Can the unbounded ``window burst / median burst`` statistic be replaced by
   a bounded, numerically stable equivalent?
2. Do verdicts survive alternate window radii and checkpoint thinning?
3. How much positive collapse mass is concentrated near the ability jump?

The bounded statistic

    q = window_burst / (window_burst + median_background_burst)

is a monotone transform of the original ratio (without the numerical epsilon).
The registered ratio threshold 5 maps exactly to q >= 5/6. Thus this audit
does not tune a new threshold.

Empirical window ranks are also reported. They are local coincidence
statistics, not mutually independent p values; adjusted values are included
only as a conservative multiplicity sensitivity analysis.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
BOUNDED_BURST_THRESHOLD = 5.0 / 6.0


@dataclass(frozen=True)
class RunSpec:
    label: str
    csv_name: str
    run: str
    expected: int
    endogenous: bool = True
    status: str = "confirmatory"


RUNS: tuple[RunSpec, ...] = (
    RunSpec("multiberts seed0 agreement", "multiberts_collapse_timeseries.csv",
            "multiberts_agreement", 1),
    RunSpec("multiberts seed1 agreement", "multiberts_collapse_timeseries_seed1.csv",
            "multiberts_agreement", 1),
    RunSpec("multiberts seed2 agreement", "multiberts_collapse_timeseries_seed2.csv",
            "multiberts_agreement", 1),
    RunSpec("multiberts seed3 agreement", "multiberts_collapse_timeseries_seed3.csv",
            "multiberts_agreement", 1),
    RunSpec("multiberts seed4 agreement", "multiberts_collapse_timeseries_seed4.csv",
            "multiberts_agreement", 1),
    RunSpec("multiberts random target", "multiberts_collapse_timeseries.csv",
            "multiberts_random_target", 0),
    RunSpec("multiberts shuffled vocabulary", "multiberts_collapse_timeseries.csv",
            "shuffled_vocab", 0),
    RunSpec("grokking MLP", "grokking_collapse_timeseries.csv", "grokking", 1),
    RunSpec("memorizer MLP", "grokking_collapse_timeseries.csv", "memorizer", 0),
    RunSpec("no structure MLP", "grokking_collapse_timeseries.csv", "no_structure", 0),
    RunSpec("prewired MLP", "grokking_collapse_timeseries.csv", "prewired", 0,
            endogenous=False),
    RunSpec("grokking transformer", "transformer_grokking_timeseries.csv",
            "transformer_grokking", 1),
    RunSpec("induction two layer", "induction_head_timeseries.csv",
            "induction_2layer", 1),
    RunSpec("induction one layer", "induction_head_timeseries.csv",
            "induction_1layer", 0),
    RunSpec("induction memorizer", "induction_head_timeseries.csv", "memorizer", 0),
    RunSpec("pythia 160m agreement", "pythia_collapse_timeseries.csv",
            "pythia_agreement", 1),
    RunSpec("pythia 160m random target", "pythia_collapse_timeseries.csv",
            "pythia_random_target", 0),
    RunSpec("pythia 160m shuffled vocabulary", "pythia_collapse_timeseries.csv",
            "shuffled_vocab", 0),
    RunSpec("pythia 410m agreement", "pythia_collapse_timeseries_410m.csv",
            "pythia_agreement", 1),
    RunSpec("pythia 410m random target", "pythia_collapse_timeseries_410m.csv",
            "pythia_random_target", 0),
    RunSpec("pythia 410m shuffled vocabulary", "pythia_collapse_timeseries_410m.csv",
            "shuffled_vocab", 0),
    RunSpec("pythia head facts", "pythia_tail_timeseries.csv", "head_facts", 1),
    RunSpec("pythia tail facts", "pythia_tail_timeseries.csv", "tail_facts", 0),
    RunSpec("pythia tail words", "pythia_tail_timeseries.csv", "tail_words", 0),
    RunSpec("multiberts reflexive", "multiberts_phenomena_timeseries.csv",
            "reflexive", 1, status="exploratory"),
    RunSpec("multiberts determiner", "multiberts_phenomena_timeseries.csv",
            "determiner", 1, status="exploratory"),
    RunSpec("multiberts facts", "multiberts_phenomena_timeseries.csv",
            "facts", 1, status="exploratory"),
)


def load_rows(spec: RunSpec) -> List[Dict[str, float]]:
    with (OUTPUTS / spec.csv_name).open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["run"] == spec.run]
    rows.sort(key=lambda row: int(row["epoch"]))
    return [
        {
            "epoch": int(row["epoch"]),
            "acc": float(row["test_acc"]),
            "entropy": float(row["test_entropy_bits"]),
            "collapse": float(row["collapse_bits"]),
        }
        for row in rows
    ]


def analyse(rows: List[Dict[str, float]], radius: int = 1) -> Dict[str, Any]:
    epochs = np.asarray([row["epoch"] for row in rows], dtype=int)
    acc = np.asarray([row["acc"] for row in rows], dtype=float)
    entropy = np.asarray([row["entropy"] for row in rows], dtype=float)
    collapse = np.asarray([row["collapse"] for row in rows], dtype=float)
    bursts = np.maximum(np.diff(collapse), 0.0)
    acc_jumps = np.diff(acc)
    anchor = int(np.argmax(acc_jumps))
    lo = max(0, anchor - radius)
    hi = min(len(bursts), anchor + radius + 1)
    window = bursts[lo:hi]
    observed = float(np.max(window)) if len(window) else 0.0
    median_background = float(np.median(bursts)) if len(bursts) else 0.0
    denominator = observed + median_background
    bounded_burst = observed / denominator if denominator > 0 else 0.0
    raw_ratio = observed / (median_background + 1e-6)
    positive_mass = float(np.sum(bursts))
    burst_share = observed / positive_mass if positive_mass > 0 else 0.0
    width = max(1, hi - lo)
    window_maxima = [
        float(np.max(bursts[start:start + width]))
        for start in range(0, len(bursts) - width + 1)
    ]
    rank = (
        sum(value >= observed - 1e-12 for value in window_maxima)
        / len(window_maxima)
        if window_maxima else 1.0
    )
    gain_lo = max(0, anchor - radius)
    gain_hi = min(len(acc) - 1, anchor + radius + 1)
    ability_gain = float(acc[gain_hi] - acc[gain_lo])
    potential = float(entropy[gain_lo])
    global_burst = int(np.argmax(bursts)) if len(bursts) else 0
    return {
        "n_checkpoints": len(rows),
        "anchor_index": anchor,
        "anchor_epoch": int(epochs[min(anchor + 1, len(epochs) - 1)]),
        "max_burst_epoch": int(epochs[min(global_burst + 1, len(epochs) - 1)]),
        "burst_lead_epochs": int(
            epochs[min(global_burst + 1, len(epochs) - 1)]
            - epochs[min(anchor + 1, len(epochs) - 1)]
        ),
        "potential_bits": potential,
        "ability_gain": ability_gain,
        "window_burst_bits": observed,
        "median_background_burst_bits": median_background,
        "raw_burstiness_ratio": raw_ratio,
        "bounded_burst_concentration": bounded_burst,
        "positive_burst_mass_share": burst_share,
        "empirical_window_rank": rank,
    }


def verdict(metrics: Dict[str, Any], endogenous: bool) -> Dict[str, Any]:
    passes = {
        "potential": metrics["potential_bits"] >= 1.0,
        "bounded_burst": (
            metrics["bounded_burst_concentration"] >= BOUNDED_BURST_THRESHOLD
        ),
        "usefulness": metrics["ability_gain"] >= 0.2,
        "endogeneity": endogenous,
    }
    return {"passes": passes, "emergent": int(all(passes.values()))}


def thin(rows: List[Dict[str, float]], factor: int, offset: int) -> List[Dict[str, float]]:
    indices = list(range(offset, len(rows), factor))
    if 0 not in indices:
        indices.insert(0, 0)
    if len(rows) - 1 not in indices:
        indices.append(len(rows) - 1)
    return [rows[index] for index in sorted(set(indices))]


def adjust_bh(values: Iterable[float]) -> List[float]:
    vals = np.asarray(list(values), dtype=float)
    order = np.argsort(vals)
    adjusted = np.empty(len(vals), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(vals) - reverse_rank + 1
        running = min(running, vals[index] * len(vals) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def adjust_holm(values: Iterable[float]) -> List[float]:
    vals = np.asarray(list(values), dtype=float)
    order = np.argsort(vals)
    adjusted = np.empty(len(vals), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(vals) - rank) * vals[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    output: Dict[str, Any] = {
        "status": "exploratory re-analysis of stored time series",
        "bounded_threshold": BOUNDED_BURST_THRESHOLD,
        "threshold_derivation": "registered ratio 5 mapped by q=r/(1+r)",
        "runs": {},
    }
    local_ranks: List[float] = []
    rank_labels: List[str] = []
    for spec in RUNS:
        rows = load_rows(spec)
        primary = analyse(rows, radius=1)
        primary_verdict = verdict(primary, spec.endogenous)
        radius_results = {}
        for radius in (0, 1, 2):
            metrics = analyse(rows, radius=radius)
            radius_results[str(radius)] = {
                "metrics": metrics,
                "verdict": verdict(metrics, spec.endogenous),
            }
        thinning_results = {}
        for factor in (2, 3, 4):
            cells = []
            for offset in range(factor):
                sampled = thin(rows, factor, offset)
                if len(sampled) < 6:
                    continue
                metrics = analyse(sampled, radius=1)
                cells.append({
                    "offset": offset,
                    "metrics": metrics,
                    "verdict": verdict(metrics, spec.endogenous),
                })
            thinning_results[str(factor)] = cells
        output["runs"][spec.label] = {
            "source": {"csv": spec.csv_name, "run": spec.run},
            "status": spec.status,
            "expected": spec.expected,
            "primary": {"metrics": primary, "verdict": primary_verdict},
            "radius_sensitivity": radius_results,
            "thinning_sensitivity": thinning_results,
        }
        if spec.expected and spec.status == "confirmatory":
            local_ranks.append(primary["empirical_window_rank"])
            rank_labels.append(spec.label)

    bh = adjust_bh(local_ranks)
    holm = adjust_holm(local_ranks)
    output["multiplicity_sensitivity"] = {
        label: {
            "local_window_rank": rank,
            "bh_adjusted": bh_value,
            "holm_adjusted": holm_value,
        }
        for label, rank, bh_value, holm_value in zip(rank_labels, local_ranks, bh, holm)
    }
    output["multiplicity_note"] = (
        "Runs are dependent, so these adjustments are conservative sensitivity "
        "summaries rather than a valid omnibus analysis."
    )

    primary_correct = [
        item["primary"]["verdict"]["emergent"] == item["expected"]
        for item in output["runs"].values()
    ]
    radius_accuracy = {}
    for radius in ("0", "1", "2"):
        radius_accuracy[radius] = float(np.mean([
            item["radius_sensitivity"][radius]["verdict"]["emergent"]
            == item["expected"]
            for item in output["runs"].values()
        ]))
    thinning_cells = []
    for item in output["runs"].values():
        for factor_cells in item["thinning_sensitivity"].values():
            thinning_cells.extend(
                cell["verdict"]["emergent"] == item["expected"]
                for cell in factor_cells
            )
    output["summary"] = {
        "n_runs": len(RUNS),
        "primary_accuracy": float(np.mean(primary_correct)),
        "radius_accuracy": radius_accuracy,
        "thinning_cell_accuracy": float(np.mean(thinning_cells)),
        "n_thinning_cells": len(thinning_cells),
        "all_primary_verdicts_match_expected": bool(all(primary_correct)),
    }

    path = OUTPUTS / "process_proxy_robustness.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
