"""Seed-level analysis for the frozen Contextual LBF confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
RNG = np.random.default_rng(20260711)
B = 20_000


def bootstrap_mean(values: List[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=float)
    draws = RNG.choice(array, size=(B, len(array)), replace=True).mean(axis=1)
    return {
        "point": float(array.mean()),
        "n_seeds": len(array),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def positive_sign_p(positives: int, n: int) -> float:
    return float(sum(math.comb(n, k) for k in range(positives, n + 1)) / 2 ** n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed-level analysis for Contextual LBF runs.")
    parser.add_argument("--input", type=Path,
                        default=OUTPUTS / "contextual_lbf_confirmation.json")
    parser.add_argument("--output", type=Path,
                        default=OUTPUTS / "contextual_lbf_confirmation_analysis.json")
    parser.add_argument("--registered_confirmation", action="store_true")
    args = parser.parse_args()
    data = json.loads(
        args.input.read_text(encoding="utf-8"))
    seeds = data["seeds"]
    learned = [entry["systems"]["learned"] for entry in seeds.values()]
    metrics = [entry["metrics"] for entry in learned]
    acquisitions = [entry["acquisition"] for entry in learned]
    usefulness = [entry["usefulness_gap"] for entry in metrics]
    selectivity = [entry["conditional_selectivity"] for entry in metrics]
    specificity = [entry["specificity_js_bits"] for entry in metrics]
    potential = [entry["potential_bits"] for entry in metrics]
    full_passes = sum(entry["verdict"]["emergent"] for entry in learned)

    nonlearned_names = (
        "initial_twin", "team_nearest", "fixed_food0", "fixed_food1")
    nonlearned = [
        entry["systems"][name]
        for entry in seeds.values() for name in nonlearned_names
    ]
    controls_rejected = sum(
        not item["verdict"]["emergent"] for item in nonlearned)
    nearest_exact = 0
    for entry in seeds.values():
        passes = entry["systems"]["team_nearest"]["verdict"]["passes"]
        failed = {name for name, passed in passes.items() if not passed}
        nearest_exact += failed == {"endogeneity", "acquisition"}

    ordered = sum(
        metric["trigger_rates"]["0"] > metric["trigger_rates"]["1"]
        for metric in metrics)
    useful_positive = sum(value > 0 for value in usefulness)
    init_acquisition_fail = sum(
        not entry["systems"]["initial_twin"]["verdict"]["passes"]["acquisition"]
        for entry in seeds.values())

    intervals = {
        "potential_bits": bootstrap_mean(potential),
        "conditional_selectivity": bootstrap_mean(selectivity),
        "specificity_js_bits": bootstrap_mean(specificity),
        "usefulness_gap": bootstrap_mean(usefulness),
        "acquisition": bootstrap_mean(acquisitions),
    }
    predictions = None
    if args.registered_confirmation:
        predictions = {
            "CLBF_C1_learned_full_pass_at_least_9_of_10": full_passes >= 9,
            "CLBF_C2_all_40_controls_rejected": controls_rejected == 40,
            "CLBF_C3_nearest_exact_route_at_least_9_of_10": nearest_exact >= 9,
            "CLBF_C4_acquisition_all_positive_and_init_all_fail": (
                all(value > 0 for value in acquisitions)
                and init_acquisition_fail == len(seeds)
            ),
            "CLBF_C5_order_all_and_usefulness_at_least_9_of_10": (
                ordered == len(seeds) and useful_positive >= 9
            ),
            "CLBF_C6_positive_bootstrap_lower_bounds": (
                intervals["acquisition"]["ci95"][0] > 0
                and intervals["usefulness_gap"]["ci95"][0] > 0
            ),
        }
    result = {
        "status": data.get("status", "contextual LBF seed-level analysis"),
        "analysis_scope": (
            "registered confirmation" if args.registered_confirmation
            else "post-confirmation extension"
        ),
        "n_seeds": len(seeds),
        "counts": {
            "learned_full_passes": full_passes,
            "nonlearned_controls_rejected": controls_rejected,
            "team_nearest_exact_failure_route": nearest_exact,
            "learned_context_ordering": ordered,
            "learned_positive_usefulness": useful_positive,
            "learned_positive_acquisition": sum(
                value > 0 for value in acquisitions),
            "initial_twins_fail_acquisition": init_acquisition_fail,
        },
        "learned_full_pass_exact_one_sided_sign_p": positive_sign_p(
            full_passes, len(seeds)),
        "seed_bootstrap_intervals": intervals,
        "predictions": predictions,
        "all_registered_predictions_pass": (
            None if predictions is None else all(predictions.values())
        ),
        "inference_note": (
            "Training seed is the population unit. Bootstrap intervals resample "
            f"{len(seeds)} seed-level summaries, not evaluation episodes."
        ),
    }
    path = args.output
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
