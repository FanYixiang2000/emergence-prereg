"""Single-signal audit on the Contextual LBF six-component runs.

This is a post-confirmation diagnostic, not a new definition. It asks whether
any one observable signal can reproduce the full six-component verdict on the
same systems, giving each signal a hindsight-optimal threshold and direction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def iter_systems(data: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for seed, entry in data["seeds"].items():
        for name, system in entry["systems"].items():
            yield f"seed{seed}:{name}", system


def finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def score_row(name: str, system: Dict[str, Any]) -> Dict[str, float]:
    metrics = system["metrics"]
    trigger_rates = metrics.get("trigger_rates", {})
    context_usefulness = metrics.get("context_usefulness", {})
    passes = system["verdict"]["passes"]
    return {
        "system": name,
        "truth_full_criterion": int(system["verdict"]["emergent"]),
        "performance_level": finite(metrics.get("natural_score")),
        "potential_bits": finite(metrics.get("potential_bits")),
        "conditional_selectivity": finite(metrics.get("conditional_selectivity")),
        "specificity_js_bits": finite(metrics.get("specificity_js_bits")),
        "usefulness_gap": finite(metrics.get("usefulness_gap")),
        "min_context_usefulness": min(
            finite(v) for v in context_usefulness.values()
        ) if context_usefulness else math.nan,
        "acquisition": finite(system.get("acquisition")),
        "trigger_rate_context0": finite(trigger_rates.get("0")),
        "trigger_rate_context1": finite(trigger_rates.get("1")),
        "endogeneity_flag": 1.0 if passes.get("endogeneity") else 0.0,
    }


def best_threshold(rows: List[Dict[str, float]], score: str) -> Dict[str, Any]:
    usable = [row for row in rows if math.isfinite(row[score])]
    if not usable:
        return {"status": "no_finite_scores"}
    labels = {row["system"]: int(row["truth_full_criterion"]) for row in usable}
    values = sorted({row[score] for row in usable})
    candidates = [values[0] - 1.0, values[-1] + 1.0]
    candidates.extend(values)
    candidates.extend((a + b) / 2 for a, b in zip(values, values[1:]))
    best: Dict[str, Any] | None = None
    for direction in (1, -1):
        for threshold in candidates:
            pred = {
                row["system"]: int(row[score] >= threshold)
                if direction == 1 else int(row[score] <= threshold)
                for row in usable
            }
            correct = sum(pred[name] == label for name, label in labels.items())
            accuracy = correct / len(usable)
            misclassified = sorted(
                name for name, label in labels.items() if pred[name] != label
            )
            candidate = {
                "accuracy": accuracy,
                "threshold": threshold,
                "direction": direction,
                "misclassified": misclassified,
            }
            if (
                best is None
                or candidate["accuracy"] > best["accuracy"]
                or (
                    candidate["accuracy"] == best["accuracy"]
                    and len(candidate["misclassified"]) < len(best["misclassified"])
                )
            ):
                best = candidate
    assert best is not None
    return best


def summarize(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [score_row(name, system) for name, system in iter_systems(data)]
    scores = [
        "performance_level",
        "potential_bits",
        "conditional_selectivity",
        "specificity_js_bits",
        "usefulness_gap",
        "min_context_usefulness",
        "acquisition",
        "trigger_rate_context0",
        "trigger_rate_context1",
        "endogeneity_flag",
    ]
    detectors = {score: best_threshold(rows, score) for score in scores}
    behavior_only = [
        "performance_level",
        "conditional_selectivity",
        "specificity_js_bits",
        "usefulness_gap",
        "min_context_usefulness",
        "trigger_rate_context0",
        "trigger_rate_context1",
    ]
    definition_internal = ["potential_bits", "acquisition", "endogeneity_flag"]
    positives = sum(row["truth_full_criterion"] for row in rows)
    return {
        "status": "post-confirmation single-signal diagnostic",
        "source": str(path),
        "truth": "full six-component verdict in the same run",
        "n_systems": len(rows),
        "n_positive_full_criterion": positives,
        "n_negative_full_criterion": len(rows) - positives,
        "detectors": detectors,
        "grouped_max_accuracy": {
            "behavior_only_single_signals": max(
                detectors[name]["accuracy"] for name in behavior_only
            ),
            "definition_internal_signals": max(
                detectors[name]["accuracy"] for name in definition_internal
            ),
        },
        "per_system_scores": rows,
        "interpretation": (
            "Each signal receives a hindsight-optimal threshold and direction. "
            "Behavior-only single signals exclude potential, acquisition and "
            "endogeneity, which are internal components of the proposed criterion "
            "rather than stand-alone prior definitions. Failures are blind spots "
            "relative to the full conjunctive verdict, not claims that the signal "
            "is useless."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=OUTPUTS / "contextual_lbf_confirmation.json")
    parser.add_argument("--output", type=Path,
                        default=OUTPUTS / "contextual_lbf_single_signal_audit.json")
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "source": result["source"],
        "n_systems": result["n_systems"],
        "n_positive_full_criterion": result["n_positive_full_criterion"],
        "detector_accuracies": {
            name: round(item["accuracy"], 3)
            for name, item in result["detectors"].items()
            if "accuracy" in item
        },
    }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
