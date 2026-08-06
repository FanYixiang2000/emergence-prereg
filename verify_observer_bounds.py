"""Numerically verify observer/estimator bounds used by the framework.

The proofs are algebraic and live in THEORY.md. This script stress-tests:

1. partition refinement: entropy and JS cannot decrease when a coarse basin
   map is refined, while value is invariant if it is coarse-measurable;
2. model-based do-gap error: value-gap error is bounded by the sum of total
   variation errors times the value range;
3. entropy continuity under observer-model error (Fannes--Audenaert);
4. exact equivalence of the unbounded and bounded burstiness thresholds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np


OUTPUTS = Path(__file__).resolve().parent / "outputs"
RNG = np.random.default_rng(20260711)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def binary_entropy(x: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def coarse(p: np.ndarray, groups: np.ndarray, k: int) -> np.ndarray:
    return np.asarray([np.sum(p[groups == index]) for index in range(k)])


def random_distribution(n: int) -> np.ndarray:
    return RNG.dirichlet(np.ones(n))


def partition_trials(n_trials: int = 10_000) -> Dict[str, float]:
    violations_entropy = violations_js = violations_value = 0
    max_entropy_slack = max_js_slack = max_value_gap = 0.0
    for _ in range(n_trials):
        n_fine = int(RNG.integers(4, 15))
        n_coarse = int(RNG.integers(2, min(5, n_fine) + 1))
        groups = np.arange(n_fine) % n_coarse
        RNG.shuffle(groups)
        # Ensure every coarse cell is represented.
        groups[:n_coarse] = np.arange(n_coarse)
        p, q = random_distribution(n_fine), random_distribution(n_fine)
        pc, qc = coarse(p, groups, n_coarse), coarse(q, groups, n_coarse)
        h_gap = entropy(p) - entropy(pc)
        js_gap = js(p, q) - js(pc, qc)
        values_coarse = RNG.uniform(-2, 3, size=n_coarse)
        values_fine = values_coarse[groups]
        value_gap = abs(
            (float(p @ values_fine) - float(q @ values_fine))
            - (float(pc @ values_coarse) - float(qc @ values_coarse))
        )
        violations_entropy += h_gap < -1e-10
        violations_js += js_gap < -1e-10
        violations_value += value_gap > 1e-10
        max_entropy_slack = max(max_entropy_slack, h_gap)
        max_js_slack = max(max_js_slack, js_gap)
        max_value_gap = max(max_value_gap, value_gap)
    return {
        "trials": n_trials,
        "entropy_refinement_violations": violations_entropy,
        "js_refinement_violations": violations_js,
        "coarse_measurable_value_violations": violations_value,
        "max_entropy_refinement_gain_bits": max_entropy_slack,
        "max_js_refinement_gain_bits": max_js_slack,
        "max_value_invariance_error": max_value_gap,
    }


def model_error_trials(n_trials: int = 10_000) -> Dict[str, float]:
    value_violations = entropy_violations = 0
    max_value_ratio = max_entropy_ratio = 0.0
    for _ in range(n_trials):
        k = int(RNG.integers(2, 15))
        p_a, p_b = random_distribution(k), random_distribution(k)
        q_a, q_b = random_distribution(k), random_distribution(k)
        values = RNG.uniform(-4, 7, size=k)
        value_range = float(np.max(values) - np.min(values))
        true_gap = float(p_a @ values - p_b @ values)
        model_gap = float(q_a @ values - q_b @ values)
        bound = value_range * (tv(p_a, q_a) + tv(p_b, q_b))
        error = abs(true_gap - model_gap)
        value_violations += error > bound + 1e-10
        if bound > 0:
            max_value_ratio = max(max_value_ratio, error / bound)

        epsilon = tv(p_a, q_a)
        # Fannes--Audenaert applies for epsilon <= 1 - 1/k.
        if epsilon <= 1 - 1 / k:
            entropy_bound = binary_entropy(epsilon) + epsilon * math.log2(k - 1)
            entropy_error = abs(entropy(p_a) - entropy(q_a))
            entropy_violations += entropy_error > entropy_bound + 1e-10
            if entropy_bound > 0:
                max_entropy_ratio = max(max_entropy_ratio, entropy_error / entropy_bound)
    return {
        "trials": n_trials,
        "value_gap_bound_violations": value_violations,
        "max_value_error_to_bound_ratio": max_value_ratio,
        "entropy_continuity_violations": entropy_violations,
        "max_entropy_error_to_bound_ratio": max_entropy_ratio,
    }


def bounded_burst_equivalence() -> Dict[str, object]:
    data = json.loads((OUTPUTS / "process_proxy_robustness.json").read_text())
    mismatches = []
    for label, item in data["runs"].items():
        metrics = item["primary"]["metrics"]
        ratio_pass = metrics["raw_burstiness_ratio"] >= 5.0
        bounded_pass = metrics["bounded_burst_concentration"] >= 5.0 / 6.0
        if ratio_pass != bounded_pass:
            mismatches.append(label)
    return {
        "n_runs": len(data["runs"]),
        "threshold_ratio": 5.0,
        "threshold_bounded": 5.0 / 6.0,
        "mismatches": mismatches,
        "equivalent_on_all_runs": not mismatches,
    }


def main() -> None:
    result = {
        "partition_refinement": partition_trials(),
        "model_based_intervention_error": model_error_trials(),
        "bounded_burst_equivalence": bounded_burst_equivalence(),
    }
    result["all_checks_pass"] = (
        result["partition_refinement"]["entropy_refinement_violations"] == 0
        and result["partition_refinement"]["js_refinement_violations"] == 0
        and result["partition_refinement"]["coarse_measurable_value_violations"] == 0
        and result["model_based_intervention_error"]["value_gap_bound_violations"] == 0
        and result["model_based_intervention_error"]["entropy_continuity_violations"] == 0
        and result["bounded_burst_equivalence"]["equivalent_on_all_runs"]
    )
    path = OUTPUTS / "observer_bounds_verification.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
