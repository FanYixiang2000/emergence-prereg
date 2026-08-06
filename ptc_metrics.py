"""Metrics for Potential-Trigger-Collapse experiments."""

from __future__ import annotations

from collections import Counter
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from ptc_gridworld import BASINS, Trajectory


EPS = 1e-12


def normalize_counts(counts: Mapping[str, int]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: value / total for key, value in counts.items()}


def basin_distribution(trajectories: Sequence[Trajectory]) -> Dict[str, float]:
    counts = {basin: 0 for basin in BASINS}
    for trajectory in trajectories:
        counts[trajectory.basin] += 1
    return normalize_counts(counts)


def entropy(distribution: Mapping[str, float]) -> float:
    return -sum(
        probability * math.log(probability + EPS, 2)
        for probability in distribution.values()
        if probability > 0
    )


def effective_modes(distribution: Mapping[str, float]) -> float:
    """Perplexity-style count of active basins."""

    return 2 ** entropy(distribution)


def kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    return sum(
        p.get(key, 0.0) * math.log((p.get(key, 0.0) + EPS) / (q.get(key, 0.0) + EPS), 2)
        for key in set(p) | set(q)
        if p.get(key, 0.0) > 0
    )


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p) | set(q)
    midpoint = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def event_order_concentration(trajectories: Sequence[Trajectory]) -> float:
    """Return the probability mass of the most common event sequence."""

    if not trajectories:
        return 0.0
    counter = Counter(trajectory.events for trajectory in trajectories)
    return counter.most_common(1)[0][1] / len(trajectories)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_distribution(trajectories: Sequence[Trajectory]) -> Dict[str, float]:
    distribution = basin_distribution(trajectories)
    summary = {
        "entropy_bits": entropy(distribution),
        "effective_modes": effective_modes(distribution),
        "event_order_concentration": event_order_concentration(trajectories),
        "team_return_mean": mean(t.team_return for t in trajectories),
        "individual_conflict_mean": mean(t.individual_conflict for t in trajectories),
        "trigger_rate": mean(1.0 if t.trigger_used else 0.0 for t in trajectories),
    }
    for basin in BASINS:
        summary[f"p_{basin}"] = distribution[basin]
    return summary


def potential_trigger_collapse(
    prior: Sequence[Trajectory],
    trigger: Sequence[Trajectory],
    non_trigger: Sequence[Trajectory],
) -> Dict[str, float]:
    """Measure latent potential before and after a candidate trigger.

    The index is intentionally transparent:
    - potential is the number of effective future basins before intervention.
    - trigger_effect is JS divergence between prior and trigger-conditioned future.
    - collapse is entropy reduction from prior to trigger-conditioned future.
    - trigger_specificity checks that the trigger differs from a non-trigger action.
    """

    prior_dist = basin_distribution(prior)
    trigger_dist = basin_distribution(trigger)
    non_trigger_dist = basin_distribution(non_trigger)
    prior_entropy = entropy(prior_dist)
    trigger_entropy = entropy(trigger_dist)
    non_trigger_entropy = entropy(non_trigger_dist)
    trigger_effect = js_divergence(prior_dist, trigger_dist)
    non_trigger_effect = js_divergence(prior_dist, non_trigger_dist)
    trigger_specificity = js_divergence(trigger_dist, non_trigger_dist)

    return {
        "potential_effective_modes": effective_modes(prior_dist),
        "prior_entropy_bits": prior_entropy,
        "trigger_entropy_bits": trigger_entropy,
        "non_trigger_entropy_bits": non_trigger_entropy,
        "collapse_bits": prior_entropy - trigger_entropy,
        "non_trigger_collapse_bits": prior_entropy - non_trigger_entropy,
        "trigger_effect_js_bits": trigger_effect,
        "non_trigger_effect_js_bits": non_trigger_effect,
        "trigger_specificity_js_bits": trigger_specificity,
        "sacrifice_probability_shift": (
            trigger_dist["sacrifice_rescue"] - prior_dist["sacrifice_rescue"]
        ),
        "macro_predictability_gain": (
            event_order_concentration(trigger) - event_order_concentration(prior)
        ),
        "team_return_gain_after_trigger": (
            mean(t.team_return for t in trigger) - mean(t.team_return for t in prior)
        ),
    }


def compact_metric_row(regime_name: str, metrics: Mapping[str, float]) -> Tuple[str, ...]:
    columns = (
        "regime",
        "potential_effective_modes",
        "collapse_bits",
        "trigger_effect_js_bits",
        "trigger_specificity_js_bits",
        "sacrifice_probability_shift",
        "macro_predictability_gain",
        "team_return_gain_after_trigger",
    )
    values = [regime_name]
    for column in columns[1:]:
        values.append(f"{metrics[column]:.4f}")
    return tuple(values)
