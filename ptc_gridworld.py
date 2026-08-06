"""Controlled testbed for Potential-Trigger-Collapse emergence.

The environment is intentionally small. It gives us ground truth basins:
selfish escape, explicit team coordination, sacrifice-triggered rescue, and
unstructured noise. The point is not to train a strong policy yet, but to make
the latent-possibility story measurable before moving into larger MARL systems.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BASINS = (
    "selfish_escape",
    "team_direct",
    "sacrifice_rescue",
    "failed_noise",
)


@dataclass(frozen=True)
class Trajectory:
    """A compact trajectory record used by the metric code."""

    regime: str
    basin: str
    events: Tuple[str, ...]
    rewards: Tuple[float, float]
    team_return: float
    individual_conflict: float
    trigger_used: bool


@dataclass(frozen=True)
class RegimeConfig:
    """Reward/design condition controlling the basin prior."""

    name: str
    logits: Mapping[str, float]
    description: str


REGIMES: Dict[str, RegimeConfig] = {
    "pure_individual": RegimeConfig(
        name="pure_individual",
        logits={
            "selfish_escape": 3.2,
            "team_direct": -0.4,
            "sacrifice_rescue": -3.0,
            "failed_noise": -1.0,
        },
        description="Individual reward dominates; sacrifice is locally irrational.",
    ),
    "pure_team": RegimeConfig(
        name="pure_team",
        logits={
            "selfish_escape": -2.0,
            "team_direct": 1.2,
            "sacrifice_rescue": 3.1,
            "failed_noise": -1.0,
        },
        description="Team reward erases individual conflict; sacrifice becomes expected.",
    ),
    "linear_mixed": RegimeConfig(
        name="linear_mixed",
        logits={
            "selfish_escape": 1.8,
            "team_direct": 1.6,
            "sacrifice_rescue": 0.1,
            "failed_noise": -1.4,
        },
        description="Scalarized reward behaves like a predictable trade-off.",
    ),
    "dense_shaping": RegimeConfig(
        name="dense_shaping",
        logits={
            "selfish_escape": -1.8,
            "team_direct": 3.4,
            "sacrifice_rescue": 0.2,
            "failed_noise": -1.5,
        },
        description="Process rewards directly point to a safe coordination mode.",
    ),
    "uncertain_preference": RegimeConfig(
        name="uncertain_preference",
        logits={
            "selfish_escape": 1.1,
            "team_direct": 0.9,
            "sacrifice_rescue": 1.0,
            "failed_noise": -1.7,
        },
        description="Competing futures stay viable before the trigger is selected.",
    ),
}


EVENT_LIBRARY: Dict[str, Tuple[str, ...]] = {
    "selfish_escape": (
        "a0_keep_resource",
        "a0_exit_safe_lane",
        "a1_face_enemy_alone",
        "team_collects_small_reward",
    ),
    "team_direct": (
        "a0_group_with_a1",
        "both_take_visible_bridge",
        "enemy_pressure_shared",
        "team_collects_medium_reward",
    ),
    "sacrifice_rescue": (
        "a0_step_on_sacrifice_switch",
        "hidden_gate_opens",
        "enemy_locked_on_a0",
        "a1_reaches_high_value_goal",
        "team_collects_delayed_reward",
    ),
    "failed_noise": (
        "agents_split_without_plan",
        "enemy_blocks_middle",
        "no_macro_structure_forms",
        "team_collects_low_reward",
    ),
}


REWARD_LIBRARY: Dict[str, Tuple[float, float]] = {
    "selfish_escape": (6.0, -2.0),
    "team_direct": (3.0, 3.0),
    "sacrifice_rescue": (-5.0, 14.0),
    "failed_noise": (0.0, 0.0),
}


def softmax(logits: Mapping[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = {key: value / temperature for key, value in logits.items()}
    max_logit = max(scaled.values())
    weights = {key: math.exp(value - max_logit) for key, value in scaled.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def sample_categorical(
    probabilities: Mapping[str, float], rng: random.Random
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last_key = ""
    for key, probability in probabilities.items():
        cumulative += probability
        last_key = key
        if threshold <= cumulative:
            return key
    return last_key


def forced_trigger_logits(base_logits: Mapping[str, float]) -> Dict[str, float]:
    """Condition the future on the sacrifice switch being selected."""

    logits = dict(base_logits)
    logits["sacrifice_rescue"] = max(logits.values()) + 3.5
    logits["selfish_escape"] -= 2.2
    logits["team_direct"] -= 0.8
    logits["failed_noise"] -= 1.4
    return logits


def forced_non_trigger_logits(base_logits: Mapping[str, float]) -> Dict[str, float]:
    """Condition the future on avoiding the sacrifice switch."""

    logits = dict(base_logits)
    logits["sacrifice_rescue"] -= 4.0
    logits["selfish_escape"] += 0.8
    logits["team_direct"] += 0.4
    return logits


def rollout(
    regime: RegimeConfig,
    rng: random.Random,
    forced_action: Optional[str] = None,
    temperature: float = 1.0,
) -> Trajectory:
    """Sample one future trajectory from a reward/design condition.

    forced_action can be:
    - None: pre-trigger latent future distribution.
    - "trigger": condition on the sacrifice switch.
    - "non_trigger": condition on avoiding the switch.
    """

    logits = dict(regime.logits)
    if forced_action == "trigger":
        logits = forced_trigger_logits(logits)
    elif forced_action == "non_trigger":
        logits = forced_non_trigger_logits(logits)
    elif forced_action is not None:
        raise ValueError(f"unknown forced_action: {forced_action}")

    probabilities = softmax(logits, temperature=temperature)
    basin = sample_categorical(probabilities, rng)
    rewards = REWARD_LIBRARY[basin]
    team_return = sum(rewards)
    individual_conflict = abs(rewards[0] - rewards[1])
    return Trajectory(
        regime=regime.name,
        basin=basin,
        events=EVENT_LIBRARY[basin],
        rewards=rewards,
        team_return=team_return,
        individual_conflict=individual_conflict,
        trigger_used=basin == "sacrifice_rescue",
    )


def sample_trajectories(
    regime_name: str,
    n: int,
    seed: int,
    forced_action: Optional[str] = None,
    temperature: float = 1.0,
) -> List[Trajectory]:
    if regime_name not in REGIMES:
        known = ", ".join(sorted(REGIMES))
        raise KeyError(f"unknown regime '{regime_name}', expected one of: {known}")
    rng = random.Random(seed)
    regime = REGIMES[regime_name]
    return [
        rollout(
            regime=regime,
            rng=rng,
            forced_action=forced_action,
            temperature=temperature,
        )
        for _ in range(n)
    ]


def all_regime_names() -> Sequence[str]:
    return tuple(REGIMES.keys())


def basin_counts(trajectories: Iterable[Trajectory]) -> Dict[str, int]:
    counts = {basin: 0 for basin in BASINS}
    for trajectory in trajectories:
        counts[trajectory.basin] += 1
    return counts
