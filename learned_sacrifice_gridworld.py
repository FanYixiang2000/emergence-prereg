"""Learnable sacrifice gridworld for PTC evidence.

This file upgrades the first scripted basin generator into a tiny learned MDP.
The policy is still tabular and intentionally transparent, but the basin
distribution now comes from Q-learning under different reward/design regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ptc_gridworld import EVENT_LIBRARY, Trajectory
from ptc_metrics import potential_trigger_collapse, summarize_distribution


START_ACTIONS = (
    "selfish_escape",
    "team_direct",
    "sacrifice_switch",
    "wander",
)

SWITCH_ACTIONS = (
    "rescue_goal",
    "hesitate",
)

CONTEXTS = (
    "fixed",
    "self_preservation",
    "visible_teamwork",
    "latent_sacrifice",
)

SUMMARY_COLUMNS = (
    "regime",
    "potential_effective_modes",
    "natural_trigger_rate",
    "natural_sacrifice_rate",
    "trigger_choice_tension",
    "collapse_bits",
    "trigger_effect_js_bits",
    "trigger_specificity_js_bits",
    "sacrifice_probability_shift",
    "endogenous_emergence_score",
    "macro_predictability_gain",
    "team_return_gain_after_trigger",
)


@dataclass(frozen=True)
class StepResult:
    state: str
    rewards: Tuple[float, float]
    done: bool
    event: str
    basin: Optional[str]


@dataclass(frozen=True)
class RewardRegime:
    name: str
    description: str


REGIMES: Dict[str, RewardRegime] = {
    "pure_individual": RewardRegime(
        name="pure_individual",
        description="Learner optimizes agent-0 local reward only.",
    ),
    "pure_team": RewardRegime(
        name="pure_team",
        description="Learner optimizes total team reward; conflict is erased.",
    ),
    "linear_mixed": RewardRegime(
        name="linear_mixed",
        description="Learner optimizes a fixed scalarization of self and team reward.",
    ),
    "dense_shaping": RewardRegime(
        name="dense_shaping",
        description="Learner receives process shaping toward visible teamwork.",
    ),
    "uncertain_preference": RewardRegime(
        name="uncertain_preference",
        description="Episode-level preference context preserves competing futures.",
    ),
}


QTable = Dict[Tuple[str, str], Dict[str, float]]


class SacrificeMDP:
    """A minimal MDP where a costly switch opens a delayed rescue path."""

    def reset(self) -> str:
        return "start"

    def actions(self, state: str) -> Tuple[str, ...]:
        if state == "start":
            return START_ACTIONS
        if state == "after_switch":
            return SWITCH_ACTIONS
        return ()

    def step(self, state: str, action: str) -> StepResult:
        if state == "start":
            if action == "selfish_escape":
                return StepResult(
                    state="terminal",
                    rewards=(6.0, -2.0),
                    done=True,
                    event="a0_exit_safe_lane",
                    basin="selfish_escape",
                )
            if action == "team_direct":
                return StepResult(
                    state="terminal",
                    rewards=(3.0, 3.0),
                    done=True,
                    event="both_take_visible_bridge",
                    basin="team_direct",
                )
            if action == "sacrifice_switch":
                return StepResult(
                    state="after_switch",
                    rewards=(-2.0, 0.0),
                    done=False,
                    event="a0_step_on_sacrifice_switch",
                    basin=None,
                )
            if action == "wander":
                return StepResult(
                    state="terminal",
                    rewards=(0.0, 0.0),
                    done=True,
                    event="agents_split_without_plan",
                    basin="failed_noise",
                )
        if state == "after_switch":
            if action == "rescue_goal":
                return StepResult(
                    state="terminal",
                    rewards=(-3.0, 14.0),
                    done=True,
                    event="a1_reaches_high_value_goal",
                    basin="sacrifice_rescue",
                )
            if action == "hesitate":
                return StepResult(
                    state="terminal",
                    rewards=(-2.0, 1.0),
                    done=True,
                    event="enemy_blocks_middle",
                    basin="failed_noise",
                )
        raise ValueError(f"invalid transition: state={state}, action={action}")


def sample_context(regime: str, rng: random.Random) -> str:
    if regime != "uncertain_preference":
        return "fixed"
    threshold = rng.random()
    if threshold < 0.34:
        return "self_preservation"
    if threshold < 0.67:
        return "visible_teamwork"
    return "latent_sacrifice"


def scalar_reward(
    regime: str,
    context: str,
    state: str,
    action: str,
    rewards: Tuple[float, float],
) -> float:
    r0, r1 = rewards
    team = r0 + r1
    if regime == "pure_individual":
        return r0
    if regime == "pure_team":
        return team
    if regime == "linear_mixed":
        return 0.65 * r0 + 0.35 * team
    if regime == "dense_shaping":
        shaping = 0.0
        if state == "start" and action == "team_direct":
            shaping += 5.5
        if state == "start" and action == "sacrifice_switch":
            shaping -= 2.0
        return team + shaping
    if regime == "uncertain_preference":
        if context == "self_preservation":
            return r0
        if context == "visible_teamwork":
            shaping = 5.0 if state == "start" and action == "team_direct" else 0.0
            return team + shaping
        if context == "latent_sacrifice":
            delayed_bonus = 4.0 if action == "rescue_goal" else 0.0
            return team + delayed_bonus
    raise ValueError(f"unknown reward regime/context: {regime}/{context}")


def q_values(q_table: QTable, state: str, context: str, actions: Sequence[str]) -> Dict[str, float]:
    key = (state, context)
    if key not in q_table:
        q_table[key] = {action: 0.0 for action in actions}
    for action in actions:
        q_table[key].setdefault(action, 0.0)
    return q_table[key]


def choose_epsilon_greedy(
    q_table: QTable,
    state: str,
    context: str,
    actions: Sequence[str],
    epsilon: float,
    rng: random.Random,
) -> str:
    if rng.random() < epsilon:
        return rng.choice(tuple(actions))
    values = q_values(q_table, state, context, actions)
    max_value = max(values[action] for action in actions)
    best = [action for action in actions if values[action] == max_value]
    return rng.choice(best)


def choose_softmax(
    q_table: QTable,
    state: str,
    context: str,
    actions: Sequence[str],
    temperature: float,
    rng: random.Random,
    forced_start_action: Optional[str] = None,
) -> str:
    if state == "start" and forced_start_action is not None:
        if forced_start_action == "trigger":
            return "sacrifice_switch"
        if forced_start_action == "non_trigger":
            allowed = tuple(action for action in actions if action != "sacrifice_switch")
            return choose_softmax(q_table, state, context, allowed, temperature, rng)
        raise ValueError(f"unknown forced_start_action: {forced_start_action}")

    values = q_values(q_table, state, context, actions)
    if temperature <= 0:
        max_value = max(values[action] for action in actions)
        best = [action for action in actions if values[action] == max_value]
        return rng.choice(best)
    max_value = max(values[action] for action in actions)
    weights = [
        math.exp((values[action] - max_value) / temperature)
        for action in actions
    ]
    total = sum(weights)
    threshold = rng.random()
    cumulative = 0.0
    for action, weight in zip(actions, weights):
        cumulative += weight / total
        if threshold <= cumulative:
            return action
    return actions[-1]


def train_q_policy(
    regime: str,
    episodes: int,
    seed: int,
    alpha: float = 0.25,
    gamma: float = 0.95,
    epsilon_start: float = 0.35,
    epsilon_end: float = 0.03,
) -> QTable:
    if regime not in REGIMES:
        raise KeyError(f"unknown regime: {regime}")
    rng = random.Random(seed)
    env = SacrificeMDP()
    q_table: QTable = {}

    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - episode / max(1, episodes)
        )
        context = sample_context(regime, rng)
        state = env.reset()
        done = False
        while not done:
            actions = env.actions(state)
            action = choose_epsilon_greedy(q_table, state, context, actions, epsilon, rng)
            result = env.step(state, action)
            reward = scalar_reward(regime, context, state, action, result.rewards)
            current_values = q_values(q_table, state, context, actions)
            if result.done:
                bootstrap = 0.0
            else:
                next_actions = env.actions(result.state)
                next_values = q_values(q_table, result.state, context, next_actions)
                bootstrap = max(next_values[action] for action in next_actions)
            current_values[action] += alpha * (
                reward + gamma * bootstrap - current_values[action]
            )
            state = result.state
            done = result.done
    return q_table


def events_for_basin(basin: str, observed_events: Iterable[str]) -> Tuple[str, ...]:
    if basin == "sacrifice_rescue":
        return EVENT_LIBRARY[basin]
    if basin in EVENT_LIBRARY:
        return EVENT_LIBRARY[basin]
    return tuple(observed_events)


def evaluate_policy(
    q_table: QTable,
    regime: str,
    episodes: int,
    seed: int,
    temperature: float,
    forced_start_action: Optional[str] = None,
) -> List[Trajectory]:
    rng = random.Random(seed)
    env = SacrificeMDP()
    trajectories: List[Trajectory] = []
    for _ in range(episodes):
        context = sample_context(regime, rng)
        state = env.reset()
        done = False
        total_rewards = [0.0, 0.0]
        observed_events: List[str] = []
        basin = "failed_noise"
        trigger_used = False
        while not done:
            actions = env.actions(state)
            action = choose_softmax(
                q_table=q_table,
                state=state,
                context=context,
                actions=actions,
                temperature=temperature,
                rng=rng,
                forced_start_action=forced_start_action,
            )
            if state == "start" and action == "sacrifice_switch":
                trigger_used = True
            result = env.step(state, action)
            observed_events.append(result.event)
            total_rewards[0] += result.rewards[0]
            total_rewards[1] += result.rewards[1]
            if result.basin is not None:
                basin = result.basin
            state = result.state
            done = result.done

        rewards = (total_rewards[0], total_rewards[1])
        trajectories.append(
            Trajectory(
                regime=regime,
                basin=basin,
                events=events_for_basin(basin, observed_events),
                rewards=rewards,
                team_return=sum(rewards),
                individual_conflict=abs(rewards[0] - rewards[1]),
                trigger_used=trigger_used and basin == "sacrifice_rescue",
            )
        )
    return trajectories


def run_learned_regime(
    regime: str,
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    eval_temperature: float,
) -> Dict[str, object]:
    q_table = train_q_policy(regime=regime, episodes=train_episodes, seed=seed)
    prior = evaluate_policy(
        q_table=q_table,
        regime=regime,
        episodes=eval_episodes,
        seed=seed + 100_001,
        temperature=eval_temperature,
        forced_start_action=None,
    )
    trigger = evaluate_policy(
        q_table=q_table,
        regime=regime,
        episodes=eval_episodes,
        seed=seed + 200_001,
        temperature=eval_temperature,
        forced_start_action="trigger",
    )
    non_trigger = evaluate_policy(
        q_table=q_table,
        regime=regime,
        episodes=eval_episodes,
        seed=seed + 300_001,
        temperature=eval_temperature,
        forced_start_action="non_trigger",
    )
    metrics = potential_trigger_collapse(prior, trigger, non_trigger)
    natural_trigger_rate = sum(1.0 if t.trigger_used else 0.0 for t in prior) / len(prior)
    natural_sacrifice_rate = sum(
        1.0 if t.basin == "sacrifice_rescue" else 0.0 for t in prior
    ) / len(prior)
    trigger_choice_tension = 4.0 * natural_trigger_rate * (1.0 - natural_trigger_rate)
    # This score is not a final metric; it is a diagnostic for the strongest
    # evidence pattern: multimodal potential, non-saturated endogenous trigger
    # choice, and trigger-specific future reorganization. Saturated trigger use
    # is closer to ordinary team-reward optimization than surprising emergence.
    metrics["natural_trigger_rate"] = natural_trigger_rate
    metrics["natural_sacrifice_rate"] = natural_sacrifice_rate
    metrics["trigger_choice_tension"] = trigger_choice_tension
    metrics["endogenous_emergence_score"] = (
        metrics["potential_effective_modes"]
        * trigger_choice_tension
        * metrics["trigger_specificity_js_bits"]
    )
    return {
        "regime": regime,
        "description": REGIMES[regime].description,
        "prior": summarize_distribution(prior),
        "trigger": summarize_distribution(trigger),
        "non_trigger": summarize_distribution(non_trigger),
        "ptc": metrics,
        "q_start": {
            context: q_table.get(("start", context), {})
            for context in CONTEXTS
            if ("start", context) in q_table
        },
    }


def write_learned_outputs(
    results: Sequence[Mapping[str, object]],
    output_dir: Path,
    train_episodes: int,
    eval_episodes: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "results": list(results),
    }
    (output_dir / "learned_ptc_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "learned_ptc_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        for item in results:
            writer.writerow(learned_metric_row(str(item["regime"]), item["ptc"]))  # type: ignore[arg-type]


def print_summary(results: Sequence[Mapping[str, object]]) -> None:
    print(",".join(SUMMARY_COLUMNS))
    for item in results:
        print(",".join(learned_metric_row(str(item["regime"]), item["ptc"])))  # type: ignore[arg-type]


def learned_metric_row(regime_name: str, metrics: Mapping[str, float]) -> Tuple[str, ...]:
    values = [regime_name]
    for column in SUMMARY_COLUMNS[1:]:
        values.append(f"{metrics[column]:.4f}")
    return tuple(values)


def run_all_learned(
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    eval_temperature: float,
    output_dir: Path,
    regimes: Optional[Sequence[str]] = None,
) -> List[Mapping[str, object]]:
    selected = tuple(regimes) if regimes is not None else tuple(REGIMES)
    results = [
        run_learned_regime(
            regime=regime,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            seed=seed + idx * 10_000,
            eval_temperature=eval_temperature,
        )
        for idx, regime in enumerate(selected)
    ]
    write_learned_outputs(
        results=results,
        output_dir=output_dir,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        seed=seed,
    )
    print_summary(results)
    return results
