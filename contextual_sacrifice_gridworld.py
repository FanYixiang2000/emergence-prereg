"""Contextual selective-trigger benchmark.

This benchmark addresses the key weakness of the first spatial task: a policy
can get high return by always sacrificing. Here, the episode mode is visible:

- rescue mode: the switch is locally costly but opens the high-value goal.
- bridge mode: the switch is a decoy; direct teamwork is better.

The target phenomenon is not "learn to sacrifice". It is selective triggering:
a locally costly action is chosen only when it becomes retrospectively important.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ptc_gridworld import EVENT_LIBRARY, Trajectory
from ptc_metrics import potential_trigger_collapse, summarize_distribution


Position = Tuple[int, int]
Mode = str
State = Tuple[Mode, Position, Position, bool, bool, int]
JointAction = Tuple[str, str]
QTable = Dict[Tuple[State, str], Dict[JointAction, float]]


GRID_SIZE = 5
MAX_STEPS = 10
A0_START = (0, 2)
A1_START = (0, 4)
SWITCH = (1, 1)
SAFE_EXIT = (4, 2)
TEAM_A0 = (2, 3)
TEAM_A1 = (2, 4)
HIGH_GOAL = (4, 4)
MODES = ("rescue", "bridge")

MOVES: Dict[str, Position] = {
    "stay": (0, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

JOINT_ACTIONS: Tuple[JointAction, ...] = tuple(
    (a0, a1) for a0 in MOVES for a1 in MOVES
)

REGIMES = (
    "pure_individual",
    "pure_team",
    "linear_mixed",
    "dense_shaping",
    "uncertain_preference",
    "random_noise",
)

PREFERENCE_CONTEXTS = (
    "fixed",
    "self_preservation",
    "visible_teamwork",
    "latent_sacrifice",
)

SUMMARY_COLUMNS = (
    "regime",
    "natural_team_return_mean",
    "rescue_success_rate",
    "bridge_success_rate",
    "over_sacrifice_rate",
    "selective_trigger_score",
    "potential_effective_modes",
    "natural_trigger_rate",
    "trigger_choice_tension",
    "counterfactual_necessity",
    "retrospective_importance",
    "endogenous_emergence_score",
)


@dataclass(frozen=True)
class StepResult:
    state: State
    rewards: Tuple[float, float]
    done: bool
    events: Tuple[str, ...]


class ContextualSacrificeEnv:
    def __init__(self, mode: Mode):
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode

    def reset(self) -> State:
        return (self.mode, A0_START, A1_START, False, False, 0)

    def step(self, state: State, action: JointAction) -> StepResult:
        mode, a0_pos, a1_pos, gate_open, switch_used, t = state
        next_a0 = move_position(a0_pos, action[0])
        next_a1 = move_position(a1_pos, action[1])
        events: List[str] = []
        rewards = [0.0, 0.0]

        if next_a0 == SWITCH and not switch_used:
            switch_used = True
            gate_open = True
            rewards[0] -= 2.0
            if mode == "rescue":
                events.extend(("a0_step_on_sacrifice_switch", "hidden_gate_opens"))
            else:
                events.extend(("a0_step_on_decoy_switch", "decoy_switch_opens"))

        done = False
        if next_a1 == HIGH_GOAL and gate_open:
            if mode == "rescue":
                rewards[0] -= 3.0
                rewards[1] += 16.0
                events.append("a1_reaches_high_value_goal")
            else:
                rewards[0] -= 3.0
                rewards[1] -= 3.0
                events.append("a1_hits_decoy_goal")
            done = True
        elif next_a0 == SAFE_EXIT and not switch_used:
            rewards[0] += 6.0
            rewards[1] -= 2.0
            events.append("a0_reaches_safe_exit")
            done = True
        elif next_a0 == TEAM_A0 and next_a1 == TEAM_A1:
            if mode == "bridge":
                rewards[0] += 4.0
                rewards[1] += 4.0
            else:
                rewards[0] += 2.0
                rewards[1] += 2.0
            events.append("both_take_visible_bridge")
            done = True

        next_t = t + 1
        if next_t >= MAX_STEPS and not done:
            events.append("episode_timeout")
            done = True

        return StepResult(
            state=(mode, next_a0, next_a1, gate_open, switch_used, next_t),
            rewards=(rewards[0], rewards[1]),
            done=done,
            events=tuple(events),
        )


def move_position(pos: Position, move: str) -> Position:
    dx, dy = MOVES[move]
    return (
        min(GRID_SIZE - 1, max(0, pos[0] + dx)),
        min(GRID_SIZE - 1, max(0, pos[1] + dy)),
    )


def sample_mode(rng: random.Random, episode: Optional[int] = None) -> Mode:
    if episode is not None:
        return MODES[episode % len(MODES)]
    return rng.choice(MODES)


def sample_preference_context(regime: str, rng: random.Random, episode: int) -> str:
    if regime != "uncertain_preference":
        return "fixed"
    return (
        "self_preservation",
        "visible_teamwork",
        "latent_sacrifice",
    )[episode % 3]


def scalar_reward(
    regime: str,
    preference_context: str,
    rewards: Tuple[float, float],
    events: Sequence[str],
    rng: random.Random,
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
        shaping = 2.0 if "both_take_visible_bridge" in events else 0.0
        shaping -= 2.0 if "a0_step_on_sacrifice_switch" in events else 0.0
        shaping -= 2.0 if "a0_step_on_decoy_switch" in events else 0.0
        return team + shaping
    if regime == "uncertain_preference":
        if preference_context == "self_preservation":
            return r0
        if preference_context == "visible_teamwork":
            shaping = 2.0 if "both_take_visible_bridge" in events else 0.0
            return team + shaping
        if preference_context == "latent_sacrifice":
            bonus = 4.0 if "a1_reaches_high_value_goal" in events else 0.0
            return team + bonus
    if regime == "random_noise":
        return team + rng.gauss(0.0, 4.0)
    raise ValueError(f"unknown regime/context: {regime}/{preference_context}")


def q_values(q_table: QTable, state: State, context: str) -> Dict[JointAction, float]:
    key = (state, context)
    if key not in q_table:
        q_table[key] = {action: 0.0 for action in JOINT_ACTIONS}
    return q_table[key]


def choose_epsilon_greedy(
    q_table: QTable,
    state: State,
    context: str,
    epsilon: float,
    rng: random.Random,
) -> JointAction:
    if rng.random() < epsilon:
        return rng.choice(JOINT_ACTIONS)
    values = q_values(q_table, state, context)
    max_value = max(values.values())
    best = [action for action, value in values.items() if value == max_value]
    return rng.choice(best)


def choose_softmax(
    q_table: QTable,
    state: State,
    context: str,
    temperature: float,
    rng: random.Random,
    forced_trigger: Optional[str],
) -> JointAction:
    _, a0_pos, _, _, switch_used, _ = state
    values = q_values(q_table, state, context)
    actions = list(JOINT_ACTIONS)
    if forced_trigger == "trigger" and not switch_used:
        actions = [action for action in actions if moves_toward_switch(a0_pos, action[0])]
    elif forced_trigger == "non_trigger" and not switch_used:
        actions = [action for action in actions if not moves_toward_switch(a0_pos, action[0])]
        if not actions:
            actions = list(JOINT_ACTIONS)
    if temperature <= 0:
        max_value = max(values[action] for action in actions)
        best = [action for action in actions if values[action] == max_value]
        return rng.choice(best)
    max_value = max(values[action] for action in actions)
    weights = [math.exp((values[action] - max_value) / temperature) for action in actions]
    total = sum(weights)
    threshold = rng.random()
    cumulative = 0.0
    for action, weight in zip(actions, weights):
        cumulative += weight / total
        if threshold <= cumulative:
            return action
    return actions[-1]


def moves_toward_switch(a0_pos: Position, move: str) -> bool:
    candidate = move_position(a0_pos, move)
    return manhattan(candidate, SWITCH) < manhattan(a0_pos, SWITCH)


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def train_policy(
    regime: str,
    episodes: int,
    seed: int,
    alpha: float = 0.28,
    gamma: float = 0.96,
    epsilon_start: float = 0.45,
    epsilon_end: float = 0.04,
) -> QTable:
    rng = random.Random(seed)
    q_table: QTable = {}
    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - episode / max(1, episodes)
        )
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        preference_context = sample_preference_context(regime, rng, episode)
        done = False
        while not done:
            action = choose_epsilon_greedy(q_table, state, preference_context, epsilon, rng)
            result = env.step(state, action)
            reward = scalar_reward(
                regime, preference_context, result.rewards, result.events, rng
            )
            values = q_values(q_table, state, preference_context)
            if result.done:
                bootstrap = 0.0
            else:
                bootstrap = max(q_values(q_table, result.state, preference_context).values())
            values[action] += alpha * (reward + gamma * bootstrap - values[action])
            state = result.state
            done = result.done
    return q_table


def classify_basin(events: Sequence[str]) -> str:
    event_set = set(events)
    if "a1_reaches_high_value_goal" in event_set and "hidden_gate_opens" in event_set:
        return "sacrifice_rescue"
    if "both_take_visible_bridge" in event_set:
        return "team_direct"
    if "a0_reaches_safe_exit" in event_set:
        return "selfish_escape"
    return "failed_noise"


def canonical_events(basin: str, events: Sequence[str]) -> Tuple[str, ...]:
    if basin in EVENT_LIBRARY:
        return EVENT_LIBRARY[basin]
    return tuple(events)


def evaluate_policy(
    q_table: QTable,
    regime: str,
    episodes: int,
    seed: int,
    temperature: float,
    forced_trigger: Optional[str] = None,
) -> List[Trajectory]:
    rng = random.Random(seed)
    trajectories: List[Trajectory] = []
    for episode in range(episodes):
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        preference_context = sample_preference_context(regime, rng, episode)
        events: List[str] = []
        total_rewards = [0.0, 0.0]
        done = False
        while not done:
            action = choose_softmax(
                q_table, state, preference_context, temperature, rng, forced_trigger
            )
            result = env.step(state, action)
            events.extend(result.events)
            total_rewards[0] += result.rewards[0]
            total_rewards[1] += result.rewards[1]
            state = result.state
            done = result.done
        basin = classify_basin(events)
        rewards = (total_rewards[0], total_rewards[1])
        trajectories.append(
            Trajectory(
                regime=regime,
                basin=basin,
                events=canonical_events(basin, events),
                rewards=rewards,
                team_return=sum(rewards),
                individual_conflict=abs(rewards[0] - rewards[1]),
                trigger_used=(
                    "a0_step_on_sacrifice_switch" in events
                    and basin == "sacrifice_rescue"
                ),
            )
        )
    return trajectories


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_regime(
    regime: str,
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    eval_temperature: float,
) -> Dict[str, object]:
    q_table = train_policy(regime, train_episodes, seed)
    prior = evaluate_policy(q_table, regime, eval_episodes, seed + 100_003, eval_temperature)
    trigger = evaluate_policy(
        q_table,
        regime,
        eval_episodes,
        seed + 200_003,
        eval_temperature,
        forced_trigger="trigger",
    )
    non_trigger = evaluate_policy(
        q_table,
        regime,
        eval_episodes,
        seed + 300_003,
        eval_temperature,
        forced_trigger="non_trigger",
    )
    metrics = potential_trigger_collapse(prior, trigger, non_trigger)
    natural_trigger_rate = average(1.0 if t.trigger_used else 0.0 for t in prior)
    rescue_success_rate = average(1.0 if t.basin == "sacrifice_rescue" else 0.0 for t in prior)
    bridge_success_rate = average(1.0 if t.basin == "team_direct" else 0.0 for t in prior)
    over_sacrifice_rate = average(
        1.0 if t.basin == "failed_noise" and t.rewards[0] < 0 else 0.0 for t in prior
    )
    trigger_choice_tension = 4.0 * natural_trigger_rate * (1.0 - natural_trigger_rate)
    natural_team_return_mean = average(t.team_return for t in prior)
    non_sacrifice_return = average(t.team_return for t in prior if t.basin != "sacrifice_rescue")
    sacrifice_return = average(t.team_return for t in prior if t.basin == "sacrifice_rescue")
    if rescue_success_rate == 0.0:
        sacrifice_return = non_sacrifice_return
    retrospective_importance = sacrifice_return - non_sacrifice_return
    counterfactual_necessity = (
        average(t.team_return for t in trigger)
        - average(t.team_return for t in non_trigger)
    )
    selective_trigger_score = rescue_success_rate * bridge_success_rate * 4.0
    metrics.update(
        {
            "natural_team_return_mean": natural_team_return_mean,
            "rescue_success_rate": rescue_success_rate,
            "bridge_success_rate": bridge_success_rate,
            "over_sacrifice_rate": over_sacrifice_rate,
            "selective_trigger_score": selective_trigger_score,
            "natural_trigger_rate": natural_trigger_rate,
            "trigger_choice_tension": trigger_choice_tension,
            "counterfactual_necessity": counterfactual_necessity,
            "retrospective_importance": retrospective_importance,
            "endogenous_emergence_score": (
                metrics["potential_effective_modes"]
                * trigger_choice_tension
                * metrics["trigger_specificity_js_bits"]
            ),
        }
    )
    return {
        "regime": regime,
        "prior": summarize_distribution(prior),
        "trigger": summarize_distribution(trigger),
        "non_trigger": summarize_distribution(non_trigger),
        "ptc": metrics,
    }


def metric_row(regime: str, metrics: Mapping[str, float]) -> Tuple[str, ...]:
    return tuple([regime] + [f"{metrics[column]:.4f}" for column in SUMMARY_COLUMNS[1:]])


def run_all(
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    eval_temperature: float,
    output_dir: Path,
    regimes: Sequence[str],
) -> List[Mapping[str, object]]:
    results = [
        run_regime(
            regime=regime,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            seed=seed + idx * 10_000,
            eval_temperature=eval_temperature,
        )
        for idx, regime in enumerate(regimes)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "results": results,
    }
    (output_dir / "contextual_ptc_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "contextual_ptc_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        for item in results:
            writer.writerow(metric_row(str(item["regime"]), item["ptc"]))  # type: ignore[arg-type]
    print(",".join(SUMMARY_COLUMNS))
    for item in results:
        print(",".join(metric_row(str(item["regime"]), item["ptc"])))  # type: ignore[arg-type]
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contextual selective-trigger benchmark.")
    parser.add_argument("--train_episodes", type=int, default=100000)
    parser.add_argument("--eval_episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--regimes", nargs="*", default=list(REGIMES), choices=list(REGIMES))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        eval_temperature=args.eval_temperature,
        output_dir=args.output_dir,
        regimes=args.regimes,
    )
    print(f"\nWrote {args.output_dir / 'contextual_ptc_results.json'}")
    print(f"Wrote {args.output_dir / 'contextual_ptc_summary.csv'}")


if __name__ == "__main__":
    main()
