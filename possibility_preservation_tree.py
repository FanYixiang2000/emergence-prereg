"""Possibility preservation benchmark.

This benchmark isolates the mathematical core of the theory:

    locally optimal now can be globally suboptimal later.

The start state offers a tempting cash-out action with the best immediate reward.
The alternative is locally costly, but it preserves an option to adapt after the
future context is revealed. This converts "emergence" into measurable option
value: keeping multiple futures alive can dominate greedy local optimality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ptc_gridworld import EVENT_LIBRARY, Trajectory
from ptc_metrics import (
    potential_trigger_collapse,
    summarize_distribution,
)


POLICIES = (
    "myopic_greedy",
    "always_trigger",
    "always_direct",
    "random_preserve",
    "possibility_preserving",
)

CONTEXTS = ("trigger_needed", "direct_needed")

SUMMARY_COLUMNS = (
    "policy",
    "expected_return",
    "immediate_reward",
    "option_value",
    "local_optimality_gap",
    "potential_effective_modes",
    "natural_trigger_rate",
    "trigger_choice_tension",
    "contextual_success_rate",
    "counterfactual_necessity",
    "retrospective_importance",
    "endogenous_emergence_score",
)


def average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def sample_context(rng: random.Random) -> str:
    return rng.choice(CONTEXTS)


def events_for_basin(basin: str) -> Tuple[str, ...]:
    return EVENT_LIBRARY.get(basin, ("possibility_preserved", "context_revealed"))


def rollout(policy: str, rng: random.Random, forced_action: str | None = None) -> Trajectory:
    """Roll out one finite-horizon tree trajectory.

    forced_action applies after the preserve-option branch:
    - "trigger": force the trigger action.
    - "non_trigger": force the direct action.
    """

    if policy not in POLICIES:
        raise KeyError(f"unknown policy: {policy}")

    context = sample_context(rng)

    if policy == "myopic_greedy":
        return Trajectory(
            regime=policy,
            basin="selfish_escape",
            events=("cash_out_now", "future_options_closed"),
            rewards=(5.0, 0.0),
            team_return=5.0,
            individual_conflict=5.0,
            trigger_used=False,
        )

    # preserve_option is locally costly but reveals the context.
    immediate_reward = -1.0
    if forced_action == "trigger":
        second_action = "trigger"
    elif forced_action == "non_trigger":
        second_action = "direct"
    elif policy == "always_trigger":
        second_action = "trigger"
    elif policy == "always_direct":
        second_action = "direct"
    elif policy == "random_preserve":
        second_action = rng.choice(("trigger", "direct"))
    elif policy == "possibility_preserving":
        second_action = "trigger" if context == "trigger_needed" else "direct"
    else:
        raise ValueError(f"policy cannot preserve: {policy}")

    success = (
        (context == "trigger_needed" and second_action == "trigger")
        or (context == "direct_needed" and second_action == "direct")
    )
    if second_action == "trigger" and success:
        basin = "sacrifice_rescue"
        future_reward = 13.0
        events = (
            "preserve_option",
            "context_reveals_trigger_needed",
            "activation_trigger_selected",
            "high_value_future_realized",
        )
    elif second_action == "direct" and success:
        basin = "team_direct"
        future_reward = 9.0
        events = (
            "preserve_option",
            "context_reveals_direct_needed",
            "direct_path_selected",
            "stable_future_realized",
        )
    else:
        basin = "failed_noise"
        future_reward = 0.0
        events = (
            "preserve_option",
            f"context_reveals_{context}",
            f"wrong_{second_action}_selected",
            "future_option_wasted",
        )

    team_return = immediate_reward + future_reward
    trigger_used = second_action == "trigger" and success
    return Trajectory(
        regime=policy,
        basin=basin,
        events=events_for_basin(basin) if success else events,
        rewards=(immediate_reward, future_reward),
        team_return=team_return,
        individual_conflict=abs(immediate_reward - future_reward),
        trigger_used=trigger_used,
    )


def sample_trajectories(
    policy: str,
    episodes: int,
    seed: int,
    forced_action: str | None = None,
) -> List[Trajectory]:
    rng = random.Random(seed)
    return [rollout(policy, rng, forced_action=forced_action) for _ in range(episodes)]


def local_immediate_reward(policy: str) -> float:
    return 5.0 if policy == "myopic_greedy" else -1.0


def run_policy(policy: str, episodes: int, seed: int) -> Dict[str, object]:
    prior = sample_trajectories(policy, episodes, seed)
    if policy == "myopic_greedy":
        # A greedy cash-out policy has already closed the option. These
        # counterfactuals compare against itself to keep the schema valid.
        trigger = sample_trajectories(policy, episodes, seed + 100_003)
        non_trigger = sample_trajectories(policy, episodes, seed + 200_003)
    else:
        trigger = sample_trajectories(policy, episodes, seed + 100_003, "trigger")
        non_trigger = sample_trajectories(policy, episodes, seed + 200_003, "non_trigger")

    metrics = potential_trigger_collapse(prior, trigger, non_trigger)
    expected_return = average(t.team_return for t in prior)
    immediate_reward = local_immediate_reward(policy)
    natural_trigger_rate = average(1.0 if t.trigger_used else 0.0 for t in prior)
    trigger_choice_tension = 4.0 * natural_trigger_rate * (1.0 - natural_trigger_rate)
    contextual_success_rate = average(
        1.0 if t.basin in ("sacrifice_rescue", "team_direct") else 0.0 for t in prior
    )
    non_success_return = average(
        t.team_return for t in prior if t.basin not in ("sacrifice_rescue", "team_direct")
    )
    success_return = average(
        t.team_return for t in prior if t.basin in ("sacrifice_rescue", "team_direct")
    )
    if contextual_success_rate == 0.0:
        success_return = non_success_return
    retrospective_importance = success_return - non_success_return
    counterfactual_necessity = (
        average(t.team_return for t in trigger)
        - average(t.team_return for t in non_trigger)
    )
    metrics.update(
        {
            "expected_return": expected_return,
            "immediate_reward": immediate_reward,
            "option_value": expected_return - 5.0,
            "local_optimality_gap": 5.0 - immediate_reward,
            "natural_trigger_rate": natural_trigger_rate,
            "trigger_choice_tension": trigger_choice_tension,
            "contextual_success_rate": contextual_success_rate,
            "counterfactual_necessity": counterfactual_necessity,
            "retrospective_importance": retrospective_importance,
            "endogenous_emergence_score": (
                metrics["potential_effective_modes"]
                * trigger_choice_tension
                * contextual_success_rate
            ),
        }
    )
    return {
        "policy": policy,
        "prior": summarize_distribution(prior),
        "trigger": summarize_distribution(trigger),
        "non_trigger": summarize_distribution(non_trigger),
        "ptc": metrics,
    }


def metric_row(policy: str, metrics: Mapping[str, float]) -> Tuple[str, ...]:
    return tuple([policy] + [f"{metrics[column]:.4f}" for column in SUMMARY_COLUMNS[1:]])


def run_all(episodes: int, seed: int, output_dir: Path) -> List[Mapping[str, object]]:
    results = [
        run_policy(policy=policy, episodes=episodes, seed=seed + idx * 10_000)
        for idx, policy in enumerate(POLICIES)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "episodes": episodes, "results": results}
    (output_dir / "possibility_tree_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "possibility_tree_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        for item in results:
            writer.writerow(metric_row(str(item["policy"]), item["ptc"]))  # type: ignore[arg-type]
    print(",".join(SUMMARY_COLUMNS))
    for item in results:
        print(",".join(metric_row(str(item["policy"]), item["ptc"])))  # type: ignore[arg-type]
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Possibility preservation tree.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(episodes=args.episodes, seed=args.seed, output_dir=args.output_dir)
    print(f"\nWrote {args.output_dir / 'possibility_tree_results.json'}")
    print(f"Wrote {args.output_dir / 'possibility_tree_summary.csv'}")


if __name__ == "__main__":
    main()
