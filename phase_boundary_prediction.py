"""Prospective phase-boundary prediction: the criterion as a predictive theory.

A definition that only classifies after the fact is a taxonomy. A definition
that predicts *where* emergence appears as a control parameter varies is a
theory. This experiment fixes a control parameter, derives the phase
boundaries in closed form BEFORE training anything, and then trains
independent learners across the parameter grid to test the predictions.

Control parameter
-----------------
G = the reward delivered to agent 1 when it reaches the high-value goal in
rescue mode (the internal benchmark hardcodes G = 16). Everything else about
the contextual sacrifice environment is unchanged.

Closed-form payoff accounting (from the environment source, not from data):

    trigger path in rescue mode:  team return = (-2 switch) + (-3 + G) = G - 5
    best non-trigger alternative: bridge = 2 + 2 = 4  (= safe exit 6 - 2 = 4)
    trigger path in bridge mode:  team return = -8 (decoy), independent of G

Training contexts of the uncertain_preference regime value the trigger as:

    self_preservation:  r0 = -5 vs +6  -> never triggers, any G
    latent_sacrifice:   (G - 5) + 4 bonus vs 4    -> triggers iff G > 5
    visible_teamwork:   (G - 5) vs 4 + 2 shaping  -> triggers iff G > 11

Pre-registered predictions (before any training run; contexts rotate 1/3
each, modes alternate rescue/bridge 1/2 each; thresholds copied unchanged
from criterion_ablation_battery):

    P1  Behavioral onset: natural trigger rate jumps from ~0 to ~1/6
        strictly between G = 5 and the next grid point (latent_sacrifice
        context starts triggering in rescue mode).
    P2  Usefulness sign: the counterfactual-necessity gap (behavior minus
        do_non_trigger, in team return) crosses zero at G = 9
        (G - 5 vs the alternative 4). Negative for G < 9, positive for
        G > 9.
    P3  Criterion acceptance: the full five-component verdict flips from
        reject to accept exactly at the usefulness boundary: reject for
        G <= 9, accept for G > 9. Rejection route for G <= 5 is
        selectivity (tension < 0.5 at trigger rate ~0); rejection route
        for 5 < G <= 9 is usefulness (selective but value-destroying,
        the wrong_selector pattern arising naturally from payoffs).
    P4  Second onset: near G > 11 the visible_teamwork context also starts
        triggering, lifting the natural trigger rate toward ~1/3.

    Grid: G in {3, 5, 7, 9, 11, 13, 16}. Boundary points G = 5, 9, 11 are
    ties in the closed-form accounting; learning noise decides them, and we
    do not count them for or against the predictions. Tension at trigger
    rate 1/6 is 4*(1/6)*(5/6) = 0.556, only 0.056 above the 0.5 threshold,
    so P3's middle phase is the most fragile registered prediction.

Everything below measures; nothing below re-derives or adjusts predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from contextual_sacrifice_gridworld import (
    HIGH_GOAL,
    MAX_STEPS,
    MODES,
    SAFE_EXIT,
    SWITCH,
    TEAM_A0,
    TEAM_A1,
    ContextualSacrificeEnv,
    StepResult,
    choose_epsilon_greedy,
    move_position,
    q_values,
    sample_mode,
    sample_preference_context,
    scalar_reward,
)
from within_episode_collapse_probe import (
    TRIGGER_EVENTS,
    choose_with_intervention,
    entropy,
    estimate_future,
    js,
    mean,
    probe_contexts,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

THRESHOLDS = {
    "potential_bits": 0.5,
    "selectivity_tension": 0.5,
    "specificity_js": 0.2,
    "usefulness_gap": 0.0,
}


class ParamGoalEnv(ContextualSacrificeEnv):
    """Contextual sacrifice env with a parametrized high-goal reward G."""

    def __init__(self, mode: str, goal_reward: float):
        super().__init__(mode)
        self.goal_reward = goal_reward

    def step(self, state, action):  # type: ignore[override]
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
                rewards[1] += self.goal_reward
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


def train_param_policy(goal_reward: float, episodes: int, seed: int,
                       alpha: float = 0.28, gamma: float = 0.96,
                       epsilon_start: float = 0.45, epsilon_end: float = 0.04):
    regime = "uncertain_preference"
    rng = random.Random(seed)
    q_table: Dict = {}
    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - episode / max(1, episodes)
        )
        mode = sample_mode(rng, episode)
        env = ParamGoalEnv(mode, goal_reward)
        state = env.reset()
        context = sample_preference_context(regime, rng, episode)
        done = False
        while not done:
            action = choose_epsilon_greedy(q_table, state, context, epsilon, rng)
            result = env.step(state, action)
            reward = scalar_reward(regime, context, result.rewards, result.events, rng)
            values = q_values(q_table, state, context)
            bootstrap = 0.0 if result.done else max(
                q_values(q_table, result.state, context).values()
            )
            values[action] += alpha * (reward + gamma * bootstrap - values[action])
            state = result.state
            done = result.done
    return q_table


def natural_trigger(q_table, goal_reward: float, mode: str, episode_idx: int,
                    temperature: float, rng: random.Random) -> bool:
    env = ParamGoalEnv(mode, goal_reward)
    state = env.reset()
    context = sample_preference_context("uncertain_preference", rng, episode_idx)
    events: List[str] = []
    done = False
    while not done:
        action = choose_with_intervention(q_table, state, context, temperature, rng, None)
        result = env.step(state, action)
        events.extend(result.events)
        state = result.state
        done = result.done
    return any(event in events for event in TRIGGER_EVENTS)


def measure_point(goal_reward: float, train_episodes: int, probe_episodes: int,
                  samples: int, temperature: float, probe_temperature: float,
                  seed: int) -> Dict[str, float]:
    q_table = train_param_policy(goal_reward, train_episodes, seed)
    contexts = probe_contexts("uncertain_preference")
    h0_values: List[float] = []
    js_values: List[float] = []
    gap_values: List[float] = []
    flags: List[float] = []
    for episode in range(probe_episodes):
        rng = random.Random(seed + episode * 17)
        mode = MODES[episode % len(MODES)]
        env = ParamGoalEnv(mode, goal_reward)
        state = env.reset()
        behavior_dist, behavior_return = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention=None,
        )
        h0_values.append(entropy(behavior_dist))
        do_t_dist, _ = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_trigger",
        )
        do_n_dist, do_n_ret = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_non_trigger",
        )
        js_values.append(js(do_t_dist, do_n_dist))
        gap_values.append(behavior_return - do_n_ret)
        flags.append(
            1.0 if natural_trigger(q_table, goal_reward, mode, episode, temperature, rng)
            else 0.0
        )
    p = mean(flags)
    row = {
        "goal_reward": goal_reward,
        "h0_bits": mean(h0_values),
        "natural_trigger_rate": p,
        "selectivity_tension": 4.0 * p * (1.0 - p),
        "specificity_js": mean(js_values),
        "usefulness_gap": mean(gap_values),
    }
    passes = {
        "potential": row["h0_bits"] >= THRESHOLDS["potential_bits"],
        "selectivity": row["selectivity_tension"] >= THRESHOLDS["selectivity_tension"],
        "specificity": row["specificity_js"] >= THRESHOLDS["specificity_js"],
        "usefulness": row["usefulness_gap"] > THRESHOLDS["usefulness_gap"],
        "endogeneity": True,
    }
    row.update({f"pass_{k}": int(v) for k, v in passes.items()})
    row["accepted"] = int(all(passes.values()))
    return row


def predicted_verdict(goal_reward: float) -> Optional[int]:
    """Closed-form prediction; None marks registered boundary ties."""
    if goal_reward in (5.0, 9.0, 11.0):
        return None
    return int(goal_reward > 9.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prospective phase-boundary prediction.")
    parser.add_argument("--goal_rewards", type=str, default="3,5,7,9,11,13,16")
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--probe_episodes", type=int, default=24)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--probe_temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=8011)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    grid = [float(x) for x in args.goal_rewards.split(",") if x.strip()]
    rows: List[Dict[str, float]] = []
    for idx, goal_reward in enumerate(grid):
        print(f"=== G = {goal_reward} (training {args.train_episodes} episodes) ===")
        row = measure_point(
            goal_reward, args.train_episodes, args.probe_episodes, args.samples,
            args.temperature, args.probe_temperature, args.seed + idx * 10_000,
        )
        predicted = predicted_verdict(goal_reward)
        row["predicted_accept"] = -1 if predicted is None else predicted
        row["prediction_match"] = (
            -1 if predicted is None else int(int(row["accepted"]) == predicted)
        )
        rows.append(row)
        print(
            f"  rate {row['natural_trigger_rate']:.3f} tension {row['selectivity_tension']:.3f} "
            f"gap {row['usefulness_gap']:+.3f} accepted {row['accepted']} "
            f"predicted {row['predicted_accept']}"
        )

    non_tie = [row for row in rows if row["predicted_accept"] != -1]
    match_rate = mean([float(row["prediction_match"]) for row in non_tie])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "phase_boundary_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (args.output_dir / "phase_boundary_summary.json").write_text(
        json.dumps(
            {
                "thresholds": THRESHOLDS,
                "registered_boundaries": {
                    "behavioral_onset_G": 5.0,
                    "usefulness_sign_G": 9.0,
                    "second_onset_G": 11.0,
                },
                "non_tie_match_rate": match_rate,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nG,rate,tension,gap,accepted,predicted,match")
    for row in rows:
        print(
            f"{row['goal_reward']},{row['natural_trigger_rate']:.3f},"
            f"{row['selectivity_tension']:.3f},{row['usefulness_gap']:+.3f},"
            f"{row['accepted']},{row['predicted_accept']},{row['prediction_match']}"
        )
    print(f"\nNon-tie prediction match rate: {match_rate:.3f}")
    print(f"Wrote {args.output_dir / 'phase_boundary_summary.json'}")


if __name__ == "__main__":
    main()
