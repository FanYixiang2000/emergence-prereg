"""Learned representation-jump probe on the contextual selective-trigger task.

This is a stricter version of `learned_representation_jump_probe.py`. The same
switch is useful in rescue mode but harmful in bridge mode, so a representation
jump only supports emergence if it is paired with selective trigger use and
future-basin collapse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from contextual_sacrifice_gridworld import (
    A0_START,
    A1_START,
    HIGH_GOAL,
    JOINT_ACTIONS,
    MODES,
    PREFERENCE_CONTEXTS,
    REGIMES,
    SAFE_EXIT,
    SWITCH,
    TEAM_A0,
    TEAM_A1,
    ContextualSacrificeEnv,
    choose_epsilon_greedy,
    evaluate_policy,
    q_values,
    sample_mode,
    sample_preference_context,
    scalar_reward,
)
from ptc_metrics import potential_trigger_collapse


OUTPUTS = Path(__file__).resolve().parent / "outputs"
BASINS = ("sacrifice_rescue", "team_direct", "selfish_escape", "failed_noise")
PROBE_STATES = tuple(
    (mode, a0, a1, gate, used, t)
    for mode in MODES
    for a0, a1, gate, used, t in (
        (A0_START, A1_START, False, False, 0),
        (SWITCH, A1_START, True, True, 2),
        (SAFE_EXIT, A1_START, False, False, 4),
        (TEAM_A0, TEAM_A1, False, False, 4),
        ((3, 1), HIGH_GOAL, True, True, 5),
    )
)

QTable = Dict[Tuple[object, str], Dict[Tuple[str, str], float]]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def normalize(counts: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(counts.get(basin, 0.0), 0.0) for basin in BASINS)
    if total <= 0:
        return {basin: 1.0 / len(BASINS) for basin in BASINS}
    return {basin: max(counts.get(basin, 0.0), 0.0) / total for basin in BASINS}


def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    eps = 1e-12
    return sum(p[basin] * math.log((p[basin] + eps) / (q[basin] + eps), 2) for basin in BASINS if p[basin] > 0)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return 0.0
    return numerator / (denom_x * denom_y)


def normalized_l2(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def representation_vector(q_table: QTable) -> List[float]:
    values: List[float] = []
    for context in PREFERENCE_CONTEXTS:
        for state in PROBE_STATES:
            action_values = q_values(q_table, state, context)  # type: ignore[arg-type]
            values.extend(action_values[action] for action in JOINT_ACTIONS)
    return values


def basin_distribution(q_table: QTable, regime: str, episodes: int, seed: int, temperature: float) -> Dict[str, float]:
    trajectories = evaluate_policy(
        q_table=q_table,  # type: ignore[arg-type]
        regime=regime,
        episodes=episodes,
        seed=seed,
        temperature=temperature,
    )
    counts = {basin: 0.0 for basin in BASINS}
    for trajectory in trajectories:
        counts[trajectory.basin] += 1.0
    return normalize(counts)


def train_between_checkpoints(
    q_table: QTable,
    regime: str,
    start_episode: int,
    end_episode: int,
    total_episodes: int,
    rng: random.Random,
) -> None:
    alpha = 0.28
    gamma = 0.96
    epsilon_start = 0.45
    epsilon_end = 0.04
    for episode in range(start_episode, end_episode):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - episode / max(1, total_episodes)
        )
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        preference_context = sample_preference_context(regime, rng, episode)
        done = False
        while not done:
            action = choose_epsilon_greedy(q_table, state, preference_context, epsilon, rng)  # type: ignore[arg-type]
            result = env.step(state, action)
            reward = scalar_reward(regime, preference_context, result.rewards, result.events, rng)
            values = q_values(q_table, state, preference_context)  # type: ignore[arg-type]
            bootstrap = 0.0 if result.done else max(q_values(q_table, result.state, preference_context).values())  # type: ignore[arg-type]
            values[action] += alpha * (reward + gamma * bootstrap - values[action])
            state = result.state
            done = result.done


def final_ptc_scores(q_table: QTable, regime: str, episodes: int, temperature: float, seed: int) -> Dict[str, float]:
    prior = evaluate_policy(q_table, regime, episodes, seed + 10_003, temperature)  # type: ignore[arg-type]
    trigger = evaluate_policy(q_table, regime, episodes, seed + 20_003, temperature, forced_trigger="trigger")  # type: ignore[arg-type]
    non_trigger = evaluate_policy(q_table, regime, episodes, seed + 30_003, temperature, forced_trigger="non_trigger")  # type: ignore[arg-type]
    ptc = potential_trigger_collapse(prior, trigger, non_trigger)
    rescue_success = mean(1.0 if trajectory.basin == "sacrifice_rescue" else 0.0 for trajectory in prior)
    bridge_success = mean(1.0 if trajectory.basin == "team_direct" else 0.0 for trajectory in prior)
    natural_trigger = mean(1.0 if trajectory.trigger_used else 0.0 for trajectory in prior)
    tension = 4.0 * natural_trigger * (1.0 - natural_trigger)
    selective = 4.0 * rescue_success * bridge_success
    endogenous = float(ptc["potential_effective_modes"]) * tension * float(ptc["trigger_specificity_js_bits"])
    return {
        "rescue_success_rate": rescue_success,
        "bridge_success_rate": bridge_success,
        "natural_trigger_rate": natural_trigger,
        "trigger_choice_tension": tension,
        "selective_trigger_score": selective,
        "trigger_specificity_js_bits": float(ptc["trigger_specificity_js_bits"]),
        "potential_effective_modes": float(ptc["potential_effective_modes"]),
        "ptc_endogenous_emergence_score": endogenous,
    }


def run_regime(
    regime: str,
    checkpoints: Sequence[int],
    eval_episodes: int,
    eval_temperature: float,
    seed: int,
) -> Tuple[List[Dict[str, float | str]], Dict[str, float | str]]:
    rng = random.Random(seed)
    q_table: QTable = {}
    total_episodes = max(checkpoints)
    rows: List[Dict[str, float | str]] = []
    initial_dist: Optional[Dict[str, float]] = None
    previous_collapse = 0.0
    previous_rep: Optional[List[float]] = None
    previous_checkpoint = 0
    bursts: List[float] = []
    jumps: List[float] = []

    for checkpoint in checkpoints:
        train_between_checkpoints(q_table, regime, previous_checkpoint, checkpoint, total_episodes, rng)
        dist = basin_distribution(q_table, regime, eval_episodes, seed + checkpoint + 100_003, eval_temperature)
        if initial_dist is None:
            initial_dist = dict(dist)
        collapse = kl(dist, initial_dist)
        burst = max(collapse - previous_collapse, 0.0) if rows else 0.0
        rep = representation_vector(q_table)
        jump = normalized_l2(rep, previous_rep) if previous_rep is not None else 0.0
        row: Dict[str, float | str] = {
            "regime": regime,
            "checkpoint": float(checkpoint),
            "collapse_kl": collapse,
            "collapse_burst": burst,
            "q_representation_jump": jump,
        }
        row.update({f"p_{basin}": dist[basin] for basin in BASINS})
        rows.append(row)
        bursts.append(burst)
        jumps.append(jump)
        previous_checkpoint = checkpoint
        previous_collapse = collapse
        previous_rep = rep

    max_burst = max(bursts)
    max_jump = max(jumps)
    burst_t = float(bursts.index(max_burst))
    jump_t = float(jumps.index(max_jump))
    alignment = 1.0 / (1.0 + abs(burst_t - jump_t))
    corr = pearson(bursts, jumps)
    rep_score = max_burst * max_jump * max(corr, 0.0) * alignment
    ptc = final_ptc_scores(q_table, regime, eval_episodes, eval_temperature, seed)
    gated = rep_score * ptc["selective_trigger_score"] * ptc["ptc_endogenous_emergence_score"]
    summary: Dict[str, float | str] = {
        "regime": regime,
        "final_collapse_kl": float(rows[-1]["collapse_kl"]),
        "max_collapse_burst": max_burst,
        "max_q_representation_jump": max_jump,
        "burst_jump_correlation": corr,
        "peak_alignment": alignment,
        "learned_representation_bridge_score": rep_score,
        "ptc_gated_contextual_bridge_score": gated,
    }
    summary.update(ptc)
    return rows, summary


def parse_checkpoints(raw: str) -> Tuple[int, ...]:
    checkpoints = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    return checkpoints if checkpoints and checkpoints[0] == 0 else (0, *checkpoints)


def run_all(
    regimes: Sequence[str],
    checkpoints: Sequence[int],
    eval_episodes: int,
    eval_temperature: float,
    seed: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, float | str]] = []
    summaries: List[Dict[str, float | str]] = []
    for idx, regime in enumerate(regimes):
        rows, summary = run_regime(regime, checkpoints, eval_episodes, eval_temperature, seed + idx * 50_000)
        all_rows.extend(rows)
        summaries.append(summary)

    with (output_dir / "contextual_learned_representation_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    with (output_dir / "contextual_learned_representation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    (output_dir / "contextual_learned_representation_summary.json").write_text(
        json.dumps({"summary": summaries}, indent=2),
        encoding="utf-8",
    )
    print("regime,rep_score,selective,ptc,gated,rescue,bridge")
    for row in summaries:
        print(
            f"{row['regime']},{float(row['learned_representation_bridge_score']):.6f},"
            f"{float(row['selective_trigger_score']):.4f},"
            f"{float(row['ptc_endogenous_emergence_score']):.6f},"
            f"{float(row['ptc_gated_contextual_bridge_score']):.6f},"
            f"{float(row['rescue_success_rate']):.4f},{float(row['bridge_success_rate']):.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contextual learned representation probe.")
    parser.add_argument("--regimes", nargs="*", default=["pure_team", "dense_shaping", "uncertain_preference", "random_noise"], choices=list(REGIMES))
    parser.add_argument("--checkpoints", default="0,5000,15000,30000,60000")
    parser.add_argument("--eval_episodes", type=int, default=900)
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(
        regimes=args.regimes,
        checkpoints=parse_checkpoints(args.checkpoints),
        eval_episodes=args.eval_episodes,
        eval_temperature=args.eval_temperature,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\nWrote {args.output_dir / 'contextual_learned_representation_summary.csv'}")
    print(f"Wrote {args.output_dir / 'contextual_learned_representation_timeseries.csv'}")


if __name__ == "__main__":
    main()
