"""Probe learned representation jumps against future-basin collapse.

This experiment addresses a likely reviewer concern: the previous
representation-jump bridge used controlled basin embeddings, not a learned
agent's internal representation. Here we use the tabular spatial Q learner and
treat fixed-state Q vectors as the learned representation.

At training checkpoints we measure:

    P_k(B)   = rollout distribution over future basins at checkpoint k
    C_k      = KL(P_k(B) || P_0(B))
    B_k      = max(C_k - C_{k-1}, 0)
    R_k      = concatenated Q-values on a fixed probe-state set
    J_k      = ||R_k - R_{k-1}||_2 / sqrt(dim(R))

The key test is whether the regime that scores highest under PTC also shows
alignment between basin-collapse bursts and learned-representation jumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from spatial_sacrifice_gridworld import (
    A0_START,
    A1_START,
    JOINT_ACTIONS,
    REGIMES,
    SAFE_EXIT,
    SWITCH,
    TEAM_A0,
    TEAM_A1,
    SpatialSacrificeEnv,
    choose_epsilon_greedy,
    evaluate_policy,
    q_values,
    scalar_reward,
    stratified_context,
)
from ptc_metrics import potential_trigger_collapse


OUTPUTS = Path(__file__).resolve().parent / "outputs"
BASINS = ("sacrifice_rescue", "team_direct", "selfish_escape", "failed_noise")
PROBE_CONTEXTS = ("fixed", "self_preservation", "visible_teamwork", "latent_sacrifice")
PROBE_STATES = (
    (A0_START, A1_START, False, False, 0),
    (SWITCH, A1_START, True, True, 2),
    (SAFE_EXIT, A1_START, False, False, 4),
    (TEAM_A0, TEAM_A1, False, False, 4),
    ((2, 1), (2, 4), False, False, 3),
    ((3, 1), (3, 4), True, True, 5),
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
        counts[trajectory.basin] = counts.get(trajectory.basin, 0.0) + 1.0
    return normalize(counts)


def representation_vector(q_table: QTable) -> List[float]:
    values: List[float] = []
    for context in PROBE_CONTEXTS:
        for state in PROBE_STATES:
            action_values = q_values(q_table, state, context)  # type: ignore[arg-type]
            values.extend(action_values[action] for action in JOINT_ACTIONS)
    return values


def normalized_l2(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def train_between_checkpoints(
    q_table: QTable,
    regime: str,
    start_episode: int,
    end_episode: int,
    total_episodes: int,
    rng: random.Random,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
) -> None:
    env = SpatialSacrificeEnv()
    for episode in range(start_episode, end_episode):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, 1.0 - episode / max(1, total_episodes)
        )
        context = stratified_context(regime, episode, rng)
        state = env.reset()
        done = False
        while not done:
            action = choose_epsilon_greedy(q_table, state, context, epsilon, rng)  # type: ignore[arg-type]
            result = env.step(state, action)
            reward = scalar_reward(regime, context, state, action, result.rewards, result.events, rng)
            values = q_values(q_table, state, context)  # type: ignore[arg-type]
            if result.done:
                bootstrap = 0.0
            else:
                bootstrap = max(q_values(q_table, result.state, context).values())  # type: ignore[arg-type]
            values[action] += alpha * (reward + gamma * bootstrap - values[action])
            state = result.state
            done = result.done


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
    previous_checkpoint = 0
    previous_rep: Optional[List[float]] = None
    previous_collapse = 0.0
    initial_dist: Optional[Dict[str, float]] = None
    rep_jumps: List[float] = []
    collapse_bursts: List[float] = []

    for checkpoint in checkpoints:
        train_between_checkpoints(
            q_table=q_table,
            regime=regime,
            start_episode=previous_checkpoint,
            end_episode=checkpoint,
            total_episodes=total_episodes,
            rng=rng,
            alpha=0.28,
            gamma=0.96,
            epsilon_start=0.45,
            epsilon_end=0.04,
        )
        dist = basin_distribution(
            q_table=q_table,
            regime=regime,
            episodes=eval_episodes,
            seed=seed + 1_000_003 + checkpoint,
            temperature=eval_temperature,
        )
        if initial_dist is None:
            initial_dist = dict(dist)
        collapse = kl(dist, initial_dist)
        collapse_burst = max(collapse - previous_collapse, 0.0) if rows else 0.0
        rep = representation_vector(q_table)
        rep_jump = normalized_l2(rep, previous_rep) if previous_rep is not None else 0.0
        row: Dict[str, float | str] = {
            "regime": regime,
            "checkpoint": float(checkpoint),
            "collapse_kl": collapse,
            "collapse_burst": collapse_burst,
            "q_representation_jump": rep_jump,
        }
        row.update({f"p_{basin}": dist[basin] for basin in BASINS})
        rows.append(row)
        rep_jumps.append(rep_jump)
        collapse_bursts.append(collapse_burst)
        previous_checkpoint = checkpoint
        previous_rep = rep
        previous_collapse = collapse

    max_burst = max(collapse_bursts)
    max_jump = max(rep_jumps)
    burst_t = float(collapse_bursts.index(max_burst))
    jump_t = float(rep_jumps.index(max_jump))
    alignment = 1.0 / (1.0 + abs(burst_t - jump_t))
    corr = pearson(collapse_bursts, rep_jumps)
    final = rows[-1]
    learned_bridge_score = max_burst * max_jump * max(corr, 0.0) * alignment
    prior = evaluate_policy(
        q_table=q_table,  # type: ignore[arg-type]
        regime=regime,
        episodes=eval_episodes,
        seed=seed + 2_000_003,
        temperature=eval_temperature,
    )
    trigger = evaluate_policy(
        q_table=q_table,  # type: ignore[arg-type]
        regime=regime,
        episodes=eval_episodes,
        seed=seed + 3_000_003,
        temperature=eval_temperature,
        forced_trigger="trigger",
    )
    non_trigger = evaluate_policy(
        q_table=q_table,  # type: ignore[arg-type]
        regime=regime,
        episodes=eval_episodes,
        seed=seed + 4_000_003,
        temperature=eval_temperature,
        forced_trigger="non_trigger",
    )
    ptc = potential_trigger_collapse(prior, trigger, non_trigger)
    natural_trigger_rate = mean(1.0 if trajectory.trigger_used else 0.0 for trajectory in prior)
    trigger_choice_tension = 4.0 * natural_trigger_rate * (1.0 - natural_trigger_rate)
    endogenous_emergence_score = (
        float(ptc["potential_effective_modes"])
        * trigger_choice_tension
        * float(ptc["trigger_specificity_js_bits"])
    )
    ptc_gated_bridge_score = learned_bridge_score * endogenous_emergence_score
    summary = {
        "regime": regime,
        "final_collapse_kl": float(final["collapse_kl"]),
        "max_collapse_burst": max_burst,
        "max_q_representation_jump": max_jump,
        "burst_jump_correlation": corr,
        "peak_alignment": alignment,
        "final_p_sacrifice_rescue": float(final["p_sacrifice_rescue"]),
        "final_p_team_direct": float(final["p_team_direct"]),
        "final_p_selfish_escape": float(final["p_selfish_escape"]),
        "learned_representation_bridge_score": learned_bridge_score,
        "natural_trigger_rate": natural_trigger_rate,
        "trigger_choice_tension": trigger_choice_tension,
        "trigger_specificity_js_bits": float(ptc["trigger_specificity_js_bits"]),
        "potential_effective_modes": float(ptc["potential_effective_modes"]),
        "ptc_endogenous_emergence_score": endogenous_emergence_score,
        "ptc_gated_representation_bridge_score": ptc_gated_bridge_score,
    }
    return rows, summary


def parse_checkpoints(raw: str) -> Tuple[int, ...]:
    checkpoints = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    if not checkpoints or checkpoints[0] != 0:
        checkpoints = (0, *checkpoints)
    return checkpoints


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
        rows, summary = run_regime(
            regime=regime,
            checkpoints=checkpoints,
            eval_episodes=eval_episodes,
            eval_temperature=eval_temperature,
            seed=seed + idx * 20_000,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    with (output_dir / "learned_representation_jump_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    with (output_dir / "learned_representation_jump_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    (output_dir / "learned_representation_jump_summary.json").write_text(
        json.dumps({"summary": summaries}, indent=2),
        encoding="utf-8",
    )
    print("regime,final_collapse,max_burst,max_q_jump,corr,align,rep_score,ptc_score,gated_score")
    for row in summaries:
        print(
            f"{row['regime']},{float(row['final_collapse_kl']):.4f},"
            f"{float(row['max_collapse_burst']):.4f},{float(row['max_q_representation_jump']):.4f},"
            f"{float(row['burst_jump_correlation']):.4f},{float(row['peak_alignment']):.4f},"
            f"{float(row['learned_representation_bridge_score']):.6f},"
            f"{float(row['ptc_endogenous_emergence_score']):.6f},"
            f"{float(row['ptc_gated_representation_bridge_score']):.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learned Q-representation jump probe.")
    parser.add_argument("--regimes", nargs="*", default=["pure_team", "dense_shaping", "uncertain_preference", "random_noise"], choices=list(REGIMES))
    parser.add_argument("--checkpoints", default="0,500,1000,2000,4000,8000")
    parser.add_argument("--eval_episodes", type=int, default=700)
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=313)
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
    print(f"\nWrote {args.output_dir / 'learned_representation_jump_summary.csv'}")
    print(f"Wrote {args.output_dir / 'learned_representation_jump_timeseries.csv'}")


if __name__ == "__main__":
    main()
