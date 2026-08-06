"""Within-episode future-distribution collapse from learned rollouts.

This is the mechanism-level experiment the roadmap repeatedly flags as the
strongest missing evidence. Instead of comparing summary statistics across
training checkpoints, we estimate the future-basin distribution P_t(B | s_t)
at every step of real evaluation episodes by Monte Carlo rollouts under the
learned policy, then test three claims:

1. **Localization**: the largest stepwise collapse
   KL(P_{t+1} || P_t) coincides with the trigger step (stepping on the switch),
   not with an arbitrary step.
2. **Counterfactual usefulness**: at the trigger decision point, rollouts that
   take the trigger reach a different future basin than rollouts forbidden from
   triggering, and in rescue mode the triggered future has higher return.
3. **Harmful collapse control**: in bridge mode the same physical action is a
   decoy; the collapse still happens but the counterfactual return gap flips
   sign, so collapse alone is not evidence of useful emergence.

All distributions come from the learned tabular policy itself; nothing is
hand-scripted at probe time.

Two design points matter for measuring *potential* honestly:

- The observer does not know the latent preference context of an episode, so
  future distributions marginalize over contexts (epistemic openness), while
  the behaving agent follows one sampled context.
- Probe rollouts use a higher softmax temperature than the behaving policy so
  aleatoric openness is not erased by a fully converged deterministic policy.
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
    JOINT_ACTIONS,
    MAX_STEPS,
    MODES,
    REGIMES,
    SWITCH,
    ContextualSacrificeEnv,
    choose_softmax,
    classify_basin,
    move_position,
    q_values,
    sample_preference_context,
    train_policy,
)


def probe_contexts(regime: str) -> Tuple[str, ...]:
    if regime == "uncertain_preference":
        return ("self_preservation", "visible_teamwork", "latent_sacrifice")
    return ("fixed",)


OUTPUTS = Path(__file__).resolve().parent / "outputs"
BASINS = ("sacrifice_rescue", "team_direct", "selfish_escape", "failed_noise")
TRIGGER_EVENTS = ("a0_step_on_sacrifice_switch", "a0_step_on_decoy_switch")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def normalize(counts: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(counts.get(basin, 0.0), 0.0) for basin in BASINS)
    if total <= 0:
        return {basin: 1.0 / len(BASINS) for basin in BASINS}
    return {basin: max(counts.get(basin, 0.0), 0.0) / total for basin in BASINS}


def entropy(p: Mapping[str, float]) -> float:
    eps = 1e-12
    return -sum(p[b] * math.log(p[b] + eps, 2) for b in BASINS if p[b] > 0)


def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    eps = 1e-12
    return sum(p[b] * math.log((p[b] + eps) / (q[b] + eps), 2) for b in BASINS if p[b] > 0)


def js(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    m = {b: 0.5 * (p[b] + q[b]) for b in BASINS}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def choose_with_intervention(
    q_table,
    state,
    context: str,
    temperature: float,
    rng: random.Random,
    intervention: Optional[str],
) -> Tuple[str, str]:
    """Softmax action choice under a minimal do-operator.

    - ``do_trigger``: while the switch is unused, agent 0 must move toward it.
    - ``do_non_trigger``: agent 0 may move freely but never lands on the switch
      cell itself (minimal restriction; ordinary paths are untouched).
    """

    _, a0_pos, _, _, switch_used, _ = state
    values = q_values(q_table, state, context)
    actions = list(JOINT_ACTIONS)
    if intervention == "do_trigger" and not switch_used:
        toward = [
            action for action in actions
            if manhattan(move_position(a0_pos, action[0]), SWITCH) < manhattan(a0_pos, SWITCH)
        ]
        if toward:
            actions = toward
    elif intervention == "do_non_trigger" and not switch_used:
        actions = [
            action for action in actions
            if move_position(a0_pos, action[0]) != SWITCH
        ] or actions

    max_value = max(values[action] for action in actions)
    if temperature <= 0:
        best = [action for action in actions if values[action] == max_value]
        return rng.choice(best)
    weights = [math.exp((values[action] - max_value) / temperature) for action in actions]
    total = sum(weights)
    threshold = rng.random()
    cumulative = 0.0
    for action, weight in zip(actions, weights):
        cumulative += weight / total
        if threshold <= cumulative:
            return action
    return actions[-1]


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def estimate_future(
    q_table,
    env: ContextualSacrificeEnv,
    state,
    contexts: Sequence[str],
    events_so_far: Sequence[str],
    temperature: float,
    samples: int,
    rng: random.Random,
    intervention: Optional[str] = None,
) -> Tuple[Dict[str, float], float]:
    """Monte Carlo estimate of (future-basin distribution, mean team return).

    The distribution marginalizes over the given contexts: the observer does
    not know which latent preference the episode runs under.
    """

    counts = {basin: 0.0 for basin in BASINS}
    returns: List[float] = []
    for sample_idx in range(samples):
        context = contexts[sample_idx % len(contexts)]
        events = list(events_so_far)
        current = state
        total = 0.0
        done = current[5] >= MAX_STEPS
        while not done:
            action = choose_with_intervention(q_table, current, context, temperature, rng, intervention)
            result = env.step(current, action)
            events.extend(result.events)
            total += result.rewards[0] + result.rewards[1]
            current = result.state
            done = result.done
        counts[classify_basin(events)] += 1.0
        returns.append(total)
    return normalize(counts), mean(returns)


def probe_episode(
    q_table,
    regime: str,
    mode: str,
    episode_idx: int,
    temperature: float,
    probe_temperature: float,
    samples: int,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    env = ContextualSacrificeEnv(mode)
    state = env.reset()
    context = sample_preference_context(regime, rng, episode_idx)
    contexts = probe_contexts(regime)
    events: List[str] = []
    step_rows: List[Dict[str, float]] = []
    trigger_t: Optional[int] = None
    counterfactual: Optional[Dict[str, float]] = None

    # Intervention contrast at the initial decision point, computed for every
    # episode: what futures open if we force the trigger versus forbid it?
    do_trigger_dist, do_trigger_return = estimate_future(
        q_table, env, state, contexts, events, probe_temperature, samples, rng,
        intervention="do_trigger",
    )
    do_non_trigger_dist, do_non_trigger_return = estimate_future(
        q_table, env, state, contexts, events, probe_temperature, samples, rng,
        intervention="do_non_trigger",
    )
    intervention = {
        "intervention_js": js(do_trigger_dist, do_non_trigger_dist),
        "intervention_return_gap": do_trigger_return - do_non_trigger_return,
        "do_trigger_p_rescue": do_trigger_dist["sacrifice_rescue"],
        "do_non_trigger_p_rescue": do_non_trigger_dist["sacrifice_rescue"],
    }

    t = 0
    done = False
    while not done:
        dist, _ = estimate_future(
            q_table, env, state, contexts, events, probe_temperature, samples, rng
        )
        action = choose_softmax(q_table, state, context, temperature, rng, None)
        result = env.step(state, action)
        hit_trigger = any(event in result.events for event in TRIGGER_EVENTS)
        if hit_trigger and trigger_t is None:
            trigger_t = t
            factual_dist, factual_return = estimate_future(
                q_table, env, state, contexts, events, probe_temperature, samples, rng,
                intervention="do_trigger",
            )
            cf_dist, cf_return = estimate_future(
                q_table, env, state, contexts, events, probe_temperature, samples, rng,
                intervention="do_non_trigger",
            )
            counterfactual = {
                "counterfactual_js": js(factual_dist, cf_dist),
                "counterfactual_return_gap": factual_return - cf_return,
                "factual_p_rescue": factual_dist["sacrifice_rescue"],
                "cf_p_rescue": cf_dist["sacrifice_rescue"],
            }
        step_rows.append(
            {
                "t": float(t),
                "entropy": entropy(dist),
                "is_trigger_step": 1.0 if hit_trigger else 0.0,
                **{f"p_{basin}": dist[basin] for basin in BASINS},
            }
        )
        events.extend(result.events)
        state = result.state
        done = result.done
        t += 1

    # Stepwise collapse between consecutive pre-step distributions.
    collapses: List[float] = []
    for i in range(1, len(step_rows)):
        p_next = {b: step_rows[i][f"p_{b}"] for b in BASINS}
        p_prev = {b: step_rows[i - 1][f"p_{b}"] for b in BASINS}
        collapses.append(kl(p_next, p_prev))
    max_collapse_step = (collapses.index(max(collapses)) + 1) if collapses else -1
    return {
        "mode": mode,
        "basin": classify_basin(events),
        "trigger_t": trigger_t,
        "steps": step_rows,
        "collapses": collapses,
        "max_collapse_step": max_collapse_step,
        "counterfactual": counterfactual,
        "intervention": intervention,
    }


def summarize(regime: str, mode: str, episodes: Sequence[Mapping[str, object]]) -> Dict[str, float | str]:
    triggered = [ep for ep in episodes if ep["trigger_t"] is not None]
    trigger_collapses: List[float] = []
    non_trigger_collapses: List[float] = []
    localization_hits: List[float] = []
    for ep in triggered:
        collapses: List[float] = ep["collapses"]  # type: ignore[assignment]
        trigger_t: int = ep["trigger_t"]  # type: ignore[assignment]
        # collapse index i corresponds to the change entering step i (post action at i-1)
        post_trigger_index = trigger_t + 1
        for i, value in enumerate(collapses, start=1):
            if i == post_trigger_index:
                trigger_collapses.append(value)
            else:
                non_trigger_collapses.append(value)
        if collapses:
            localization_hits.append(1.0 if ep["max_collapse_step"] == post_trigger_index else 0.0)
    cf_rows = [ep["counterfactual"] for ep in triggered if ep["counterfactual"]]
    iv_rows = [ep["intervention"] for ep in episodes if ep["intervention"]]
    entropy_start = mean(float(ep["steps"][0]["entropy"]) for ep in episodes)  # type: ignore[index]
    entropy_end = mean(float(ep["steps"][-1]["entropy"]) for ep in episodes)  # type: ignore[index]
    return {
        "regime": regime,
        "mode": mode,
        "n_episodes": float(len(episodes)),
        "trigger_rate": len(triggered) / max(len(episodes), 1),
        "initial_future_entropy": entropy_start,
        "final_future_entropy": entropy_end,
        "mean_collapse_at_trigger": mean(trigger_collapses),
        "mean_collapse_elsewhere": mean(non_trigger_collapses),
        "collapse_localization_rate": mean(localization_hits),
        "counterfactual_js": mean(float(row["counterfactual_js"]) for row in cf_rows),
        "counterfactual_return_gap": mean(float(row["counterfactual_return_gap"]) for row in cf_rows),
        "factual_p_rescue": mean(float(row["factual_p_rescue"]) for row in cf_rows),
        "cf_p_rescue": mean(float(row["cf_p_rescue"]) for row in cf_rows),
        "intervention_js": mean(float(row["intervention_js"]) for row in iv_rows),
        "intervention_return_gap": mean(float(row["intervention_return_gap"]) for row in iv_rows),
        "do_trigger_p_rescue": mean(float(row["do_trigger_p_rescue"]) for row in iv_rows),
        "do_non_trigger_p_rescue": mean(float(row["do_non_trigger_p_rescue"]) for row in iv_rows),
    }


def aligned_rows(regime: str, episodes: Sequence[Mapping[str, object]]) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for ep in episodes:
        trigger_t = ep["trigger_t"]
        if trigger_t is None:
            continue
        collapses: List[float] = ep["collapses"]  # type: ignore[assignment]
        for i, value in enumerate(collapses, start=1):
            rows.append(
                {
                    "regime": regime,
                    "mode": str(ep["mode"]),
                    "steps_after_trigger": float(i - (int(trigger_t) + 1)),
                    "stepwise_collapse_kl": value,
                }
            )
    return rows


def run_all(
    regimes: Sequence[str],
    train_episodes: int,
    probe_episodes: int,
    samples: int,
    temperature: float,
    probe_temperature: float,
    seed: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, float | str]] = []
    aligned: List[Dict[str, float | str]] = []
    for reg_idx, regime in enumerate(regimes):
        q_table = train_policy(regime, train_episodes, seed + reg_idx * 10_000)
        for mode in MODES:
            episodes = [
                probe_episode(
                    q_table,
                    regime,
                    mode,
                    episode_idx=ep,
                    temperature=temperature,
                    probe_temperature=probe_temperature,
                    samples=samples,
                    seed=seed + reg_idx * 10_000 + ep * 7 + (0 if mode == "rescue" else 3),
                )
                for ep in range(probe_episodes)
            ]
            summaries.append(summarize(regime, mode, episodes))
            aligned.extend(aligned_rows(regime, episodes))

    with (output_dir / "within_episode_collapse_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    with (output_dir / "within_episode_collapse_aligned.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aligned[0].keys()))
        writer.writeheader()
        for row in aligned:
            writer.writerow(row)
    (output_dir / "within_episode_collapse_summary.json").write_text(
        json.dumps({"summary": summaries}, indent=2),
        encoding="utf-8",
    )
    print("regime,mode,trig_rate,H0,iv_js,iv_return_gap,do_trig_p_rescue,do_nontrig_p_rescue")
    for row in summaries:
        print(
            f"{row['regime']},{row['mode']},{float(row['trigger_rate']):.3f},"
            f"{float(row['initial_future_entropy']):.4f},"
            f"{float(row['intervention_js']):.4f},{float(row['intervention_return_gap']):.4f},"
            f"{float(row['do_trigger_p_rescue']):.4f},{float(row['do_non_trigger_p_rescue']):.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Within-episode future-distribution collapse probe.")
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=["pure_team", "uncertain_preference"],
        choices=list(REGIMES),
    )
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--probe_episodes", type=int, default=40)
    parser.add_argument("--samples", type=int, default=45)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--probe_temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1013)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(
        regimes=args.regimes,
        train_episodes=args.train_episodes,
        probe_episodes=args.probe_episodes,
        samples=args.samples,
        temperature=args.temperature,
        probe_temperature=args.probe_temperature,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\nWrote {args.output_dir / 'within_episode_collapse_summary.csv'}")
    print(f"Wrote {args.output_dir / 'within_episode_collapse_aligned.csv'}")


if __name__ == "__main__":
    main()
