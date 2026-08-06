"""Criterion-ablation battery: is every component of the definition necessary?

A definition is hard to attack when two things hold:

1. The full criterion classifies a battery of ground-truth-labeled systems
   correctly.
2. Removing any single component admits at least one named counterexample.

The battery contains seven *measured* systems (learned policies probed with
Monte Carlo rollouts, not hand-written numbers):

- latent_conditional: uncertain-preference learner, natural behavior. The only
  ground-truth emergent system (open potential, selective trigger, useful and
  trigger-specific collapse, no process-level reward guidance).
- converged_team: pure-team learner. Useful and trigger-specific, but the
  possibility space already collapsed during training (low initial entropy).
- shaped_process: dense-shaping learner. Behavior can look similar, but the
  trigger step itself carries an explicit process reward, so the structure is
  prespecified by design.
- noise_policy: random-noise learner. IMPORTANT: a behavioral audit (rescue
  trigger rate 0.985, bridge trigger rate 0.0, per-mode returns 10.8 / 7.7)
  shows this learner genuinely acquired the selective conditional-sacrifice
  structure despite sigma=4 reward noise. Its ground-truth label is therefore
  1 (structure-based labeling), not 0 (regime-name-based labeling). The
  battery records per-mode trigger rates so the label is data-backed.
- untrained_uniform: untrained policy. Maximal openness, no structure.
- blind_trigger: uncertain-preference policy forced to always trigger.
  Saturated trigger choice; usefulness mixes helpful and harmful modes.
- harmful_decoy: forced trigger evaluated only in bridge mode, where the same
  action is a decoy. Collapse without usefulness.

A ninth system pins the usefulness component uniquely:

- wrong_selector: natural (selective) behavior in rescue mode but forced
  triggering in bridge mode. It keeps open potential, mid-range trigger
  tension, and strong trigger-specific collapse -- yet its collapse destroys
  value on average. Only the usefulness component excludes it.

An eighth system pins the selectivity component uniquely:

- useful_habit: forced always-trigger evaluated only in rescue mode, where
  triggering is always beneficial. Open potential (latent contexts), useful,
  trigger-specific -- but zero choice tension. This is the "blind but lucky"
  pattern seen in the external team-reward sacrifice data; our definition
  excludes it because a constant reflex is not a selected response to a
  latent context.

Pre-registered thresholds (fixed before running, recorded here):

    potential:    H0 >= 0.5 bits
    selectivity:  4 p (1 - p) >= 0.5   (natural trigger rate p in ~[0.15, 0.85])
    specificity:  JS(do-trigger, do-non-trigger) >= 0.2 bits
    usefulness:   counterfactual necessity > 0, i.e. the system's own
                  (selective) behavior outperforms the same system forbidden
                  from triggering
    endogeneity:  no explicit process reward on the trigger step (design flag)

Usefulness is deliberately *not* the blind marginal gap between forced
triggering and forbidden triggering: a selective system should be credited for
triggering only where it helps. Counterfactual necessity captures exactly the
value of the emergent selective structure.

The full criterion requires all five. Each ablation drops one component.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from contextual_sacrifice_gridworld import (
    ContextualSacrificeEnv,
    MODES,
    choose_softmax,
    classify_basin,
    sample_preference_context,
    train_policy,
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

COMPONENTS = ("potential", "selectivity", "specificity", "usefulness", "endogeneity")


def run_natural_episode(
    q_table,
    mode: str,
    regime: str,
    episode_idx: int,
    temperature: float,
    rng: random.Random,
    behavior_intervention: Optional[str],
) -> bool:
    """Simulate one behavior episode; return whether the trigger occurred."""

    env = ContextualSacrificeEnv(mode)
    state = env.reset()
    context = sample_preference_context(regime, rng, episode_idx)
    events: List[str] = []
    done = False
    while not done:
        if behavior_intervention is None:
            action = choose_softmax(q_table, state, context, temperature, rng, None)
        else:
            action = choose_with_intervention(
                q_table, state, context, temperature, rng, behavior_intervention
            )
        result = env.step(state, action)
        events.extend(result.events)
        state = result.state
        done = result.done
    return any(event in events for event in TRIGGER_EVENTS)


def measure_system(
    name: str,
    q_table,
    regime: str,
    modes: Sequence[str],
    behavior_intervention: Optional[str],
    prespecified: bool,
    ground_truth: int,
    probe_episodes: int,
    samples: int,
    temperature: float,
    probe_temperature: float,
    seed: int,
) -> Dict[str, float | str | int]:
    contexts = probe_contexts(regime)
    h0_values: List[float] = []
    js_values: List[float] = []
    gap_values: List[float] = []
    trigger_flags: List[float] = []

    def resolve_behavior(mode: str) -> Optional[str]:
        if isinstance(behavior_intervention, dict):
            return behavior_intervention.get(mode)
        return behavior_intervention

    for episode in range(probe_episodes):
        rng = random.Random(seed + episode * 17)
        mode = modes[episode % len(modes)]
        mode_behavior = resolve_behavior(mode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        behavior_dist, behavior_return = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention=mode_behavior,
        )
        h0_values.append(entropy(behavior_dist))
        do_t_dist, _do_t_ret = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_trigger",
        )
        do_n_dist, do_n_ret = estimate_future(
            q_table, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_non_trigger",
        )
        js_values.append(js(do_t_dist, do_n_dist))
        # Counterfactual necessity: the system's own behavior versus the same
        # system forbidden from triggering. Selective systems are credited for
        # triggering only where it helps; blind systems inherit the harm.
        gap_values.append(behavior_return - do_n_ret)
        trigger_flags.append(
            1.0 if run_natural_episode(
                q_table, mode, regime, episode, temperature, rng, mode_behavior
            ) else 0.0
        )
    p_trigger = mean(trigger_flags)
    rescue_flags = [
        flag for episode, flag in enumerate(trigger_flags)
        if modes[episode % len(modes)] == "rescue"
    ]
    bridge_flags = [
        flag for episode, flag in enumerate(trigger_flags)
        if modes[episode % len(modes)] == "bridge"
    ]
    return {
        "system": name,
        "ground_truth_emergent": ground_truth,
        "prespecified": 1 if prespecified else 0,
        "h0_bits": mean(h0_values),
        "natural_trigger_rate": p_trigger,
        "rescue_trigger_rate": mean(rescue_flags),
        "bridge_trigger_rate": mean(bridge_flags),
        "selectivity_tension": 4.0 * p_trigger * (1.0 - p_trigger),
        "specificity_js": mean(js_values),
        "usefulness_gap": mean(gap_values),
    }


def component_passes(row: Mapping[str, float | str | int]) -> Dict[str, bool]:
    return {
        "potential": float(row["h0_bits"]) >= THRESHOLDS["potential_bits"],
        "selectivity": float(row["selectivity_tension"]) >= THRESHOLDS["selectivity_tension"],
        "specificity": float(row["specificity_js"]) >= THRESHOLDS["specificity_js"],
        "usefulness": float(row["usefulness_gap"]) > THRESHOLDS["usefulness_gap"],
        "endogeneity": int(row["prespecified"]) == 0,
    }


def classify(passes: Mapping[str, bool], dropped: Optional[Sequence[str]]) -> int:
    excluded = set(dropped or ())
    return int(all(value for key, value in passes.items() if key not in excluded))


def run_battery(
    train_episodes: int,
    probe_episodes: int,
    samples: int,
    temperature: float,
    probe_temperature: float,
    seed: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = {
        regime: train_policy(regime, train_episodes, seed + idx * 10_000)
        for idx, regime in enumerate(
            ("uncertain_preference", "pure_team", "dense_shaping", "random_noise")
        )
    }
    untrained: Dict = {}

    system_specs = (
        ("latent_conditional", policies["uncertain_preference"], "uncertain_preference",
         list(MODES), None, False, 1),
        ("converged_team", policies["pure_team"], "pure_team",
         list(MODES), None, False, 0),
        ("shaped_process", policies["dense_shaping"], "dense_shaping",
         list(MODES), None, True, 0),
        # Structure-based label: the behavioral audit shows this learner
        # acquired genuine selective triggering despite reward noise.
        ("noise_policy", policies["random_noise"], "random_noise",
         list(MODES), None, False, 1),
        ("untrained_uniform", untrained, "pure_team",
         list(MODES), None, False, 0),
        ("blind_trigger", policies["uncertain_preference"], "uncertain_preference",
         list(MODES), "do_trigger", False, 0),
        ("harmful_decoy", policies["uncertain_preference"], "uncertain_preference",
         ["bridge"], "do_trigger", False, 0),
        ("useful_habit", policies["uncertain_preference"], "uncertain_preference",
         ["rescue"], "do_trigger", False, 0),
        ("wrong_selector", policies["uncertain_preference"], "uncertain_preference",
         list(MODES), {"rescue": None, "bridge": "do_trigger"}, False, 0),
    )

    rows: List[Dict[str, float | str | int]] = []
    for idx, (name, q_table, regime, modes, behavior, prespec, label) in enumerate(system_specs):
        rows.append(
            measure_system(
                name, q_table, regime, modes, behavior, prespec, label,
                probe_episodes=probe_episodes,
                samples=samples,
                temperature=temperature,
                probe_temperature=probe_temperature,
                seed=seed + idx * 5_000,
            )
        )

    matrix_rows: List[Dict[str, float | str | int]] = []
    criteria: List[Tuple[str, Optional[Sequence[str]]]] = (
        [("full", None)]
        + [(f"drop_{component}", [component]) for component in COMPONENTS]
        + [
            # Single-observable baselines: what happens if emergence is defined
            # by just one popular signal?
            ("only_potential", [c for c in COMPONENTS if c != "potential"]),
            ("only_specificity", [c for c in COMPONENTS if c != "specificity"]),
            ("only_usefulness", [c for c in COMPONENTS if c != "usefulness"]),
        ]
    )
    accuracy: Dict[str, float] = {}
    errors: Dict[str, List[str]] = {}
    for criterion_name, dropped in criteria:
        correct = 0
        wrong: List[str] = []
        for row in rows:
            passes = component_passes(row)
            predicted = classify(passes, dropped)
            if predicted == int(row["ground_truth_emergent"]):
                correct += 1
            else:
                wrong.append(str(row["system"]))
        accuracy[criterion_name] = correct / len(rows)
        errors[criterion_name] = wrong

    for row in rows:
        passes = component_passes(row)
        matrix_row: Dict[str, float | str | int] = {
            "system": row["system"],
            "ground_truth_emergent": row["ground_truth_emergent"],
            **{f"pass_{component}": int(passes[component]) for component in COMPONENTS},
            "full_criterion": classify(passes, None),
        }
        matrix_rows.append(matrix_row)

    with (output_dir / "criterion_battery_measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (output_dir / "criterion_battery_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        for row in matrix_rows:
            writer.writerow(row)
    (output_dir / "criterion_battery_summary.json").write_text(
        json.dumps(
            {
                "thresholds": THRESHOLDS,
                "accuracy": accuracy,
                "misclassified": errors,
                "measurements": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("system,truth,H0,tension,js,gap,full_prediction")
    for row, matrix_row in zip(rows, matrix_rows):
        print(
            f"{row['system']},{row['ground_truth_emergent']},{float(row['h0_bits']):.3f},"
            f"{float(row['selectivity_tension']):.3f},{float(row['specificity_js']):.3f},"
            f"{float(row['usefulness_gap']):.3f},{matrix_row['full_criterion']}"
        )
    print("\ncriterion,accuracy,misclassified")
    for criterion_name, _ in criteria:
        wrong = ";".join(errors[criterion_name]) or "-"
        print(f"{criterion_name},{accuracy[criterion_name]:.3f},{wrong}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Criterion ablation battery.")
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--probe_episodes", type=int, default=24)
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--probe_temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=6011)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_battery(
        train_episodes=args.train_episodes,
        probe_episodes=args.probe_episodes,
        samples=args.samples,
        temperature=args.temperature,
        probe_temperature=args.probe_temperature,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\nWrote {args.output_dir / 'criterion_battery_matrix.csv'}")
    print(f"Wrote {args.output_dir / 'criterion_battery_summary.json'}")


if __name__ == "__main__":
    main()
