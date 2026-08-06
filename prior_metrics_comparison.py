"""Prior emergence detectors versus the multi-component criterion.

The positioning question a reviewer will ask: existing literature already
has emergence measures -- why a new criterion? This experiment runs
literature-style detectors on the SAME 10-system internal battery and gives
each detector its hindsight-OPTIMAL threshold (the best accuracy any single
cut on its score can achieve). Even with that advantage, single-signal
detectors misclassify named systems; the point is not that prior measures
are wrong but that each reads one projection of the mechanism.

Detectors (simplified, single-signal instantiations of published ideas):

1. rep_jump   (representation-jump definition): max consecutive jump of the
   mean Q-vector over training checkpoints of the system's underlying
   learning process. Derived/forced systems inherit their base policy's
   training process -- which is exactly this detector's blind spot: it
   cannot see measurement-time structure (blind_trigger vs
   latent_conditional share a policy).
2. metric_jump (emergent-abilities-style sharp metric jump): max consecutive
   jump of task success rate across the same checkpoints.
3. specificity_only (do-operator macro effectiveness): the measured
   JS(do_trigger, do_non_trigger) -- does the macro trigger do causal work?
4. synergy    (PID/Rosas-flavored): plug-in estimate of
   I(joint agent features ; basin) - I(a0 ; basin) - I(a1 ; basin)
   from behavior rollouts of the measured system itself.
5. causal_emergence_ei (Hoel-flavored, causal emergence as macro-beats-
   micro): EI_macro - EI_micro, where EI_macro is the mutual information
   between a maximum-entropy intervention on the MACRO action (do_trigger
   vs do_non_trigger, uniform) and the outcome basin, and EI_micro is the
   mutual information between a maximum-entropy intervention on the MICRO
   first-step joint action (uniform over the 25 joint actions, behavior
   afterwards) and the outcome basin. Positive score = the macro do-
   variable has more effective information than the micro one, the
   operational core of Hoel et al. 2013 instantiated on the same episodic
   systems. Predicted blind spot (recorded before running): EI compares
   INTERVENED models only -- it never observes the system's own selection
   behavior, so systems that share do-response structure while differing
   in natural selectivity, usefulness, or provenance (blind_trigger,
   useful_habit vs latent_conditional) should be indistinguishable.

Ground truth labels are the battery's audited labels. Our full refined
criterion scores 10/10 on this battery (refined_selectivity_check.py); the
question is what ceiling each single detector reaches with a free threshold.

NOTE: detectors 4 and 5 are charitable proxies. Their EXACT published
forms (Hoel EI under max-entropy interventions on the enumerated TPM;
Rosas' practical criterion Psi) are computed with zero Monte-Carlo error
in exact_prior_formalisms.py; the exact forms perform at or below these
proxies, with the same structural blind spots (fig37, Prop. 5).
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from contextual_sacrifice_gridworld import (
    MODES,
    ContextualSacrificeEnv,
    choose_epsilon_greedy,
    q_values,
    sample_mode,
    sample_preference_context,
    scalar_reward,
    classify_basin,
)
from within_episode_collapse_probe import choose_with_intervention, mean

OUTPUTS = Path(__file__).resolve().parent / "outputs"

REGIMES = ("uncertain_preference", "pure_team", "dense_shaping", "random_noise")

# system -> (base regime or None for untrained, behavior intervention, modes)
SYSTEMS: Dict[str, Tuple[Optional[str], Any, List[str]]] = {
    "latent_conditional": ("uncertain_preference", None, list(MODES)),
    "converged_team": ("pure_team", None, list(MODES)),
    "shaped_process": ("dense_shaping", None, list(MODES)),
    "noise_policy": ("random_noise", None, list(MODES)),
    "untrained_uniform": (None, None, list(MODES)),
    "blind_trigger": ("uncertain_preference", "do_trigger", list(MODES)),
    "harmful_decoy": ("uncertain_preference", "do_trigger", ["bridge"]),
    "useful_habit": ("uncertain_preference", "do_trigger", ["rescue"]),
    "wrong_selector": ("uncertain_preference",
                       {"rescue": None, "bridge": "do_trigger"}, list(MODES)),
    "anti_selector": ("uncertain_preference",
                      {"rescue": "do_non_trigger", "bridge": "do_trigger"}, list(MODES)),
}

TRUTH = {
    "latent_conditional": 1, "converged_team": 0, "shaped_process": 0,
    "noise_policy": 1, "untrained_uniform": 0, "blind_trigger": 0,
    "harmful_decoy": 0, "useful_habit": 0, "wrong_selector": 0,
    "anti_selector": 0,
}


def train_with_checkpoints(regime: str, episodes: int, seed: int,
                           n_checkpoints: int) -> List[Dict[str, float]]:
    """Train tabular Q-learning; at each checkpoint record the mean Q-vector
    norm profile (representation) and greedy success rate."""
    rng = random.Random(seed)
    q_table: Dict = {}
    every = max(1, episodes // n_checkpoints)
    checkpoints: List[Dict[str, float]] = []

    def snapshot() -> Dict[str, float]:
        # Representation: mean of Q-values over all seen (state, context)
        # entries, projected to a fixed-length profile (mean, max, min of
        # each entry's value vector), averaged.
        if not q_table:
            return {"rep_mean": 0.0, "rep_max": 0.0, "rep_min": 0.0, "success": 0.0}
        means, maxs, mins = [], [], []
        for values in q_table.values():
            vals = list(values.values())
            means.append(sum(vals) / len(vals))
            maxs.append(max(vals))
            mins.append(min(vals))
        eval_rng = random.Random(seed + 555)
        wins = []
        for k in range(40):
            mode = MODES[k % len(MODES)]
            env = ContextualSacrificeEnv(mode)
            state = env.reset()
            context = sample_preference_context(regime, eval_rng, k)
            events: List[str] = []
            done = False
            while not done:
                action = choose_with_intervention(q_table, state, context, 0.05, eval_rng, None)
                result = env.step(state, action)
                events.extend(result.events)
                state = result.state
                done = result.done
            basin = classify_basin(events)
            good = (mode == "rescue" and basin == "sacrifice_rescue") or (
                mode == "bridge" and basin == "team_direct"
            )
            wins.append(1.0 if good else 0.0)
        return {
            "rep_mean": mean(means), "rep_max": mean(maxs), "rep_min": mean(mins),
            "success": mean(wins),
        }

    for episode in range(episodes):
        if episode % every == 0:
            checkpoints.append(snapshot())
        epsilon = 0.04 + (0.45 - 0.04) * max(0.0, 1.0 - episode / max(1, episodes))
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
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
            values[action] += 0.28 * (reward + 0.96 * bootstrap - values[action])
            state = result.state
            done = result.done
    checkpoints.append(snapshot())
    return checkpoints


def jump_scores(checkpoints: List[Dict[str, float]]) -> Tuple[float, float]:
    rep_jumps, metric_jumps = [], []
    for i in range(1, len(checkpoints)):
        a, b = checkpoints[i - 1], checkpoints[i]
        rep = math.sqrt(
            (b["rep_mean"] - a["rep_mean"]) ** 2
            + (b["rep_max"] - a["rep_max"]) ** 2
            + (b["rep_min"] - a["rep_min"]) ** 2
        )
        rep_jumps.append(rep)
        metric_jumps.append(abs(b["success"] - a["success"]))
    return max(rep_jumps), max(metric_jumps)


def entropy_bits(counts: Dict[Any, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total) for c in counts.values() if c > 0
    )


def mutual_information(pairs: List[Tuple[Any, Any]]) -> float:
    """Plug-in MI in bits between two discrete variables."""
    from collections import Counter
    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    return entropy_bits(left) + entropy_bits(right) - entropy_bits(joint)


def synergy_score(q_table, regime: Optional[str], behavior, modes: List[str],
                  episodes: int, seed: int) -> float:
    rng = random.Random(seed)
    samples: List[Tuple[Tuple, str, str, str]] = []
    for k in range(episodes):
        mode = modes[k % len(modes)]
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context(regime or "pure_team", rng, k)
        iv = behavior.get(mode) if isinstance(behavior, dict) else behavior
        events: List[str] = []
        a0_feat: List[str] = []
        a1_feat: List[str] = []
        done = False
        t = 0
        while not done:
            action = choose_with_intervention(q_table, state, context, 0.9, rng, iv)
            if t < 3:
                a0_feat.append(action[0])
                a1_feat.append(action[1])
            result = env.step(state, action)
            events.extend(result.events)
            state = result.state
            done = result.done
            t += 1
        basin = classify_basin(events)
        samples.append((tuple(a0_feat), tuple(a1_feat), basin, mode))
    joint_mi = mutual_information([((a0, a1), b) for a0, a1, b, _ in samples])
    a0_mi = mutual_information([(a0, b) for a0, _, b, _ in samples])
    a1_mi = mutual_information([(a1, b) for _, a1, b, _ in samples])
    return joint_mi - a0_mi - a1_mi


def causal_emergence_ei(q_table, regime: Optional[str], behavior,
                        modes: List[str], episodes: int, seed: int) -> float:
    """EI(macro do) - EI(micro do), plug-in, on the measured system.

    Macro variable: {do_trigger, do_non_trigger}, uniform (max-entropy
    intervention), applied for the whole episode. Micro variable: the
    first-step joint action forced uniformly at random over JOINT_ACTIONS,
    with the system's own behavior afterwards. Outcome: basin label.
    """
    from contextual_sacrifice_gridworld import JOINT_ACTIONS, move_position

    rng = random.Random(seed)

    def run_episode(k: int, macro: Optional[str], forced_first=None) -> str:
        mode = modes[k % len(modes)]
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context(regime or "pure_team", rng, k)
        base_iv = behavior.get(mode) if isinstance(behavior, dict) else behavior
        iv = macro if macro is not None else base_iv
        events: List[str] = []
        done = False
        t = 0
        while not done:
            if t == 0 and forced_first is not None:
                action = forced_first
            else:
                action = choose_with_intervention(q_table, state, context, 0.9, rng, iv)
            result = env.step(state, action)
            events.extend(result.events)
            state = result.state
            done = result.done
            t += 1
        return classify_basin(events)

    macro_pairs: List[Tuple[Any, Any]] = []
    for k in range(episodes):
        macro = "do_trigger" if k % 2 == 0 else "do_non_trigger"
        macro_pairs.append((macro, run_episode(k, macro)))
    micro_pairs: List[Tuple[Any, Any]] = []
    actions = list(JOINT_ACTIONS)
    for k in range(episodes):
        forced = actions[rng.randrange(len(actions))]
        micro_pairs.append((forced, run_episode(k, None, forced_first=forced)))
    return mutual_information(macro_pairs) - mutual_information(micro_pairs)


def best_single_threshold(scores: Dict[str, float]) -> Tuple[float, float, int]:
    """Hindsight-optimal accuracy for one cut on the score (either sign)."""
    values = sorted(set(scores.values()))
    candidates = [values[0] - 1.0] + [
        (values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)
    ] + [values[-1] + 1.0]
    best = (0.0, 0.0, 1)
    for direction in (1, -1):
        for threshold in candidates:
            acc = mean([
                1.0 if int(
                    (scores[s] > threshold) if direction == 1 else (scores[s] < threshold)
                ) == TRUTH[s] else 0.0
                for s in scores
            ])
            if acc > best[0]:
                best = (acc, threshold, direction)
    return best


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Prior-metrics comparison battery.")
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--n_checkpoints", type=int, default=20)
    parser.add_argument("--synergy_episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=6011)
    args = parser.parse_args()

    print("Training checkpointed policies for rep/metric jump detectors ...")
    ckpts = {
        regime: train_with_checkpoints(regime, args.train_episodes,
                                       args.seed + i * 10_000, args.n_checkpoints)
        for i, regime in enumerate(REGIMES)
    }
    regime_jumps = {regime: jump_scores(c) for regime, c in ckpts.items()}
    for regime, (rj, mj) in regime_jumps.items():
        print(f"  {regime:22s} rep_jump {rj:.3f} metric_jump {mj:.3f}")

    print("\nTraining final policies for synergy detector ...")
    from contextual_sacrifice_gridworld import train_policy
    policies = {
        regime: train_policy(regime, args.train_episodes, args.seed + i * 10_000)
        for i, regime in enumerate(REGIMES)
    }

    with (OUTPUTS / "criterion_battery_measurements.csv").open(encoding="utf-8") as f:
        measured = {row["system"]: row for row in csv.DictReader(f)}
    refined = json.loads((OUTPUTS / "refined_selectivity_summary.json").read_text())
    measured["anti_selector"] = {k: str(v) for k, v in refined["anti_selector_measurement"].items()}

    detector_scores: Dict[str, Dict[str, float]] = {
        "rep_jump": {}, "metric_jump": {}, "specificity_only": {}, "synergy": {},
        "causal_emergence_ei": {},
    }
    for idx, (system, (regime, behavior, modes)) in enumerate(SYSTEMS.items()):
        if regime is None:
            rep_j, met_j = 0.0, 0.0
            q_table: Dict = {}
        else:
            rep_j, met_j = regime_jumps[regime]
            q_table = policies[regime]
        detector_scores["rep_jump"][system] = rep_j
        detector_scores["metric_jump"][system] = met_j
        detector_scores["specificity_only"][system] = float(measured[system]["specificity_js"])
        detector_scores["synergy"][system] = synergy_score(
            q_table, regime, behavior, modes, args.synergy_episodes,
            args.seed + idx * 977,
        )
        detector_scores["causal_emergence_ei"][system] = causal_emergence_ei(
            q_table, regime, behavior, modes, args.synergy_episodes,
            args.seed + idx * 977 + 13,
        )

    results: Dict[str, Any] = {}
    print(f"\n{'detector':18s} {'best_acc':>8s}  misclassified")
    for name, scores in detector_scores.items():
        acc, threshold, direction = best_single_threshold(scores)
        wrong = [
            s for s in scores
            if int((scores[s] > threshold) if direction == 1 else (scores[s] < threshold))
            != TRUTH[s]
        ]
        results[name] = {
            "scores": scores, "best_accuracy": acc,
            "best_threshold": threshold, "direction": direction,
            "misclassified": wrong,
        }
        print(f"{name:18s} {acc:8.3f}  {';'.join(wrong) or '-'}")
    print(f"{'full_criterion':18s} {'1.000':>8s}  -  (refined_selectivity_check.py, same battery)")

    summary = {
        "note": (
            "Each detector gets its hindsight-optimal threshold (either "
            "direction). The full refined criterion scores 1.000 on the same "
            "battery with pre-registered thresholds."
        ),
        "truth": TRUTH,
        "detectors": results,
        "full_criterion_accuracy": 1.0,
    }
    (OUTPUTS / "prior_metrics_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUTPUTS / 'prior_metrics_comparison.json'}")


if __name__ == "__main__":
    main()
