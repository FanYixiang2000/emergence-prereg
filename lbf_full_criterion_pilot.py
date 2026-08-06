"""Exploratory pilot for a full six-component criterion in LBF.

This pilot uses only the three already-probed LBF policy seeds. It is designed
to decide whether a prospectively frozen fresh-seed confirmation is feasible;
it is not confirmatory evidence.

Context is defined without outcome labels from initial geometry:

- food 0 efficient: both agents' summed approach distance is lower for food 0;
- food 1 efficient: the reverse.

The candidate trigger is "food 0 is consumed first". Conditional selectivity
is the difference in that natural trigger probability across the two contexts.
Specificity compares do-commit-to-food0 with do-commit-to-food1. Usefulness is
the context-correct intervention's gain in time-discounted full-clearance
utility. Acquisition is conditional-selectivity gain over the same network at
initialization. Training uses only the environment reward, so endogeneity is a
design/provenance component.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from lbf_collapse_probe import (
    MAX_STEPS,
    Controller,
    FoodIndex,
    PolicyNet,
    PROBE_TEMPERATURE,
    food_positions,
    make_raw_env,
    obs_all,
    world_restore,
    world_snapshot,
)


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

THRESHOLDS = {
    "potential_bits": 0.5,
    "conditional_selectivity": 0.3,
    "specificity_js_bits": 0.1,
    "usefulness": 0.0,
    "acquisition": 0.2,
}


def entropy_bits(counts: Dict[Tuple[int, ...], int]) -> float:
    total = sum(counts.values())
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values() if count > 0
    )


def js_bits(a: Dict[Tuple[int, ...], int],
            b: Dict[Tuple[int, ...], int]) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    out = 0.0
    for key in keys:
        p = a.get(key, 0) / ta
        q = b.get(key, 0) / tb
        m = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log2(p / m)
        if q > 0:
            out += 0.5 * q * math.log2(q / m)
    return out


def approach_cost(env, food: Tuple[int, int]) -> int:
    return sum(
        max(0, abs(player.position[0] - food[0])
            + abs(player.position[1] - food[1]) - 1)
        for player in env.players
    )


def context_for(env, findex: FoodIndex) -> Optional[int]:
    if len(findex.positions) != 2:
        return None
    costs = [approach_cost(env, food) for food in findex.positions]
    if costs[0] == costs[1]:
        return None
    return int(costs[1] < costs[0])  # identity of lower-cost food


def rollout(env, snap, findex: FoodIndex, controller: Controller,
            rng: random.Random, target_idx: Optional[int] = None
            ) -> Tuple[Tuple[int, ...], float]:
    world_restore(env, snap)
    order = list(snap[3])
    before = set(food_positions(env))
    target = findex.positions[target_idx] if target_idx is not None else None
    while env.current_step < MAX_STEPS and env.field.sum() > 0:
        active = None
        if target is not None and target in before:
            active = {0: {"type": "do_commit", "target": target}}
        actions = controller.act(env, obs_all(env), rng, active)
        env.step(actions)
        after = set(food_positions(env))
        if after != before:
            order.extend(findex.consumed_now(before, after))
            before = after
    cleared = len(order) == 2
    utility = (1.0 - env.current_step / MAX_STEPS) if cleared else 0.0
    return tuple(order), utility


def distribution(env, snap, findex: FoodIndex, controller: Controller,
                 rng: random.Random, n: int, target_idx: Optional[int] = None
                 ) -> Dict[str, object]:
    counts: Dict[Tuple[int, ...], int] = {}
    utilities: List[float] = []
    for _ in range(n):
        order, utility = rollout(
            env, snap, findex, controller, rng, target_idx=target_idx)
        counts[order] = counts.get(order, 0) + 1
        utilities.append(utility)
    p_food0_first = sum(
        count for order, count in counts.items() if order and order[0] == 0
    ) / n
    return {
        "counts": counts,
        "p_food0_first": p_food0_first,
        "mean_utility": float(np.mean(utilities)),
    }


def load_trained(seed: int) -> PolicyNet:
    net = PolicyNet()
    net.load_state_dict(torch.load(
        OUTPUTS / f"lbf_net_seed{seed}.pt", map_location="cpu",
        weights_only=True))
    net.eval()
    return net


def initialization_twin(seed: int) -> PolicyNet:
    torch.manual_seed(seed)
    net = PolicyNet()
    net.eval()
    return net


def measure(seed: int, n_states: int, natural_rollouts: int,
            probe_rollouts: int) -> Dict[str, object]:
    trained = load_trained(seed)
    initial = initialization_twin(seed)
    natural = Controller("policy", trained, 1.0)
    natural_init = Controller("policy", initial, 1.0)
    probe = Controller("policy", trained, PROBE_TEMPERATURE)
    rng = random.Random(91_117 * seed + 7)
    env = make_raw_env(seed + 3_000_000)
    sim = make_raw_env(seed + 4_000_000)

    rows: List[Dict[str, object]] = []
    candidate = 0
    episode = 0
    # Oversample because geometrically tied starts are excluded.
    while len(rows) < n_states and candidate < n_states * 10:
        env.reset(seed=5_000_000 + 10_003 * seed + candidate)
        candidate += 1
        findex = FoodIndex(env)
        context = context_for(env, findex)
        if context is None:
            continue
        snap = world_snapshot(env, ())
        d_nat = distribution(
            sim, snap, findex, natural, rng, natural_rollouts)
        d_init = distribution(
            sim, snap, findex, natural_init, rng, natural_rollouts)
        d_probe = distribution(
            sim, snap, findex, probe, rng, probe_rollouts)
        d0 = distribution(
            sim, snap, findex, probe, rng, probe_rollouts, target_idx=0)
        d1 = distribution(
            sim, snap, findex, probe, rng, probe_rollouts, target_idx=1)
        efficient = d0 if context == 0 else d1
        inefficient = d1 if context == 0 else d0
        rows.append({
            "episode": episode,
            "context_efficient_food": context,
            "p_food0_first_trained": d_nat["p_food0_first"],
            "p_food0_first_initial": d_init["p_food0_first"],
            "potential_bits": entropy_bits(d_probe["counts"]),
            "specificity_js_bits": js_bits(d0["counts"], d1["counts"]),
            "utility_do_efficient": efficient["mean_utility"],
            "utility_do_inefficient": inefficient["mean_utility"],
            "usefulness_gap": (
                efficient["mean_utility"] - inefficient["mean_utility"]
            ),
        })
        episode += 1

    by_context = {
        context: [row for row in rows
                  if row["context_efficient_food"] == context]
        for context in (0, 1)
    }
    rates_trained = {
        context: float(np.mean([
            row["p_food0_first_trained"] for row in subset
        ])) for context, subset in by_context.items()
    }
    rates_initial = {
        context: float(np.mean([
            row["p_food0_first_initial"] for row in subset
        ])) for context, subset in by_context.items()
    }
    separation = abs(rates_trained[0] - rates_trained[1])
    init_separation = abs(rates_initial[0] - rates_initial[1])
    metrics = {
        "n_states": len(rows),
        "context_counts": {
            str(context): len(subset) for context, subset in by_context.items()
        },
        "potential_bits": float(np.mean([
            row["potential_bits"] for row in rows
        ])),
        "natural_trigger_rates": {
            str(key): value for key, value in rates_trained.items()
        },
        "initial_trigger_rates": {
            str(key): value for key, value in rates_initial.items()
        },
        "conditional_selectivity": separation,
        "initial_conditional_selectivity": init_separation,
        "acquisition": separation - init_separation,
        "specificity_js_bits": float(np.mean([
            row["specificity_js_bits"] for row in rows
        ])),
        "usefulness_gap": float(np.mean([
            row["usefulness_gap"] for row in rows
        ])),
    }
    passes = {
        "potential": metrics["potential_bits"] >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": (
            metrics["conditional_selectivity"]
            >= THRESHOLDS["conditional_selectivity"]
        ),
        "specificity": (
            metrics["specificity_js_bits"] >= THRESHOLDS["specificity_js_bits"]
        ),
        "usefulness": metrics["usefulness_gap"] > THRESHOLDS["usefulness"],
        "endogeneity": True,
        "acquisition": metrics["acquisition"] >= THRESHOLDS["acquisition"],
    }
    return {
        "metrics": metrics,
        "passes": passes,
        "full_six_component_verdict": int(all(passes.values())),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--natural_rollouts", type=int, default=16)
    parser.add_argument("--probe_rollouts", type=int, default=24)
    parser.add_argument("--tag", default="pilot")
    args = parser.parse_args()

    result = {
        "status": "exploratory pilot on previously measured policy seeds",
        "thresholds": THRESHOLDS,
        "seeds": {},
    }
    for seed in args.seeds:
        print(f"measuring seed {seed}", flush=True)
        measured = measure(
            seed, args.states, args.natural_rollouts, args.probe_rollouts)
        result["seeds"][str(seed)] = measured
        print(json.dumps({
            "metrics": measured["metrics"],
            "passes": measured["passes"],
            "verdict": measured["full_six_component_verdict"],
        }, indent=2), flush=True)
    result["summary"] = {
        "full_verdicts": {
            seed: item["full_six_component_verdict"]
            for seed, item in result["seeds"].items()
        },
        "all_pass": all(
            item["full_six_component_verdict"]
            for item in result["seeds"].values()
        ),
    }
    path = OUTPUTS / f"lbf_full_criterion_{args.tag}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
