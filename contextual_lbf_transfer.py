"""Contextual LBF pilot and fresh-seed six-component transfer.

This uses the unmodified LBF dynamics, observations, sparse collection reward
and PPO implementation. The wrapper balances two finite-horizon layouts:
food 0 is initially nearer to the team, or food 1 is nearer. The context label
is never supplied to the controller; geometry is visible through the standard
observation. The trigger is collecting food 0 first.

Runs tagged ``pilot`` are design-feasibility analyses. A confirmatory run must
use fresh seeds after CONTEXTUAL_LBF_PREREGISTRATION.md is frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import lbforaging  # noqa: F401
import numpy as np
import torch

import lbf_collapse_probe as base


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
ENV_ID = "Foraging-5x5-2p-2f-coop-v3"
HORIZON = 15
CONTEXTS = (0, 1)  # identity of the geometrically nearer food
BASINS = (
    "win_food0", "win_food1", "loss_food0", "loss_food1",
)
THRESHOLDS = {
    "potential_bits": 0.5,
    "conditional_selectivity": 0.5,
    "specificity_js_bits": 0.2,
    "usefulness_gap": 0.0,
    "acquisition": 0.3,
}

# (players, food0, food1); food identities are lexicographic positions, as in
# the original LBF probe. Two layouts per context avoid a single fixed board.
LAYOUTS = {
    0: (
        (((1, 1), (2, 1)), (1, 2), (3, 3)),
        (((2, 3), (3, 3)), (2, 2), (3, 1)),
    ),
    1: (
        (((2, 2), (3, 2)), (1, 1), (3, 3)),
        (((1, 3), (2, 3)), (1, 1), (2, 2)),
    ),
}


def safe_actions(env, actions: Sequence[int]) -> Tuple[int, ...]:
    safe: List[int] = []
    for player, action in zip(env.players, actions):
        valid = {candidate.value for candidate in env._valid_actions[player]}
        if action not in valid:
            action = 0
        if action == 5 and env.adjacent_food_location(*player.position) is None:
            action = 0
        safe.append(action)
    return tuple(safe)


def make_contextual_env(seed: int, context: Optional[int] = None):
    env = gym.make(ENV_ID).unwrapped
    original_reset = env.reset
    original_step = env.step
    env.context_override = context

    def contextual_reset(self, *, seed: Optional[int] = None, options=None):
        actual_seed = int(seed if seed is not None else 0)
        original_reset(seed=actual_seed, options=options)
        ctx = (
            int(self.context_override)
            if self.context_override is not None
            else actual_seed % 2
        )
        layout = LAYOUTS[ctx][(actual_seed // 2) % len(LAYOUTS[ctx])]
        players, food0, food1 = layout
        self.field.fill(0)
        self.field[food0] = 2
        self.field[food1] = 2
        for player, position in zip(self.players, players):
            player.position = position
            player.level = 1
            player.reward = 0
        self.current_step = 0
        self._max_episode_steps = HORIZON
        self._game_over = False
        self.context = ctx
        self._gen_valid_moves()
        return self._make_gym_obs(), {}

    env.reset = types.MethodType(contextual_reset, env)

    def contextual_step(self, actions):
        return original_step(safe_actions(self, actions))

    env.step = types.MethodType(contextual_step, env)
    env.reset(seed=seed)
    return env


def entropy(dist: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def js(p: Dict[str, float], q: Dict[str, float]) -> float:
    out = 0.0
    for key in set(p) | set(q):
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {basin: counts.get(basin, 0) / total for basin in BASINS}


class TeamController:
    def __init__(self, kind: str, net: Optional[base.PolicyNet] = None):
        self.kind = kind
        self.net = net
        self.policy = (
            base.Controller("policy", net, 1.0) if net is not None else None
        )

    def target_index(self, env, findex: base.FoodIndex) -> int:
        live = {
            index for index, position in enumerate(findex.positions)
            if position in set(base.food_positions(env))
        }
        if not live:
            return 0
        if self.kind == "fixed_food0" and 0 in live:
            return 0
        if self.kind == "fixed_food1" and 1 in live:
            return 1
        costs = []
        for index, food in enumerate(findex.positions):
            cost = sum(
                abs(player.position[0] - food[0])
                + abs(player.position[1] - food[1])
                for player in env.players
            )
            costs.append(cost if index in live else float("inf"))
        return int(np.argmin(costs))

    def act(self, env, obs, rng: random.Random, findex: base.FoodIndex,
            forced_target: Optional[int]) -> Tuple[int, ...]:
        if forced_target is not None:
            target = findex.positions[forced_target]
            return cooperative_actions(env, target)
        if self.policy is not None:
            return self.policy.act(env, obs, rng)
        target = findex.positions[self.target_index(env, findex)]
        return cooperative_actions(env, target)


def move_toward_cell(env, agent_idx: int, goal: Tuple[int, int]) -> int:
    player = env.players[agent_idx]
    if player.position == goal:
        return 0
    row, col = player.position
    candidates = []
    for action, (dr, dc) in {
        1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1),
    }.items():
        distance = abs(row + dr - goal[0]) + abs(col + dc - goal[1])
        candidates.append((distance, action))
    valid = {action.value for action in env._valid_actions[player]}
    for _distance, action in sorted(candidates):
        if action in valid:
            return action
    return 0


def cooperative_actions(env, target: Tuple[int, int]) -> Tuple[int, ...]:
    if all(base.adjacent(player.position, target) for player in env.players):
        return tuple(5 for _ in env.players)
    adjacent_cells = [
        (target[0] + dr, target[1] + dc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= target[0] + dr < 5 and 0 <= target[1] + dc < 5
    ]
    best = None
    for first in adjacent_cells:
        for second in adjacent_cells:
            if first == second:
                continue
            cost = (
                abs(env.players[0].position[0] - first[0])
                + abs(env.players[0].position[1] - first[1])
                + abs(env.players[1].position[0] - second[0])
                + abs(env.players[1].position[1] - second[1])
            )
            candidate = (cost, first, second)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return (0, 0)
    return (
        move_toward_cell(env, 0, best[1]),
        move_toward_cell(env, 1, best[2]),
    )


def run_episode(controller: TeamController, context: int, seed: int,
                intervention: Optional[str]) -> Dict[str, Any]:
    env = make_contextual_env(seed, context)
    findex = base.FoodIndex(env)
    rng = random.Random(seed * 104_729 + 17)
    before = set(base.food_positions(env))
    order: List[int] = []
    total_reward = 0.0
    step_index = 0
    target = None
    if intervention == "do_trigger":
        target = 0
    elif intervention == "do_non_trigger":
        target = 1

    while not env.game_over and env.field.sum() > 0:
        # The intervention is minimal and releases after its target is eaten.
        active_target = (
            target if target is not None
            and findex.positions[target] in before else None
        )
        actions = controller.act(
            env, base.obs_all(env), rng, findex, active_target)
        _obs, rewards, _term, _trunc, _info = env.step(
            safe_actions(env, actions))
        total_reward += (base.GAMMA ** step_index) * float(np.mean(rewards))
        step_index += 1
        after = set(base.food_positions(env))
        if after != before:
            order.extend(findex.consumed_now(before, after))
            before = after

    win = len(order) == 2
    trigger = bool(order and order[0] == 0)
    basin = (
        ("win_" if win else "loss_")
        + ("food0" if trigger else "food1")
    )
    return {
        "basin": basin,
        "win": int(win),
        "trigger": int(trigger),
        "context": context,
        "steps": int(env.current_step),
        "score": total_reward,
        "order": order,
    }


def evaluate(controller: TeamController, n_eval: int,
             seed_offset: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for context in CONTEXTS:
        for episode in range(n_eval):
            paired_seed = seed_offset + 10_000 * context + episode
            for mode in (None, "do_trigger", "do_non_trigger"):
                row = run_episode(controller, context, paired_seed, mode)
                row["mode"] = mode or "natural"
                row["episode"] = episode
                rows.append(row)

    def subset(mode: str, context: Optional[int] = None):
        return [
            row for row in rows
            if row["mode"] == mode
            and (context is None or row["context"] == context)
        ]

    natural = subset("natural")
    do_trigger = subset("do_trigger")
    do_non_trigger = subset("do_non_trigger")
    natural_counts = {
        basin: sum(row["basin"] == basin for row in natural)
        for basin in BASINS
    }
    trigger_counts = {
        basin: sum(row["basin"] == basin for row in do_trigger)
        for basin in BASINS
    }
    non_counts = {
        basin: sum(row["basin"] == basin for row in do_non_trigger)
        for basin in BASINS
    }
    trigger_rates = {
        str(context): float(np.mean([
            row["trigger"] for row in subset("natural", context)
        ]))
        for context in CONTEXTS
    }
    separation = abs(trigger_rates["0"] - trigger_rates["1"])
    metrics = {
        "n_eval_per_context": n_eval,
        "potential_bits": entropy(normalize_counts(natural_counts)),
        "trigger_rates": trigger_rates,
        "conditional_selectivity": separation,
        "specificity_js_bits": js(
            normalize_counts(trigger_counts), normalize_counts(non_counts)),
        "natural_score": float(np.mean([row["score"] for row in natural])),
        "do_non_trigger_score": float(np.mean([
            row["score"] for row in do_non_trigger
        ])),
        "usefulness_gap": float(
            np.mean([row["score"] for row in natural])
            - np.mean([row["score"] for row in do_non_trigger])
        ),
        "context_usefulness": {
            str(context): float(
                np.mean([row["score"] for row in subset("natural", context)])
                - np.mean([
                    row["score"]
                    for row in subset("do_non_trigger", context)
                ])
            )
            for context in CONTEXTS
        },
    }
    return {"metrics": metrics, "rows": rows}


def component_verdict(metrics: Dict[str, Any], endogenous: bool,
                      acquisition: float) -> Dict[str, Any]:
    passes = {
        "potential": metrics["potential_bits"] >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": (
            metrics["conditional_selectivity"]
            >= THRESHOLDS["conditional_selectivity"]
        ),
        "specificity": (
            metrics["specificity_js_bits"] >= THRESHOLDS["specificity_js_bits"]
        ),
        "usefulness": (
            metrics["usefulness_gap"] > THRESHOLDS["usefulness_gap"]
        ),
        "endogeneity": endogenous,
        "acquisition": acquisition >= THRESHOLDS["acquisition"],
    }
    return {"passes": passes, "emergent": int(all(passes.values()))}


def train_contextual(seed: int, train_episodes: int) -> base.PolicyNet:
    original_factory = base.make_raw_env
    original_horizon = base.MAX_STEPS
    original_train_episodes = base.TRAIN_EPISODES
    try:
        base.make_raw_env = lambda env_seed: make_contextual_env(env_seed)
        base.MAX_STEPS = HORIZON
        base.TRAIN_EPISODES = train_episodes
        return base.train_ppo(seed)
    finally:
        base.make_raw_env = original_factory
        base.MAX_STEPS = original_horizon
        base.TRAIN_EPISODES = original_train_episodes


def initial_twin(seed: int) -> base.PolicyNet:
    torch.manual_seed(seed)
    return base.PolicyNet()


def run_seed(seed: int, train_episodes: int, n_eval: int,
             keep_rows: bool) -> Dict[str, Any]:
    net = train_contextual(seed, train_episodes)
    torch.save(net.state_dict(), OUTPUTS / f"contextual_lbf_net_seed{seed}.pt")
    learned = evaluate(
        TeamController("policy", net), n_eval, 8_000_000 + seed * 100_000)
    init = evaluate(
        TeamController("policy", initial_twin(seed)), n_eval,
        8_000_000 + seed * 100_000)
    acquisition = (
        learned["metrics"]["conditional_selectivity"]
        - init["metrics"]["conditional_selectivity"]
    )
    controls = {
        "team_nearest": evaluate(
            TeamController("team_nearest"), n_eval,
            8_000_000 + seed * 100_000),
        "fixed_food0": evaluate(
            TeamController("fixed_food0"), n_eval,
            8_000_000 + seed * 100_000),
        "fixed_food1": evaluate(
            TeamController("fixed_food1"), n_eval,
            8_000_000 + seed * 100_000),
    }
    systems: Dict[str, Any] = {
        "learned": learned,
        "initial_twin": init,
        **controls,
    }
    for name, measured in systems.items():
        if name == "learned":
            endogeneity, acq = True, acquisition
        elif name == "initial_twin":
            endogeneity, acq = True, 0.0
        else:
            endogeneity, acq = False, 0.0
        measured["acquisition"] = acq
        measured["verdict"] = component_verdict(
            measured["metrics"], endogeneity, acq)
        if not keep_rows:
            measured.pop("rows", None)
    expected = {
        "learned": 1, "initial_twin": 0,
        "team_nearest": 0, "fixed_food0": 0, "fixed_food1": 0,
    }
    return {
        "systems": systems,
        "expected": expected,
        "all_expected": all(
            systems[name]["verdict"]["emergent"] == label
            for name, label in expected.items()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[101])
    parser.add_argument("--train_episodes", type=int, default=8_000)
    parser.add_argument("--n_eval", type=int, default=40)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--keep_rows", action="store_true")
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    torch.set_num_threads(16)
    result: Dict[str, Any] = {
        "status": (
            "exploratory design pilot" if "pilot" in args.tag
            else "prospectively frozen fresh-seed run"
        ),
        "environment": ENV_ID,
        "horizon": HORIZON,
        "thresholds": THRESHOLDS,
        "train_episodes": args.train_episodes,
        "n_eval_per_context": args.n_eval,
        "seeds": {},
    }
    for seed in args.seeds:
        print(f"Contextual LBF seed {seed}", flush=True)
        result["seeds"][str(seed)] = run_seed(
            seed, args.train_episodes, args.n_eval, args.keep_rows)
        learned = result["seeds"][str(seed)]["systems"]["learned"]
        print(json.dumps({
            "metrics": learned["metrics"],
            "acquisition": learned["acquisition"],
            "verdict": learned["verdict"],
        }, indent=2), flush=True)
    result["summary"] = {
        "n_seeds": len(args.seeds),
        "learned_passes": sum(
            seed["systems"]["learned"]["verdict"]["emergent"]
            for seed in result["seeds"].values()
        ),
        "all_systems_expected_by_seed": {
            name: seed["all_expected"]
            for name, seed in result["seeds"].items()
        },
    }
    path = OUTPUTS / f"contextual_lbf_{args.tag}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
