"""Prospective TWO-DIMENSIONAL phase diagram: (goal reward, context mix).

The one-dimensional G sweep predicted verdict flips along a payoff axis.
A sharper test of the framework as a predictive theory is a control
surface with two independent axes whose interaction produces a
non-rectangular acceptance region. Axis 1 is the rescue goal reward G
(as before). Axis 2 is the training probability w of the
latent-sacrifice context; the remaining probability is split equally
between self-preservation and visible-teamwork.

Closed-form accounting (from the environment source, BEFORE training):

    trigger valued in latent_sacrifice   iff G > 5   ((G-5)+4 vs 4)
    trigger valued in visible_teamwork   iff G > 11  ((G-5) vs 4+2)
    trigger never valued in self_preservation
    usefulness sign (team return)        crosses 0 at G = 9
    natural trigger rate (modes alternate rescue/bridge 1/2):
        p = w/2              for 5 < G <= 11
        p = (1+w)/4          for G > 11 (vt also triggers; w_vt=(1-w)/2)
    selectivity tension 4p(1-p) >= 0.5  iff p in [0.1464, 0.8536]

Derived verdict surface (five informative components, thresholds copied
unchanged from the 1-D experiment):

    G <= 5              reject everywhere (no trigger; selectivity route)
    5 < G <= 9          reject everywhere (usefulness route where the
                        tension passes; selectivity route where w/2 <
                        0.1464, i.e. w < 0.293)
    9 < G <= 11         ACCEPT iff w >= 0.293 (tension boundary);
                        reject via selectivity below it
    G > 11              ACCEPT everywhere (p = (1+w)/4 in [0.25, 0.5],
                        tension >= 0.75)

Grid and cell predictions (frozen before any training):

    G in {3, 7, 10, 13} x w in {0.10, 0.25, 0.40, 0.70}

    G=3 :  reject, reject, reject, reject
    G=7 :  reject, reject, reject, reject
    G=10:  reject, reject, ACCEPT, ACCEPT
    G=13:  ACCEPT, ACCEPT, ACCEPT, ACCEPT

    Registered fragile cells (declared, not scored): (G=10, w=0.25) has
    predicted tension 0.4375, only 0.0625 below threshold, and learning
    noise in per-context trigger rates can flip it; every other cell has
    a predicted margin > 0.09 on its deciding component.

Primary endpoint (P2D-1): all 15 non-fragile cells match the derived
verdict by seed-majority (3/5 seeds). Secondary (P2D-2): the acceptance
region is non-rectangular exactly as derived -- within G=10 the verdict
changes along w, and within w<=0.25 it changes along G.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from contextual_sacrifice_gridworld import (
    MODES,
    choose_epsilon_greedy,
    q_values,
    sample_mode,
    scalar_reward,
)
from phase_boundary_prediction import (
    ParamGoalEnv,
    THRESHOLDS,
    natural_trigger,
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

G_GRID = (3.0, 7.0, 10.0, 13.0)
W_GRID = (0.10, 0.25, 0.40, 0.70)
SEEDS = (9501, 9502, 9503, 9504, 9505)
TRAIN_EPISODES = 60_000
PROBE_EPISODES = 24
SAMPLES = 36
TEMPERATURE = 0.25
PROBE_TEMPERATURE = 0.9

PREDICTED = {
    (3.0, 0.10): 0, (3.0, 0.25): 0, (3.0, 0.40): 0, (3.0, 0.70): 0,
    (7.0, 0.10): 0, (7.0, 0.25): 0, (7.0, 0.40): 0, (7.0, 0.70): 0,
    (10.0, 0.10): 0, (10.0, 0.25): 0, (10.0, 0.40): 1, (10.0, 0.70): 1,
    (13.0, 0.10): 1, (13.0, 0.25): 1, (13.0, 0.40): 1, (13.0, 0.70): 1,
}
FRAGILE = {(10.0, 0.25)}


def weighted_context(w: float, rng: random.Random) -> str:
    u = rng.random()
    if u < w:
        return "latent_sacrifice"
    if u < w + (1.0 - w) / 2.0:
        return "self_preservation"
    return "visible_teamwork"


def train_policy_2d(goal_reward: float, w: float, episodes: int,
                    seed: int) -> Dict:
    rng = random.Random(seed)
    ctx_rng = random.Random(seed + 99_991)
    q_table: Dict = {}
    for episode in range(episodes):
        epsilon = 0.04 + (0.45 - 0.04) * max(0.0, 1.0 - episode / episodes)
        mode = sample_mode(rng, episode)
        env = ParamGoalEnv(mode, goal_reward)
        state = env.reset()
        context = weighted_context(w, ctx_rng)
        done = False
        while not done:
            action = choose_epsilon_greedy(q_table, state, context,
                                           epsilon, rng)
            result = env.step(state, action)
            reward = scalar_reward("uncertain_preference", context,
                                   result.rewards, result.events, rng)
            values = q_values(q_table, state, context)
            bootstrap = 0.0 if result.done else max(
                q_values(q_table, result.state, context).values())
            values[action] += 0.28 * (reward + 0.96 * bootstrap
                                      - values[action])
            state = result.state
            done = result.done
    return q_table


def natural_trigger_2d(q_table, goal_reward: float, w: float, mode: str,
                       episode_idx: int, rng: random.Random) -> bool:
    env = ParamGoalEnv(mode, goal_reward)
    state = env.reset()
    ctx_rng = random.Random(episode_idx * 7919 + 5)
    context = weighted_context(w, ctx_rng)
    events: List[str] = []
    done = False
    while not done:
        action = choose_with_intervention(q_table, state, context,
                                          TEMPERATURE, rng, None)
        result = env.step(state, action)
        events.extend(result.events)
        state = result.state
        done = result.done
    return any(e in events for e in TRIGGER_EVENTS)


def measure_cell(goal_reward: float, w: float, seed: int) -> Dict:
    q_table = train_policy_2d(goal_reward, w, TRAIN_EPISODES, seed)
    contexts = probe_contexts("uncertain_preference")
    h0_values, js_values, gap_values, flags = [], [], [], []
    for episode in range(PROBE_EPISODES):
        rng = random.Random(seed + episode * 17)
        mode = MODES[episode % len(MODES)]
        env = ParamGoalEnv(mode, goal_reward)
        state = env.reset()
        behavior_dist, behavior_return = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention=None)
        h0_values.append(entropy(behavior_dist))
        do_t, _ = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention="do_trigger")
        do_n, do_n_ret = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention="do_non_trigger")
        js_values.append(js(do_t, do_n))
        gap_values.append(behavior_return - do_n_ret)
        flags.append(1.0 if natural_trigger_2d(
            q_table, goal_reward, w, mode, episode, rng) else 0.0)
    p = mean(flags)
    row = {
        "h0_bits": mean(h0_values),
        "natural_trigger_rate": p,
        "selectivity_tension": 4.0 * p * (1.0 - p),
        "specificity_js": mean(js_values),
        "usefulness_gap": mean(gap_values),
    }
    passes = {
        "potential": row["h0_bits"] >= THRESHOLDS["potential_bits"],
        "selectivity": row["selectivity_tension"]
        >= THRESHOLDS["selectivity_tension"],
        "specificity": row["specificity_js"] >= THRESHOLDS["specificity_js"],
        "usefulness": row["usefulness_gap"] > THRESHOLDS["usefulness_gap"],
    }
    row["accepted"] = int(all(passes.values()))
    row["failed"] = [k for k, ok in passes.items() if not ok]
    return row


def main() -> None:
    report = {
        "status": ("prospective 2-D phase diagram; derivation, grid, "
                   "predictions and the single fragile cell frozen in the "
                   "docstring before any training"),
        "grid": {"G": list(G_GRID), "w": list(W_GRID)},
        "predicted": {f"{g}|{w}": PREDICTED[(g, w)]
                      for g in G_GRID for w in W_GRID},
        "fragile_cells": [f"{g}|{w}" for (g, w) in FRAGILE],
        "cells": {},
    }
    scored = 0
    matched = 0
    for g in G_GRID:
        for w in W_GRID:
            key = f"{g}|{w}"
            seed_rows = {}
            accepts = 0
            for seed in SEEDS:
                row = measure_cell(g, w, seed)
                seed_rows[str(seed)] = row
                accepts += row["accepted"]
            majority = int(accepts >= 3)
            cell = {
                "seeds": seed_rows,
                "accepts": accepts,
                "majority_verdict": majority,
                "predicted": PREDICTED[(g, w)],
                "fragile": (g, w) in FRAGILE,
                "match": majority == PREDICTED[(g, w)],
            }
            report["cells"][key] = cell
            if (g, w) not in FRAGILE:
                scored += 1
                matched += int(cell["match"])
            print(f"G={g} w={w}: accepts {accepts}/5, majority {majority}, "
                  f"predicted {PREDICTED[(g, w)]}, "
                  f"{'MATCH' if cell['match'] else 'MISS'}"
                  f"{' (fragile, unscored)' if (g, w) in FRAGILE else ''}",
                  flush=True)

    g10 = [report["cells"][f"10.0|{w}"]["majority_verdict"] for w in W_GRID]
    low_w = [report["cells"][f"{g}|0.1"]["majority_verdict"] for g in G_GRID]
    report["registered_outcomes"] = {
        "P2D1_nonfragile_match": f"{matched}/{scored}",
        "P2D1_pass": matched == scored,
        "P2D2_nonrectangular": (g10[0] == 0 and g10[-1] == 1
                                and low_w[0] == 0 and low_w[-1] == 1),
    }
    out = OUTPUTS / "phase_2d_prediction.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
