"""Predictive validity of the early continuous profile.

The sharpest test that the continuous record is science rather than
description: measured EARLY in training (25% of episodes), does the
causal-magnitude component predict which seeds will END as accepted
adaptive emergence -- and does it predict better than early
performance? Prediction beats postdiction; a score that only restates
the final verdict has limited value.

Population: fresh seeds on four 2-D surface cells chosen (from the
stored phase_2d results, disclosed) BECAUSE their outcomes vary across
seeds: (G, w) in {(7, 0.7), (10, 0.4), (13, 0.1), (13, 0.25)}, five
fresh seeds each (9601..9605 offsets per cell; the stored 2-D seeds are
excluded). Each policy is trained exactly as in the frozen 2-D
protocol, with one snapshot of the Q-table at 25% of episodes.

Early predictors (measured on the 25% snapshot, blind to the future):
    M_early     mean do-trigger vs do-block JS on probe states;
    perf_early  natural mean team return (the "early performance"
                baseline predictor);
    S_early     selectivity tension;
    U_early     usefulness gap.

Final outcome (measured on the finished policy with the frozen
five-component rule of the 2-D protocol): accepted / rejected, plus the
final usefulness gap.

Registered predictions (frozen before running):
    PV-1  AUROC(M_early -> final acceptance) >= 0.75;
    PV-2  AUROC(M_early) > AUROC(perf_early);
    PV-3  Spearman(M_early, final usefulness gap) >= 0.5.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from contextual_sacrifice_gridworld import (
    MODES,
    choose_epsilon_greedy,
    q_values,
    sample_mode,
    scalar_reward,
    sample_preference_context,
)
from phase_boundary_prediction import ParamGoalEnv, THRESHOLDS
from phase_2d_prediction import (
    weighted_context,
    natural_trigger_2d,
    measure_cell,
)
from within_episode_collapse_probe import (
    choose_with_intervention,
    entropy,
    estimate_future,
    js,
    mean,
    probe_contexts,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

CELLS = ((7.0, 0.7), (10.0, 0.4), (13.0, 0.1), (13.0, 0.25))
SEED_BASE = 9600
N_SEEDS = 5
TRAIN_EPISODES = 60_000
SNAPSHOT_FRAC = 0.25
PROBE_EPISODES = 16
SAMPLES = 36
PROBE_TEMPERATURE = 0.9


def train_with_snapshot(g: float, w: float, seed: int):
    rng = random.Random(seed)
    ctx_rng = random.Random(seed + 99_991)
    q_table: Dict = {}
    snapshot = None
    snap_at = int(TRAIN_EPISODES * SNAPSHOT_FRAC)
    for episode in range(TRAIN_EPISODES):
        if episode == snap_at:
            snapshot = copy.deepcopy(q_table)
        epsilon = 0.04 + (0.45 - 0.04) * max(
            0.0, 1.0 - episode / TRAIN_EPISODES)
        mode = sample_mode(rng, episode)
        env = ParamGoalEnv(mode, g)
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
    return snapshot, q_table


def early_predictors(q_table: Dict, g: float, seed: int) -> Dict:
    contexts = probe_contexts("uncertain_preference")
    js_vals, gap_vals, h_vals, returns = [], [], [], []
    for episode in range(PROBE_EPISODES):
        rng = random.Random(seed + episode * 17)
        mode = MODES[episode % len(MODES)]
        env = ParamGoalEnv(mode, g)
        state = env.reset()
        nat_dist, nat_ret = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention=None)
        do_t, _ = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention="do_trigger")
        do_n, do_n_ret = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention="do_non_trigger")
        h_vals.append(entropy(nat_dist))
        js_vals.append(js(do_t, do_n))
        gap_vals.append(nat_ret - do_n_ret)
        returns.append(nat_ret)
    return {
        "M_early": mean(js_vals),
        "perf_early": mean(returns),
        "H_early": mean(h_vals),
        "U_early": mean(gap_vals),
    }


def final_outcome(q_table: Dict, g: float, w: float, seed: int) -> Dict:
    """Frozen 2-D cell measurement on the finished table (reuses the
    protocol's estimator but on an existing table)."""
    contexts = probe_contexts("uncertain_preference")
    h_vals, js_vals, gap_vals, flags = [], [], [], []
    for episode in range(24):
        rng = random.Random(seed + episode * 17)
        mode = MODES[episode % len(MODES)]
        env = ParamGoalEnv(mode, g)
        state = env.reset()
        nat_dist, nat_ret = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention=None)
        do_t, _ = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention="do_trigger")
        do_n, do_n_ret = estimate_future(
            q_table, env, state, contexts, [], PROBE_TEMPERATURE,
            SAMPLES, rng, intervention="do_non_trigger")
        h_vals.append(entropy(nat_dist))
        js_vals.append(js(do_t, do_n))
        gap_vals.append(nat_ret - do_n_ret)
        flags.append(1.0 if natural_trigger_2d(q_table, g, w, mode,
                                               episode, rng) else 0.0)
    p = mean(flags)
    row = {
        "h0_bits": mean(h_vals),
        "selectivity_tension": 4.0 * p * (1.0 - p),
        "specificity_js": mean(js_vals),
        "usefulness_gap": mean(gap_vals),
    }
    accepted = int(
        row["h0_bits"] >= THRESHOLDS["potential_bits"]
        and row["selectivity_tension"] >= THRESHOLDS["selectivity_tension"]
        and row["specificity_js"] >= THRESHOLDS["specificity_js"]
        and row["usefulness_gap"] > THRESHOLDS["usefulness_gap"])
    row["accepted"] = accepted
    return row


def auroc(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(xs, ys) -> float:
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    rows = []
    for g, w in CELLS:
        for k in range(N_SEEDS):
            seed = SEED_BASE + int(g) * 100 + int(w * 100) + k
            print(f"predictive validity: cell ({g},{w}) seed {seed}",
                  flush=True)
            snapshot, final_table = train_with_snapshot(g, w, seed)
            early = early_predictors(snapshot, g, seed + 7)
            final = final_outcome(final_table, g, w, seed + 13)
            rows.append({"cell": f"{g}|{w}", "seed": seed,
                         **early, **final})
            print(f"  M_early {early['M_early']:.3f} "
                  f"perf_early {early['perf_early']:.2f} -> "
                  f"accepted {final['accepted']} "
                  f"useful {final['usefulness_gap']:.2f}", flush=True)

    labels = [r["accepted"] for r in rows]
    auc_m = auroc([r["M_early"] for r in rows], labels)
    auc_perf = auroc([r["perf_early"] for r in rows], labels)
    auc_u = auroc([r["U_early"] for r in rows], labels)
    sp = spearman([r["M_early"] for r in rows],
                  [r["usefulness_gap"] for r in rows])
    report = {
        "status": ("predictive validity of the early continuous "
                   "profile; predictions PV-1..PV-3 frozen in the "
                   "docstring; cells chosen from stored 2-D outcome "
                   "variance (disclosed)"),
        "rows": rows,
        "n": len(rows),
        "acceptance_rate": float(np.mean(labels)),
        "auroc_M_early": auc_m,
        "auroc_perf_early": auc_perf,
        "auroc_U_early": auc_u,
        "spearman_M_early_final_usefulness": sp,
        "registered_outcomes": {
            "PV1_auroc_M_early_ge_0.75": bool(auc_m >= 0.75),
            "PV2_M_beats_perf": bool(auc_m > auc_perf),
            "PV3_spearman_ge_0.5": bool(sp >= 0.5),
        },
    }
    out = OUTPUTS / "predictive_validity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"AUROC M {auc_m:.3f} vs perf {auc_perf:.3f}; "
          f"Spearman {sp:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
