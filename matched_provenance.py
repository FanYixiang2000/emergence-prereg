"""Matched-behaviour provenance: four routes to one macro-structure.

Reviewer objections addressed. (i) Endogeneity/acquisition could be
"reading the experiment log, not the system": if two systems behave
identically, what does the criterion actually measure? (ii) The strength
gradient could conflate outcome rarity, discovery difficulty and
structural magnitude. Both are answered by constructing four systems
whose FINAL behaviour is matched as closely as the environment allows,
differing only in provenance:

    script         hand-coded deterministic controller (a0 to the switch,
                   a1 to the high goal);
    bc_clone       tabular behaviour cloning on script rollouts;
    shaped         Q-learning, reward names the trigger (as in the
                   strength battery);
    outcome_only   Q-learning, sparse outcome reward only.

Declared decomposition (each reported separately, never merged into one
"strength" number):

    outcome rarity          -log2 P(pattern) under the frozen uniform
                            reference (identical for all four by
                            construction);
    provenance rarity /     -log2 mean pattern probability over the first
    discovery difficulty    quarter of that system's own training trace,
                            and first checkpoint with pattern
                            probability >= 0.5;
    structural magnitude    do-contrast specificity (JS between forced
                            trigger and forced non-trigger basins) and
                            the team-return do-gap, measured on the final
                            policy.

Disclosed design pilot (one script + one BC seed, run before freezing
these predictions): natural behaviour of the clone matches the script
exactly (pattern probability 1.0 for both) but its do-contrast
specificity is far below the script's (0.37 vs 1.00), because forcing
the clone off its training distribution exposes states it never
memorized. The predictions below were frozen AFTER that pilot and
incorporate it.

Registered predictions (frozen before the confirmatory run; 5 seeds for
each trained provenance; script is deterministic):

    MP-1  natural behaviour matched: final pattern probability >= 0.9
          for every system/seed;
    MP-2  structural magnitude matched for genuinely learned policies:
          shaped and outcome_only specificity JS within +-0.15 of the
          script's value on every seed (same macro-structure, same
          causal load);
    MP-3  binary acquisition separates static from trained provenance:
          script acquisition = 0; bc_clone, shaped and outcome_only all
          >= 0.3 (a behaviour-cloned script counts as acquired relative
          to its declared training signal -- the boundary the Methods
          states, here measured);
    MP-4  provenance rarity separates what binary acquisition cannot:
          seed-mean ordering bc_clone < shaped < outcome_only;
    MP-5  the clone is counterfactually distinguishable from its source
          despite matched natural behaviour: bc_clone specificity JS
          below the script's by > 0.15 on every seed (the do-contrast
          reads the system, not the experiment log).

Reading if these hold: behavioural components measure the structure;
do-contrasts distinguish memorized from robust structure even at
matched natural behaviour; acquisition measures trained-vs-static
provenance at the declared system boundary; provenance rarity measures
how much of the structure the training signal injected. Four different
questions, reported separately rather than compressed into one number.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from contextual_sacrifice_gridworld import (
    ContextualSacrificeEnv,
    HIGH_GOAL,
    JOINT_ACTIONS,
    MOVES,
    SWITCH,
    choose_softmax,
    classify_basin,
    manhattan,
    move_position,
    moves_toward_switch,
)
from strength_gradient_battery import (
    EVAL_EPISODES,
    MODE,
    PATTERN_BASIN,
    SEEDS,
    train_with_pattern_trace,
    uniform_pattern_probability,
)
import strength_gradient_battery as sgb

OUTPUTS = Path(__file__).resolve().parent / "outputs"

PROBE_TEMPERATURE = 0.25
BC_EPISODES = 2_000
BC_CHECK_EVERY = 100


def step_toward(pos, goal) -> str:
    best = "stay"
    best_d = manhattan(pos, goal)
    for move in MOVES:
        cand = move_position(pos, move)
        if manhattan(cand, goal) < best_d:
            best, best_d = move, manhattan(cand, goal)
    return best


def script_action(state) -> tuple:
    _mode, a0, a1, gate_open, switch_used, _t = state
    a0_move = "stay" if switch_used else step_toward(a0, SWITCH)
    a1_move = step_toward(a1, HIGH_GOAL) if gate_open or not switch_used \
        else step_toward(a1, HIGH_GOAL)
    return (a0_move, a1_move)


Policy = Callable[[Any, random.Random, Optional[str]], tuple]


def forced_filter(state, action, forced: Optional[str],
                  rng: random.Random) -> tuple:
    """Apply the same trigger forcing used by the benchmark probes."""
    _mode, a0, _a1, _gate, switch_used, _t = state
    if forced is None or switch_used:
        return action
    candidates = [a for a in JOINT_ACTIONS if a[1] == action[1]]
    if forced == "trigger":
        toward = [a for a in candidates if moves_toward_switch(a0, a[0])]
        return rng.choice(toward) if toward else action
    away = [a for a in candidates if not moves_toward_switch(a0, a[0])]
    return rng.choice(away) if away else action


def script_policy(state, rng, forced):
    return forced_filter(state, script_action(state), forced, rng)


def make_bc_policy(table: Dict) -> Policy:
    def policy(state, rng, forced):
        action = table.get(state)
        if action is None:
            action = rng.choice(JOINT_ACTIONS)
        return forced_filter(state, action, forced, rng)
    return policy


def make_q_policy(q_table: Dict) -> Policy:
    def policy(state, rng, forced):
        forced_arg = {"trigger": "trigger",
                      "non_trigger": "non_trigger"}.get(forced)
        return choose_softmax(q_table, state, "fixed", PROBE_TEMPERATURE,
                              rng, forced_arg)
    return policy


def run_episode(policy: Policy, rng: random.Random,
                forced: Optional[str]) -> Dict[str, Any]:
    env = ContextualSacrificeEnv(MODE)
    state = env.reset()
    events: List[str] = []
    team = 0.0
    done = False
    while not done:
        action = policy(state, rng, forced)
        result = env.step(state, action)
        events.extend(result.events)
        team += result.rewards[0] + result.rewards[1]
        state = result.state
        done = result.done
    return {"basin": classify_basin(events), "team": team}


def entropy_bits(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c > 0) if total else 0.0


def js_bits(p: Dict[str, float], q: Dict[str, float]) -> float:
    out = 0.0
    for key in set(p) | set(q):
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def distribution(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["basin"]] = counts.get(row["basin"], 0) + 1
    return {b: c / len(rows) for b, c in counts.items()}


def tv(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def measure(policy: Policy, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed + 313)
    natural = [run_episode(policy, rng, None)
               for _ in range(EVAL_EPISODES)]
    do_t = [run_episode(policy, rng, "trigger")
            for _ in range(EVAL_EPISODES)]
    do_n = [run_episode(policy, rng, "non_trigger")
            for _ in range(EVAL_EPISODES)]
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    return {
        "pattern_probability": sum(
            r["basin"] == PATTERN_BASIN for r in natural) / len(natural),
        "basin_distribution": distribution(natural),
        "specificity_js_bits": js_bits(distribution(do_t),
                                       distribution(do_n)),
        "usefulness_do_gap": mean(natural, "team") - mean(do_n, "team"),
    }


def train_bc_with_trace(seed: int) -> tuple:
    """Tabular behaviour cloning on script rollouts, with pattern trace."""
    rng = random.Random(seed)
    table: Dict = {}
    trace: List[float] = []
    for episode in range(BC_EPISODES):
        if episode % BC_CHECK_EVERY == 0:
            probe = make_bc_policy(dict(table))
            probe_rng = random.Random(seed + 777 + len(trace))
            hits = sum(
                run_episode(probe, probe_rng, None)["basin"] == PATTERN_BASIN
                for _ in range(60))
            trace.append((hits + 0.5) / 61.0)
        env = ContextualSacrificeEnv(MODE)
        state = env.reset()
        done = False
        while not done:
            action = script_action(state)
            table[state] = action
            result = env.step(state, action)
            state = result.state
            done = result.done
        _ = rng.random()
    return table, trace


def provenance_stats(trace: List[float]) -> Dict[str, Any]:
    quarter = max(1, len(trace) // 4)
    c_prov = -math.log2(sum(trace[:quarter]) / quarter)
    discovery = next((i for i, p in enumerate(trace) if p >= 0.5), None)
    return {"c_prov_bits": c_prov, "discovery_checkpoint": discovery,
            "n_checkpoints": len(trace)}


def main() -> None:
    p_uniform = uniform_pattern_probability(4242)
    script_metrics = measure(script_policy, 11)

    report: Dict[str, Any] = {
        "status": ("matched-behaviour provenance; predictions MP-1..MP-4 "
                   "frozen in the docstring"),
        "outcome_rarity_bits_uniform_reference": -math.log2(p_uniform),
        "systems": {
            "script": {
                "metrics": script_metrics,
                "acquisition": 0.0,
                "c_prov_bits": 0.0,
                "note": "deterministic hand rule; its own initialization",
            },
        },
    }

    mp1_fail = int(script_metrics["pattern_probability"] < 0.9)
    mp2_fail = 0
    mp3_ok = {"script": True}
    per_prov_cprov: Dict[str, List[float]] = {}

    for provenance in ("bc_clone", "shaped", "outcome_only"):
        seeds_out = {}
        for seed in SEEDS:
            print(f"{provenance} seed {seed}", flush=True)
            if provenance == "bc_clone":
                table, trace = train_bc_with_trace(seed)
                policy = make_bc_policy(table)
                init_metrics = measure(make_bc_policy({}), seed + 7)
            else:
                sgb.N_CHECKPOINTS = 20
                trace = train_with_pattern_trace(provenance, 60_000, seed)
                q_final = None  # retrain returns trace only; measure fresh
                # train again capturing the table via the module helper
                q_table = train_q_table(provenance, seed)
                policy = make_q_policy(q_table)
                init_metrics = measure(make_q_policy({}), seed + 7)
            metrics = measure(policy, seed)
            acquisition = (metrics["pattern_probability"]
                           - init_metrics["pattern_probability"])
            stats = provenance_stats(trace)
            seeds_out[str(seed)] = {
                "metrics": metrics, "acquisition": acquisition, **stats}
            per_prov_cprov.setdefault(provenance, []).append(
                stats["c_prov_bits"])
            mp1_fail += int(metrics["pattern_probability"] < 0.9)
            if provenance in ("shaped", "outcome_only"):
                mp2_fail += int(abs(metrics["specificity_js_bits"]
                                    - script_metrics["specificity_js_bits"])
                                > 0.15)
            mp3_ok.setdefault(provenance, True)
            if acquisition < 0.3:
                mp3_ok[provenance] = False
            print(f"  p {metrics['pattern_probability']:.2f} "
                  f"spec {metrics['specificity_js_bits']:.2f} "
                  f"acq {acquisition:.2f} "
                  f"C_prov {stats['c_prov_bits']:.2f}", flush=True)
        report["systems"][provenance] = {"seeds": seeds_out}

    mean = lambda xs: sum(xs) / len(xs)
    order = [mean(per_prov_cprov[p])
             for p in ("bc_clone", "shaped", "outcome_only")]
    mp5_ok = all(
        script_metrics["specificity_js_bits"]
        - entry["metrics"]["specificity_js_bits"] > 0.15
        for entry in report["systems"]["bc_clone"]["seeds"].values())
    report["seed_mean_c_prov_bits"] = {
        p: mean(per_prov_cprov[p])
        for p in ("bc_clone", "shaped", "outcome_only")}
    report["registered_outcomes"] = {
        "MP1_behaviour_matched": mp1_fail == 0,
        "MP2_structural_magnitude_matched": mp2_fail == 0,
        "MP3_acquisition_boundary": all(mp3_ok.values()),
        "MP4_provenance_rarity_ordering": order[0] < order[1] < order[2],
        "MP5_clone_counterfactually_distinguishable": mp5_ok,
        "mp2_deviations": mp2_fail,
    }
    out = OUTPUTS / "matched_provenance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(json.dumps(report["seed_mean_c_prov_bits"], indent=2))
    print(f"Wrote {out}")


def train_q_table(provenance: str, seed: int) -> Dict:
    """Q-learning identical to the strength battery, returning the table."""
    from contextual_sacrifice_gridworld import (
        choose_epsilon_greedy, q_values)
    from strength_gradient_battery import provenance_reward
    rng = random.Random(seed)
    q_table: Dict = {}
    episodes = 60_000
    for episode in range(episodes):
        epsilon = 0.04 + (0.45 - 0.04) * max(0.0, 1.0 - episode / episodes)
        env = ContextualSacrificeEnv(MODE)
        state = env.reset()
        done = False
        while not done:
            action = choose_epsilon_greedy(q_table, state, "fixed",
                                           epsilon, rng)
            result = env.step(state, action)
            reward = provenance_reward(provenance, result.rewards,
                                       result.events)
            values = q_values(q_table, state, "fixed")
            bootstrap = 0.0 if result.done else max(
                q_values(q_table, result.state, "fixed").values())
            values[action] += 0.28 * (reward + 0.96 * bootstrap
                                      - values[action])
            state = result.state
            done = result.done
    return q_table


if __name__ == "__main__":
    main()
