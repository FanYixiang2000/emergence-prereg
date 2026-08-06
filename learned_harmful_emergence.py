"""Learned harmful emergence: structural emergence with a negative value.

Reviewer objection addressed: the framework's harmful-collapse exemplar
(harmful_decoy) is a FORCED construction, so "emergent but harmful" was a
category the layered definition allowed but never demonstrated on a
learned system. Here a policy LEARNS, from its own declared reward, a
context-selective, counterfactually load-bearing, acquired macro-structure
that is valuable to the declared beneficiary and destructive to the team.

Construction (benchmark dynamics unchanged; only the training reward is
provenance-specific, as in the strength-gradient battery):

    exploiter reward  r0 + 14.0 * [a1_hits_decoy_goal]
                      (agent 0 is paid when its partner is lured onto the
                      decoy goal; the bonus makes the decoy path strictly dominate
                      the safe exit and the visible bridge for the
                      exploiter: net +9 vs +6 vs +4). The
                      decoy path only exists in bridge mode, so the
                      acquired trigger structure is mode-selective for
                      free -- but in the HARMFUL direction.

Declared value functions (both fixed before training):

    u_private   the exploiter's own return including the bonus;
    u_team      plain team return (r0 + r1), the value used everywhere
                else in the paper.

Registered predictions (frozen before running; 5 seeds):

    HE-1  the learned policy triggers in bridge mode (rate >= 0.8) and
          not in rescue mode (rate <= 0.2) on >= 4/5 seeds:
          context-selective in the harmful direction.
    HE-2  structural components pass on >= 4/5 seeds: potential >= 0.5
          bits, per-mode trigger separation >= 0.5, do-contrast
          specificity >= 0.2 bits.
    HE-3  the usefulness do-contrast is POSITIVE under u_private and
          NEGATIVE under u_team on >= 4/5 seeds: the same structural
          collapse is adaptive for the declared beneficiary and
          maladaptive for the team.
    HE-4  acquisition >= 0.3 on >= 4/5 seeds (initialization twin =
          untrained table, which never triggers selectively).

Expected classification under the layered definition: structural
emergence PASSES; adaptive qualification depends on the declared value
subject -- which turns observer/value dependence from a liability into
the framework's expressive content.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from contextual_sacrifice_gridworld import (
    ContextualSacrificeEnv,
    MODES,
    choose_epsilon_greedy,
    choose_softmax,
    classify_basin,
    q_values,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

SEEDS = (9301, 9302, 9303, 9304, 9305)
EPISODES = 60_000
EVAL_EPISODES = 200
PROBE_TEMPERATURE = 0.25
BASINS = ("sacrifice_rescue", "team_direct", "selfish_escape",
          "failed_noise", "decoy_trap")


def classify_with_decoy(events) -> str:
    if "a1_hits_decoy_goal" in set(events):
        return "decoy_trap"
    return classify_basin(events)


def exploiter_reward(rewards, events) -> float:
    bonus = 14.0 if "a1_hits_decoy_goal" in events else 0.0
    return rewards[0] + bonus


def train(seed: int) -> Dict:
    rng = random.Random(seed)
    q_table: Dict = {}
    for episode in range(EPISODES):
        epsilon = 0.04 + (0.45 - 0.04) * max(0.0, 1.0 - episode / EPISODES)
        mode = MODES[episode % 2]
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        done = False
        while not done:
            action = choose_epsilon_greedy(q_table, state, "fixed",
                                           epsilon, rng)
            result = env.step(state, action)
            reward = exploiter_reward(result.rewards, result.events)
            values = q_values(q_table, state, "fixed")
            bootstrap = 0.0 if result.done else max(
                q_values(q_table, result.state, "fixed").values())
            values[action] += 0.28 * (reward + 0.96 * bootstrap
                                      - values[action])
            state = result.state
            done = result.done
    return q_table


def run_episode(q_table: Dict, mode: str, rng: random.Random,
                forced: Optional[str]) -> Dict[str, Any]:
    env = ContextualSacrificeEnv(mode)
    state = env.reset()
    events: List[str] = []
    r0_total = 0.0
    team_total = 0.0
    private_total = 0.0
    done = False
    while not done:
        action = choose_softmax(q_table, state, "fixed",
                                PROBE_TEMPERATURE, rng, forced)
        result = env.step(state, action)
        events.extend(result.events)
        r0_total += result.rewards[0]
        team_total += result.rewards[0] + result.rewards[1]
        private_total += exploiter_reward(result.rewards, result.events)
        state = result.state
        done = result.done
    return {
        "basin": classify_with_decoy(events),
        "trigger": any(e in ("a0_step_on_sacrifice_switch",
                             "a0_step_on_decoy_switch") for e in events),
        "u_team": team_total,
        "u_private": private_total,
    }


def entropy_bits(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c > 0)


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
    total = len(rows)
    return {b: counts.get(b, 0) / total for b in BASINS}


def measure(q_table: Dict, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed + 555)
    rows = {mode: {kind: [run_episode(q_table, mode, rng, kind)
                          for _ in range(EVAL_EPISODES)]
                   for kind in (None, "trigger", "non_trigger")}
            for mode in MODES}
    natural = rows["rescue"][None] + rows["bridge"][None]
    do_t = rows["rescue"]["trigger"] + rows["bridge"]["trigger"]
    do_n = rows["rescue"]["non_trigger"] + rows["bridge"]["non_trigger"]
    counts: Dict[str, int] = {}
    for row in natural:
        counts[row["basin"]] = counts.get(row["basin"], 0) + 1
    trig_rescue = sum(r["trigger"] for r in rows["rescue"][None]) / EVAL_EPISODES
    trig_bridge = sum(r["trigger"] for r in rows["bridge"][None]) / EVAL_EPISODES

    def mean(rows_in, key):
        return sum(r[key] for r in rows_in) / len(rows_in)

    return {
        "potential_bits": entropy_bits(counts),
        "trigger_rate_rescue": trig_rescue,
        "trigger_rate_bridge": trig_bridge,
        "selectivity_separation": abs(trig_bridge - trig_rescue),
        "specificity_js_bits": js_bits(distribution(do_t),
                                       distribution(do_n)),
        "usefulness_private": mean(natural, "u_private")
        - mean(do_n, "u_private"),
        "usefulness_team": mean(natural, "u_team") - mean(do_n, "u_team"),
    }


def main() -> None:
    report: Dict[str, Any] = {
        "status": ("learned harmful emergence with two declared value "
                   "functions; predictions HE-1..HE-4 frozen in the "
                   "docstring"),
        "seeds": {},
    }
    he1 = he2 = he3 = he4 = 0
    for seed in SEEDS:
        print(f"training exploiter, seed {seed}", flush=True)
        q_table = train(seed)
        metrics = measure(q_table, seed)
        untrained = measure({}, seed + 7)
        acquisition = (metrics["selectivity_separation"]
                       - untrained["selectivity_separation"])
        metrics["acquisition"] = acquisition
        metrics["untrained_separation"] = untrained["selectivity_separation"]
        report["seeds"][str(seed)] = metrics
        he1 += int(metrics["trigger_rate_bridge"] >= 0.8
                   and metrics["trigger_rate_rescue"] <= 0.2)
        he2 += int(metrics["potential_bits"] >= 0.5
                   and metrics["selectivity_separation"] >= 0.5
                   and metrics["specificity_js_bits"] >= 0.2)
        he3 += int(metrics["usefulness_private"] > 0
                   and metrics["usefulness_team"] < 0)
        he4 += int(acquisition >= 0.3)
        print(f"  trig bridge {metrics['trigger_rate_bridge']:.2f} "
              f"rescue {metrics['trigger_rate_rescue']:.2f} | "
              f"pot {metrics['potential_bits']:.2f} "
              f"spec {metrics['specificity_js_bits']:.2f} | "
              f"U_priv {metrics['usefulness_private']:+.2f} "
              f"U_team {metrics['usefulness_team']:+.2f} | "
              f"acq {acquisition:.2f}", flush=True)

    n = len(SEEDS)
    report["registered_outcomes"] = {
        "HE1_harmful_direction_selectivity": f"{he1}/{n}",
        "HE2_structural_components": f"{he2}/{n}",
        "HE3_value_sign_split": f"{he3}/{n}",
        "HE4_acquisition": f"{he4}/{n}",
        "HE1_pass": he1 >= 4, "HE2_pass": he2 >= 4,
        "HE3_pass": he3 >= 4, "HE4_pass": he4 >= 4,
    }
    report["reading"] = (
        "The same learned structure is structurally emergent under both "
        "value functions, adaptive under the declared beneficiary's value "
        "and maladaptive under the team value: adaptivity is relative to "
        "the declared value subject, which the layered definition states "
        "and this system now measures."
    )
    out = OUTPUTS / "learned_harmful_emergence.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
