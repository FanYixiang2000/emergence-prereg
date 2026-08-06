"""Simulator--world-model closure: verdicts from imperfect learned models.

Reviewer objection addressed: in real systems futures cannot be simulated
exactly, so the framework must show that verdicts remain reliable when the
rollout law is a LEARNED world model -- and must say "undecidable" rather
than guess when the model is too wrong. The theory supplies a bound
(|Delta_P u - Delta_Q u| <= R(eps_a + eps_a')); this experiment closes the
loop empirically, using ensemble disagreement as the practical error
estimate available WITHOUT simulator access.

Setup (contextual sacrifice gridworld; simulator = exact environment):

    policy         Q-learning under uncertain_preference (one per seed);
    ground truth   components measured with simulator rollouts (the
                   standard within-episode probe);
    world models   tabular MLE transition tables P(s', events, r | s, a)
                   fitted on K on-policy trajectories,
                   K in {200, 1000, 5000, 20000}; 5 independent data
                   seeds per K (the model ensemble);
    measurement    potential, do-specificity and usefulness recomputed
                   with each world model replacing the simulator in
                   probe rollouts; verdict with frozen thresholds
                   (potential 0.5 bits, specificity 0.2 bits,
                   usefulness > 0);
    error proxy    epsilon_hat = mean pairwise TV between the basin
                   distributions of the 5 models at the same K
                   (no simulator needed);
    margin rule    a component is DECIDABLE at error epsilon_hat iff its
                   measured margin to threshold exceeds epsilon_hat *
                   SCALE (declared: SCALE = 2 for probability-scale
                   components; usefulness margin compared to
                   epsilon_hat * R with R = declared return range 30);
                   verdicts with any undecidable component are
                   abstentions, not classifications.

Registered predictions (frozen before running; 3 policy seeds):

    WM-1  calibration improves with data: seed-mean TV(model, simulator)
          over natural-basin distributions decreases monotonically in K;
    WM-2  at K = 20000 every non-abstaining model verdict matches the
          simulator verdict (no silent wrong verdict at high data);
    WM-3  across ALL K, the margin rule catches every mismatch: no
          model both disagrees with the simulator verdict AND declares
          itself decidable on the disagreeing component.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from contextual_sacrifice_gridworld import (
    ContextualSacrificeEnv,
    MODES,
    choose_epsilon_greedy,
    classify_basin,
    q_values,
    sample_mode,
    sample_preference_context,
    scalar_reward,
    train_policy,
)
from within_episode_collapse_probe import choose_with_intervention

OUTPUTS = Path(__file__).resolve().parent / "outputs"

POLICY_SEEDS = (9701, 9702, 9703)
K_GRID = (200, 1000, 5000, 20000)
MODEL_SEEDS = (1, 2, 3, 4, 5)
PROBE_EPISODES = 300
PROBE_TEMPERATURE = 0.9
THRESHOLDS = {"potential_bits": 0.5, "specificity_js": 0.2,
              "usefulness_gap": 0.0}
RETURN_RANGE = 30.0
TV_SCALE = 2.0
BASINS = ("sacrifice_rescue", "team_direct", "selfish_escape",
          "failed_noise")


def entropy_bits(dist: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


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


def tv(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def collect_trajectories(q_table: Dict, n: int, seed: int):
    """On-policy trajectories for world-model fitting."""
    rng = random.Random(seed)
    data = []
    for episode in range(n):
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context("uncertain_preference", rng,
                                            episode)
        done = False
        while not done:
            action = choose_with_intervention(
                q_table, state, context, PROBE_TEMPERATURE, rng, None)
            result = env.step(state, action)
            data.append((state, action,
                         (result.state, result.events, result.rewards,
                          result.done)))
            state = result.state
            done = result.done
    return data


class WorldModel:
    """Tabular MLE model of (s, a) -> (s', events, rewards, done)."""

    def __init__(self, data):
        self.table: Dict[Tuple, Counter] = defaultdict(Counter)
        for state, action, outcome in data:
            key = (state, action)
            self.table[key][(outcome[0], outcome[1], outcome[2],
                             outcome[3])] += 1
        self.misses = 0
        self.queries = 0

    def step(self, state, action, rng: random.Random):
        self.queries += 1
        key = (state, action)
        counter = self.table.get(key)
        if not counter:
            self.misses = self.misses + 1
            return None
        outcomes = list(counter)
        weights = [counter[o] for o in outcomes]
        total = sum(weights)
        u = rng.random() * total
        acc = 0.0
        for outcome, weight in zip(outcomes, weights):
            acc += weight
            if u <= acc:
                return outcome
        return outcomes[-1]


def probe(q_table: Dict, stepper, seed: int) -> Dict[str, Any]:
    """Components from rollouts under a stepper (simulator or model)."""
    def rollout(mode: str, episode: int, forced: Optional[str]):
        rng = random.Random(seed + episode * 31 + (hash(forced) % 97))
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context("uncertain_preference", rng,
                                            episode)
        events: List[str] = []
        team = 0.0
        done = False
        while not done:
            action = choose_with_intervention(
                q_table, state, context, PROBE_TEMPERATURE, rng, forced)
            outcome = stepper(state, action, rng)
            if outcome is None:
                events.append("model_incomplete")
                break
            next_state, ev, rewards, done = outcome
            events.extend(ev)
            team += rewards[0] + rewards[1]
            state = next_state
        return {"basin": classify_basin(events), "team": team,
                "incomplete": "model_incomplete" in events}

    rows = {kind: [rollout(MODES[e % 2], e, kind)
                   for e in range(PROBE_EPISODES)]
            for kind in (None, "do_trigger", "do_non_trigger")}

    def dist(kind):
        counts: Dict[str, int] = {}
        for row in rows[kind]:
            counts[row["basin"]] = counts.get(row["basin"], 0) + 1
        total = len(rows[kind])
        return {b: counts.get(b, 0) / total for b in BASINS}

    mean = lambda kind, key: (sum(r[key] for r in rows[kind])
                              / len(rows[kind]))
    incomplete = sum(r["incomplete"] for k in rows for r in rows[k])
    return {
        "natural_dist": dist(None),
        "potential_bits": entropy_bits(dist(None)),
        "specificity_js": js_bits(dist("do_trigger"),
                                  dist("do_non_trigger")),
        "usefulness_gap": mean(None, "team")
        - mean("do_non_trigger", "team"),
        "incomplete_rollouts": incomplete,
    }


def verdict_and_margins(metrics: Dict[str, Any]):
    margins = {
        "potential_bits": metrics["potential_bits"]
        - THRESHOLDS["potential_bits"],
        "specificity_js": metrics["specificity_js"]
        - THRESHOLDS["specificity_js"],
        "usefulness_gap": metrics["usefulness_gap"]
        - THRESHOLDS["usefulness_gap"],
    }
    verdict = int(all(m > 0 for m in margins.values()))
    return verdict, margins


def main() -> None:
    report: Dict[str, Any] = {
        "status": ("simulator--world-model closure; predictions WM-1..3 "
                   "and the margin rule frozen in the docstring"),
        "policy_seeds": {},
    }
    wm1_ok = True
    wm2_violations = 0
    wm3_violations = 0
    for pseed in POLICY_SEEDS:
        print(f"policy seed {pseed}: training", flush=True)
        q_table = train_policy("uncertain_preference", 60_000, pseed)
        sim_metrics = probe(
            q_table, lambda s, a, r: (lambda res: (res.state, res.events,
                                                   res.rewards, res.done))(
                ContextualSacrificeEnv(s[0]).step(s, a)), pseed)
        sim_verdict, _ = verdict_and_margins(sim_metrics)
        entry: Dict[str, Any] = {
            "simulator": {**sim_metrics, "verdict": sim_verdict},
            "K": {},
        }
        prev_tv = None
        for K in K_GRID:
            models = []
            for mseed in MODEL_SEEDS:
                data = collect_trajectories(q_table, K,
                                            pseed * 1000 + mseed)
                models.append(WorldModel(data))
            model_rows = []
            for i, model in enumerate(models):
                metrics = probe(q_table, model.step, pseed + 17 * i)
                v, margins = verdict_and_margins(metrics)
                model_rows.append({"metrics": metrics, "verdict": v,
                                   "margins": margins})
            dists = [r["metrics"]["natural_dist"] for r in model_rows]
            eps_hat = (sum(tv(dists[i], dists[j])
                           for i in range(len(dists))
                           for j in range(i + 1, len(dists)))
                       / (len(dists) * (len(dists) - 1) / 2))
            sim_tv = sum(tv(d, sim_metrics["natural_dist"])
                         for d in dists) / len(dists)
            for row in model_rows:
                decidable = {
                    "potential_bits": abs(row["margins"]["potential_bits"])
                    > TV_SCALE * eps_hat,
                    "specificity_js": abs(row["margins"]["specificity_js"])
                    > TV_SCALE * eps_hat,
                    "usefulness_gap": abs(row["margins"]["usefulness_gap"])
                    > RETURN_RANGE * eps_hat,
                }
                row["decidable"] = decidable
                row["abstain"] = not all(decidable.values())
                if row["verdict"] != sim_verdict:
                    if K == K_GRID[-1] and not row["abstain"]:
                        wm2_violations += 1
                    disagree_components = [
                        c for c in decidable
                        if (row["margins"][c] > 0) != (
                            verdict_and_margins(sim_metrics)[1][c] > 0)]
                    if any(decidable[c] for c in disagree_components):
                        wm3_violations += 1
            entry["K"][str(K)] = {
                "ensemble_tv_error": eps_hat,
                "mean_tv_to_simulator": sim_tv,
                "models": model_rows,
                "verdict_matches": sum(
                    r["verdict"] == sim_verdict for r in model_rows),
                "abstentions": sum(r["abstain"] for r in model_rows),
            }
            if prev_tv is not None and sim_tv > prev_tv + 1e-9:
                wm1_ok = False
            prev_tv = sim_tv
            print(f"  K={K}: eps_hat {eps_hat:.4f}, tv_sim {sim_tv:.4f}, "
                  f"match {entry['K'][str(K)]['verdict_matches']}/5, "
                  f"abstain {entry['K'][str(K)]['abstentions']}/5",
                  flush=True)
        report["policy_seeds"][str(pseed)] = entry

    report["registered_outcomes"] = {
        "WM1_calibration_monotone": bool(wm1_ok),
        "WM2_no_silent_wrong_verdict_at_K20000": wm2_violations == 0,
        "WM3_margin_rule_catches_all_mismatches": wm3_violations == 0,
        "wm2_violations": wm2_violations,
        "wm3_violations": wm3_violations,
    }
    out = OUTPUTS / "world_model_closure.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
