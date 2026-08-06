"""Construct calibration of the continuous emergence profile.

Two decisive checks that the graded quantities measure DIFFERENT
things, on systems whose generating mechanism is exactly known.

Part 1 -- orthogonal (alpha, beta) grid.
    Synthetic collapse curves C_{alpha,beta}(t) = beta * sigmoid(alpha
    (t - t0)) with per-step future-basin laws constructed so that beta
    controls the total causal reorganization (final JS between do-laws)
    and alpha controls how suddenly it is acquired in training time.
    Registered expectations (declared here): M responds to beta and not
    to alpha; A responds to alpha and not to beta. The FIRST frozen
    formalization used Spearman rank correlations on the cross terms
    and FAILED (retained): the cross-axis series are practically
    constant (range < 1e-3), so their rank correlation is the rank of
    numerical jitter (values up to 1.0 by chance). The disclosed
    corrected rule scores responsiveness by effect size + rank
    (range > 0.1 AND Spearman >= 0.9) and orthogonality by practical
    constancy (range < 0.02). Both the original miss and the corrected
    outcome are reported.

Part 2 -- intervention dose-response.
    On five trained outcome-only gridworld policies (fresh seeds), the
    do-block intervention is applied with strength lambda in
    {0, .25, .5, .75, 1} (the forced non-trigger filter fires with
    probability lambda at each step before the switch). Registered
    expectations: the measured causal magnitude M(lambda') between
    do-trigger and lambda'-strength block increases monotonically in
    lambda' (Spearman >= 0.9 per seed mean), and the value contrast
    Delta U(lambda) decreases monotonically in lambda -- the continuous
    score tracks graded causal destruction rather than being a
    repackaged binary.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

import emergence_profile as ep
from contextual_sacrifice_gridworld import (
    ContextualSacrificeEnv,
    classify_basin,
    choose_softmax,
)
from strength_gradient_battery import MODE, SEEDS
from matched_provenance import train_q_table

OUTPUTS = Path(__file__).resolve().parent / "outputs"

ALPHAS = (0.5, 1.0, 2.0, 4.0)
BETAS = (0.2, 0.4, 0.6, 0.8)
T_GRID = 41
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EVAL_EPISODES = 300
PROBE_TEMPERATURE = 0.25


# ------------------------------------------------ part 1: (alpha, beta)

def synthetic_system(alpha: float, beta: float):
    """Two-basin process: do-trigger shifts mass beta toward basin 1 at
    the end of training; the shift is acquired with steepness alpha."""
    t0 = T_GRID // 2
    js_series = []
    entropy_series = []
    for t in range(T_GRID):
        frac = 1.0 / (1.0 + math.exp(-alpha * (t - t0)))
        shift = beta * frac
        p_trig = np.array([0.5 - shift / 2, 0.5 + shift / 2])
        p_block = np.array([0.5 + shift / 2, 0.5 - shift / 2])
        mix = 0.5 * (p_trig + p_block)
        js = (0.5 * sum(p * math.log2(p / m)
                        for p, m in zip(p_trig, mix) if p > 0)
              + 0.5 * sum(p * math.log2(p / m)
                          for p, m in zip(p_block, mix) if p > 0))
        js_series.append(js)
        h_natural = -sum(p * math.log2(p) for p in
                         0.5 * (p_trig + p_block) if p > 0)
        entropy_series.append(h_natural + (1 - frac) * beta)
    return js_series, entropy_series


def spearman(xs, ys) -> float:
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def part1() -> Dict:
    grid = {}
    for a in ALPHAS:
        for b in BETAS:
            js_series, entropy_series = synthetic_system(a, b)
            m = ep.magnitude_norm(js_series[-1])
            abrupt = ep.abruptness(entropy_series)
            grid[f"{a}|{b}"] = {"M": m, "A": abrupt}
    def series(axis, fixed, key):
        if axis == "beta":
            return [grid[f"{fixed}|{b}"][key] for b in BETAS]
        return [grid[f"{a}|{fixed}"][key] for a in ALPHAS]

    def responds(values, xs):
        rng_v = float(max(values) - min(values))
        sp = float(spearman(xs, values))
        return {"range": rng_v, "spearman": sp,
                "responsive": bool(rng_v > 0.1 and sp >= 0.9)}

    def flat(values):
        rng_v = float(max(values) - min(values))
        return {"range": rng_v, "practically_constant": bool(rng_v < 0.02)}

    m_beta = [responds(series("beta", a, "M"), BETAS) for a in ALPHAS]
    a_alpha = [responds(series("alpha", b, "A"), ALPHAS) for b in BETAS]
    m_alpha = [flat(series("alpha", b, "M")) for b in BETAS]
    a_beta = [flat(series("beta", a, "A")) for a in ALPHAS]
    return {
        "grid": grid,
        "M_responds_to_beta": m_beta,
        "A_responds_to_alpha": a_alpha,
        "M_flat_in_alpha": m_alpha,
        "A_flat_in_beta": a_beta,
        "rule": ("responsive: range > 0.1 and Spearman >= 0.9; "
                 "orthogonal: cross-axis range < 0.02 (rank statistics "
                 "are undefined on practically constant series)"),
        "CAL1_original_spearman_rule": "FAIL, retained (rank of "
                                       "jitter on constant series)",
        "CAL1_pass": bool(
            all(r["responsive"] for r in m_beta)
            and all(r["responsive"] for r in a_alpha)
            and all(f["practically_constant"] for f in m_alpha)
            and all(f["practically_constant"] for f in a_beta)),
    }


# --------------------------------------------- part 2: dose-response

def run_episode_partial_block(q_table, rng: random.Random,
                              lam: float) -> Dict:
    """Non-trigger forcing applied each pre-switch step w.p. lambda."""
    env = ContextualSacrificeEnv(MODE)
    state = env.reset()
    events: List[str] = []
    team = 0.0
    done = False
    while not done:
        forced = "non_trigger" if rng.random() < lam else None
        action = choose_softmax(q_table, state, "fixed",
                                PROBE_TEMPERATURE, rng, forced)
        result = env.step(state, action)
        events.extend(result.events)
        team += result.rewards[0] + result.rewards[1]
        state = result.state
        done = result.done
    return {"basin": classify_basin(events), "team": team}


def dist(rows: List[Dict]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["basin"]] = counts.get(r["basin"], 0) + 1
    return {b: c / len(rows) for b, c in counts.items()}


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


def run_episode_forced_trigger(q_table, rng: random.Random) -> Dict:
    env = ContextualSacrificeEnv(MODE)
    state = env.reset()
    events: List[str] = []
    team = 0.0
    done = False
    while not done:
        action = choose_softmax(q_table, state, "fixed",
                                PROBE_TEMPERATURE, rng, "trigger")
        result = env.step(state, action)
        events.extend(result.events)
        team += result.rewards[0] + result.rewards[1]
        state = result.state
        done = result.done
    return {"basin": classify_basin(events), "team": team}


def part2() -> Dict:
    seeds_out = {}
    for seed in SEEDS:
        print(f"dose-response: training outcome_only seed {seed}",
              flush=True)
        q_table = train_q_table("outcome_only", seed)
        rng = random.Random(seed + 4242)
        trig_rows = [run_episode_forced_trigger(q_table, rng)
                     for _ in range(EVAL_EPISODES)]
        natural_team = None
        rows_by_lambda = {}
        for lam in LAMBDAS:
            rows = [run_episode_partial_block(q_table, rng, lam)
                    for _ in range(EVAL_EPISODES)]
            rows_by_lambda[lam] = rows
            if lam == 0.0:
                natural_team = float(np.mean([r["team"] for r in rows]))
        m_curve = []
        du_curve = []
        for lam in LAMBDAS:
            rows = rows_by_lambda[lam]
            m = ep.magnitude_norm(js_bits(dist(trig_rows), dist(rows)))
            du = natural_team - float(np.mean([r["team"] for r in rows]))
            m_curve.append(m)
            du_curve.append(du)
        seeds_out[str(seed)] = {
            "lambdas": list(LAMBDAS),
            "M_vs_forced_trigger": m_curve,
            "value_degradation": du_curve,
            "spearman_M_lambda": spearman(LAMBDAS, m_curve),
            "spearman_dU_lambda": spearman(LAMBDAS, du_curve),
        }
        print(f"  M curve {['%.3f' % m for m in m_curve]} "
              f"dU curve {['%.2f' % d for d in du_curve]}", flush=True)
    mean_m = float(np.mean([seeds_out[str(s)]["spearman_M_lambda"]
                            for s in SEEDS]))
    mean_du = float(np.mean([seeds_out[str(s)]["spearman_dU_lambda"]
                             for s in SEEDS]))
    return {
        "seeds": seeds_out,
        "mean_spearman_M_lambda": mean_m,
        "mean_spearman_dU_lambda": mean_du,
        "CAL2_pass": bool(mean_m >= 0.9 and mean_du >= 0.9),
    }


def main() -> None:
    report = {
        "status": ("construct calibration of the continuous profile; "
                   "expectations declared in the docstring"),
        "part1_orthogonal_grid": part1(),
    }
    print("CAL1 (orthogonality):",
          report["part1_orthogonal_grid"]["CAL1_pass"], flush=True)
    report["part2_dose_response"] = part2()
    print("CAL2 (dose-response):",
          report["part2_dose_response"]["CAL2_pass"], flush=True)
    out = OUTPUTS / "profile_calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
