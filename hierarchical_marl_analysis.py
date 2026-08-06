"""Seed-aware uncertainty and power audit for deep-MARL do-contrasts.

Evaluation episodes are nested within trained policy seeds. This script avoids
treating episodes as independent training replications by reporting:

- per-seed means, medians and positive fractions;
- exact seed-level sign tests;
- leave-one-seed-out summaries;
- a two-stage cluster bootstrap (seeds, then episodes);
- an approximate random-intercept ICC/effective sample size;
- prospective seed-count planning under an explicit normal random-effects
  assumption.

With only three observed seeds, population-level results remain exploratory.
The power calculation is a planning tool, not evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
RNG = np.random.default_rng(20260711)
B = 20_000


def percentile_interval(values: np.ndarray) -> List[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def extract_simple_spread(extra_path: Path | None = None) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for seed in (11, 22, 33):
        data = json.loads(
            (OUTPUTS / f"deep_marl_collapse_mappo_seed{seed}.json").read_text()
        )
        cond = data["conditions"][f"trained_seed{seed}"]
        out[str(seed)] = np.asarray([
            episode["p_win_do_commit"] - episode["p_win_do_block"]
            for episode in cond["episodes"]
        ], dtype=float)
    if extra_path is not None and extra_path.exists():
        data = json.loads(extra_path.read_text())
        for name, cond in data["conditions"].items():
            if not name.startswith("trained_seed"):
                continue
            seed = name.replace("trained_seed", "")
            out[seed] = np.asarray([
                episode["p_win_do_commit"] - episode["p_win_do_block"]
                for episode in cond["episodes"]
            ], dtype=float)
    return out


def extract_lbf(extra_path: Path | None = None) -> Dict[str, np.ndarray]:
    data = json.loads((OUTPUTS / "lbf_collapse_main.json").read_text())
    out: Dict[str, np.ndarray] = {}
    for seed in (11, 22, 33):
        cond = data["conditions"][f"trained_seed{seed}"]
        out[str(seed)] = np.asarray([
            episode["p_win_do_commit"] - episode["p_win_do_block"]
            for episode in cond["episodes"]
        ], dtype=float)
    if extra_path is not None and extra_path.exists():
        data = json.loads(extra_path.read_text())
        for name, cond in data["conditions"].items():
            if not name.startswith("trained_seed"):
                continue
            seed = name.replace("trained_seed", "")
            out[seed] = np.asarray([
                episode["p_win_do_commit"] - episode["p_win_do_block"]
                for episode in cond["episodes"]
            ], dtype=float)
    return out


def exact_seed_sign_p(values: np.ndarray) -> float:
    wins = int(np.sum(values > 0))
    losses = int(np.sum(values < 0))
    n = wins + losses
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def cluster_bootstrap(seed_episodes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    labels = list(seed_episodes)
    k = len(labels)
    means = np.empty(B)
    medians = np.empty(B)
    seed_mean_stats = np.empty(B)
    for draw in range(B):
        selected = RNG.choice(labels, size=k, replace=True)
        pooled = []
        selected_means = []
        for label in selected:
            values = seed_episodes[label]
            resampled = RNG.choice(values, size=len(values), replace=True)
            pooled.extend(resampled.tolist())
            selected_means.append(float(np.mean(resampled)))
        means[draw] = np.mean(pooled)
        medians[draw] = np.median(pooled)
        seed_mean_stats[draw] = np.mean(selected_means)
    pooled_original = np.concatenate(list(seed_episodes.values()))
    return {
        "pooled_mean": float(np.mean(pooled_original)),
        "pooled_mean_cluster_ci95": percentile_interval(means),
        "pooled_median": float(np.median(pooled_original)),
        "pooled_median_cluster_ci95": percentile_interval(medians),
        "mean_of_seed_means": float(np.mean([
            np.mean(values) for values in seed_episodes.values()
        ])),
        "mean_of_seed_means_cluster_ci95": percentile_interval(seed_mean_stats),
        "bootstrap_probability_mean_positive": float(np.mean(seed_mean_stats > 0)),
        "note": (
            "Two-stage bootstrap resamples three observed policy seeds and then "
            "episodes; with K=3 it cannot establish broad training-population generality."
        ),
    }


def random_intercept_diagnostics(seed_episodes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    arrays = list(seed_episodes.values())
    k = len(arrays)
    m = float(np.mean([len(values) for values in arrays]))
    seed_means = np.asarray([np.mean(values) for values in arrays])
    within_var = float(np.mean([np.var(values, ddof=1) for values in arrays]))
    observed_between = float(np.var(seed_means, ddof=1))
    between_var = max(0.0, observed_between - within_var / m)
    denominator = between_var + within_var
    icc = between_var / denominator if denominator > 0 else 0.0
    n_episodes = sum(len(values) for values in arrays)
    design_effect = 1.0 + (m - 1.0) * icc
    effective_n = n_episodes / design_effect
    return {
        "within_seed_variance": within_var,
        "between_seed_variance_method_of_moments": between_var,
        "icc": icc,
        "mean_cluster_size": m,
        "episode_count": n_episodes,
        "design_effect": design_effect,
        "episode_equivalent_effective_n": effective_n,
        "warning": "Variance components are highly unstable with only three seeds.",
    }


def power_plan(seed_effects: np.ndarray) -> Dict[str, Any]:
    mean_effect = float(np.mean(seed_effects))
    sd_effect = float(np.std(seed_effects, ddof=1))
    plans: Dict[str, Any] = {}
    for n_seeds in (5, 8, 10, 12, 15, 20):
        if sd_effect == 0:
            power = 1.0 if mean_effect > 0 else 0.0
        else:
            sims = RNG.normal(mean_effect, sd_effect, size=(20_000, n_seeds))
            sample_mean = np.mean(sims, axis=1)
            sample_sd = np.std(sims, axis=1, ddof=1)
            t_stat = sample_mean / (sample_sd / np.sqrt(n_seeds))
            threshold = stats.t.ppf(0.95, df=n_seeds - 1)
            power = float(np.mean(t_stat > threshold))
        plans[str(n_seeds)] = power
    return {
        "assumed_seed_effect_mean": mean_effect,
        "assumed_seed_effect_sd": sd_effect,
        "one_sided_alpha": 0.05,
        "simulated_power": plans,
        "warning": (
            "Planning simulation assumes normally distributed seed effects and "
            "uses mean/SD estimated from only three seeds."
        ),
    }


def analyse_domain(seed_episodes: Dict[str, np.ndarray]) -> Dict[str, Any]:
    per_seed = {}
    for seed, values in seed_episodes.items():
        per_seed[seed] = {
            "n_episodes": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "positive_fraction": float(np.mean(values > 0)),
            "negative_fraction": float(np.mean(values < 0)),
            "zero_fraction": float(np.mean(values == 0)),
        }
    seed_means = np.asarray([entry["mean"] for entry in per_seed.values()])
    seed_medians = np.asarray([entry["median"] for entry in per_seed.values()])
    leave_one_out = {}
    seeds = list(per_seed)
    for omitted in seeds:
        kept = [per_seed[seed]["mean"] for seed in seeds if seed != omitted]
        leave_one_out[omitted] = float(np.mean(kept))
    return {
        "per_seed": per_seed,
        "seed_level": {
            "mean_effects": seed_means.tolist(),
            "median_effects": seed_medians.tolist(),
            "mean_of_seed_means": float(np.mean(seed_means)),
            "median_of_seed_medians": float(np.median(seed_medians)),
            "exact_one_sided_sign_p_seed_means": exact_seed_sign_p(seed_means),
            "exact_one_sided_sign_p_seed_medians": exact_seed_sign_p(seed_medians),
            "leave_one_seed_out_mean": leave_one_out,
        },
        "cluster_bootstrap": cluster_bootstrap(seed_episodes),
        "random_intercept_diagnostics": random_intercept_diagnostics(seed_episodes),
        "power_planning": power_plan(seed_means),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple_spread_extra", type=Path)
    parser.add_argument("--lbf_extra", type=Path)
    parser.add_argument("--output", type=Path,
                        default=OUTPUTS / "hierarchical_marl_analysis.json")
    args = parser.parse_args()
    result = {
        "status": "exploratory seed-aware re-analysis and prospective planning",
        "simple_spread": analyse_domain(extract_simple_spread(args.simple_spread_extra)),
        "lbf": analyse_domain(extract_lbf(args.lbf_extra)),
        "inference_note": (
            "Episodes quantify evaluation uncertainty conditional on policies. "
            "Training-population inference is seed-level; exact sign-test "
            "resolution is limited by the number of trained policy seeds."
        ),
    }
    path = args.output
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for domain in ("simple_spread", "lbf"):
        seed = result[domain]["seed_level"]
        boot = result[domain]["cluster_bootstrap"]
        print(
            f"{domain}: seed means={seed['mean_effects']}, "
            f"seed-sign p={seed['exact_one_sided_sign_p_seed_means']:.3f}, "
            f"cluster CI={boot['mean_of_seed_means_cluster_ci95']}"
        )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
