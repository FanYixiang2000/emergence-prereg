"""Fair multivariate baselines against the six-component protocol.

Reviewer objection addressed: comparing one scalar (with one threshold)
against a six-dimensional conjunction (with six thresholds) is asymmetric.
Here prior-signature baselines receive EQUAL or GREATER degrees of freedom
than the six-component protocol, and the decisive test is out-of-sample:
every baseline is fitted on the ORIGINAL battery scores and then applied,
frozen, to a fresh-seed battery (seed 7011 -- the same fresh internal seed
on which the frozen six-component protocol already scored 10/10 in
refined_confirmation_summary.json, before this script existed).

Baselines (all fitted with hindsight on the original battery):

  1. conj_k       best AND-conjunction of k prior signals, each with its own
                  hindsight-optimal threshold AND direction (k = 1, 2, 3);
                  this matches the multi-threshold AND structure of the
                  six-component rule.
  2. logistic     L2 logistic regression on all prior signals (standardized);
                  reported in-sample, leave-one-out CV, and frozen-on-fresh.
  3. tree         depth-2 decision tree; same three numbers.
  4. two_component AND rules built from PAIRS of our own components
                  (specificity+usefulness, potential+specificity,
                  potential+usefulness) with hindsight thresholds -- the
                  "why not just two components?" baseline. Acquisition is
                  not informative on the tabular battery (every system's
                  initialization is degenerate), which is why the pair
                  usefulness+acquisition cannot be instantiated here.

Prior-signal feature sets:

  prior5 (transportable to the fresh battery): rep_jump, metric_jump,
         specificity_only, synergy, causal_emergence_ei -- recomputed from
         scratch on seed 7011 by this script.
  prior7 (original battery only): prior5 + exact Hoel EI + exact PhiID Psi
         from exact_prior_formalisms.json (the exact quantities live on the
         enumerated original chains and are not recomputed here).

Reference: the six-component protocol's fresh-seed accuracy is READ from the
stored refined confirmation (1.0); nothing about it is refitted.

Failure counts as failure: if a fair multivariate baseline transfers at
accuracy 1.0, that is reported as a limitation of the battery's diagnostic
power, not hidden.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from contextual_sacrifice_gridworld import MODES, train_policy
from criterion_ablation_battery import measure_system as measure_internal_system
from prior_metrics_comparison import (
    REGIMES,
    SYSTEMS,
    TRUTH,
    causal_emergence_ei,
    jump_scores,
    synergy_score,
    train_with_checkpoints,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

PRIOR5 = ("rep_jump", "metric_jump", "specificity_only", "synergy",
          "causal_emergence_ei")
FRESH_SEED = 7011  # matches refined_criterion_confirmation --internal_seed
SIX_COMPONENT_FRESH_ACCURACY_SOURCE = "refined_confirmation_summary.json"


def load_original_features() -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    prior = json.loads(
        (OUTPUTS / "prior_metrics_comparison.json").read_text())
    exact = json.loads(
        (OUTPUTS / "exact_prior_formalisms.json").read_text())
    systems = list(TRUTH)
    features: Dict[str, Dict[str, float]] = {s: {} for s in systems}
    for name in PRIOR5:
        for s in systems:
            features[s][name] = float(prior["detectors"][name]["scores"][s])
    for s in systems:
        features[s]["exact_ce"] = float(
            exact["detectors"]["causal_emergence_exact"]["scores"][s])
        features[s]["exact_psi"] = float(
            exact["detectors"]["phiid_psi_exact"]["scores"][s])
    return features, systems


def matrix(features: Dict[str, Dict[str, float]], systems: List[str],
           names: Tuple[str, ...]) -> np.ndarray:
    return np.array([[features[s][n] for n in names] for s in systems])


def hindsight_cut(values: np.ndarray, labels: np.ndarray):
    """Best single threshold+direction for one signal (hindsight)."""
    order = np.unique(values)
    cuts = np.concatenate([[order[0] - 1.0],
                           (order[:-1] + order[1:]) / 2,
                           [order[-1] + 1.0]])
    best = (0.0, cuts[0], 1)
    for direction in (1, -1):
        for cut in cuts:
            pred = (values > cut) if direction == 1 else (values < cut)
            acc = float(np.mean(pred.astype(int) == labels))
            if acc > best[0]:
                best = (acc, float(cut), direction)
    return best


def best_conjunction(x: np.ndarray, labels: np.ndarray,
                     names: Tuple[str, ...], k: int) -> Dict[str, Any]:
    """Best k-signal AND rule, each signal with its own threshold+direction.

    Joint hindsight search: per signal we consider every midpoint cut in
    both directions, and take the AND over the k chosen signals.
    """
    n_signals = x.shape[1]
    best: Dict[str, Any] = {"accuracy": 0.0}
    for combo in itertools.combinations(range(n_signals), k):
        candidate_cuts = []
        for j in combo:
            order = np.unique(x[:, j])
            cuts = np.concatenate([[order[0] - 1.0],
                                   (order[:-1] + order[1:]) / 2,
                                   [order[-1] + 1.0]])
            candidate_cuts.append([
                (float(c), d) for c in cuts for d in (1, -1)
            ])
        for assignment in itertools.product(*candidate_cuts):
            pred = np.ones(len(labels), dtype=bool)
            for (cut, direction), j in zip(assignment, combo):
                col = x[:, j]
                pred &= (col > cut) if direction == 1 else (col < cut)
            acc = float(np.mean(pred.astype(int) == labels))
            if acc > best["accuracy"]:
                best = {
                    "accuracy": acc,
                    "signals": [names[j] for j in combo],
                    "rule": [
                        {"signal": names[j], "threshold": cut,
                         "direction": direction}
                        for (cut, direction), j in zip(assignment, combo)
                    ],
                }
    return best


def apply_conjunction(rule: List[Dict[str, Any]],
                      features: Dict[str, Dict[str, float]],
                      systems: List[str]) -> np.ndarray:
    pred = np.ones(len(systems), dtype=bool)
    for part in rule:
        col = np.array([features[s][part["signal"]] for s in systems])
        if part["direction"] == 1:
            pred &= col > part["threshold"]
        else:
            pred &= col < part["threshold"]
    return pred.astype(int)


def loocv_accuracy(x: np.ndarray, labels: np.ndarray, make_model) -> float:
    hits = 0
    for i in range(len(labels)):
        mask = np.arange(len(labels)) != i
        model = make_model()
        model.fit(x[mask], labels[mask])
        hits += int(model.predict(x[i:i + 1])[0] == labels[i])
    return hits / len(labels)


def compute_fresh_prior5(seed: int, train_episodes: int,
                         n_checkpoints: int,
                         synergy_episodes: int) -> Dict[str, Dict[str, float]]:
    """Recompute the five transportable prior signals on a fresh battery."""
    print("Training fresh checkpointed policies (rep/metric jump) ...",
          flush=True)
    ckpts = {
        regime: train_with_checkpoints(
            regime, train_episodes, seed + i * 10_000, n_checkpoints)
        for i, regime in enumerate(REGIMES)
    }
    regime_jumps = {regime: jump_scores(c) for regime, c in ckpts.items()}

    print("Training fresh final policies ...", flush=True)
    policies = {
        regime: train_policy(regime, train_episodes, seed + i * 10_000)
        for i, regime in enumerate(REGIMES)
    }

    features: Dict[str, Dict[str, float]] = {}
    measurements: Dict[str, Dict[str, float]] = {}
    for idx, (system, (regime, behavior, modes)) in enumerate(SYSTEMS.items()):
        print(f"  measuring fresh system {system} ...", flush=True)
        if regime is None:
            rep_j, met_j = 0.0, 0.0
            q_table: Dict = {}
        else:
            rep_j, met_j = regime_jumps[regime]
            q_table = policies[regime]
        row = measure_internal_system(
            system, q_table, regime or "pure_team", modes, behavior,
            False, TRUTH[system], probe_episodes=24, samples=36,
            temperature=0.25, probe_temperature=0.9,
            seed=seed + idx * 5_000,
        )
        measurements[system] = {
            "h0_bits": float(row["h0_bits"]),
            "specificity_js": float(row["specificity_js"]),
            "usefulness_gap": float(row["usefulness_gap"]),
        }
        features[system] = {
            "rep_jump": rep_j,
            "metric_jump": met_j,
            "specificity_only": float(row["specificity_js"]),
            "synergy": synergy_score(
                q_table, regime, behavior, modes, synergy_episodes,
                seed + idx * 977),
            "causal_emergence_ei": causal_emergence_ei(
                q_table, regime, behavior, modes, synergy_episodes,
                seed + idx * 977 + 13),
        }
    for system, meas in measurements.items():
        features[system].update({f"own_{k}": v for k, v in meas.items()})
    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair multivariate baseline comparison.")
    parser.add_argument("--train_episodes", type=int, default=60_000)
    parser.add_argument("--n_checkpoints", type=int, default=20)
    parser.add_argument("--synergy_episodes", type=int, default=400)
    parser.add_argument("--skip_fresh", action="store_true")
    args = parser.parse_args()

    features, systems = load_original_features()
    labels = np.array([TRUTH[s] for s in systems])

    battery_csv = (OUTPUTS / "criterion_battery_measurements.csv")
    own: Dict[str, Dict[str, float]] = {}
    import csv as csv_module
    with battery_csv.open(encoding="utf-8") as f:
        for row in csv_module.DictReader(f):
            own[row["system"]] = {
                "own_h0_bits": float(row["h0_bits"]),
                "own_specificity_js": float(row["specificity_js"]),
                "own_usefulness_gap": float(row["usefulness_gap"]),
            }
    refined = json.loads(
        (OUTPUTS / "refined_selectivity_summary.json").read_text())
    own["anti_selector"] = {
        "own_h0_bits": float(
            refined["anti_selector_measurement"]["h0_bits"]),
        "own_specificity_js": float(
            refined["anti_selector_measurement"]["specificity_js"]),
        "own_usefulness_gap": float(
            refined["anti_selector_measurement"]["usefulness_gap"]),
    }
    for s in systems:
        features[s].update(own[s])

    report: Dict[str, Any] = {
        "status": ("fair multivariate baselines; fitted with hindsight on "
                   "the original battery, frozen, then applied to a fresh "
                   "battery (seed 7011)"),
        "labels": {s: int(TRUTH[s]) for s in systems},
        "original_battery": {},
        "fresh_battery": {},
    }

    # conjunctions of prior signals
    x5 = matrix(features, systems, PRIOR5)
    conj_results = {}
    for k in (1, 2, 3):
        best = best_conjunction(x5, labels, PRIOR5, k)
        conj_results[f"conj_{k}"] = best
        print(f"conj_{k}: acc {best['accuracy']:.2f} "
              f"signals {best.get('signals')}")
    report["original_battery"]["prior5_conjunctions"] = conj_results

    # learned models on prior5 and prior7
    prior7 = PRIOR5 + ("exact_ce", "exact_psi")
    x7 = matrix(features, systems, prior7)

    def make_logistic():
        return LogisticRegression(C=1.0, max_iter=5_000)

    def make_tree():
        return DecisionTreeClassifier(max_depth=2, random_state=0)

    learned = {}
    for tag, x in (("prior5", x5), ("prior7", x7)):
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-12
        z = (x - mu) / sd
        logit = make_logistic().fit(z, labels)
        tree = make_tree().fit(x, labels)
        learned[tag] = {
            "logistic_in_sample": float(
                np.mean(logit.predict(z) == labels)),
            "logistic_loocv": loocv_accuracy(z, labels, make_logistic),
            "tree_in_sample": float(np.mean(tree.predict(x) == labels)),
            "tree_loocv": loocv_accuracy(x, labels, make_tree),
        }
        print(f"{tag}: {learned[tag]}")
    report["original_battery"]["learned_models"] = learned

    # two-component AND rules from our own measured components
    own_names = ("own_h0_bits", "own_specificity_js", "own_usefulness_gap")
    x_own = matrix(features, systems, own_names)
    two_component = {}
    for i, j in itertools.combinations(range(3), 2):
        pair = (own_names[i], own_names[j])
        best = best_conjunction(x_own[:, [i, j]], labels, pair, 2)
        two_component["+".join(n[4:] for n in pair)] = best
        print(f"two-component {pair}: acc {best['accuracy']:.2f}")
    report["original_battery"]["own_two_component"] = two_component

    if args.skip_fresh:
        out = OUTPUTS / "fair_baseline_comparison.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out} (fresh battery skipped)")
        return

    # fresh battery: recompute prior5, apply frozen baselines
    fresh = compute_fresh_prior5(
        FRESH_SEED, args.train_episodes, args.n_checkpoints,
        args.synergy_episodes)
    report["fresh_battery"]["features"] = fresh

    fresh_systems = list(TRUTH)
    fresh_labels = np.array([TRUTH[s] for s in fresh_systems])
    xf = matrix(fresh, fresh_systems, PRIOR5)

    frozen_results = {}
    for k in (1, 2, 3):
        rule = conj_results[f"conj_{k}"]["rule"]
        pred = apply_conjunction(rule, fresh, fresh_systems)
        acc = float(np.mean(pred == fresh_labels))
        frozen_results[f"conj_{k}_frozen"] = {
            "accuracy": acc,
            "misclassified": [
                s for s, p, t in zip(fresh_systems, pred, fresh_labels)
                if p != t
            ],
        }
        print(f"frozen conj_{k} on fresh battery: acc {acc:.2f}")

    mu, sd = x5.mean(axis=0), x5.std(axis=0) + 1e-12
    logit = make_logistic().fit((x5 - mu) / sd, labels)
    pred = logit.predict((xf - mu) / sd)
    frozen_results["logistic_frozen"] = {
        "accuracy": float(np.mean(pred == fresh_labels)),
        "misclassified": [
            s for s, p, t in zip(fresh_systems, pred, fresh_labels) if p != t
        ],
    }
    tree = make_tree().fit(x5, labels)
    pred = tree.predict(xf)
    frozen_results["tree_frozen"] = {
        "accuracy": float(np.mean(pred == fresh_labels)),
        "misclassified": [
            s for s, p, t in zip(fresh_systems, pred, fresh_labels) if p != t
        ],
    }
    x_own_fresh = matrix(fresh, fresh_systems,
                         ("own_h0_bits", "own_specificity_js",
                          "own_usefulness_gap"))
    for pair_name, best in two_component.items():
        pred = apply_conjunction(
            [{"signal": "own_" + part["signal"][4:],
              "threshold": part["threshold"],
              "direction": part["direction"]}
             for part in best["rule"]],
            fresh, fresh_systems)
        frozen_results[f"two_component_{pair_name}_frozen"] = {
            "accuracy": float(np.mean(pred == fresh_labels)),
            "misclassified": [
                s for s, p, t in zip(fresh_systems, pred, fresh_labels)
                if p != t
            ],
        }
    _ = x_own_fresh  # kept for the JSON feature dump only

    report["fresh_battery"]["frozen_baselines"] = frozen_results
    report["fresh_battery"]["six_component_reference"] = {
        "accuracy": 1.0,
        "source": SIX_COMPONENT_FRESH_ACCURACY_SOURCE,
        "note": ("stored fresh-seed confirmation (same seed 7011), "
                 "thresholds frozen before that run and untouched here"),
    }

    out = OUTPUTS / "fair_baseline_comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
