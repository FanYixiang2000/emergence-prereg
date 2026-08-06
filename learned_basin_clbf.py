"""Machine-discovered basins: the full criterion without hand-made partitions.

Reviewer objection addressed: in every full-criterion domain the basin
partition was written by the authors, so the possibility space could be an
artifact of authorial understanding. Here the partition is DISCOVERED by
clustering raw trajectory summaries, with no semantic labels, and the full
six-component criterion is re-scored on all 75 stored Contextual-LBF
systems with frozen thresholds.

Observer construction (declared before running):
    For each policy seed, k-means (k=4, fixed init) is fitted on the pooled
    NATURAL episodes of all five systems of that seed, using only raw
    per-episode trajectory features -- first-consumed-food identity
    (one-hot over food0/food1/none, a trajectory event, not a verdict
    label), number of foods collected, and normalized episode length.
    The fitted clusters are then used as the basin partition for every
    measurement of that seed, including do-episodes.

Identifiability check (required by the measurement contract): the
discovered partition must resolve the candidate macro-structure, measured
as mutual information between cluster id and the trigger event >= 0.1 bits
on natural episodes. Partitions failing this are reported, not silently
replaced.

Only potential and specificity depend on the partition; selectivity,
usefulness, endogeneity and acquisition are re-measured from the same
fresh episodes with unchanged definitions.

Registered predictions (frozen before running):
    LB-1  all 60 control verdicts remain rejections;
    LB-2  learned systems accepted on >= 13/15 seeds;
    LB-3  verdict agreement with the stored hand-basin verdicts on
          >= 70/75 systems.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans

import contextual_lbf_transfer as clbf
import lbf_collapse_probe as base

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

N_EVAL = 80
SEED_STREAM = 11_000_000
K_CLUSTERS = 4
MI_IDENTIFIABILITY_BITS = 0.1
THRESHOLDS = clbf.THRESHOLDS

CONFIRMATION_SEEDS = (1101, 1102, 1103, 1104, 1105,
                      1106, 1107, 1108, 1109, 1110)
EXTENSION_SEEDS = (1201, 1202, 1203, 1204, 1205)


def episode_features(row: Dict[str, Any]) -> List[float]:
    order = row["order"]
    first = order[0] if order else None
    return [
        1.0 if first == 0 else 0.0,
        1.0 if first == 1 else 0.0,
        1.0 if first is None else 0.0,
        len(order) / 2.0,
        row["steps"] / 15.0,
    ]


def collect_rows(controller: clbf.TeamController,
                 seed_offset: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for context in clbf.CONTEXTS:
        for episode in range(N_EVAL):
            paired_seed = seed_offset + 10_000 * context + episode
            for mode in (None, "do_trigger", "do_non_trigger"):
                row = clbf.run_episode(controller, context, paired_seed, mode)
                row["mode"] = mode or "natural"
                rows.append(row)
    return rows


def entropy_bits(counts: Dict[Any, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total)
                for c in counts.values() if c > 0)


def js_bits(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    out = 0.0
    for key in set(p) | set(q):
        a, b = p.get(key, 0.0), q.get(key, 0.0)
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def cluster_distribution(rows: List[Dict[str, Any]],
                         labels: np.ndarray) -> Dict[int, float]:
    counts: Dict[int, int] = {}
    for lab in labels:
        counts[int(lab)] = counts.get(int(lab), 0) + 1
    total = len(labels)
    return {k: v / total for k, v in counts.items()}


def mutual_information_bits(labels: np.ndarray,
                            trigger: np.ndarray) -> float:
    joint: Dict[Any, int] = {}
    left: Dict[Any, int] = {}
    right: Dict[Any, int] = {}
    for a, b in zip(labels, trigger):
        joint[(int(a), int(b))] = joint.get((int(a), int(b)), 0) + 1
        left[int(a)] = left.get(int(a), 0) + 1
        right[int(b)] = right.get(int(b), 0) + 1
    return (entropy_bits(left) + entropy_bits(right)
            - entropy_bits(joint))


def metrics_with_clusters(rows: List[Dict[str, Any]],
                          km: KMeans) -> Dict[str, Any]:
    feats = np.array([episode_features(r) for r in rows])
    labels = km.predict(feats)
    natural_idx = [i for i, r in enumerate(rows) if r["mode"] == "natural"]
    trig_idx = [i for i, r in enumerate(rows) if r["mode"] == "do_trigger"]
    non_idx = [i for i, r in enumerate(rows)
               if r["mode"] == "do_non_trigger"]

    natural_rows = [rows[i] for i in natural_idx]
    trigger_rates = {
        str(c): float(np.mean([r["trigger"] for r in natural_rows
                               if r["context"] == c]))
        for c in clbf.CONTEXTS
    }
    natural_counts: Dict[int, int] = {}
    for i in natural_idx:
        natural_counts[int(labels[i])] = (
            natural_counts.get(int(labels[i]), 0) + 1)
    return {
        "potential_bits": entropy_bits(natural_counts),
        "conditional_selectivity": abs(
            trigger_rates["0"] - trigger_rates["1"]),
        "trigger_rates": trigger_rates,
        "specificity_js_bits": js_bits(
            cluster_distribution(rows, labels[trig_idx]),
            cluster_distribution(rows, labels[non_idx])),
        "usefulness_gap": float(
            np.mean([rows[i]["score"] for i in natural_idx])
            - np.mean([rows[i]["score"] for i in non_idx])),
        "identifiability_mi_bits": mutual_information_bits(
            labels[natural_idx],
            np.array([r["trigger"] for r in natural_rows])),
    }


def verdict(metrics: Dict[str, Any], endogenous: bool,
            acquisition: float) -> Dict[str, Any]:
    passes = {
        "potential": metrics["potential_bits"] >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": (
            metrics["conditional_selectivity"]
            >= THRESHOLDS["conditional_selectivity"]),
        "specificity": (
            metrics["specificity_js_bits"]
            >= THRESHOLDS["specificity_js_bits"]),
        "usefulness": metrics["usefulness_gap"] > THRESHOLDS["usefulness_gap"],
        "endogeneity": endogenous,
        "acquisition": acquisition >= THRESHOLDS["acquisition"],
    }
    return {"passes": passes, "emergent": int(all(passes.values())),
            "failed": [k for k, ok in passes.items() if not ok]}


def load_net(seed: int) -> base.PolicyNet:
    net = base.PolicyNet()
    net.load_state_dict(torch.load(
        OUTPUTS / f"contextual_lbf_net_seed{seed}.pt", map_location="cpu"))
    net.eval()
    return net


def stored_verdicts() -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for path, seeds in (
        (OUTPUTS / "contextual_lbf_confirmation.json", CONFIRMATION_SEEDS),
        (OUTPUTS / "contextual_lbf_extension.json", EXTENSION_SEEDS),
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        for seed in seeds:
            out[str(seed)] = {
                name: int(s["verdict"]["emergent"])
                for name, s in data["seeds"][str(seed)]["systems"].items()
            }
    return out


def run_seed(seed: int) -> Dict[str, Any]:
    offset = SEED_STREAM + seed * 100_000
    learned_net = load_net(seed)
    twin = clbf.initial_twin(seed)
    controllers = {
        "learned": clbf.TeamController("policy", learned_net),
        "initial_twin": clbf.TeamController("policy", twin),
        "team_nearest": clbf.TeamController("team_nearest"),
        "fixed_food0": clbf.TeamController("fixed_food0"),
        "fixed_food1": clbf.TeamController("fixed_food1"),
    }
    all_rows = {name: collect_rows(ctrl, offset)
                for name, ctrl in controllers.items()}

    pooled_natural = [r for rows in all_rows.values() for r in rows
                      if r["mode"] == "natural"]
    feats = np.array([episode_features(r) for r in pooled_natural])
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10,
                random_state=seed).fit(feats)

    metrics = {name: metrics_with_clusters(rows, km)
               for name, rows in all_rows.items()}
    acquisition = (metrics["learned"]["conditional_selectivity"]
                   - metrics["initial_twin"]["conditional_selectivity"])
    out: Dict[str, Any] = {}
    for name in controllers:
        endo = name in ("learned", "initial_twin")
        acq = acquisition if name == "learned" else 0.0
        out[name] = {
            "metrics": metrics[name],
            "acquisition": acq,
            "verdict": verdict(metrics[name], endo, acq),
        }
    return out


def main() -> None:
    random.seed(0)
    torch.set_num_threads(8)
    expected = {"learned": 1, "initial_twin": 0, "team_nearest": 0,
                "fixed_food0": 0, "fixed_food1": 0}
    stored = stored_verdicts()

    seeds = list(CONFIRMATION_SEEDS) + list(EXTENSION_SEEDS)
    results: Dict[str, Any] = {}
    learned_accepted = 0
    controls_rejected = 0
    agreement = 0
    total = 0
    identifiable = 0
    for seed in seeds:
        print(f"learned basins, seed {seed}", flush=True)
        results[str(seed)] = run_seed(seed)
        for name, entry in results[str(seed)].items():
            v = entry["verdict"]["emergent"]
            total += 1
            agreement += int(v == stored[str(seed)][name])
            if name == "learned":
                learned_accepted += v
                identifiable += int(
                    entry["metrics"]["identifiability_mi_bits"]
                    >= MI_IDENTIFIABILITY_BITS)
            else:
                controls_rejected += int(v == 0)
            marker = "" if v == expected[name] else "  <-- differs"
            print(f"  {name:14s} verdict {v} "
                  f"pot {entry['metrics']['potential_bits']:.2f} "
                  f"spec {entry['metrics']['specificity_js_bits']:.2f} "
                  f"mi {entry['metrics']['identifiability_mi_bits']:.2f}"
                  f"{marker}", flush=True)

    summary = {
        "status": ("machine-discovered basins (k-means on raw trajectory "
                   "features, no semantic labels); frozen thresholds; "
                   "predictions LB-1..LB-3 frozen in the docstring"),
        "observer": {
            "k": K_CLUSTERS,
            "features": ["first_food_onehot(3)", "n_collected", "steps/15"],
            "fit_population": "pooled natural episodes of the seed's five "
                              "systems",
            "identifiability_rule": f">= {MI_IDENTIFIABILITY_BITS} bits "
                                    "MI(cluster; trigger)",
        },
        "registered_outcomes": {
            "LB1_controls_rejected": f"{controls_rejected}/{4 * len(seeds)}",
            "LB2_learned_accepted": f"{learned_accepted}/{len(seeds)}",
            "LB3_agreement_with_hand_basins": f"{agreement}/{total}",
            "learned_partitions_identifiable": f"{identifiable}/{len(seeds)}",
            "LB1_pass": controls_rejected == 4 * len(seeds),
            "LB2_pass": learned_accepted >= 13,
            "LB3_pass": agreement >= 70,
        },
        "seeds": results,
    }
    out = OUTPUTS / "learned_basin_clbf.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
