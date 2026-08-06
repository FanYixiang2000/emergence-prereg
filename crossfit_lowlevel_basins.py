"""Cross-fitted basin discovery from low-level trajectories only.

Reviewer objection addressed: the first machine-basin audit clustered
author-chosen behavioural summaries (first-consumed-food identity), which
is close to hand-made semantics. Here the inputs are LOW-LEVEL
state-action trajectories only -- per-step agent coordinates, per-step
actions, and the per-step reward sequence. No food identities, no
first-consumption events, no outcome summaries.

Cross-fitting: for each policy seed, the partition is fitted on the
even-indexed natural episodes of all five systems (train half) and then
FROZEN; all components are computed from the odd-indexed natural episodes
and all do-episodes (test half). The number of clusters is chosen on the
train half by silhouette score over k in 2..8, never by verdict outcome.

Methods compared (all fitted identically): k-means, Gaussian mixture,
Ward agglomerative (nearest-centroid assignment for held-out episodes),
and PCA-10 + k-means. The reported quantity is the fraction of
system-verdicts that match the hand-basin protocol, per method and
jointly -- Pr(same verdict | reasonable unsupervised method).

Registered predictions (frozen before running):
    XF-1  every fitted partition is identifiable
          (MI(cluster; trigger) >= 0.1 bits on held-out natural episodes)
          for >= 13/15 seeds per method;
    XF-2  controls are rejected in >= 58/60 verdicts per method;
    XF-3  mean verdict agreement with the hand-basin protocol across the
          four methods >= 0.85.

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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import contextual_lbf_transfer as clbf
import lbf_collapse_probe as base

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

N_EVAL = 80
SEED_STREAM = 13_000_000
K_RANGE = range(2, 9)
MI_IDENTIFIABILITY_BITS = 0.1
THRESHOLDS = clbf.THRESHOLDS
HORIZON = 15
N_ACTIONS = 6

CONFIRMATION_SEEDS = (1101, 1102, 1103, 1104, 1105,
                      1106, 1107, 1108, 1109, 1110)
EXTENSION_SEEDS = (1201, 1202, 1203, 1204, 1205)


def run_episode_lowlevel(controller: clbf.TeamController, context: int,
                         seed: int, intervention) -> Dict[str, Any]:
    """clbf.run_episode with per-step low-level recording."""
    env = clbf.make_contextual_env(seed, context)
    findex = base.FoodIndex(env)
    rng = random.Random(seed * 104_729 + 17)
    before = set(base.food_positions(env))
    order: List[int] = []
    total_reward = 0.0
    step_index = 0
    target = {"do_trigger": 0, "do_non_trigger": 1}.get(intervention)

    positions: List[List[float]] = []
    actions_taken: List[List[int]] = []
    rewards_seq: List[float] = []

    while not env.game_over and env.field.sum() > 0:
        active_target = (
            target if target is not None
            and findex.positions[target] in before else None
        )
        acts = controller.act(
            env, base.obs_all(env), rng, findex, active_target)
        acts = clbf.safe_actions(env, acts)
        positions.append([float(c) for p in env.players
                          for c in p.position])
        actions_taken.append(list(acts))
        _obs, rewards, _term, _trunc, _info = env.step(acts)
        step_reward = float(np.mean(rewards))
        rewards_seq.append(step_reward)
        total_reward += (base.GAMMA ** step_index) * step_reward
        step_index += 1
        after = set(base.food_positions(env))
        if after != before:
            order.extend(findex.consumed_now(before, after))
            before = after

    win = len(order) == 2
    trigger = bool(order and order[0] == 0)
    return {
        "trigger": int(trigger),
        "win": int(win),
        "context": context,
        "score": total_reward,
        "positions": positions,
        "actions": actions_taken,
        "rewards": rewards_seq,
    }


def lowlevel_features(row: Dict[str, Any]) -> np.ndarray:
    """Pad per-step coordinates/actions/rewards to fixed length; no
    outcome summaries."""
    pos = row["positions"]
    last = pos[-1] if pos else [0.0, 0.0, 0.0, 0.0]
    pos_padded = (pos + [last] * HORIZON)[:HORIZON]
    action_hist = np.zeros(2 * N_ACTIONS)
    for acts in row["actions"]:
        for agent, act in enumerate(acts):
            action_hist[agent * N_ACTIONS + int(act)] += 1
    if row["actions"]:
        action_hist /= len(row["actions"])
    rewards = (row["rewards"] + [0.0] * HORIZON)[:HORIZON]
    return np.concatenate([
        np.array(pos_padded).ravel() / 5.0,
        action_hist,
        np.array(rewards),
    ])


def collect(controller: clbf.TeamController,
            seed_offset: int) -> List[Dict[str, Any]]:
    rows = []
    for context in clbf.CONTEXTS:
        for episode in range(N_EVAL):
            paired_seed = seed_offset + 10_000 * context + episode
            for mode in (None, "do_trigger", "do_non_trigger"):
                row = run_episode_lowlevel(controller, context,
                                           paired_seed, mode)
                row["mode"] = mode or "natural"
                row["episode"] = episode
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


def mi_bits(a: np.ndarray, b: np.ndarray) -> float:
    joint: Dict[Any, int] = {}
    left: Dict[Any, int] = {}
    right: Dict[Any, int] = {}
    for x, y in zip(a, b):
        joint[(int(x), int(y))] = joint.get((int(x), int(y)), 0) + 1
        left[int(x)] = left.get(int(x), 0) + 1
        right[int(y)] = right.get(int(y), 0) + 1
    return entropy_bits(left) + entropy_bits(right) - entropy_bits(joint)


class FittedPartition:
    """Uniform predict() facade over the four clustering methods."""

    def __init__(self, method: str, train_x: np.ndarray, rng_seed: int):
        self.method = method
        self.scaler = StandardScaler().fit(train_x)
        z = self.scaler.transform(train_x)
        self.pca = None
        if method == "pca_kmeans":
            self.pca = PCA(n_components=10, random_state=rng_seed).fit(z)
            z = self.pca.transform(z)
        best = None
        for k in K_RANGE:
            labels, model = self._fit_k(z, k, rng_seed)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(z, labels)
            if best is None or score > best[0]:
                best = (score, k, model)
        self.silhouette, self.k, self.model = best

    def _fit_k(self, z: np.ndarray, k: int, rng_seed: int):
        if self.method in ("kmeans", "pca_kmeans"):
            model = KMeans(n_clusters=k, n_init=10,
                           random_state=rng_seed).fit(z)
            return model.labels_, model
        if self.method == "gmm":
            model = GaussianMixture(n_components=k, n_init=3,
                                    random_state=rng_seed).fit(z)
            return model.predict(z), model
        if self.method == "ward":
            agg = AgglomerativeClustering(n_clusters=k).fit(z)
            centroids = np.vstack([
                z[agg.labels_ == c].mean(axis=0) for c in range(k)])
            return agg.labels_, centroids
        raise ValueError(self.method)

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x)
        if self.pca is not None:
            z = self.pca.transform(z)
        if self.method == "ward":
            d = ((z[:, None, :] - self.model[None, :, :]) ** 2).sum(-1)
            return d.argmin(axis=1)
        return self.model.predict(z)


def components_from_labels(rows: List[Dict[str, Any]],
                           labels: np.ndarray) -> Dict[str, Any]:
    idx = {m: [i for i, r in enumerate(rows) if r["mode"] == m]
           for m in ("natural", "do_trigger", "do_non_trigger")}
    natural_rows = [rows[i] for i in idx["natural"]]
    trigger_rates = {
        str(c): float(np.mean([r["trigger"] for r in natural_rows
                               if r["context"] == c]))
        for c in clbf.CONTEXTS
    }

    def dist(indices):
        counts: Dict[int, int] = {}
        for i in indices:
            counts[int(labels[i])] = counts.get(int(labels[i]), 0) + 1
        total = len(indices)
        return {k: v / total for k, v in counts.items()}

    counts_nat: Dict[int, int] = {}
    for i in idx["natural"]:
        counts_nat[int(labels[i])] = counts_nat.get(int(labels[i]), 0) + 1
    return {
        "potential_bits": entropy_bits(counts_nat),
        "conditional_selectivity": abs(trigger_rates["0"]
                                       - trigger_rates["1"]),
        "specificity_js_bits": js_bits(dist(idx["do_trigger"]),
                                       dist(idx["do_non_trigger"])),
        "usefulness_gap": float(
            np.mean([rows[i]["score"] for i in idx["natural"]])
            - np.mean([rows[i]["score"] for i in idx["do_non_trigger"]])),
        "identifiability_mi_bits": mi_bits(
            labels[idx["natural"]],
            np.array([r["trigger"] for r in natural_rows])),
    }


def verdict(metrics: Dict[str, Any], endogenous: bool,
            acquisition: float) -> int:
    passes = (
        metrics["potential_bits"] >= THRESHOLDS["potential_bits"]
        and metrics["conditional_selectivity"]
        >= THRESHOLDS["conditional_selectivity"]
        and metrics["specificity_js_bits"]
        >= THRESHOLDS["specificity_js_bits"]
        and metrics["usefulness_gap"] > THRESHOLDS["usefulness_gap"]
        and endogenous
        and acquisition >= THRESHOLDS["acquisition"]
    )
    return int(passes)


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


def run_seed(seed: int, methods) -> Dict[str, Any]:
    offset = SEED_STREAM + seed * 100_000
    controllers = {
        "learned": clbf.TeamController("policy", load_net(seed)),
        "initial_twin": clbf.TeamController("policy",
                                            clbf.initial_twin(seed)),
        "team_nearest": clbf.TeamController("team_nearest"),
        "fixed_food0": clbf.TeamController("fixed_food0"),
        "fixed_food1": clbf.TeamController("fixed_food1"),
    }
    all_rows = {name: collect(ctrl, offset)
                for name, ctrl in controllers.items()}

    train_x = np.vstack([
        lowlevel_features(r)
        for rows in all_rows.values() for r in rows
        if r["mode"] == "natural" and r["episode"] % 2 == 0
    ])
    test_rows = {name: [r for r in rows
                        if r["mode"] != "natural" or r["episode"] % 2 == 1]
                 for name, rows in all_rows.items()}

    out: Dict[str, Any] = {}
    for method in methods:
        part = FittedPartition(method, train_x, seed)
        per_system: Dict[str, Any] = {}
        sel = {}
        for name, rows in test_rows.items():
            feats = np.vstack([lowlevel_features(r) for r in rows])
            metrics = components_from_labels(rows, part.predict(feats))
            per_system[name] = metrics
            sel[name] = metrics["conditional_selectivity"]
        acquisition = sel["learned"] - sel["initial_twin"]
        verdicts = {}
        for name, metrics in per_system.items():
            endo = name in ("learned", "initial_twin")
            acq = acquisition if name == "learned" else 0.0
            verdicts[name] = verdict(metrics, endo, acq)
        out[method] = {
            "k": part.k,
            "silhouette": float(part.silhouette),
            "metrics": per_system,
            "verdicts": verdicts,
            "learned_identifiable": bool(
                per_system["learned"]["identifiability_mi_bits"]
                >= MI_IDENTIFIABILITY_BITS),
        }
    return out


def main() -> None:
    random.seed(0)
    torch.set_num_threads(8)
    methods = ("kmeans", "gmm", "ward", "pca_kmeans")
    stored = stored_verdicts()
    expected = {"learned": 1, "initial_twin": 0, "team_nearest": 0,
                "fixed_food0": 0, "fixed_food1": 0}

    seeds = list(CONFIRMATION_SEEDS) + list(EXTENSION_SEEDS)
    results: Dict[str, Any] = {}
    tallies = {m: {"identifiable": 0, "controls_rejected": 0,
                   "agreement": 0, "learned_accepted": 0}
               for m in methods}
    for seed in seeds:
        print(f"cross-fit low-level basins, seed {seed}", flush=True)
        results[str(seed)] = run_seed(seed, methods)
        for method in methods:
            entry = results[str(seed)][method]
            tallies[method]["identifiable"] += int(
                entry["learned_identifiable"])
            for name, v in entry["verdicts"].items():
                tallies[method]["agreement"] += int(
                    v == stored[str(seed)][name])
                if name == "learned":
                    tallies[method]["learned_accepted"] += v
                else:
                    tallies[method]["controls_rejected"] += int(v == 0)
            print(f"  {method:11s} k={entry['k']} "
                  f"learned={entry['verdicts']['learned']} "
                  f"controls_ok="
                  f"{sum(1 for n, v in entry['verdicts'].items() if n != 'learned' and v == 0)}/4",
                  flush=True)

    n = len(seeds)
    agreement_rates = {m: tallies[m]["agreement"] / (5 * n)
                       for m in methods}
    summary = {
        "status": ("cross-fitted basin discovery from low-level "
                   "trajectories only (coordinates, actions, rewards); "
                   "k by silhouette; predictions XF-1..XF-3 frozen in the "
                   "docstring"),
        "per_method": {
            m: {
                "identifiable_seeds": f"{tallies[m]['identifiable']}/{n}",
                "controls_rejected": f"{tallies[m]['controls_rejected']}/{4 * n}",
                "learned_accepted": f"{tallies[m]['learned_accepted']}/{n}",
                "verdict_agreement_with_hand_basins":
                    f"{tallies[m]['agreement']}/{5 * n}",
            } for m in methods
        },
        "registered_outcomes": {
            "XF1_pass": all(tallies[m]["identifiable"] >= 13
                            for m in methods),
            "XF2_pass": all(tallies[m]["controls_rejected"] >= 58
                            for m in methods),
            "XF3_mean_agreement": float(np.mean(list(
                agreement_rates.values()))),
            "XF3_pass": float(np.mean(list(
                agreement_rates.values()))) >= 0.85,
        },
        "seeds": results,
    }
    out = OUTPUTS / "crossfit_lowlevel_basins.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["per_method"], indent=2))
    print(json.dumps(summary["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
