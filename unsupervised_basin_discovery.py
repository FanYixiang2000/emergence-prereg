"""Unsupervised basin discovery from raw event sequences.

Reviewer objection: "your future basins are hand-designed labels, so the
possibility-collapse metrics are circular." This experiment answers it by
recovering basins without labels:

1. Roll out the learned policy and record *raw* event sequences (not the
   canonical event library used elsewhere).
2. Embed each trajectory as a binary event-presence vector plus outcome
   features that any observer could log (episode length, team return sign).
3. Cluster with k-means (pure python, multiple restarts).
4. Only afterwards, compare discovered clusters to the hand labels (purity,
   per-cluster majority), and recompute the potential/collapse quantities from
   cluster labels to check they agree with the hand-label versions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from contextual_sacrifice_gridworld import (
    ContextualSacrificeEnv,
    REGIMES,
    choose_softmax,
    classify_basin,
    sample_mode,
    sample_preference_context,
    train_policy,
)


OUTPUTS = Path(__file__).resolve().parent / "outputs"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def entropy_from_probs(probs: Sequence[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(p + eps, 2) for p in probs if p > 0)


def collect_trajectories(
    q_table,
    regime: str,
    episodes: int,
    temperature: float,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for episode in range(episodes):
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context(regime, rng, episode)
        events: List[str] = []
        total = 0.0
        steps = 0
        done = False
        while not done:
            action = choose_softmax(q_table, state, context, temperature, rng, None)
            result = env.step(state, action)
            events.extend(result.events)
            total += result.rewards[0] + result.rewards[1]
            state = result.state
            steps += 1
            done = result.done
        rows.append(
            {
                "raw_events": tuple(events),
                "hand_basin": classify_basin(events),
                "team_return": total,
                "steps": steps,
            }
        )
    return rows


def build_features(rows: Sequence[Mapping[str, object]]) -> Tuple[List[List[float]], List[str]]:
    vocabulary = sorted({event for row in rows for event in row["raw_events"]})  # type: ignore[union-attr]
    features: List[List[float]] = []
    for row in rows:
        present = set(row["raw_events"])  # type: ignore[arg-type]
        vector = [1.0 if event in present else 0.0 for event in vocabulary]
        vector.append(float(row["steps"]) / 10.0)  # type: ignore[arg-type]
        vector.append(1.0 if float(row["team_return"]) > 0 else 0.0)  # type: ignore[arg-type]
        features.append(vector)
    return features, vocabulary


def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(
    features: Sequence[Sequence[float]],
    k: int,
    seed: int,
    restarts: int = 8,
    iterations: int = 40,
) -> List[int]:
    best_assignment: Optional[List[int]] = None
    best_cost = float("inf")
    n = len(features)
    dim = len(features[0])
    for restart in range(restarts):
        rng = random.Random(seed + restart * 977)
        centers = [list(features[rng.randrange(n)]) for _ in range(k)]
        assignment = [0] * n
        for _ in range(iterations):
            changed = False
            for i, vector in enumerate(features):
                distances = [squared_distance(vector, center) for center in centers]
                choice = distances.index(min(distances))
                if choice != assignment[i]:
                    assignment[i] = choice
                    changed = True
            for c in range(k):
                members = [features[i] for i in range(n) if assignment[i] == c]
                if members:
                    centers[c] = [
                        mean(member[d] for member in members) for d in range(dim)
                    ]
            if not changed:
                break
        cost = sum(
            squared_distance(features[i], centers[assignment[i]]) for i in range(n)
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = list(assignment)
    assert best_assignment is not None
    return best_assignment


def purity(clusters: Sequence[int], labels: Sequence[str]) -> float:
    total = len(labels)
    correct = 0
    for cluster in set(clusters):
        members = [labels[i] for i in range(total) if clusters[i] == cluster]
        correct += Counter(members).most_common(1)[0][1]
    return correct / max(total, 1)


def effective_modes(probs: Sequence[float]) -> float:
    return 2.0 ** entropy_from_probs(probs)


def distribution(labels: Sequence[object]) -> List[float]:
    counts = Counter(labels)
    total = sum(counts.values())
    return [count / total for count in counts.values()]


def run_regime(
    regime: str,
    train_episodes: int,
    eval_episodes: int,
    temperature: float,
    seed: int,
    k: int,
) -> Tuple[Dict[str, float | str], List[Dict[str, object]]]:
    q_table = train_policy(regime, train_episodes, seed)
    rows = collect_trajectories(q_table, regime, eval_episodes, temperature, seed + 55_501)
    features, _vocabulary = build_features(rows)
    clusters = kmeans(features, k=k, seed=seed + 777)
    hand_labels = [str(row["hand_basin"]) for row in rows]

    cluster_purity = purity(clusters, hand_labels)
    hand_modes = effective_modes(distribution(hand_labels))
    cluster_modes = effective_modes(distribution(clusters))
    cluster_majority: Dict[int, str] = {}
    for cluster in set(clusters):
        members = [hand_labels[i] for i in range(len(rows)) if clusters[i] == cluster]
        cluster_majority[cluster] = Counter(members).most_common(1)[0][0]

    detail = [
        {
            "regime": regime,
            "cluster": cluster,
            "majority_hand_basin": cluster_majority[cluster],
            "size": sum(1 for c in clusters if c == cluster),
        }
        for cluster in sorted(set(clusters))
    ]
    summary: Dict[str, float | str] = {
        "regime": regime,
        "n_trajectories": float(len(rows)),
        "n_clusters_used": float(len(set(clusters))),
        "cluster_purity": cluster_purity,
        "hand_effective_modes": hand_modes,
        "cluster_effective_modes": cluster_modes,
        "modes_absolute_error": abs(hand_modes - cluster_modes),
    }
    return summary, detail


def run_all(
    regimes: Sequence[str],
    train_episodes: int,
    eval_episodes: int,
    temperature: float,
    seed: int,
    k: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, float | str]] = []
    details: List[Dict[str, object]] = []
    for idx, regime in enumerate(regimes):
        summary, detail = run_regime(
            regime, train_episodes, eval_episodes, temperature, seed + idx * 30_000, k
        )
        summaries.append(summary)
        details.extend(detail)

    with (output_dir / "unsupervised_basin_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    with (output_dir / "unsupervised_basin_clusters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        for row in details:
            writer.writerow(row)
    (output_dir / "unsupervised_basin_summary.json").write_text(
        json.dumps({"summary": summaries}, indent=2),
        encoding="utf-8",
    )
    print("regime,purity,hand_modes,cluster_modes,abs_error")
    for row in summaries:
        print(
            f"{row['regime']},{float(row['cluster_purity']):.4f},"
            f"{float(row['hand_effective_modes']):.4f},"
            f"{float(row['cluster_effective_modes']):.4f},"
            f"{float(row['modes_absolute_error']):.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unsupervised basin discovery.")
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=["pure_team", "uncertain_preference", "random_noise"],
        choices=list(REGIMES),
    )
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--eval_episodes", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=4241)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(
        regimes=args.regimes,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        temperature=args.temperature,
        seed=args.seed,
        k=args.k,
        output_dir=args.output_dir,
    )
    print(f"\nWrote {args.output_dir / 'unsupervised_basin_summary.csv'}")
    print(f"Wrote {args.output_dir / 'unsupervised_basin_clusters.csv'}")


if __name__ == "__main__":
    main()
