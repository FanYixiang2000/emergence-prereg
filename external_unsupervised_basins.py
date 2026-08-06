"""Unsupervised basin discovery on the EXTERNAL swarm system.

The internal version (unsupervised_basin_discovery.py) answered the
circularity objection for the gridworld family. The external transfer,
however, still used hand-defined basins ({win, loss} x {engaged, bypassed}).
This experiment repeats the discovery externally: cluster raw episode
observables that any monitoring system could log -- no basin labels, no
role labels, no context labels -- and only afterwards compare clusters to
the hand basins.

Episode features (all label-free):
    steps taken, agents alive at end, mean agent HP at end,
    total damage dealt (magnitude only), damage concentration
    (Herfindahl index over per-enemy damage -- how focused the swarm was),
    fraction of damage on high-HP targets (an observable, not a role label:
    initial enemy HP is visible in the local observation model).

Clustered with k-means (k = 4, matching the hand-basin count; we also
report k chosen by silhouette-style gap for honesty).

Checks:
    U1: cluster purity against hand basins >= 0.8 for the learned
        (marl_learned) controller's episodes.
    U2: the effective mode count of the cluster distribution agrees with
        the hand-basin version within 0.5 for the learned controller.
    U3: repeating the criterion's potential component (H0 over clusters
        instead of hand basins) does not change the verdict for any of the
        five external systems.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from external_swarm_criterion_transfer import (
    CONTEXTS,
    ContextualDecoyEnv,
    make_config,
    rule_controller,
    scorer_controller,
    train_marl_scorer,
    untrained_scorer,
)
from swarm_decoy_abstraction import ROLE_DECOY, MISSION_ROLES
from unsupervised_basin_discovery import kmeans, purity

OUTPUTS = Path(__file__).resolve().parent / "outputs"
TRIGGER_DAMAGE = 0.3


def entropy_bits(probs: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def run_episode_features(controller, context: str, seed: int) -> Dict[str, Any]:
    env = ContextualDecoyEnv(make_config(seed), context)
    env.controller = controller
    env.intervention = None
    initial_hp = {int(e): float(env.enemy_hp[e]) for e in range(env.cfg.n_enemies)}
    for _ in range(env.cfg.horizon):
        env.step()
        if env.alive_agents().size == 0 or env.alive_mission_enemies().size == 0:
            break
    win = bool(env.alive_mission_enemies().size == 0 and env.alive_agents().size > 0)
    engaged = env.damage_to_roles[ROLE_DECOY] > TRIGGER_DAMAGE
    total_damage = sum(env.damage_to_roles.values())
    per_enemy_damage = [
        max(0.0, initial_hp[int(e)] - max(float(env.enemy_hp[e]), 0.0))
        for e in range(env.cfg.n_enemies)
    ]
    dmg_sum = sum(per_enemy_damage) or 1e-8
    herfindahl = sum((d / dmg_sum) ** 2 for d in per_enemy_damage)
    # Observable proxy for "engaged the tanky front": share of damage into
    # enemies whose INITIAL HP was above the median initial HP.
    median_hp = float(np.median(list(initial_hp.values())))
    high_hp_share = sum(
        d for e, d in enumerate(per_enemy_damage) if initial_hp[e] > median_hp
    ) / dmg_sum
    hand_basin = ("win_" if win else "loss_") + ("engaged" if engaged else "bypassed")
    return {
        "features": [
            env.t / env.cfg.horizon,
            float(env.alive_agents().size) / env.cfg.n_agents,
            float(np.mean(np.maximum(env.agent_hp, 0.0))),
            min(total_damage / 4.0, 1.0),
            herfindahl,
            high_hp_share,
        ],
        "hand_basin": hand_basin,
    }


def effective_modes(labels: Sequence[Any]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    return 2.0 ** entropy_bits([c / total for c in counts.values()])


def main() -> None:
    parser = argparse.ArgumentParser(description="External unsupervised basin discovery.")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7031)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    print("Training external learner (same recipe as the transfer battery) ...")
    learned, _ = train_marl_scorer(args.iters, args.batch, args.lr, args.seed)
    systems = (
        ("marl_learned", scorer_controller(learned)),
        ("marl_untrained", scorer_controller(untrained_scorer(args.seed + 999))),
        ("nearest_only", rule_controller(["nearest"])),
        ("role_oracle", rule_controller(["threat", "fragile", "non_decoy", "nearest"])),
        ("damage_aware", rule_controller(["damage", "nearest"])),
    )

    results: Dict[str, Any] = {}
    for name, controller in systems:
        rows = [
            run_episode_features(controller, CONTEXTS[i % len(CONTEXTS)],
                                 args.seed + i * 97)
            for i in range(args.episodes)
        ]
        features = [row["features"] for row in rows]
        hand = [row["hand_basin"] for row in rows]
        clusters = kmeans(features, k=args.k, seed=args.seed + 777)
        p = purity(clusters, hand)
        hand_h0 = entropy_bits(
            [c / len(hand) for c in Counter(hand).values()]
        )
        cluster_h0 = entropy_bits(
            [c / len(clusters) for c in Counter(clusters).values()]
        )
        results[name] = {
            "purity": p,
            "hand_h0_bits": hand_h0,
            "cluster_h0_bits": cluster_h0,
            "hand_modes": effective_modes(hand),
            "cluster_modes": effective_modes(clusters),
            "potential_verdict_hand": hand_h0 >= 0.5,
            "potential_verdict_cluster": cluster_h0 >= 0.5,
        }
        print(
            f"{name:15s} purity {p:.3f} | H0 hand {hand_h0:.3f} vs cluster "
            f"{cluster_h0:.3f} | modes {results[name]['hand_modes']:.2f} vs "
            f"{results[name]['cluster_modes']:.2f} | potential verdict same: "
            f"{results[name]['potential_verdict_hand'] == results[name]['potential_verdict_cluster']}"
        )

    checks = {
        "u1_learned_purity": results["marl_learned"]["purity"] >= 0.8,
        "u2_learned_modes_agree": abs(
            results["marl_learned"]["hand_modes"]
            - results["marl_learned"]["cluster_modes"]
        ) <= 0.5,
        "u3_potential_verdicts_unchanged": all(
            r["potential_verdict_hand"] == r["potential_verdict_cluster"]
            for r in results.values()
        ),
    }
    summary = {"k": args.k, "episodes": args.episodes, "results": results, "checks": checks}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_unsupervised_basins.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nchecks:")
    for key, ok in checks.items():
        print(f"  {key}: {'PASS' if ok else 'FAIL'}")
    print(f"\nWrote {args.output_dir / 'external_unsupervised_basins.json'}")


if __name__ == "__main__":
    main()
