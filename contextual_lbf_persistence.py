"""Persistence of the acquired Contextual LBF macro-structure.

Protocol frozen in PERSISTENCE_PREREGISTRATION.md BEFORE any perturbed
evaluation. Saved confirmation policies only; no retraining. Perturbations:
fresh eval seeds (P0), novel interior layouts (P1), shorter/longer horizon
(P2/P3), Gaussian observation noise sigma 0.05/0.10/0.20 (P4/P5/P6).
Measured: conditional selectivity and usefulness gap per system, plus
retention relative to the system's own P0 selectivity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import contextual_lbf_transfer as clbf
import lbf_collapse_probe as base

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

SEEDS = [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110]
N_EVAL = 40
EVAL_OFFSET = 9_000_000

# MECHANICS FIX (recorded): the first P1 run used layout tuples violating
# the benchmark's lexicographic food-identity convention (FoodIndex sorts
# positions, so declared food0/food1 swapped in two variants and the context
# semantics inverted mechanically). The botched run is preserved as
# contextual_lbf_persistence_layoutbug.json. Layouts below satisfy the
# preregistration's declared spec: food0 < food1 lexicographically AND the
# nearer food's identity equals the context.
NOVEL_LAYOUTS = {
    0: (
        (((1, 2), (2, 2)), (2, 1), (3, 3)),
        (((1, 1), (2, 1)), (1, 2), (3, 3)),
    ),
    1: (
        (((2, 3), (3, 3)), (1, 1), (3, 2)),
        (((3, 2), (2, 3)), (1, 1), (3, 3)),
    ),
}


def team_cost(players, food) -> int:
    return sum(abs(p[0] - food[0]) + abs(p[1] - food[1]) for p in players)


def assert_layout_semantics(layouts) -> None:
    for ctx, variants in layouts.items():
        for players, food0, food1 in variants:
            # lexicographic identity convention (FoodIndex sorts positions)
            assert tuple(food0) < tuple(food1), (
                "declared food0 must be lexicographically first",
                ctx, food0, food1)
            costs = (team_cost(players, food0), team_cost(players, food1))
            nearer = int(costs[1] < costs[0])
            assert nearer == ctx, (ctx, players, food0, food1, costs)
            assert costs[0] != costs[1], ("ambiguous context", ctx)


class NoisyTeamController(clbf.TeamController):
    """Adds i.i.d. Gaussian observation noise before the neural policy."""

    def __init__(self, kind, net=None, sigma: float = 0.0,
                 noise_seed: int = 0):
        super().__init__(kind, net)
        self.sigma = sigma
        self.noise_rng = np.random.default_rng(noise_seed)

    def act(self, env, obs, rng, findex, forced_target):
        if self.sigma > 0 and self.policy is not None and forced_target is None:
            obs = [o + self.noise_rng.normal(0, self.sigma, size=o.shape)
                   .astype(o.dtype) for o in obs]
        return super().act(env, obs, rng, findex, forced_target)


def lean_evaluate(controller, n_eval: int, seed_offset: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for context in clbf.CONTEXTS:
        for episode in range(n_eval):
            paired_seed = seed_offset + 10_000 * context + episode
            for mode in (None, "do_non_trigger"):
                row = clbf.run_episode(controller, context, paired_seed, mode)
                row["mode"] = mode or "natural"
                rows.append(row)
    natural = [r for r in rows if r["mode"] == "natural"]
    non = [r for r in rows if r["mode"] == "do_non_trigger"]
    rate = {
        c: float(np.mean([r["trigger"] for r in natural if r["context"] == c]))
        for c in clbf.CONTEXTS
    }
    return {
        "trigger_rates": rate,
        "conditional_selectivity": abs(rate[0] - rate[1]),
        "usefulness_gap": float(np.mean([r["score"] for r in natural])
                                - np.mean([r["score"] for r in non])),
    }


def load_net(seed: int) -> base.PolicyNet:
    net = base.PolicyNet()
    net.load_state_dict(torch.load(
        OUTPUTS / f"contextual_lbf_net_seed{seed}.pt", weights_only=True))
    net.eval()
    return net


def twin(seed: int) -> base.PolicyNet:
    torch.manual_seed(seed)
    return base.PolicyNet()


PERTURBATIONS = ["P0_baseline", "P1_novel_layouts", "P2_horizon12",
                 "P3_horizon18", "P4_noise005", "P5_noise010", "P6_noise020"]


def apply_perturbation(name: str):
    """Returns (layouts, horizon, sigma)."""
    layouts, horizon, sigma = clbf.LAYOUTS, 15, 0.0
    if name == "P1_novel_layouts":
        layouts = NOVEL_LAYOUTS
    elif name == "P2_horizon12":
        horizon = 12
    elif name == "P3_horizon18":
        horizon = 18
    elif name == "P4_noise005":
        sigma = 0.05
    elif name == "P5_noise010":
        sigma = 0.10
    elif name == "P6_noise020":
        sigma = 0.20
    return layouts, horizon, sigma


def main() -> None:
    assert_layout_semantics(clbf.LAYOUTS)
    assert_layout_semantics(NOVEL_LAYOUTS)
    frozen_layouts, frozen_horizon = clbf.LAYOUTS, clbf.HORIZON

    systems: Dict[str, Any] = {}
    for seed in SEEDS:
        systems[f"learned_{seed}"] = ("learned", load_net(seed), seed)
        systems[f"twin_{seed}"] = ("twin", twin(seed), seed)
    for kind in ("team_nearest", "fixed_food0", "fixed_food1"):
        systems[kind] = ("scripted", None, 0)

    results: Dict[str, Dict[str, Any]] = {name: {} for name in systems}
    for pert in PERTURBATIONS:
        layouts, horizon, sigma = apply_perturbation(pert)
        clbf.LAYOUTS = layouts
        clbf.HORIZON = horizon
        try:
            for name, (group, net, seed) in systems.items():
                if group == "scripted":
                    controller = clbf.TeamController(name)
                else:
                    controller = NoisyTeamController(
                        "policy", net, sigma=sigma,
                        noise_seed=seed * 7 + hash(pert) % 10_000)
                offset = EVAL_OFFSET + (seed or 0) * 100_000
                results[name][pert] = lean_evaluate(controller, N_EVAL, offset)
            print(f"{pert}: done", flush=True)
        finally:
            clbf.LAYOUTS = frozen_layouts
            clbf.HORIZON = frozen_horizon

    # ---------------- registered predictions ----------------
    learned = [f"learned_{s}" for s in SEEDS]
    non_p6 = [p for p in PERTURBATIONS if p not in ("P0_baseline",
                                                    "P6_noise020")]
    ps1_counts = {}
    for pert in non_p6:
        ok = sum(
            results[n][pert]["conditional_selectivity"]
            >= 0.5 * results[n]["P0_baseline"]["conditional_selectivity"]
            for n in learned)
        ps1_counts[pert] = ok
    ps1 = all(v >= 8 for v in ps1_counts.values())

    ps2_per_policy = {
        n: sum(results[n][p]["usefulness_gap"] > 0 for p in non_p6
               + ["P0_baseline", "P6_noise020"])
        for n in learned
    }
    ps2 = sum(v >= 5 for v in ps2_per_policy.values()) >= 8

    ps3 = all(
        results[f"twin_{s}"][p]["conditional_selectivity"] < 0.5
        for s in SEEDS for p in PERTURBATIONS)

    noise_means = [
        float(np.mean([results[n][p]["conditional_selectivity"]
                       for n in learned]))
        for p in ("P0_baseline", "P4_noise005", "P5_noise010", "P6_noise020")
    ]
    ps4 = all(a >= b - 1e-9 for a, b in zip(noise_means, noise_means[1:]))

    summary = {
        "status": "prospectively frozen persistence extension "
                  "(saved policies; no retraining)",
        "protocol": "PERSISTENCE_PREREGISTRATION.md",
        "n_eval_per_context": N_EVAL,
        "results": results,
        "predictions": {
            "PS1_structure_persists": {"pass": ps1, "counts": ps1_counts},
            "PS2_usefulness_persists": {"pass": ps2,
                                        "per_policy": ps2_per_policy},
            "PS3_twins_stay_below": {"pass": ps3},
            "PS4_graceful_noise_degradation": {"pass": ps4,
                                               "means": noise_means},
        },
        "all_pass": all([ps1, ps2, ps3, ps4]),
    }
    out = OUTPUTS / "contextual_lbf_persistence.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["predictions"], indent=2))
    print("all_pass:", summary["all_pass"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
