"""Joint-collapse curve on the real Overcooked learning system.

Registered in COLLAPSE_SOURCE_PREREGISTRATION.md Part 2 (frozen before
run). Measures the theory's PRIMARY object directly: the contraction of
the effective joint action possibility space over training
(10^n -> 2^n), with the nested source ladder

    C_total = C_individual + C_env + C_relational   (+ C_high = 0, n=2)

on the eight checkpoints saved by overcooked_genesis_curve.py
(seed 93001). Declared environment variable E = layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUTS = ("cramped_room", "asymmetric_advantages")
CHECKPOINT_STEPS = (40_000, 80_000, 160_000, 320_000, 640_000,
                    1_000_000, 1_500_000, 2_000_000)
N_EPISODES = 30
HORIZON = 200
N_ACTIONS = len(Action.ALL_ACTIONS)
N_JOINT = N_ACTIONS * N_ACTIONS
N_BOOT = 1000
EPS = 1e-15


def entropy(p: np.ndarray) -> float:
    q = p[p > EPS]
    return float(-(q * np.log2(q)).sum())


def rollout_joint_counts(net, seed: int):
    """Per-layout episode-level joint action count tables, plus the
    per-episode macro basin (first potter x delivery)."""
    import overcooked_transition_certificate as otc
    policy = oc.TeamPolicy("net", net=net)
    tables: Dict[str, List[np.ndarray]] = {}
    basins: List[str] = []
    for li, layout in enumerate(LAYOUTS):
        env = oc.make_env(layout)
        rng = random.Random(seed + li * 7_777)
        ep_tables = []
        for ep in range(N_EPISODES):
            env.reset()
            counts = np.zeros((N_ACTIONS, N_ACTIONS))
            sparse_total = 0.0
            for t in range(HORIZON):
                obs = oc.featurize(env)
                torch.manual_seed(seed * 1_000 + li * 100_000
                                  + ep * 1_000 + t)
                actions = policy.actions(env, obs, rng)
                a0 = Action.ALL_ACTIONS.index(actions[0])
                a1 = Action.ALL_ACTIONS.index(actions[1])
                counts[a0, a1] += 1
                _s, sparse_r, done, _info = env.step(actions)
                sparse_total += sparse_r
                if done:
                    break
            ep_tables.append(counts)
            first = otc.first_potter_from_stats(env)
            basins.append(otc.basin_from_continuation(first, sparse_total))
        tables[layout] = ep_tables
    return tables, basins


def ladder_from_tables(per_layout: Dict[str, np.ndarray]) -> Dict[str, float]:
    """per_layout: layout -> pooled (6,6) count table."""
    pe = {}
    for layout, counts in per_layout.items():
        total = counts.sum()
        pe[layout] = counts / total if total > 0 else np.full_like(
            counts, 1.0 / counts.size)
    p_mix = sum(pe.values()) / len(pe)
    h_p = entropy(p_mix.ravel())
    h_q0 = math.log2(N_JOINT)
    m0 = p_mix.sum(axis=1)
    m1 = p_mix.sum(axis=0)
    qi = np.outer(m0, m1)
    h_qi = entropy(qi.ravel())
    qe = sum(np.outer(t.sum(axis=1), t.sum(axis=0))
             for t in pe.values()) / len(pe)
    h_qe = entropy(qe.ravel())
    return {
        "H_Q0": h_q0, "H_QI": h_qi, "H_QE": h_qe, "H_P": h_p,
        "C_individual": h_q0 - h_qi,
        "C_env": h_qi - h_qe,
        "C_relational": h_qe - h_p,
        "C_high": 0.0,  # degenerate by construction for n=2
        "C_total": h_q0 - h_p,
        "openness": h_p / h_q0,
        "collapse_norm": 1.0 - h_p / h_q0,
    }


def bootstrap_ci(tables: Dict[str, List[np.ndarray]], seed: int,
                 key: str) -> List[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        pooled = {}
        for layout, eps_tables in tables.items():
            picks = rng.integers(0, len(eps_tables), size=len(eps_tables))
            pooled[layout] = sum(eps_tables[p] for p in picks)
        vals.append(ladder_from_tables(pooled)[key])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=95001)
    ap.add_argument("--curve-tag", default="curve")
    ap.add_argument("--curve-seed", type=int, default=93001)
    ap.add_argument("--grid", default=None,
                    help="comma-separated checkpoint steps; default is "
                         "the original 8-point grid")
    args = ap.parse_args()

    grid = (tuple(int(x) for x in args.grid.split(","))
            if args.grid else CHECKPOINT_STEPS)
    torch.set_num_threads(int(os.environ.get("OC_THREADS", "4")))
    genesis_path = (OUTPUTS /
                    f"overcooked_genesis_curve_{args.curve_tag}"
                    f"_s{args.curve_seed}.json")
    genesis = json.loads(genesis_path.read_text(encoding="utf-8"))

    curve = {}
    for idx, ck in enumerate(grid):
        path = (OUTPUTS / f"overcooked_genesis_{args.curve_tag}"
                f"_s{args.curve_seed}_{ck}.pt")
        net = PolicyNet()
        net.load_state_dict(torch.load(path, weights_only=True,
                                       map_location="cpu"))
        net.eval()
        tables, basins = rollout_joint_counts(net, args.seed + idx * 100)
        pooled = {layout: sum(t) for layout, t in tables.items()}
        row = ladder_from_tables(pooled)
        row["collapse_norm_ci95"] = bootstrap_ci(
            tables, args.seed + idx, "collapse_norm")
        row["C_relational_ci95"] = bootstrap_ci(
            tables, args.seed + idx + 50, "C_relational")
        # macro-branch (basin) openness from these same rollouts
        from collections import Counter as _Counter
        bc = _Counter(basins)
        pb = np.array([bc.get(b, 0) for b in sorted(set(basins) | set(
            f"{p}_{d}" for p in ("pot0", "pot1", "potnone")
            for d in ("deliver", "nodeliver")))], dtype=float)
        pb = pb / pb.sum()
        row["macro_basin_entropy_bits"] = entropy(pb)
        row["real_score"] = genesis["curve"][str(ck)]["real_score"]
        row["G_js_bits"] = genesis["curve"][str(ck)]["G_js_bits"]
        curve[str(ck)] = row
        print(f"ckpt {ck}: Cbar={row['collapse_norm']:.4f} "
              f"CI={row['collapse_norm_ci95']} "
              f"C_ind={row['C_individual']:.3f} "
              f"C_env={row['C_env']:.3f} "
              f"C_rel={row['C_relational']:.3f} "
              f"Hmacro={row['macro_basin_entropy_bits']:.3f} "
              f"score={row['real_score']:.1f}", flush=True)

    cks = [str(c) for c in grid]
    cbar = [curve[c]["collapse_norm"] for c in cks]
    crel = [curve[c]["C_relational"] for c in cks]
    comps = ["C_individual", "C_env", "C_relational"]
    min_comp = min(curve[c][k] for c in cks for k in comps)
    deltas = [cbar[i + 1] - cbar[i] for i in range(len(cbar) - 1)]
    biggest = int(np.argmax(deltas))
    interval = (grid[biggest], grid[biggest + 1])
    takeoff = {(640_000, 1_000_000), (1_000_000, 1_500_000)}
    adjacent = {(320_000, 640_000), (1_500_000, 2_000_000)}

    hmacro = [curve[c]["macro_basin_entropy_bits"] for c in cks]
    # DG-1 overlap test (grid-agnostic): does the largest-collapse
    # interval intersect the registered takeoff window [640k, 1.0M]?
    dg1_overlap = interval[0] < 1_000_000 and interval[1] > 640_000
    outcomes = {
        "JC1_collapse_direction": cbar[-1] > cbar[0],
        "JC2_ladder_sanity": min_comp >= -0.02,
        "JC3_burst_at_takeoff": ((interval in takeoff
                                  or interval in adjacent)
                                 if grid == CHECKPOINT_STEPS
                                 else dg1_overlap),
        "JC4_relational_growth": crel[-1] > crel[0],
        "JC5_macro_collapse": hmacro[-1] < hmacro[0],
    }
    report = {
        "status": ("joint-collapse curve on saved learning checkpoints; "
                   "registered in COLLAPSE_SOURCE_PREREGISTRATION.md "
                   "Part 2; PRIMARY possibility-collapse measurement, "
                   "pilot scale (one seed)"),
        "declared": {
            "E": "layout", "joint_variable": "per-step (a0,a1), 36 cells",
            "episodes_per_layout": N_EPISODES, "horizon": HORIZON,
            "C_high": "identically 0 for n=2 (declared degeneracy; "
                      "needs >=3 agents)",
        },
        "checkpoint_grid": list(grid),
        "curve": curve,
        "largest_collapse_interval": list(interval),
        "collapse_deltas": [round(d, 5) for d in deltas],
        "registered_outcomes": outcomes,
    }
    suffix = "" if args.curve_tag == "curve" else f"_{args.curve_tag}"
    out = (OUTPUTS /
           f"overcooked_joint_collapse{suffix}_s{args.curve_seed}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"largest collapse interval: {interval}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
