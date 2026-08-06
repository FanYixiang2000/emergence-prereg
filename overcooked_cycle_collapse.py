"""EP-CYCLE: cycle-aligned within-episode collapse on the learned
2M policy.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Cycles are delimited by delivery events; snapshots at cycle
phases {0, 0.25, 0.5, 0.75}; openness is the basin entropy of 24
cloned fixed-horizon (H=100) continuations, with the unchanged EP
basin definition (first-potter x deliver).
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf
import torch

import overcooked_criterion as oc
import overcooked_transition_certificate as otc
from overcooked_pilot import PolicyNet
from overcooked_genesis_comparison import LEARNED_CKPT

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUTS = ("cramped_room", "asymmetric_advantages")
N_EPISODES = 6
EP_LEN = 400
N_CONT = 24
HORIZON = 100
MIN_CYCLE = 20
PHASES = (0.0, 0.25, 0.5, 0.75)


def basin_entropy(basins: List[str]) -> float:
    c = Counter(basins)
    n = sum(c.values())
    return float(-sum((v / n) * math.log2(v / n)
                      for v in c.values() if v > 0))


def rollout_with_snapshots(policy, layout: str, seed: int):
    """One episode: returns per-step states (deep copies) and the
    list of delivery step indices."""
    env = oc.make_env(layout)
    env.reset()
    rng = random.Random(seed)
    states, deliveries = [], []
    for t in range(EP_LEN):
        states.append(copy.deepcopy(env.state))
        actions = otc.policy_actions(policy, env, rng,
                                     seed * 1_000_000 + t)
        _s, sparse_r, done, _info = env.step(actions)
        if sparse_r > 0:
            deliveries.append(t)
        if done:
            break
    return states, deliveries


def main() -> None:
    torch.set_num_threads(2)
    net = PolicyNet()
    net.load_state_dict(torch.load(LEARNED_CKPT, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    policy = oc.TeamPolicy("net", net=net)

    rows: List[Dict] = []
    for li, layout in enumerate(LAYOUTS):
        for ep in range(N_EPISODES):
            seed = 64_000 + li * 100 + ep
            states, deliveries = rollout_with_snapshots(
                policy, layout, seed)
            bounds = [0] + [d + 1 for d in deliveries]
            cycles = []
            for k in range(len(bounds) - 1):
                start, end = bounds[k], bounds[k + 1]
                if end - start >= MIN_CYCLE and end <= len(states):
                    cycles.append((k, start, end))
            print(f"{layout} ep{ep}: {len(deliveries)} deliveries, "
                  f"{len(cycles)} kept cycles", flush=True)
            for k, start, end in cycles:
                L = end - start
                for phi in PHASES:
                    t = start + int(round(phi * L))
                    t = min(t, len(states) - 1)
                    basins = [otc.continue_from_state(
                        policy, layout, states[t],
                        65_000 + li * 1_000_000 + ep * 50_000
                        + k * 1_000 + int(phi * 100) * 10 + c,
                        HORIZON)["basin"] for c in range(N_CONT)]
                    h = basin_entropy(basins)
                    rows.append({"layout": layout, "episode": ep,
                                 "cycle": k, "post_delivery": k > 0,
                                 "cycle_len": L, "phase": phi,
                                 "t_abs": t, "entropy_bits": round(h, 4),
                                 "basin_counts": dict(Counter(basins))})
                    print(f"  cyc{k} phi={phi}: H={h:.3f}", flush=True)

    med = {}
    for phi in PHASES:
        vals = [r["entropy_bits"] for r in rows if r["phase"] == phi]
        med[str(phi)] = float(np.median(vals))
    seq = [med[str(p)] for p in PHASES]

    post0 = [r["entropy_bits"] for r in rows
             if r["phase"] == 0.0 and r["post_delivery"]]
    late = [r["entropy_bits"] for r in rows if r["phase"] == 0.75]
    reopen_gap = (float(np.median(post0)) - float(np.median(late))
                  if post0 and late else None)

    epc1 = bool(all(b <= a + 1e-9 for a, b in zip(seq, seq[1:]))
                and seq[-1] < seq[0])
    epc2 = bool(reopen_gap is not None and reopen_gap > 0.2)
    epc3 = bool(seq[-1] <= 0.5 * seq[0])

    outcomes = {"EPC1_within_cycle_commitment": epc1,
                "EPC2_boundary_reopening": epc2,
                "EPC3_substantial": epc3,
                "median_by_phase": med,
                "reopen_gap_bits": (round(reopen_gap, 4)
                                    if reopen_gap is not None else None)}
    report = {
        "status": ("EP-CYCLE cycle-aligned within-episode collapse; "
                   "registered before run; EP basin definition "
                   "unchanged; fixed continuation horizon"),
        "config": {"layouts": LAYOUTS, "episodes": N_EPISODES,
                   "ep_len": EP_LEN, "n_cont": N_CONT,
                   "horizon": HORIZON, "min_cycle": MIN_CYCLE,
                   "phases": PHASES},
        "n_snapshots": len(rows),
        "per_snapshot": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "overcooked_cycle_collapse.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
