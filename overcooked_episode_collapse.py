"""EP: episode-time collapse on the learned 2M policy.

Registered in V2_ALIGNMENT_PREREGISTRATION.md, Wave 4. The two-
timescale claim: the same openness instrument that measured learning-
time collapse must show WITHIN-EPISODE commitment. From stored
mid-episode snapshots at in-episode times t in {0,40,80,120,160},
cloned continuations with FIXED horizon H = 200 (removing the
time-budget confound) give the macro-basin distribution at each t;
its entropy is the within-episode openness.

EP-1: median basin entropy non-increasing in t.
EP-2: median entropy at t=160 < 50% of median entropy at t=0.
"""

from __future__ import annotations

import json
import math
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
SNAP_TS = (0, 40, 80, 120, 160)
N_EPISODES = 8          # snapshot-bearing episodes per layout
N_CONT = 24             # cloned continuations per snapshot
HORIZON = 200           # FIXED continuation horizon at every t
BASINS = [f"{p}_{d}" for p in ("pot0", "pot1", "potnone")
          for d in ("deliver", "nodeliver")]


def basin_entropy(basins: List[str]) -> float:
    c = Counter(basins)
    n = sum(c.values())
    return float(-sum((v / n) * math.log2(v / n)
                      for v in c.values() if v > 0))


def main() -> None:
    torch.set_num_threads(2)
    net = PolicyNet()
    net.load_state_dict(torch.load(LEARNED_CKPT, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    policy = oc.TeamPolicy("net", net=net)

    per_snapshot: List[Dict] = []
    for li, layout in enumerate(LAYOUTS):
        snaps, _ghosts = otc.collect_snapshots(
            policy, layout, 61_000 + li, N_EPISODES, SNAP_TS, HORIZON)
        print(f"{layout}: {len(snaps)} snapshots", flush=True)
        for si, snap in enumerate(snaps):
            basins = []
            for k in range(N_CONT):
                row = otc.continue_from_state(
                    policy, layout, snap["state"],
                    62_000 + li * 100_000 + si * 1_000 + k, HORIZON)
                basins.append(row["basin"])
            per_snapshot.append({
                "layout": layout, "episode": snap["episode"],
                "t": snap["t"],
                "basin_counts": dict(Counter(basins)),
                "entropy_bits": basin_entropy(basins),
            })
            print(f"  ep{snap['episode']} t={snap['t']}: "
                  f"H={per_snapshot[-1]['entropy_bits']:.3f} "
                  f"{per_snapshot[-1]['basin_counts']}", flush=True)

    medians = {}
    for t in SNAP_TS:
        vals = [r["entropy_bits"] for r in per_snapshot if r["t"] == t]
        medians[str(t)] = float(np.median(vals))
    seq = [medians[str(t)] for t in SNAP_TS]
    outcomes = {
        "EP1_monotone_commitment": all(b <= a + 1e-9
                                       for a, b in zip(seq, seq[1:])),
        "EP2_substantial": seq[-1] < 0.5 * seq[0],
    }
    report = {
        "status": ("EP episode-time collapse; registered Wave 4; fixed "
                   "continuation horizon H=200 at every in-episode t "
                   "removes the time-budget confound; learned 2M policy"),
        "config": {"layouts": LAYOUTS, "snapshot_ts": SNAP_TS,
                   "episodes_per_layout": N_EPISODES,
                   "continuations_per_snapshot": N_CONT,
                   "horizon": HORIZON},
        "median_entropy_bits_by_t": medians,
        "per_snapshot": per_snapshot,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "overcooked_episode_collapse.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"medians": medians, **outcomes}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
