"""TRI-C-BP: dense-grid breakpoint test on the learned high-order
channel.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Reruns the TRI-C game with fresh seeds and 81 checkpoints,
then applies the hinge (two-segment vs one-segment, Delta-BIC)
detector with 2x-thinning persistence to the joint-openness series
of the unconditional 2x2x2 bit table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from triad_relational_collapse import entropy
from triad_highorder_cue import AgentNet, play_batch, update
from breakpoint_model_comparison import fit_one_segment, \
    fit_two_segment, bic

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_UPDATES = 2000
EVAL_EVERY = 25
EVAL_ROUNDS = 4096
SEEDS = (95_311, 95_312, 95_313)
GRID = list(range(0, N_UPDATES + 1, EVAL_EVERY))


def eval_point(nets, seed: int, tag: int) -> Dict:
    gen = torch.Generator().manual_seed(seed * 10_000 + tag)
    with torch.no_grad():
        batch = play_batch(nets, EVAL_ROUNDS, gen)
    a1, a2, a3 = (x.numpy() for x in batch["acts"])
    b1, b2, b3 = a1 % 2, a2 % 2, a3 % 2
    t2 = np.zeros((2, 2, 2))
    np.add.at(t2, (b1, b2, b3), 1)
    h_p = entropy(t2 / t2.sum())
    return {"openness": round(h_p / 3.0, 6),
            "r3": round(float(batch["rews"][2].mean()), 5),
            "r_total": round(float(sum(r.mean().item()
                                       for r in batch["rews"])), 4)}


def hinge_linear(x: np.ndarray, y: np.ndarray) -> Dict:
    n = len(y)
    rss1 = fit_one_segment(x, y)
    best = None
    for bi in range(2, n - 2):
        rss2 = fit_two_segment(x, y, bi)
        if best is None or rss2 < best[1]:
            best = (bi, rss2)
    bi, rss2 = best
    delta = bic(rss1, n, 2) - bic(rss2, n, 4)
    # recover slopes of the winning hinge
    xb = x[bi]
    A = np.vstack([x, np.maximum(x - xb, 0.0), np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    s_before = float(coef[0])
    s_after = float(coef[0] + coef[1])
    return {"delta_bic": round(float(delta), 3),
            "t_star": float(x[bi]),
            "slope_before": round(s_before, 8),
            "slope_after": round(s_after, 8),
            "onset_type": bool(s_after < s_before)}


def run_seed(seed: int) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    nets = [AgentNet() for _ in range(3)]
    opts = [torch.optim.Adam(n.parameters(), lr=3e-4) for n in nets]
    gen = torch.Generator().manual_seed(seed)
    curve = {}
    for u in range(N_UPDATES + 1):
        if u % EVAL_EVERY == 0:
            curve[str(u)] = eval_point(nets, seed, u)
        if u == N_UPDATES:
            break
        update(nets, opts, play_batch(nets, 256, gen))
    print(f"seed {seed} trained; final "
          f"O={curve[str(N_UPDATES)]['openness']:.4f} "
          f"r3={curve[str(N_UPDATES)]['r3']:.3f}", flush=True)
    return curve


def adjudicate(curve: Dict) -> Dict:
    x = np.array(GRID, dtype=float)
    y = np.array([curve[str(u)]["openness"] for u in GRID])
    full = hinge_linear(x, y)
    thin_idx = list(range(0, len(GRID), 2))
    thin = hinge_linear(x[thin_idx], y[thin_idx])
    span = GRID[-1] - GRID[0]
    persists = (thin["delta_bic"] >= 10.0 and thin["onset_type"]
                and abs(thin["t_star"] - full["t_star"])
                <= 0.10 * span)
    r3_cross = next((u for u in GRID if curve[str(u)]["r3"] >= 0.9),
                    None)
    return {
        "full": full, "thinned": thin,
        "tricbp1_seed": bool(full["delta_bic"] >= 10.0
                             and full["onset_type"]),
        "tricbp2_seed": bool(persists),
        "r3_090_cross": r3_cross,
        "tricbp3_seed": bool(r3_cross is not None
                             and full["t_star"] < r3_cross),
    }


def main() -> None:
    torch.set_num_threads(4)
    seeds_out, verdicts = {}, {}
    for seed in SEEDS:
        curve = run_seed(seed)
        v = adjudicate(curve)
        seeds_out[str(seed)] = {"curve": curve, "verdict": v}
        verdicts[str(seed)] = v
        print(f"seed {seed}: dBIC={v['full']['delta_bic']} "
              f"t*={v['full']['t_star']:.0f} "
              f"onset={v['full']['onset_type']} "
              f"persist={v['tricbp2_seed']} "
              f"r3_cross={v['r3_090_cross']}", flush=True)

    p1 = [s for s, v in verdicts.items() if v["tricbp1_seed"]]
    tricbp1 = len(p1) >= 2
    tricbp2 = all(verdicts[s]["tricbp2_seed"] for s in p1) and tricbp1
    tricbp3 = all(verdicts[s]["tricbp3_seed"] for s in p1) and tricbp1
    outcomes = {"TRICBP1_onset_breakpoint": bool(tricbp1),
                "TRICBP2_persistence": bool(tricbp2),
                "TRICBP3_collapse_leads_capability": bool(tricbp3),
                "passing_seeds": p1}
    report = {"status": ("TRI-C-BP dense-grid breakpoint test; "
                         "registered before run"),
              "grid": GRID, "seeds": seeds_out,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "tri_c_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
