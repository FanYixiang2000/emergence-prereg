"""LEARN-N-EXACT: finite-size onset under exact policy-gradient learning.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
confirmatory run). N independent categorical policies optimize the
exact expected all-pairs agreement objective from tiny random logit
perturbations. Object: mean policy entropy (current-state joint
action openness). Detector: matured V3.1/V3.2 contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from kuramoto_breakpoint_r2 import truncate_at_saturation
from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
A = 6
NS = (2, 5, 10, 50)
SEEDS = (96_501, 96_502, 96_503, 96_504, 96_505)
SIGMA = 0.01
LR = 0.05
N_UPDATES = 3000
EVAL_EVERY = 25
GRID = list(range(0, N_UPDATES + 1, EVAL_EVERY))
GATE = 0.1
LOG2A = math.log2(A)


def objective(probs: torch.Tensor) -> torch.Tensor:
    """Mean pairwise agreement across policies.

    probs: (N, A), rows sum to one.
    """
    n = probs.shape[0]
    gram = probs @ probs.T
    off_diag = gram.sum() - gram.diag().sum()
    return off_diag / (n * (n - 1))


def openness(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    h = -(p * torch.log2(p.clamp_min(1e-12))).sum(dim=-1)
    return float((h / LOG2A).mean())


def run_seed(n_agents: int, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(SIGMA * torch.randn(n_agents, A))
    opt = torch.optim.Adam([logits], lr=LR)
    curve = {}
    for update in range(N_UPDATES + 1):
        if update % EVAL_EVERY == 0:
            curve[str(update)] = openness(logits)
        if update == N_UPDATES:
            break
        p = torch.softmax(logits, dim=-1)
        loss = -objective(p)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return curve


def adjudicate(curve: Dict[str, float]) -> Dict:
    x = np.array(GRID, dtype=float)
    y = np.array([curve[str(u)] for u in GRID])
    drop = float(y[0] - y[-1])
    out = {
        "drop": round(drop, 4),
        "gate_passed": bool(drop >= GATE),
        "final_openness": round(float(y[-1]), 4),
    }
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        out["b5_onset"] = False
        return out

    xw, yw, t_sat = truncate_at_saturation(x, y)
    out["t_sat"] = t_sat
    out["window_points"] = len(yw)
    if len(yw) < 8:
        out["verdict"] = "window_too_short"
        out["b5_onset"] = False
        return out

    full = hinge_linear(xw, yw)
    span = xw[-1] - xw[0]
    thin = {}
    thin_ok = True
    for parity in (0, 1):
        t = hinge_linear(xw[parity::2], yw[parity::2])
        ok = (
            t["delta_bic"] >= 2.0
            and t["onset_type"]
            and abs(t["t_star"] - full["t_star"]) <= 0.10 * span
        )
        t["ok"] = bool(ok)
        thin_ok = thin_ok and ok
        thin[f"parity{parity}"] = t

    out.update(
        {
            "hinge": full,
            "thinning": thin,
            "b5_onset": bool(
                full["delta_bic"] >= 10 and full["onset_type"] and thin_ok
            ),
        }
    )
    return out


def main() -> None:
    torch.set_num_threads(4)
    per_n = {}
    for n_agents in NS:
        rows = {}
        for seed in SEEDS:
            curve = run_seed(n_agents, seed)
            adj = adjudicate(curve)
            adj["curve"] = {k: round(v, 5) for k, v in curve.items()}
            rows[str(seed)] = adj
            hinge = adj.get("hinge", {})
            print(
                f"N={n_agents} seed {seed}: drop={adj['drop']} "
                f"b5={adj.get('b5_onset')} "
                f"t*={hinge.get('t_star')} "
                f"dBIC={hinge.get('delta_bic')} "
                f"slopes={hinge.get('slope_before')}->"
                f"{hinge.get('slope_after')}",
                flush=True,
            )
        per_n[str(n_agents)] = rows

    def onset_seeds(n_agents: int):
        return [
            seed
            for seed in SEEDS
            if per_n[str(n_agents)][str(seed)].get("b5_onset")
        ]

    lne1 = len(onset_seeds(2)) <= 2
    lne2 = len(onset_seeds(50)) >= 4

    def median_slope(n_agents: int, key: str):
        vals = [
            abs(per_n[str(n_agents)][str(seed)]["hinge"][key])
            for seed in onset_seeds(n_agents)
        ]
        return float(np.median(vals)) if vals else None

    onset_sizes = [n for n in (5, 10, 50) if len(onset_seeds(n)) >= 3]
    if 5 in onset_sizes:
        law_sizes = onset_sizes
    else:
        law_sizes = [n for n in (10, 50) if n in onset_sizes]

    pre = [median_slope(n, "slope_before") for n in law_sizes]
    post = [median_slope(n, "slope_after") for n in law_sizes]
    lne3 = bool(
        len(law_sizes) >= 2
        and all(a >= b - 1e-12 for a, b in zip(pre, pre[1:]))
        and all(a <= b + 1e-12 for a, b in zip(post, post[1:]))
        and post[-1] > post[0]
    )

    outcomes = {
        "LNE1_small_N_no_onset": bool(lne1),
        "LNE2_collective_onset": bool(lne2),
        "LNE3_finite_size_law": bool(lne3),
        "onset_counts": {str(n): len(onset_seeds(n)) for n in NS},
        "law_sizes": law_sizes,
        "median_pre_slope": {
            str(n): median_slope(n, "slope_before") for n in (5, 10, 50)
        },
        "median_post_slope": {
            str(n): median_slope(n, "slope_after") for n in (5, 10, 50)
        },
    }
    report = {
        "status": (
            "LEARN-N-EXACT finite-size onset under exact policy-gradient "
            "learning; registered before run"
        ),
        "config": {
            "A": A,
            "Ns": NS,
            "seeds": SEEDS,
            "sigma": SIGMA,
            "lr": LR,
            "updates": N_UPDATES,
        },
        "per_N": per_n,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_n_exact.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
