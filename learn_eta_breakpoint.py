"""LEARN-ETA-BP: learning-rate scale separation in a learned population.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this run).
Same exact all-pairs agreement game as LEARN-N-EXACT, fixed N=50,
sweeping optimizer step size. The hypothesis is that small learning
steps recover the slowly-organizing phase and onset-type B5, while a
large step saturates too quickly.
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
N_AGENTS = 50
A = 6
SEEDS = (96_601, 96_602, 96_603, 96_604, 96_605)
ETAS = (0.0005, 0.001, 0.003, 0.01, 0.05)
SIGMA = 0.01
EVAL_EVERY = 50
GATE = 0.1
LOG2A = math.log2(A)


def n_updates(eta: float) -> int:
    if eta <= 0.001:
        return 10_000
    if eta == 0.003:
        return 5_000
    return 3_000


def objective(probs: torch.Tensor) -> torch.Tensor:
    n = probs.shape[0]
    gram = probs @ probs.T
    off_diag = gram.sum() - gram.diag().sum()
    return off_diag / (n * (n - 1))


def openness(logits: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)
        h = -(p * torch.log2(p.clamp_min(1e-12))).sum(dim=-1)
        return float((h / LOG2A).mean())


def run_seed(seed: int, eta: float) -> Dict[str, float]:
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(SIGMA * torch.randn(N_AGENTS, A))
    opt = torch.optim.Adam([logits], lr=eta)
    total = n_updates(eta)
    curve = {}
    for update in range(total + 1):
        if update % EVAL_EVERY == 0:
            curve[str(update)] = openness(logits)
        if update == total:
            break
        loss = -objective(torch.softmax(logits, dim=-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return curve


def adjudicate(curve: Dict[str, float], grid) -> Dict:
    x = np.array(grid, dtype=float)
    y = np.array([curve[str(u)] for u in grid])
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
        if len(xw[parity::2]) < 5:
            t = {"verdict": "too_few_points", "ok": False}
            ok = False
        else:
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
    per_eta = {}
    for eta in ETAS:
        total = n_updates(eta)
        grid = list(range(0, total + 1, EVAL_EVERY))
        rows = {}
        for seed in SEEDS:
            curve = run_seed(seed, eta)
            adj = adjudicate(curve, grid)
            adj["curve"] = {k: round(v, 5) for k, v in curve.items()}
            rows[str(seed)] = adj
            hinge = adj.get("hinge", {})
            print(
                f"eta={eta} seed {seed}: drop={adj['drop']} "
                f"b5={adj.get('b5_onset')} "
                f"t*={hinge.get('t_star')} "
                f"dBIC={hinge.get('delta_bic')} "
                f"slopes={hinge.get('slope_before')}->"
                f"{hinge.get('slope_after')}",
                flush=True,
            )
        per_eta[str(eta)] = {"grid": grid, "seeds": rows}

    def onset_seeds(eta: float):
        return [
            s
            for s in SEEDS
            if per_eta[str(eta)]["seeds"][str(s)].get("b5_onset")
        ]

    eta1 = any(len(onset_seeds(e)) >= 4 for e in (0.001, 0.003))
    eta2 = len(onset_seeds(0.05)) <= 1

    onset_etas = [e for e in ETAS if len(onset_seeds(e)) >= 4]
    tstars = []
    for eta in onset_etas:
        vals = [
            per_eta[str(eta)]["seeds"][str(s)]["hinge"]["t_star"]
            for s in onset_seeds(eta)
        ]
        tstars.append(float(np.median(vals)))
    eta3 = bool(
        len(tstars) >= 2
        and all(a > b for a, b in zip(tstars, tstars[1:]))
    )

    outcomes = {
        "ETA1_small_step_onset": bool(eta1),
        "ETA2_large_step_no_onset": bool(eta2),
        "ETA3_timing_law": bool(eta3),
        "onset_counts": {str(e): len(onset_seeds(e)) for e in ETAS},
        "onset_etas": onset_etas,
        "median_t_star": {str(e): t for e, t in zip(onset_etas, tstars)},
    }
    report = {
        "status": (
            "LEARN-ETA-BP learning-rate scale separation in an exact "
            "policy-gradient population; registered before run"
        ),
        "config": {
            "N": N_AGENTS,
            "A": A,
            "etas": ETAS,
            "seeds": SEEDS,
            "sigma": SIGMA,
            "eval_every": EVAL_EVERY,
        },
        "per_eta": per_eta,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_eta_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
