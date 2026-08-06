"""LEARN-QUORUM-BP: learned population onset under a nonlinear quorum.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this run).
N Bernoulli policies optimize the exact expected soft-quorum payoff via
differentiable dynamic programming over the Poisson-binomial count of
agents choosing convention 1. Object: mean policy entropy.
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
NS = (2, 5, 20, 50)
SEEDS = (96_701, 96_702, 96_703, 96_704, 96_705)
SIGMA = 0.01
LR = 0.01
Q = 0.65
BETA = 20.0
N_UPDATES = 8000
EVAL_EVERY = 50
GRID = list(range(0, N_UPDATES + 1, EVAL_EVERY))
GATE = 0.1


def count_distribution(p: torch.Tensor) -> torch.Tensor:
    """Poisson-binomial distribution over k=sum Bernoulli(p_i)."""
    dist = torch.zeros(p.shape[0] + 1, dtype=p.dtype)
    dist[0] = 1.0
    for i, pi in enumerate(p):
        new = torch.zeros_like(dist)
        new[: i + 1] += dist[: i + 1] * (1 - pi)
        new[1 : i + 2] += dist[: i + 1] * pi
        dist = new
    return dist


def quorum_reward_values(n_agents: int) -> torch.Tensor:
    k = torch.arange(n_agents + 1, dtype=torch.float32)
    frac = k / n_agents
    return torch.sigmoid(BETA * (frac - Q)) + torch.sigmoid(
        BETA * ((1 - frac) - Q)
    )


def expected_reward(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    dist = count_distribution(p)
    reward = quorum_reward_values(logits.shape[0])
    return (dist * reward).sum()


def openness(logits: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.sigmoid(logits).clamp(1e-12, 1 - 1e-12)
        h = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
        return float(h.mean())


def run_seed(n_agents: int, seed: int) -> Dict:
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(SIGMA * torch.randn(n_agents))
    opt = torch.optim.Adam([logits], lr=LR)
    curve = {}
    rewards = {}
    for update in range(N_UPDATES + 1):
        if update % EVAL_EVERY == 0:
            curve[str(update)] = openness(logits)
            with torch.no_grad():
                rewards[str(update)] = float(expected_reward(logits))
        if update == N_UPDATES:
            break
        loss = -expected_reward(logits)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"openness": curve, "reward": rewards}


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
    per_n = {}
    for n_agents in NS:
        rows = {}
        for seed in SEEDS:
            curves = run_seed(n_agents, seed)
            adj = adjudicate(curves["openness"])
            reward_vals = [curves["reward"][str(u)] for u in GRID]
            r_final = reward_vals[-1]
            r_cross = next(
                (
                    GRID[i]
                    for i, r in enumerate(reward_vals)
                    if r_final > 0 and r >= 0.8 * r_final
                ),
                None,
            )
            adj["reward_080_cross"] = r_cross
            adj["leads_reward"] = bool(
                adj.get("b5_onset")
                and r_cross is not None
                and adj["hinge"]["t_star"] < r_cross
            )
            adj["curve"] = {
                k: round(v, 5) for k, v in curves["openness"].items()
            }
            adj["reward_curve"] = {
                k: round(v, 5) for k, v in curves["reward"].items()
            }
            rows[str(seed)] = adj
            hinge = adj.get("hinge", {})
            print(
                f"N={n_agents} seed {seed}: drop={adj['drop']} "
                f"b5={adj.get('b5_onset')} "
                f"t*={hinge.get('t_star')} "
                f"dBIC={hinge.get('delta_bic')} "
                f"slopes={hinge.get('slope_before')}->"
                f"{hinge.get('slope_after')} "
                f"r_cross={r_cross}",
                flush=True,
            )
        per_n[str(n_agents)] = rows

    def onset_seeds(n_agents: int):
        return [
            seed
            for seed in SEEDS
            if per_n[str(n_agents)][str(seed)].get("b5_onset")
        ]

    lq1 = len(onset_seeds(2)) <= 2
    large_hits = {
        n: len(onset_seeds(n)) for n in (20, 50)
    }
    lq2 = any(v >= 4 for v in large_hits.values())
    lq3 = all(
        per_n[str(n)][str(seed)].get("leads_reward")
        for n in (20, 50)
        for seed in onset_seeds(n)
    ) and any(large_hits.values())
    lq4 = bool(lq2)

    outcomes = {
        "LQ1_small_N_no_onset": bool(lq1),
        "LQ2_threshold_population_onset": bool(lq2),
        "LQ3_onset_leads_reward": bool(lq3),
        "LQ4_nonlinear_vs_smooth_contrast": bool(lq4),
        "onset_counts": {str(n): len(onset_seeds(n)) for n in NS},
    }
    report = {
        "status": (
            "LEARN-QUORUM-BP learned population with nonlinear quorum "
            "threshold; registered before run"
        ),
        "config": {
            "Ns": NS,
            "seeds": SEEDS,
            "sigma": SIGMA,
            "lr": LR,
            "q": Q,
            "beta": BETA,
            "updates": N_UPDATES,
        },
        "per_N": per_n,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_quorum_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
