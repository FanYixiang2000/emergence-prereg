"""LEARN-N-BP: onset-type B5 emerges with population size in a
learned multi-agent system.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). N independent REINFORCE learners play a repeated consensus
(plurality-matching) game; the joint action openness is tracked
across learning and tested for onset-type B5 as a function of N.
This is the learned analog of the ANT-COLONY-BP finite-size law and
the explanation for the OC-STATE/OCC-BP nulls (Overcooked = N=2).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from kuramoto_breakpoint_r2 import truncate_at_saturation
from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
A = 6
BATCH = 256
N_UPDATES = 1500
EVAL_EVERY = 25
GRID = list(range(0, N_UPDATES + 1, EVAL_EVERY))
NS = (2, 3, 5, 10)
SEEDS = (96_401, 96_402, 96_403)
GATE = 0.1
LOG2A = math.log2(A)


class Agent(nn.Module):
    """State-free repeated game: a learnable logit vector + baseline."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(A))
        self.v = nn.Parameter(torch.zeros(1))


def plurality_reward(acts: torch.Tensor, n_agents: int) -> torch.Tensor:
    """Return the fraction of agents choosing the round plurality action."""
    batch = acts.shape[0]
    onehot = torch.zeros(batch, n_agents, A)
    onehot.scatter_(2, acts.unsqueeze(-1), 1.0)
    counts = onehot.sum(dim=1)
    plurality = counts.max(dim=1).values
    return plurality / n_agents


def openness(agents: List[Agent]) -> float:
    """Mean normalized policy entropy across agents."""
    hs = []
    with torch.no_grad():
        for ag in agents:
            p = torch.softmax(ag.logits, dim=-1)
            h = -(p * torch.log2(p.clamp_min(1e-12))).sum()
            hs.append(float(h) / LOG2A)
    return float(np.mean(hs))


def run_seed(n_agents: int, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    agents = [Agent() for _ in range(n_agents)]
    opt = torch.optim.Adam(
        [p for ag in agents for p in ag.parameters()], lr=3e-4
    )
    curve = {}
    for update in range(N_UPDATES + 1):
        if update % EVAL_EVERY == 0:
            curve[str(update)] = openness(agents)
        if update == N_UPDATES:
            break

        dists = [
            torch.distributions.Categorical(logits=ag.logits)
            for ag in agents
        ]
        acts = torch.stack([d.sample((BATCH,)) for d in dists], dim=1)
        rew = plurality_reward(acts, n_agents)
        loss = 0.0
        for i, (ag, dist) in enumerate(zip(agents, dists)):
            adv = (rew - ag.v).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            logp = dist.log_prob(acts[:, i])
            loss = (
                loss
                - (logp * adv).mean()
                + 0.5 * ((ag.v - rew) ** 2).mean()
                - 0.01 * dist.entropy()
            )
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
                f"gate={adj['gate_passed']} b5={adj.get('b5_onset')} "
                f"t*={hinge.get('t_star')} "
                f"dBIC={hinge.get('delta_bic')} "
                f"slope_after={hinge.get('slope_after')}",
                flush=True,
            )
        per_n[str(n_agents)] = rows

    def onset_seeds(n_agents: int) -> List[int]:
        return [
            seed
            for seed in SEEDS
            if per_n[str(n_agents)][str(seed)].get("b5_onset")
        ]

    lnb1 = len(onset_seeds(2)) <= 1
    lnb2 = len(onset_seeds(10)) >= 2

    def med_slope(n_agents: int):
        vals = [
            abs(per_n[str(n_agents)][str(seed)]["hinge"]["slope_after"])
            for seed in onset_seeds(n_agents)
        ]
        return float(np.median(vals)) if vals else None

    slopes = {n: med_slope(n) for n in (3, 5, 10)}
    onset_ns = [n for n in (3, 5, 10) if slopes[n] is not None]
    lnb3 = False
    if slopes[10] is not None and onset_ns:
        smallest = min(onset_ns)
        seq = [slopes[n] for n in onset_ns]
        lnb3 = bool(
            all(a <= b + 1e-9 for a, b in zip(seq, seq[1:]))
            and slopes[10] > slopes[smallest] - 1e-9
            and 10 != smallest
        )

    outcomes = {
        "LNB1_small_N_no_onset": bool(lnb1),
        "LNB2_large_N_onset": bool(lnb2),
        "LNB3_monotone_sharpening": bool(lnb3),
        "onset_counts": {str(n): len(onset_seeds(n)) for n in NS},
        "median_post_slope": {str(n): slopes.get(n) for n in (3, 5, 10)},
    }
    report = {
        "status": (
            "LEARN-N-BP finite-size law in a learned multi-agent "
            "system; registered before run; explains the "
            "OC-STATE/OCC-BP nulls (N=2)"
        ),
        "grid": GRID,
        "per_N": per_n,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_n_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
