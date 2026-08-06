"""E1: product-matched genesis comparison.

Registered in V2_ALIGNMENT_PREREGISTRATION.md wave 3 before running.
Fixes the disclosed OTC-C4 product-matching failure: the scripted
role pair is handicapped with a declared per-agent action-noise knob
eps, calibrated ON SCORE ONLY to the stored learned product (41.0),
then both systems are certified with fresh seeds. New outputs only.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_genesis_comparison import LEARNED_CKPT, evaluate_policy

OUTPUTS = Path(__file__).resolve().parent / "outputs"
TARGET_SCORE = 41.0  # stored learned product, genesis-comparison pilot
EPS_GRID = [round(0.05 * k, 2) for k in range(1, 15)]  # 0.05 .. 0.70


class NoisyScripted:
    """Scripted role pair with per-agent per-step uniform action noise."""

    def __init__(self, eps: float, seed: int):
        self.base = oc.TeamPolicy("scripted_roles", cook_agent=0)
        self.eps = eps
        self.rng = random.Random(seed)

    def actions(self, env, obs, rng):
        acts = list(self.base.actions(env, obs, rng))
        for i in range(2):
            if self.rng.random() < self.eps:
                acts[i] = self.rng.choice(Action.ALL_ACTIONS)
        return acts


def main() -> None:
    torch.set_num_threads(2)
    calibration = {}
    print("calibrating eps on score only", flush=True)
    for gi, eps in enumerate(EPS_GRID):
        pol = NoisyScripted(eps, seed=88_000 + gi)
        row = evaluate_policy(pol, 99_101 + gi)
        calibration[str(eps)] = {"real_score": row["real_score"],
                                 "G_js_bits": row["G_js_bits"]}
        print(f"  eps={eps}: score={row['real_score']:.1f}", flush=True)

    eps_star = min(EPS_GRID,
                   key=lambda e: abs(calibration[str(e)]["real_score"]
                                     - TARGET_SCORE))
    print(f"selected eps*={eps_star}", flush=True)

    t0 = time.time()
    noisy = NoisyScripted(eps_star, seed=88_500)
    row_noisy = evaluate_policy(noisy, 99_201)

    net = PolicyNet()
    net.load_state_dict(torch.load(LEARNED_CKPT, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    row_learned = evaluate_policy(oc.TeamPolicy("net", net=net), 99_211)
    eval_min = round((time.time() - t0) / 60, 2)

    s_n, s_l = row_noisy["real_score"], row_learned["real_score"]
    g_n, g_l = row_noisy["G_js_bits"], row_learned["G_js_bits"]
    outcomes = {
        "E1_1_product_matched_factor2":
            max(s_n, s_l) <= 2.0 * max(min(s_n, s_l), 1e-9),
        "E1_2_genesis_separation": g_n < 0.5 * g_l,
    }
    report = {
        "status": ("E1 product-matched genesis comparison; registered "
                   "in V2_ALIGNMENT_PREREGISTRATION.md wave 3; "
                   "calibration used score only"),
        "target_score": TARGET_SCORE,
        "calibration": calibration,
        "eps_star": eps_star,
        "certified_noisy_scripted": row_noisy,
        "certified_learned": row_learned,
        "eval_minutes_certified": eval_min,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "overcooked_product_matched_genesis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "eps_star": eps_star,
        "score_noisy": s_n, "score_learned": s_l,
        "G_noisy": g_n, "G_learned": g_l, **outcomes}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
