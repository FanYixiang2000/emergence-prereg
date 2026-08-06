"""E1-B: source-profile comparison at matched product (ESTIMATION ONLY).

Registered in V2_ALIGNMENT_PREREGISTRATION.md, Wave 4: declared
hypothesis-generating, no directional prediction. E1's registered
falsification of single-point G stands. This measures whether the
joint-action SOURCE PROFILE (C_individual / C_env / C_relational,
E = layout) differs between the eps*-noisy scripted pair and the
learned 2M policy whose scores were matched in E1.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_genesis_comparison import LEARNED_CKPT
from overcooked_product_matched_genesis import NoisyScripted
from overcooked_joint_collapse_curve import (LAYOUTS, N_EPISODES, HORIZON,
                                             ladder_from_tables,
                                             bootstrap_ci)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_ACTIONS = len(Action.ALL_ACTIONS)


def rollout_counts(policy, seed: int) -> Dict[str, List[np.ndarray]]:
    tables: Dict[str, List[np.ndarray]] = {}
    for li, layout in enumerate(LAYOUTS):
        env = oc.make_env(layout)
        rng = random.Random(seed + li * 7_777)
        ep_tables = []
        for ep in range(N_EPISODES):
            env.reset()
            counts = np.zeros((N_ACTIONS, N_ACTIONS))
            for t in range(HORIZON):
                obs = oc.featurize(env)
                torch.manual_seed(seed * 1_000 + li * 100_000
                                  + ep * 1_000 + t)
                actions = policy.actions(env, obs, rng)
                a0 = Action.ALL_ACTIONS.index(actions[0])
                a1 = Action.ALL_ACTIONS.index(actions[1])
                counts[a0, a1] += 1
                _s, _r, done, _info = env.step(actions)
                if done:
                    break
            ep_tables.append(counts)
        tables[layout] = ep_tables
    return tables


def certify(policy, seed: int) -> Dict:
    tables = rollout_counts(policy, seed)
    pooled = {layout: sum(t) for layout, t in tables.items()}
    row = ladder_from_tables(pooled)
    for key in ("collapse_norm", "C_individual", "C_env",
                "C_relational"):
        row[f"{key}_ci95"] = bootstrap_ci(tables, seed + hash(key) %
                                          1_000, key)
    return row


def main() -> None:
    torch.set_num_threads(2)
    e1 = json.loads((OUTPUTS /
                     "overcooked_product_matched_genesis.json")
                    .read_text(encoding="utf-8"))
    eps_star = e1["eps_star"]

    noisy = NoisyScripted(eps_star, seed=88_900)
    row_noisy = certify(noisy, 97_301)
    print("noisy scripted:", {k: round(v, 4) for k, v in
                              row_noisy.items()
                              if isinstance(v, float)}, flush=True)

    net = PolicyNet()
    net.load_state_dict(torch.load(LEARNED_CKPT, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    row_learned = certify(oc.TeamPolicy("net", net=net), 97_401)
    print("learned:", {k: round(v, 4) for k, v in row_learned.items()
                       if isinstance(v, float)}, flush=True)

    def shares(row):
        tot = max(row["C_total"], 1e-9)
        return {c: row[c] / tot for c in ("C_individual", "C_env",
                                          "C_relational")}

    report = {
        "status": ("E1-B source-profile comparison at matched product; "
                   "registered Wave 4 as ESTIMATION ONLY "
                   "(hypothesis-generating, no directional prediction); "
                   "E1's falsification of single-point G stands"),
        "eps_star": eps_star,
        "scores_from_e1": {"noisy": e1["certified_noisy_scripted"]
                           ["real_score"],
                           "learned": e1["certified_learned"]
                           ["real_score"]},
        "noisy_scripted": row_noisy,
        "learned": row_learned,
        "profile_shares": {"noisy_scripted": shares(row_noisy),
                           "learned": shares(row_learned)},
    }
    out = OUTPUTS / "overcooked_source_profile_matched.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["profile_shares"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
