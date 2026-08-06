"""Observer-contract sensitivity table (CS in the V2 alignment prereg).

Recomputes the joint-action collapse ladder on the SAVED seed-93001
checkpoints (640k, 1M, 2M) under three declared environment contracts:

    E = none              (single context; env structure falls into C_rel)
    E = layout            (the registered default)
    E = layout x time-bin (5 bins of 40 steps)

Fresh rollouts, new output file; stored checkpoints and stored results
are untouched. Registered predictions CS-1, CS-2.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_joint_collapse_curve import (HORIZON, LAYOUTS,
                                             N_ACTIONS, N_EPISODES,
                                             entropy, ladder_from_tables)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
CHECKPOINTS = (640_000, 1_000_000, 2_000_000)
TIME_BIN = 40
SEED = 98_301


def rollout_context_counts(net, seed: int) -> Dict[str, np.ndarray]:
    """Counts per (layout, time-bin) context."""
    policy = oc.TeamPolicy("net", net=net)
    ctx: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((N_ACTIONS, N_ACTIONS)))
    for li, layout in enumerate(LAYOUTS):
        env = oc.make_env(layout)
        rng = random.Random(seed + li * 7_777)
        for ep in range(N_EPISODES):
            env.reset()
            for t in range(HORIZON):
                obs = oc.featurize(env)
                torch.manual_seed(seed * 1_000 + li * 100_000
                                  + ep * 1_000 + t)
                actions = policy.actions(env, obs, rng)
                a0 = Action.ALL_ACTIONS.index(actions[0])
                a1 = Action.ALL_ACTIONS.index(actions[1])
                ctx[f"{layout}|t{t // TIME_BIN}"][a0, a1] += 1
                _s, _r, done, _info = env.step(actions)
                if done:
                    break
    return dict(ctx)


def regroup(ctx: Dict[str, np.ndarray], contract: str) -> Dict[str, np.ndarray]:
    if contract == "none":
        return {"all": sum(ctx.values())}
    if contract == "layout":
        out: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((N_ACTIONS, N_ACTIONS)))
        for key, table in ctx.items():
            out[key.split("|")[0]] += table
        return dict(out)
    return ctx  # layout x time-bin


def main() -> None:
    torch.set_num_threads(2)
    table = {}
    for ck in CHECKPOINTS:
        path = OUTPUTS / f"overcooked_genesis_curve_s93001_{ck}.pt"
        net = PolicyNet()
        net.load_state_dict(torch.load(path, weights_only=True,
                                       map_location="cpu"))
        net.eval()
        ctx = rollout_context_counts(net, SEED + ck // 1_000)
        row = {}
        for contract in ("none", "layout", "layout_x_timebin"):
            grouped = regroup(ctx, contract.split("_x_")[0]
                              if contract != "layout_x_timebin"
                              else "timebin")
            ladder = ladder_from_tables(grouped)
            row[contract] = {k: round(ladder[k], 5) for k in
                             ("C_individual", "C_env", "C_relational",
                              "C_total", "collapse_norm")}
            print(f"ckpt {ck} E={contract}: "
                  f"C_ind={ladder['C_individual']:.3f} "
                  f"C_env={ladder['C_env']:.3f} "
                  f"C_rel={ladder['C_relational']:.3f}", flush=True)
        table[str(ck)] = row

    rel_layout = table["2000000"]["layout"]["C_relational"]
    rel_time = table["2000000"]["layout_x_timebin"]["C_relational"]
    cs1 = (max(rel_layout, rel_time)
           <= 3.0 * max(min(rel_layout, rel_time), 1e-9))
    cs2 = all(table[c]["none"]["C_relational"]
              > table[c]["layout"]["C_relational"] - 1e-9
              and table[c]["none"]["C_relational"]
              > table[c]["layout_x_timebin"]["C_relational"] - 1e-9
              for c in table)
    report = {
        "status": ("observer-contract sensitivity table; registered in "
                   "V2_ALIGNMENT_PREREGISTRATION.md (CS); reuses saved "
                   "seed-93001 checkpoints, fresh rollouts, no stored "
                   "output modified"),
        "contracts": ["none", "layout", "layout_x_timebin (bins of 40)"],
        "checkpoints": list(CHECKPOINTS),
        "table": table,
        "registered_outcomes": {
            "CS1_relational_stable_across_declared_E_factor3": cs1,
            "CS2_hidden_E_inflates_relational": cs2,
        },
    }
    out = OUTPUTS / "overcooked_contract_sensitivity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
