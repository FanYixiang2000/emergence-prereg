"""Overcooked genesis-comparison pilot (registered in
OVERCOOKED_TRANSITION_CONTRACT.md, V3 addendum, frozen before launch).

Four mechanisms, one certificate:

1. scripted_roles          prewired role regime (fresh-seed replication);
2. learned                 the fixed 2M self-play checkpoint
                           overcooked_transition_pilot2m_s92003.pt;
3. bc_clone_of_learned     per-agent supervised distillation of the
                           learned policy from its own trajectories
                           (copied organization, no formation history);
4. context_marginal        both agents sample independently from the
                           learned policy's (layout, time-bin) marginal
                           action table (common context kept, state
                           coupling removed by construction).

Registered outcomes OTC-C1..C4. This is a PILOT, not the confirmatory
flagship; in particular OTC-C2 is the registered honest negative that a
single-time real-vs-cut snapshot cannot separate copied from learned
genesis -- that separation is the job of the formation curve and
training-time interventions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
import overcooked_transition_certificate as otc
from overcooked_pilot import PolicyNet

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUTS = ("cramped_room", "asymmetric_advantages")
SNAPSHOT_STEPS = (20, 40, 80, 120, 160)
EPISODES_PER_LAYOUT = 10
HORIZON = 120
N_BOOT = 1000
TIME_BIN = 40
N_ACTIONS = len(Action.ALL_ACTIONS)
LEARNED_CKPT = OUTPUTS / "overcooked_transition_pilot2m_s92003.pt"


class ContextMarginalPolicy:
    """Samples each agent's action independently from the learned
    policy's (layout, time-bin, agent) marginal action distribution."""

    kind = "context_marginal"

    def __init__(self, table: Dict):
        self.table = table

    def actions(self, env, obs, rng: random.Random):
        layout = env.mdp.layout_name
        t = getattr(env.state, "timestep", 0)
        b = min(t // TIME_BIN, max(k[1] for k in self.table
                                   if k[0] == layout))
        out = []
        for agent in (0, 1):
            probs = self.table[(layout, b, agent)]
            r = rng.random()
            cum = 0.0
            idx = N_ACTIONS - 1
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    idx = i
                    break
            out.append(Action.ALL_ACTIONS[idx])
        return out


def rollout_learned(net, n_episodes: int, seed: int):
    """Self-play rollouts of the learned policy; returns per-sample
    (layout, t, agent, obs, action) records."""
    policy = oc.TeamPolicy("net", net=net)
    records = []
    for li, layout in enumerate(LAYOUTS):
        env = oc.make_env(layout)
        rng = random.Random(seed + li * 7_777)
        for ep in range(n_episodes):
            env.reset()
            for t in range(SNAPSHOT_STEPS[-1] + HORIZON):
                obs = oc.featurize(env)
                torch.manual_seed(seed * 1_000 + li * 100_000
                                  + ep * 10_000 + t)
                actions = policy.actions(env, obs, rng)
                for agent in (0, 1):
                    records.append((layout, t, agent, obs[agent],
                                    Action.ALL_ACTIONS.index(
                                        actions[agent])))
                _s, _r, done, _info = env.step(actions)
                if done:
                    break
    return records


def build_bc_clone(records, seed: int) -> PolicyNet:
    torch.manual_seed(seed)
    xs = np.stack([r[3] for r in records])
    ys = np.array([r[4] for r in records])
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.tensor(xs)
    y = torch.tensor(ys)
    for _ in range(20):
        perm = torch.randperm(len(x))
        for start in range(0, len(x), 1024):
            mb = perm[start:start + 1024]
            logits, _ = net(x[mb])
            loss = torch.nn.functional.cross_entropy(logits, y[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


def build_marginal_table(records) -> Dict:
    counts: Dict = defaultdict(lambda: np.zeros(N_ACTIONS))
    for layout, t, agent, _obs, a in records:
        counts[(layout, min(t // TIME_BIN, 6), agent)][a] += 1
    table = {}
    for key, c in counts.items():
        total = c.sum()
        table[key] = (c / total if total > 0
                      else np.ones(N_ACTIONS) / N_ACTIONS)
    return table


def evaluate_policy(policy, seed: int) -> Dict:
    all_rows = []
    for li, layout in enumerate(LAYOUTS):
        snaps, ghosts = otc.collect_snapshots(
            policy, layout, seed + li * 100_000, EPISODES_PER_LAYOUT,
            SNAPSHOT_STEPS, HORIZON)
        for i, snap in enumerate(snaps):
            pool = ghosts.get((layout, snap["t"]), [])
            if not pool:
                continue
            ghost = pool[(i + 1) % len(pool)]
            real = otc.continue_from_state(
                policy, layout, snap["state"],
                seed + 1_000_000 + li * 100_000 + i, HORIZON)
            cut = otc.continue_from_state(
                policy, layout, snap["state"],
                seed + 2_000_000 + li * 100_000 + i, HORIZON,
                ghost_actions=ghost)
            all_rows.append({"layout": layout, "t": snap["t"],
                             "real": real, "cut": cut})
    overall = otc.bin_metrics(all_rows)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(all_rows)
    for _ in range(N_BOOT):
        picks = rng.integers(0, n, size=n)
        rc = Counter(all_rows[p]["real"]["basin"] for p in picks)
        cc = Counter(all_rows[p]["cut"]["basin"] for p in picks)
        boots.append(otc.js_bits(otc.normalize_counts(rc),
                                 otc.normalize_counts(cc)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": overall["n"],
        "G_js_bits": overall["G_js_bits"],
        "G_boot_ci95": [float(lo), float(hi)],
        "C_signed_bits": overall["C_signed_bits"],
        "M_score_gain": overall["M_score_gain"],
        "real_score": overall["real_score"],
        "cut_score": overall["cut_score"],
        "partner_action_tv": overall["partner_action_tv"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=94001)
    ap.add_argument("--clone-episodes", type=int, default=30)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    torch.set_num_threads(4)
    net = PolicyNet()
    net.load_state_dict(torch.load(LEARNED_CKPT, weights_only=True,
                                   map_location="cpu"))
    net.eval()

    print("collecting learned self-play rollouts for distillation",
          flush=True)
    records = rollout_learned(net, args.clone_episodes, args.seed)
    print(f"  {len(records)} state-action samples", flush=True)
    clone_net = build_bc_clone(records, args.seed + 31)
    marg_table = build_marginal_table(records)

    systems = {
        "scripted_roles": oc.TeamPolicy("scripted_roles", cook_agent=0),
        "learned": oc.TeamPolicy("net", net=net),
        # Sampling (not argmax) so the clone keeps the teacher's action
        # stochasticity and the product stays comparable.
        "bc_clone_of_learned": oc.TeamPolicy("net", net=clone_net),
        "context_marginal": ContextMarginalPolicy(marg_table),
    }
    rows = {}
    for i, (name, pol) in enumerate(systems.items()):
        t0 = time.time()
        rows[name] = evaluate_policy(pol, args.seed + i * 10)
        rows[name]["eval_minutes"] = round((time.time() - t0) / 60, 2)
        print(f"{name}: G={rows[name]['G_js_bits']:.4f} "
              f"CI={rows[name]['G_boot_ci95']} "
              f"M={rows[name]['M_score_gain']:.2f} "
              f"score={rows[name]['real_score']:.2f}", flush=True)

    g_script = rows["scripted_roles"]["G_js_bits"]
    scores = [rows[k]["real_score"] for k in
              ("scripted_roles", "learned", "bc_clone_of_learned")]
    outcomes = {
        "OTC_C1_scripted_null_learned_positive": (
            g_script < 0.005
            and rows["learned"]["G_boot_ci95"][0] > g_script),
        "OTC_C2_clone_instantaneous_G_positive": (
            rows["bc_clone_of_learned"]["G_js_bits"] > 0.0),
        "OTC_C3_marginal_G_at_or_below_scripted": (
            rows["context_marginal"]["G_js_bits"] <= max(g_script, 0.01)),
        "OTC_C4_product_band_factor_two": (
            min(scores) > 0 and max(scores) / min(scores) <= 2.0),
    }
    report = {
        "status": ("genesis-comparison pilot; registered in "
                   "OVERCOOKED_TRANSITION_CONTRACT.md V3 addendum; "
                   "not a confirmatory flagship result"),
        "seed": args.seed,
        "learned_checkpoint": str(LEARNED_CKPT),
        "clone_training_samples": len(records),
        "systems": rows,
        "registered_outcomes": outcomes,
        "interpretation": [
            "OTC-C2 is a registered honest negative: a copied "
            "organization retains instantaneous coupling, so a "
            "single-time real-vs-cut snapshot does NOT separate copied "
            "from learned genesis; the separation lives in the "
            "formation-history curve and training-time interventions.",
            "OTC-C3/C4 document how far the marginal control is from "
            "product matching; a matched-product marginal control is "
            "future flagship work.",
        ],
    }
    out = OUTPUTS / f"overcooked_genesis_comparison_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
