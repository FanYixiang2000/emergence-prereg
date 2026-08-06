"""Overcooked real-vs-ghost transition certificate scaffold.

This is the first executable step toward the requested flagship experiment:
from the SAME Overcooked simulator state, compare a true coupled continuation
against an interaction-broken ghost-partner continuation.

It is intentionally labelled as a scaffold/smoke certificate. A learned
flagship claim requires newly saved checkpoints and a separately frozen run.
The existing round-1 JSONs are read-only summaries and cannot support this
state-level replay.

Smoke predictions frozen in OVERCOOKED_TRANSITION_CONTRACT.md:
  OTC-S1  Complete contract and normalized distributions are exported.
  OTC-S2  Ghost replay gives finite G and partner-action marginal diagnostics.
  OTC-S3  Output marks whether a learned checkpoint was supplied.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

np.Inf = np.inf
import torch
from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LAYOUTS = ("cramped_room", "asymmetric_advantages")
BASINS = tuple(f"{p}_{d}" for p in ("pot0", "pot1", "potnone")
               for d in ("deliver", "nodeliver"))
N_ACTIONS = len(Action.ALL_ACTIONS)


def entropy(p: Dict[str, float]) -> float:
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def js_bits(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        a, b = p.get(k, 0.0), q.get(k, 0.0)
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def normalize_counts(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {b: 0.0 for b in BASINS}
    return {b: counts.get(b, 0) / total for b in BASINS}


def tv_actions(a: Sequence[int], b: Sequence[int]) -> float:
    ca, cb = Counter(a), Counter(b)
    n, m = max(1, len(a)), max(1, len(b))
    return 0.5 * sum(abs(ca.get(i, 0) / n - cb.get(i, 0) / m)
                     for i in range(N_ACTIONS))


def action_index(action) -> int:
    try:
        return Action.ALL_ACTIONS.index(action)
    except ValueError:
        return N_ACTIONS - 1


def make_policy(kind: str, seed: int, checkpoint: Optional[Path]):
    if kind == "scripted_roles":
        return oc.TeamPolicy("scripted_roles", cook_agent=0), False
    if kind == "initial":
        torch.manual_seed(seed)
        net = PolicyNet()
        net.eval()
        return oc.TeamPolicy("net", net=net), False
    if kind == "checkpoint":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for kind=checkpoint")
        net = PolicyNet()
        net.load_state_dict(torch.load(checkpoint, weights_only=True,
                                       map_location="cpu"))
        net.eval()
        return oc.TeamPolicy("net", net=net), True
    raise ValueError(kind)


def reset_stats(env) -> None:
    """Keep the state but start a fresh continuation-level event log."""
    env.t = 0
    env._episode_steps = 0 if hasattr(env, "_episode_steps") else 0
    # OvercookedEnv initializes game_stats on reset; deepcopy is safest across
    # overcooked_ai_py versions because the exact nested keys are versioned.
    fresh = oc.make_env("cramped_room")
    fresh.reset()
    env.game_stats = copy.deepcopy(fresh.game_stats)


def first_potter_from_stats(env) -> Optional[int]:
    stats = env.game_stats
    pots0 = list(stats.get("potting_onion", [[], []])[0]) + \
        list(stats.get("potting_tomato", [[], []])[0])
    pots1 = list(stats.get("potting_onion", [[], []])[1]) + \
        list(stats.get("potting_tomato", [[], []])[1])
    t0 = min(pots0) if pots0 else None
    t1 = min(pots1) if pots1 else None
    if t0 is not None and (t1 is None or t0 <= t1):
        return 0
    if t1 is not None:
        return 1
    return None


def basin_from_continuation(first_potter: Optional[int],
                            sparse_total: float) -> str:
    potter = {0: "pot0", 1: "pot1", None: "potnone"}[first_potter]
    deliver = "deliver" if sparse_total > 0 else "nodeliver"
    return f"{potter}_{deliver}"


def policy_actions(policy, env, rng: random.Random, torch_seed: int):
    torch.manual_seed(torch_seed)
    obs = oc.featurize(env)
    return policy.actions(env, obs, rng)


def collect_snapshots(policy, layout: str, seed: int, n_episodes: int,
                      snapshot_steps: Sequence[int], horizon: int):
    env = oc.make_env(layout)
    rng = random.Random(seed)
    snapshots = []
    ghosts = defaultdict(list)
    wanted = set(snapshot_steps)
    for ep in range(n_episodes):
        env.reset()
        trace1: List[int] = []
        saved = []
        for t in range(max(max(snapshot_steps) + horizon + 1, horizon + 1)):
            if t in wanted:
                snap = {
                    "state": copy.deepcopy(env.state),
                    "t": t,
                    "layout": layout,
                    "episode": ep,
                }
                snapshots.append(snap)
                saved.append((t, len(snapshots) - 1, len(trace1)))
            actions = policy_actions(policy, env, rng,
                                     seed * 1_000_000 + ep * 10_000 + t)
            trace1.append(action_index(actions[1]))
            _s, _r, done, _info = env.step(actions)
            if done:
                break
        for t, _idx, start in saved:
            suffix = trace1[start:start + horizon]
            if len(suffix) >= horizon:
                ghosts[(layout, t)].append(suffix[:horizon])
    return snapshots, ghosts


def continue_from_state(policy, layout: str, state, seed: int, horizon: int,
                        ghost_actions: Optional[Sequence[int]] = None):
    env = oc.make_env(layout)
    env.reset()
    env.state = copy.deepcopy(state)
    reset_stats(env)
    rng = random.Random(seed)
    sparse_total = 0.0
    a0_hist: List[int] = []
    a1_hist: List[int] = []
    for t in range(horizon):
        actions = policy_actions(policy, env, rng, seed * 10_000 + t)
        if ghost_actions is not None:
            actions[1] = Action.ALL_ACTIONS[ghost_actions[t]]
        a0_hist.append(action_index(actions[0]))
        a1_hist.append(action_index(actions[1]))
        _s, sparse_r, done, _info = env.step(actions)
        sparse_total += sparse_r
        if done:
            break
    first = first_potter_from_stats(env)
    return {
        "basin": basin_from_continuation(first, sparse_total),
        "score": sparse_total,
        "a0": a0_hist,
        "a1": a1_hist,
    }


def bin_metrics(rows: List[Dict]) -> Dict:
    real_counts = Counter(r["real"]["basin"] for r in rows)
    cut_counts = Counter(r["cut"]["basin"] for r in rows)
    p_real = normalize_counts(real_counts)
    p_cut = normalize_counts(cut_counts)
    real_score = float(np.mean([r["real"]["score"] for r in rows]))
    cut_score = float(np.mean([r["cut"]["score"] for r in rows]))
    return {
        "n": len(rows),
        "P_real": p_real,
        "P_cut": p_cut,
        "G_js_bits": js_bits(p_real, p_cut),
        "C_signed_bits": entropy(p_cut) - entropy(p_real),
        "M_score_gain": real_score - cut_score,
        "real_score": real_score,
        "cut_score": cut_score,
        "partner_action_tv": tv_actions(
            [a for r in rows for a in r["real"]["a1"]],
            [a for r in rows for a in r["cut"]["a1"]]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("scripted_roles", "initial",
                                         "checkpoint"),
                    default="scripted_roles")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--seed", type=int, default=91001)
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    ap.add_argument("--snapshot-steps", nargs="*", type=int,
                    default=[40, 80, 120])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--tag", default="smoke")
    args = ap.parse_args()

    torch.set_num_threads(4)
    policy, learned_checkpoint = make_policy(
        args.policy, args.seed, args.checkpoint)
    all_rows = []
    by_t = defaultdict(list)
    ghost_sizes = {}

    for li, layout in enumerate(args.layouts):
        snaps, ghosts = collect_snapshots(
            policy, layout, args.seed + li * 100_000, args.episodes,
            args.snapshot_steps, args.horizon)
        for key, vals in ghosts.items():
            ghost_sizes[f"{key[0]}@{key[1]}"] = len(vals)
        for i, snap in enumerate(snaps):
            pool = ghosts.get((layout, snap["t"]), [])
            if not pool:
                continue
            # Use a different episode's suffix when possible.
            ghost = pool[(i + 1) % len(pool)]
            real = continue_from_state(
                policy, layout, snap["state"],
                args.seed + 1_000_000 + li * 100_000 + i,
                args.horizon)
            cut = continue_from_state(
                policy, layout, snap["state"],
                args.seed + 2_000_000 + li * 100_000 + i,
                args.horizon, ghost_actions=ghost)
            row = {"layout": layout, "t": snap["t"],
                   "real": real, "cut": cut}
            all_rows.append(row)
            by_t[str(snap["t"])].append(row)

    overall = bin_metrics(all_rows) if all_rows else {}
    time_bins = {t: bin_metrics(rows) for t, rows in sorted(by_t.items())}
    gs = [v["G_js_bits"] for _, v in sorted(time_bins.items(),
                                            key=lambda kv: int(kv[0]))]
    pos = [max(0.0, gs[i] - (gs[i - 1] if i else 0.0))
           for i in range(len(gs))]
    total_pos = sum(pos)
    j = max(pos) / total_pos if total_pos > 0 else 0.0

    complete = bool(all_rows) and abs(sum(overall["P_real"].values()) - 1) < 1e-9 \
        and abs(sum(overall["P_cut"].values()) - 1) < 1e-9
    finite_g = bool(overall) and math.isfinite(overall["G_js_bits"]) \
        and overall["G_js_bits"] >= 0.0
    policy_info = {
        "kind": args.policy,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "learned_checkpoint_supplied": learned_checkpoint,
    }
    report = {
        "status": ("Overcooked transition certificate scaffold; smoke "
                   "engineering result, not a learned flagship claim"),
        "contract": {
            "S": "overcooked_ai_py two-agent Overcooked",
            "phi": "(first potter after snapshot) x (delivery within H)",
            "H": args.horizon,
            "nu": {
                "layouts": args.layouts,
                "snapshot_steps": args.snapshot_steps,
                "episodes_per_layout": args.episodes,
            },
            "I": "ghost-partner replay cut of agent-1 feedback",
            "H0": ["scripted_roles", "common context-role marginal",
                   "ghost partner replay"],
        },
        "policy": policy_info,
        "ghost_library_sizes": ghost_sizes,
        "overall": overall,
        "time_bins": time_bins,
        "J_temporal_concentration": j,
        "registered_smoke_outcomes": {
            "OTC_S1_complete_normalized_export": complete,
            "OTC_S2_finite_G_and_action_marginal_diagnostic": finite_g
            and "partner_action_tv" in overall,
            "OTC_S3_checkpoint_status_explicit": "learned_checkpoint_supplied"
            in policy_info,
        },
        "limitations": [
            "Ghost replay is a one-way partner-feedback cut, not a full "
            "matched-marginal causal intervention.",
            "Smoke runs do not match success rates across mechanisms.",
            "No learned flagship claim is made unless a checkpoint is supplied.",
        ],
    }
    out = OUTPUTS / f"overcooked_transition_certificate_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall": overall,
        "J": j,
        "registered_smoke_outcomes": report["registered_smoke_outcomes"],
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
