"""Overcooked genesis-curve pilot (registered in
OVERCOOKED_TRANSITION_CONTRACT.md, V2 addendum, frozen before launch).

Trains ONE fresh self-play PPO seed with the unchanged `train_mixed`
mechanics, saves checkpoints on the declared grid, and evaluates the
state-level real-vs-ghost transition certificate at every checkpoint.

Output: the formation curves G(s), C(s), M(s), real_score(s) with a row
bootstrap CI on G, plus the registered outcomes OTC-G1..G3 and the
t_seed / t_visible comparison. This is a PILOT, not the confirmatory
flagship (one seed, coarse checkpoint grid, declared up front).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
CHECKPOINT_STEPS = (40_000, 80_000, 160_000, 320_000, 640_000,
                    1_000_000, 1_500_000, 2_000_000)
SNAPSHOT_STEPS = (20, 40, 80, 120, 160)
EPISODES_PER_LAYOUT = 10
HORIZON = 120
N_BOOT = 1000


def train_with_checkpoints(layouts, seed: int, total_steps: int,
                           checkpoint_steps, tag: str) -> Dict[int, Path]:
    """`train_mixed` mechanics, unchanged, plus checkpoint saving at the
    first batch boundary crossing each declared step count."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    envs = {name: oc.make_env(name) for name in layouts}
    layout_rng = random.Random(seed + 5)
    current = layouts[0]
    env = envs[current]
    env.reset()
    obs = oc.featurize(env)
    step_count = 0
    pending = sorted(checkpoint_steps)
    saved: Dict[int, Path] = {}

    def maybe_save():
        while pending and step_count >= pending[0]:
            ck = pending.pop(0)
            path = OUTPUTS / f"overcooked_genesis_{tag}_s{seed}_{ck}.pt"
            torch.save(net.state_dict(), path)
            saved[ck] = path
            print(f"  checkpoint {ck} saved at step {step_count}",
                  flush=True)

    while step_count < total_steps:
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        for _ in range(2048):
            x = torch.tensor(np.stack(obs))
            with torch.no_grad():
                logits, vals = net(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logps = dist.log_prob(acts)
            actions = [Action.ALL_ACTIONS[a] for a in acts.tolist()]
            _s, sparse_r, done, info = env.step(actions)
            shaped = info.get("shaped_r_by_agent", [0, 0])
            anneal = max(0.0, 1.0 - step_count / (0.6 * total_steps))
            rewards = [sparse_r + anneal * shaped[i] for i in range(2)]
            buf["obs"].append(np.stack(obs))
            buf["act"].append(acts.numpy())
            buf["logp"].append(logps.numpy())
            buf["val"].append(vals.numpy())
            buf["rew"].append(np.array(rewards, dtype=np.float32))
            buf["done"].append(done)
            step_count += 1
            if done:
                current = layouts[layout_rng.random() > 0.5]
                env = envs[current]
                env.reset()
            obs = oc.featurize(env)
        rews = np.stack(buf["rew"])
        vals = np.stack(buf["val"])
        dones = np.array(buf["done"], dtype=np.float32)
        T = len(dones)
        adv = np.zeros((T, 2), dtype=np.float32)
        last = np.zeros(2, dtype=np.float32)
        with torch.no_grad():
            _, boot = net(torch.tensor(np.stack(obs)))
        next_val = boot.numpy()
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rews[t] + 0.99 * next_val * mask - vals[t]
            last = delta + 0.99 * 0.95 * mask * last
            adv[t] = last
            next_val = vals[t]
        ret = adv + vals
        obs_b = torch.tensor(np.concatenate(
            [np.stack(buf["obs"])[:, i] for i in (0, 1)]))
        act_b = torch.tensor(np.concatenate(
            [np.stack(buf["act"])[:, i] for i in (0, 1)]))
        logp_b = torch.tensor(np.concatenate(
            [np.stack(buf["logp"])[:, i] for i in (0, 1)]))
        adv_b = torch.tensor(np.concatenate([adv[:, i] for i in (0, 1)]))
        ret_b = torch.tensor(np.concatenate([ret[:, i] for i in (0, 1)]))
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        idx = np.arange(len(obs_b))
        for _ in range(6):
            np.random.shuffle(idx)
            for start in range(0, len(idx), 512):
                mb = idx[start:start + 512]
                logits, v = net(obs_b[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act_b[mb]) - logp_b[mb])
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 0.8, 1.2) * adv_b[mb]
                loss = (-torch.min(s1, s2).mean()
                        + 0.5 * ((v - ret_b[mb]) ** 2).mean()
                        - 0.02 * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        maybe_save()
    maybe_save()
    net.eval()
    return saved


def evaluate_checkpoint(ckpt_path: Path, seed: int) -> Dict:
    """Same evaluation as the pilot2m certificate run."""
    net = PolicyNet()
    net.load_state_dict(torch.load(ckpt_path, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    policy = oc.TeamPolicy("net", net=net)
    all_rows = []
    by_t = defaultdict(list)
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
            row = {"layout": layout, "t": snap["t"], "real": real,
                   "cut": cut}
            all_rows.append(row)
            by_t[str(snap["t"])].append(row)
    overall = otc.bin_metrics(all_rows)
    # Row bootstrap CI on G.
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
    time_bins = {t: otc.bin_metrics(rows)
                 for t, rows in sorted(by_t.items())}
    return {
        "n": overall["n"],
        "G_js_bits": overall["G_js_bits"],
        "G_boot_ci95": [float(lo), float(hi)],
        "C_signed_bits": overall["C_signed_bits"],
        "M_score_gain": overall["M_score_gain"],
        "real_score": overall["real_score"],
        "cut_score": overall["cut_score"],
        "partner_action_tv": overall["partner_action_tv"],
        "time_bins": {t: {"G_js_bits": v["G_js_bits"],
                          "M_score_gain": v["M_score_gain"]}
                      for t, v in time_bins.items()},
    }


def first_crossing(values: List[float], threshold: float) -> int | None:
    for i, v in enumerate(values):
        if v >= threshold:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=93001)
    ap.add_argument("--train-steps", type=int, default=2_000_000)
    ap.add_argument("--tag", default="curve")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing checkpoints if present")
    ap.add_argument("--grid", default=None,
                    help="comma-separated checkpoint steps; default is "
                         "the original 8-point grid")
    args = ap.parse_args()

    grid = (tuple(int(x) for x in args.grid.split(","))
            if args.grid else CHECKPOINT_STEPS)
    torch.set_num_threads(int(os.environ.get("OC_THREADS", "4")))
    t0 = time.time()
    if args.skip_train:
        saved = {}
        for ck in grid:
            p = OUTPUTS / f"overcooked_genesis_{args.tag}_s{args.seed}_{ck}.pt"
            if p.exists():
                saved[ck] = p
    else:
        print(f"training seed {args.seed} for {args.train_steps} steps",
              flush=True)
        saved = train_with_checkpoints(LAYOUTS, args.seed, args.train_steps,
                                       grid, args.tag)
    train_min = round((time.time() - t0) / 60, 2)
    print(f"training done in {train_min} min; evaluating "
          f"{len(saved)} checkpoints", flush=True)

    curve = {}
    for idx, ck in enumerate(sorted(saved)):
        t1 = time.time()
        curve[str(ck)] = evaluate_checkpoint(saved[ck], args.seed + idx)
        curve[str(ck)]["eval_minutes"] = round((time.time() - t1) / 60, 2)
        print(f"  ckpt {ck}: G={curve[str(ck)]['G_js_bits']:.4f} "
              f"CI={curve[str(ck)]['G_boot_ci95']} "
              f"M={curve[str(ck)]['M_score_gain']:.2f} "
              f"score={curve[str(ck)]['real_score']:.2f}", flush=True)

    cks = sorted(int(k) for k in curve)
    g = [curve[str(k)]["G_js_bits"] for k in cks]
    score = [curve[str(k)]["real_score"] for k in cks]
    g_final, score_final = g[-1], score[-1]
    t_seed = first_crossing(g, 0.5 * g_final) if g_final > 0 else None
    t_visible = (first_crossing(score, 0.5 * score_final)
                 if score_final > 0 else None)

    outcomes = {
        "OTC_G1_all_finite_normalized": all(
            math.isfinite(curve[str(k)]["G_js_bits"])
            and curve[str(k)]["G_js_bits"] >= 0 for k in cks),
        "OTC_G2_seed_no_later_than_visible": (
            t_seed is not None and t_visible is not None
            and t_seed <= t_visible),
        "OTC_G3_final_M_positive": curve[str(cks[-1])]["M_score_gain"] > 0,
    }
    report = {
        "status": ("genesis-curve pilot; one seed; registered in "
                   "OVERCOOKED_TRANSITION_CONTRACT.md V2 addendum; "
                   "not a confirmatory flagship result"),
        "seed": args.seed,
        "train_steps": args.train_steps,
        "train_minutes": train_min,
        "checkpoint_grid": cks,
        "curve": curve,
        "t_seed_index": t_seed,
        "t_visible_index": t_visible,
        "t_seed_steps": cks[t_seed] if t_seed is not None else None,
        "t_visible_steps": cks[t_visible] if t_visible is not None else None,
        "registered_outcomes": outcomes,
        "limitations": [
            "One seed; grid coarse; no burstiness claim registered "
            "(grid dependence known from the Pythia thinning result).",
            "Ghost replay is a one-way partner-feedback cut.",
        ],
    }
    out = OUTPUTS / f"overcooked_genesis_curve_{args.tag}_s{args.seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"t_seed_steps": report["t_seed_steps"],
                      "t_visible_steps": report["t_visible_steps"],
                      "registered_outcomes": outcomes}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
