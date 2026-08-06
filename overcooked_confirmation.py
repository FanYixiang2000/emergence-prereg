"""Overcooked-AI full six-component confirmation (frozen protocol run).

DO NOT RUN before OVERCOOKED_PREREGISTRATION.md is finalized and pushed
to the externally timestamped public repository. The script refuses to
start unless --confirm-frozen is passed, as a procedural guard.

Design (fixed by the preregistration; pilots disclosed):
    contexts        two unmodified layouts, drawn per episode during
                    training (pair fixed at freeze from pilot evidence);
    training        self-play PPO, shared parameters, community shaped
                    rewards annealed to zero over 60% of steps, sparse
                    delivery reward untouched; steps fixed at freeze;
    trigger         agent 0 first to pot an ingredient;
    contract A      basins (first potter x delivery), sparse team value,
                    horizon 400, T=1 rollouts;
    contract B      success-indicator value (>= 1 delivery), horizon 300
                    (re-evaluation only; declared second contract);
    systems/seed    learned, initialization twin, scripted role pair,
                    behavioural clone of the scripted pair, fixed-role
                    (do-commit forced every episode), untrained-other;
    thresholds      copied unchanged from the frozen criterion.

Outputs one JSON with per-seed metrics, verdicts and the registered
outcome tallies (OC-1..OC-5 as frozen).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet

OUTPUTS = Path(__file__).resolve().parent / "outputs"

CONFIRMATION_SEEDS = tuple(range(77001, 77013))
N_EVAL = 40


def train_mixed(layouts, seed: int, total_steps: int) -> PolicyNet:
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
    net.eval()
    return net


def initial_twin(seed: int) -> PolicyNet:
    torch.manual_seed(seed)
    net = PolicyNet()
    net.eval()
    return net


def contract_b_metrics(policy, layouts, seed_offset: int) -> Dict:
    """Second declared contract: success value, horizon 300."""
    old_h = oc.HORIZON
    oc.HORIZON = 300
    try:
        rows: List[Dict] = []
        for ctx, layout in enumerate(layouts):
            for episode in range(N_EVAL):
                seed = seed_offset + 10_000 * ctx + episode
                for mode in (None, "do_commit", "do_block"):
                    row = oc.run_episode(policy, layout, seed, mode)
                    row["mode"] = mode or "natural"
                    row["context"] = ctx
                    row["success"] = int(row["score"] > 0)
                    rows.append(row)
    finally:
        oc.HORIZON = old_h

    def subset(mode, ctx=None):
        return [r for r in rows if r["mode"] == mode
                and (ctx is None or r["context"] == ctx)]

    def dist(rows_in):
        counts: Dict[str, int] = {}
        for r in rows_in:
            counts[r["basin"]] = counts.get(r["basin"], 0) + 1
        total = len(rows_in)
        return {b: counts.get(b, 0) / total for b in oc.BASINS}

    natural = subset("natural")
    counts: Dict[str, int] = {}
    for r in natural:
        counts[r["basin"]] = counts.get(r["basin"], 0) + 1
    trig = {str(c): float(np.mean([r["trigger"]
                                   for r in subset("natural", c)]))
            for c in (0, 1)}
    mean_s = lambda rows_in: float(np.mean([r["success"]
                                            for r in rows_in]))
    return {
        "potential_bits": oc.entropy_bits(counts),
        "conditional_selectivity": abs(trig["0"] - trig["1"]),
        "specificity_js_bits": oc.js_bits(dist(subset("do_commit")),
                                          dist(subset("do_block"))),
        "usefulness_gap": mean_s(natural) - mean_s(subset("do_block")),
    }


def run_seed(seed: int, layouts, train_steps: int,
             ckpt_prefix: str = "overcooked_confirm_s") -> Dict:
    t0 = time.time()
    net = train_mixed(layouts, seed, train_steps)
    torch.save(net.state_dict(),
               OUTPUTS / f"{ckpt_prefix}{seed}.pt")
    twin = initial_twin(seed)
    clone = oc.train_bc_clone(layouts, seed + 31)
    offset = 20_000_000 + seed * 100_000

    systems = {
        "learned": oc.TeamPolicy("net", net=net),
        "initial_twin": oc.TeamPolicy("net", net=twin),
        "scripted_roles": oc.TeamPolicy("scripted_roles", cook_agent=0),
        "bc_clone": oc.TeamPolicy("clone", net=clone),
        "untrained_other": oc.TeamPolicy("net", net=initial_twin(seed + 999)),
    }
    metrics = {name: oc.evaluate(pol, layouts, N_EVAL, offset)
               for name, pol in systems.items()}
    acquisition = (metrics["learned"]["conditional_selectivity"]
                   - metrics["initial_twin"]["conditional_selectivity"])
    out: Dict = {"train_minutes": round((time.time() - t0) / 60, 1)}
    for name, m in metrics.items():
        endo = name in ("learned", "initial_twin", "untrained_other")
        acq = acquisition if name == "learned" else 0.0
        out[name] = {"metrics": m, "acquisition": acq,
                     "verdict": oc.verdict(m, endo, acq)}
    out["learned_contract_b"] = contract_b_metrics(
        systems["learned"], layouts, offset + 5_000_000)
    out["twin_contract_b"] = contract_b_metrics(
        systems["initial_twin"], layouts, offset + 5_000_000)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-frozen", action="store_true",
                        help="assert the preregistration is frozen and "
                             "externally timestamped")
    parser.add_argument("--layouts", nargs=2, required=True)
    parser.add_argument("--train_steps", type=int, required=True)
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=list(CONFIRMATION_SEEDS))
    parser.add_argument("--tag", default="confirmation")
    parser.add_argument("--ckpt-prefix", default="overcooked_confirm_s")
    args = parser.parse_args()
    if not args.confirm_frozen:
        raise SystemExit(
            "Refusing to run: pass --confirm-frozen only after "
            "OVERCOOKED_PREREGISTRATION.md is frozen and pushed to the "
            "externally timestamped repository.")
    torch.set_num_threads(8)
    report = {
        "status": "frozen confirmation run",
        "layouts": args.layouts,
        "train_steps": args.train_steps,
        "n_eval_per_context": N_EVAL,
        "seeds": {},
    }
    out = OUTPUTS / f"overcooked_{args.tag}.json"
    for seed in args.seeds:
        print(f"=== confirmation seed {seed} ===", flush=True)
        report["seeds"][str(seed)] = run_seed(
            seed, tuple(args.layouts), args.train_steps,
            ckpt_prefix=args.ckpt_prefix)
        v = report["seeds"][str(seed)]["learned"]["verdict"]
        print(f"  learned verdict {v['emergent']} "
              f"failed {';'.join(v['failed']) or '-'}", flush=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
