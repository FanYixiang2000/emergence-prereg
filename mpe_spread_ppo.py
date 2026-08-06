"""MPE-PPO: SPREAD-REALIZATION with the Overcooked-grade PPO trainer.

Registered as an amendment in V2_ALIGNMENT_PREREGISTRATION.md before
running. Environment, measurement objects, frozen detector and
predictions are unchanged from mpe_spread_realization.py; only the
trainer is strengthened (GAE lambda 0.95, clip 0.2, 6 epochs, entropy
0.01, 3000 updates). Competence precondition frozen at conflict-episode
coverage rate >= 30%.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mpe_spread_realization import (GRID, LOG2_3, MAX_CYCLES, N,
                                    SharedPolicy, formation_eval, make_env,
                                    realization_eval, run_episode)
from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
UPDATES = 3000
EPISODES_PER_UPDATE = 32
GAMMA = 0.95
LAM = 0.95
CLIP = 0.2
EPOCHS = 6
MB = 512
LR = 3e-4
FORM_CKPT_EVERY = 150
SEEDS = (98_101, 98_202, 98_303)
COMPETENCE_MIN = 0.30


def collect_batch(env, net, ep_seed):
    """EPISODES_PER_UPDATE episodes; returns flat tensors with GAE."""
    obs_l, act_l, logp_l, val_l, adv_l, ret_l = [], [], [], [], [], []
    rew_total = 0.0
    for _ in range(EPISODES_PER_UPDATE):
        ep_seed += 1
        obs, _ = env.reset(seed=ep_seed)
        names = list(env.agents)
        eo, ea, elp, ev, er = [], [], [], [], []
        for _t in range(MAX_CYCLES):
            x = torch.tensor(np.stack([obs[a] for a in names]),
                             dtype=torch.float32)
            with torch.no_grad():
                logits, v = net(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logp = dist.log_prob(acts)
            actions = {a: int(acts[i]) for i, a in enumerate(names)}
            nobs, rews, term, trunc, _ = env.step(actions)
            r = np.array([rews[a] for a in names], dtype=np.float32)
            rew_total += float(r.sum())
            eo.append(x.numpy())
            ea.append(acts.numpy())
            elp.append(logp.numpy())
            ev.append(v.numpy())
            er.append(r)
            obs = nobs
            if all(term.values()) or all(trunc.values()):
                break
        ev = np.stack(ev)
        er = np.stack(er)
        T = len(er)
        adv = np.zeros_like(er)
        last = np.zeros(er.shape[1], dtype=np.float32)
        nxt = np.zeros(er.shape[1], dtype=np.float32)
        for t in reversed(range(T)):
            delta = er[t] + GAMMA * nxt - ev[t]
            last = delta + GAMMA * LAM * last
            adv[t] = last
            nxt = ev[t]
        ret = adv + ev
        obs_l.append(np.stack(eo).reshape(-1, 18))
        act_l.append(np.stack(ea).reshape(-1))
        logp_l.append(np.stack(elp).reshape(-1))
        adv_l.append(adv.reshape(-1))
        ret_l.append(ret.reshape(-1))
    return (torch.tensor(np.concatenate(obs_l)),
            torch.tensor(np.concatenate(act_l)),
            torch.tensor(np.concatenate(logp_l)),
            torch.tensor(np.concatenate(adv_l)),
            torch.tensor(np.concatenate(ret_l)),
            rew_total / EPISODES_PER_UPDATE, ep_seed)


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = SharedPolicy()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    env = make_env()
    form_grid, form_open = [], []
    ep_seed = seed
    mean_rew = None
    for u in range(UPDATES):
        obs_b, act_b, logp_b, adv_b, ret_b, mean_rew, ep_seed = \
            collect_batch(env, net, ep_seed)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        idx = np.arange(len(obs_b))
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, len(idx), MB):
                mb = idx[s:s + MB]
                logits, v = net(obs_b[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act_b[mb]) - logp_b[mb])
                s1 = ratio * adv_b[mb]
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_b[mb]
                loss = (-torch.min(s1, s2).mean()
                        + 0.5 * ((v - ret_b[mb]) ** 2).mean()
                        - 0.01 * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if (u + 1) % FORM_CKPT_EVERY == 0:
            form_grid.append(u + 1)
            form_open.append(formation_eval(net, seed + 500_000 + u))
            print(f"  seed {seed} upd {u+1}: R={mean_rew:.1f} "
                  f"form_open={form_open[-1]:.4f}", flush=True)
    trained = realization_eval(net, seed + 700_000)
    form_adj = adjudicate(form_grid, np.array(form_open) * LOG2_3)
    return {
        "trained": trained,
        "formation_grid": form_grid,
        "formation_openness": [round(v, 5) for v in form_open],
        "formation_b5": form_adj["b5_onset"],
        "mean_reward_final": mean_rew,
    }


def main() -> None:
    torch.set_num_threads(4)
    rows = {}
    for seed in SEEDS:
        print(f"=== seed {seed}", flush=True)
        rows[str(seed)] = run_seed(seed)
        t = rows[str(seed)]["trained"]
        print(f"seed {seed}: covRate={t['coverage_rate_conflict']:.3f} "
              f"B5={t['b5_onset']} t*={t['t_star']} dBIC={t['delta_bic']} "
              f"cover_t={t['median_cover_time']} "
              f"perms={t['n_distinct_permutations']}", flush=True)

    competent = {k: r for k, r in rows.items()
                 if r["trained"]["coverage_rate_conflict"] >= COMPETENCE_MIN}
    onset = [r for r in competent.values() if r["trained"]["b5_onset"]]
    sr2 = all(r["trained"]["t_star"] < r["trained"]["median_cover_time"]
              for r in onset
              if r["trained"]["median_cover_time"] is not None)
    outcomes = {
        "competence_met": bool(len(competent) >= 2),
        "n_competent": len(competent),
        "SR1_onset_ge_2of3_competent": bool(len(onset) >= 2),
        "SR2_collapse_before_coverage": bool(sr2 and onset),
        "SR3_ge3_permutations": bool(competent and all(
            r["trained"]["n_distinct_permutations"] >= 3
            for r in competent.values())),
        "SR5_formation_no_onset": bool(all(
            not r["formation_b5"] for r in rows.values())),
        "n_onset": len(onset),
    }
    out = OUTPUTS / "mpe_spread_ppo.json"
    out.write_text(json.dumps({
        "status": ("MPE-PPO amendment: unmodified simple_spread_v3, "
                   "PPO trainer, frozen measurement and detector; "
                   "registered before run"),
        "config": {"updates": UPDATES, "episodes_per_update":
                   EPISODES_PER_UPDATE, "lr": LR, "clip": CLIP,
                   "epochs": EPOCHS, "seeds": SEEDS,
                   "competence_min": COMPETENCE_MIN},
        "seeds": rows,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
