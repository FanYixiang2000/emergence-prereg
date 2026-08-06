"""OC-RING-INT: causal test of formation-level commitment in the standard
Overcooked coordination_ring benchmark.

Preregistered in V2_ALIGNMENT_PREREGISTRATION.md (2026-08-05) before this
file was written. Perturb stored checkpoints with unbiased parameter
noise, resume training with byte-identical mechanics for exactly 400k
steps, and test whether the direction convention flips. Prediction:
flippable while the population is globally open, locked once committed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_ring_convention import eval_checkpoint

OUTPUTS = Path(__file__).resolve().parent / "outputs"

RING_SEEDS = (95_101, 95_202, 95_303, 95_606, 95_707, 95_808, 95_909, 96_010)
TAGS = {95_101: "ring95101", 95_202: "ring95202", 95_303: "ring95303",
        95_606: "ringx95606", 95_707: "ringx95707", 95_808: "ringx95808",
        95_909: "ringx95909", 96_010: "ringx96010"}

T_EARLY = 100_000
T_LATE = 1_600_000
RESUME_STEPS = 400_000
TOTAL_STEPS = 2_000_000        # original budget; keeps anneal schedule
NOISE_SCALES = (0.25, 0.5)
COMMIT_MARGIN = 0.3
P_LO, P_HI, N_COM_MIN = 0.30, 0.70, 20


def formation_record(seed: int) -> dict:
    orig = json.load(open(OUTPUTS / "overcooked_ring_convention.json"))
    ext = json.load(open(OUTPUTS / "oc_ring_ext.json"))
    return (orig["systems"]["ring"].get(str(seed))
            or ext["ext_seeds"][str(seed)])


def t_open_checkpoint(seed: int) -> int:
    """Selection-rule checkpoint; fallback (3 non-selectable seeds):
    p_ccw closest to 0.5 with n_committed >= 20 (preregistered)."""
    rec = formation_record(seed)
    chosen = None
    for ck, row in zip(rec["grid"], rec["curves"]):
        if (P_LO <= row["p_ccw"] <= P_HI
                and row["n_committed_episodes"] >= N_COM_MIN):
            chosen = ck
    if chosen is not None:
        return chosen
    best, bd = None, 10.0
    for ck, row in zip(rec["grid"], rec["curves"]):
        if row["n_committed_episodes"] >= N_COM_MIN:
            d = abs(row["p_ccw"] - 0.5)
            if d < bd:
                best, bd = ck, d
    return best


def openness_at(seed: int, ckpt: int) -> float:
    rec = formation_record(seed)
    i = rec["grid"].index(ckpt)
    return rec["curves"][i]["circulation_openness"]


def perturb(net: PolicyNet, s: float, gen: torch.Generator) -> None:
    with torch.no_grad():
        for p in net.parameters():
            std = float(p.std()) if p.numel() > 1 else 1e-3
            p.add_(torch.randn(p.shape, generator=gen) * s * std)


def resume_training(net: PolicyNet, layout: str, rng_seed: int,
                    start_step: int, extra_steps: int) -> None:
    """train_with_checkpoints mechanics, unchanged, resumed from `net`
    at `start_step` of the original TOTAL_STEPS schedule."""
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed % (2 ** 31))
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    env = oc.make_env(layout)
    env.reset()
    obs = oc.featurize(env)
    step_count = start_step
    end_step = start_step + extra_steps
    while step_count < end_step:
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
            anneal = max(0.0, 1.0 - step_count / (0.6 * TOTAL_STEPS))
            rewards = [sparse_r + anneal * shaped[i] for i in range(2)]
            buf["obs"].append(np.stack(obs))
            buf["act"].append(acts.numpy())
            buf["logp"].append(logps.numpy())
            buf["val"].append(vals.numpy())
            buf["rew"].append(np.array(rewards, dtype=np.float32))
            buf["done"].append(done)
            step_count += 1
            if done:
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


def run_one(seed: int, ckpt: int, s: float, label: str,
            orig_dir: int) -> dict:
    path = OUTPUTS / f"overcooked_genesis_{TAGS[seed]}_s{seed}_{ckpt}.pt"
    net = PolicyNet()
    net.load_state_dict(torch.load(path, weights_only=True,
                                   map_location="cpu"))
    rng_seed = 7 * seed + ckpt + round(100 * s)
    gen = torch.Generator().manual_seed(rng_seed)
    perturb(net, s, gen)
    resume_training(net, "coordination_ring", rng_seed, ckpt, RESUME_STEPS)
    tmp = OUTPUTS / f"oc_int_{seed}_{label}_{s}.pt"
    torch.save(net.state_dict(), tmp)
    ev = eval_checkpoint(tmp, "coordination_ring", seed)
    p = ev["p_ccw"]
    new_dir = 1 if p > 0.5 else -1
    committed = abs(p - 0.5) >= COMMIT_MARGIN
    outcome = ("flip" if committed and new_dir != orig_dir
               else "held" if committed else "uncommitted")
    return {"seed": seed, "time": label, "ckpt": ckpt, "scale": s,
            "openness_at_perturbation": openness_at(seed, ckpt),
            "final_p_ccw": p, "final_soups": ev["mean_soups"],
            "outcome": outcome}


def main() -> None:
    torch.set_num_threads(4)
    runs = []
    for seed in RING_SEEDS:
        rec = formation_record(seed)
        orig_dir = 1 if rec["curves"][-1]["p_ccw"] > 0.5 else -1
        unpert_soups = rec["final_soups"]
        times = {"early": T_EARLY, "open": t_open_checkpoint(seed),
                 "late": T_LATE}
        for label, ckpt in times.items():
            for s in NOISE_SCALES:
                r = run_one(seed, ckpt, s, label, orig_dir)
                r["orig_dir"] = orig_dir
                r["unperturbed_final_soups"] = unpert_soups
                runs.append(r)
                print(f"seed {seed} {label}@{ckpt} s={s}: "
                      f"{r['outcome']} p_ccw={r['final_p_ccw']} "
                      f"soups={r['final_soups']}", flush=True)
                (OUTPUTS / "oc_ring_intervention_partial.json").write_text(
                    json.dumps(runs, indent=1))

    def rate(label, kinds):
        rr = [r for r in runs if r["time"] == label]
        return sum(r["outcome"] in kinds for r in rr), len(rr)

    open_moved, n_open = rate("open", ("flip", "uncommitted"))
    late_moved, n_late = rate("late", ("flip", "uncommitted"))
    open_flip, _ = rate("open", ("flip",))
    late_flip, _ = rate("late", ("flip",))

    from scipy.stats import fisher_exact
    _, p_fisher = fisher_exact(
        [[open_moved, n_open - open_moved],
         [late_moved, n_late - late_moved]], alternative="greater")

    flips = np.array([r["outcome"] == "flip" for r in runs], dtype=float)
    opens = np.array([r["openness_at_perturbation"] for r in runs])
    if flips.sum() in (0, len(flips)):
        auc = None
    else:
        pos, neg = opens[flips == 1], opens[flips == 0]
        auc = float(np.mean([(pi > ni) + 0.5 * (pi == ni)
                             for pi in pos for ni in neg]))

    late_rec = ([r["final_soups"] / max(r["unperturbed_final_soups"], 1e-9)
                 for r in runs if r["time"] == "late"])
    outcomes = {
        "OCI1_open_moved_gt_late_fisher_p": float(p_fisher),
        "OCI1_pass": bool(p_fisher < 0.05),
        "open_moved": f"{open_moved}/{n_open}",
        "late_moved": f"{late_moved}/{n_late}",
        "open_strict_flips": open_flip,
        "late_strict_flips": late_flip,
        "OCI2_zero_late_flips": bool(late_flip == 0),
        "OCI3_auc": auc,
        "OCI3_pass": bool(auc is not None and auc >= 0.70),
        "OCI4_late_recovery_median": float(np.median(late_rec)),
        "OCI4_pass": bool(np.median(late_rec) >= 0.5),
    }
    (OUTPUTS / "oc_ring_intervention.json").write_text(json.dumps({
        "status": ("OC-RING-INT causal commitment test; stored checkpoints, "
                   "byte-identical resumed training, registered before run"),
        "config": {"t_early": T_EARLY, "t_late": T_LATE,
                   "resume_steps": RESUME_STEPS,
                   "noise_scales": NOISE_SCALES,
                   "commit_margin": COMMIT_MARGIN},
        "runs": runs, "registered_outcomes": outcomes}, indent=1))
    print(json.dumps(outcomes, indent=1))


if __name__ == "__main__":
    main()
