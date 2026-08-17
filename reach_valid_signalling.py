"""REACH-VALID gate VS: tractable-system sanity in Lewis signalling.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Fresh seeds, disjoint from the
published LEARN-CONVENTION seeds. Ground truth: all 120 codes are
equivalent and reachable at update 0; after capability the code is
absorbing. REACH must be open where the truth is open, closed after
capability, and monotone-irrevocable in between.
"""
from __future__ import annotations

import copy
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = (717_001, 717_102, 717_203)
SNAPS = tuple(list(range(0, 1001, 100)) + [1500, 2000])
M = 8
LOG2M = math.log2(M)


def train_with_snapshots(seed):
    import torch
    import learn_convention as lc

    torch.manual_seed(seed)
    np.random.seed(seed)
    speak = torch.zeros((lc.N_AGENTS, lc.K, lc.K), requires_grad=True)
    listen = torch.zeros((lc.N_AGENTS, lc.K, lc.K), requires_grad=True)
    opt = torch.optim.Adam([speak, listen], lr=lc.LR)
    baseline = 0.0
    gen = torch.Generator().manual_seed(seed)
    snaps, succ_curve = {}, []
    for u in range(lc.UPDATES + 1):
        if u in SNAPS:
            snaps[u] = (speak.detach().clone(), listen.detach().clone(),
                        copy.deepcopy(opt.state_dict()), baseline)
        if u % lc.EVAL_EVERY == 0:
            succ_curve.append((u, lc.mutual_success(speak, listen)))
        if u == lc.UPDATES:
            break
        baseline = train_step(lc, speak, listen, opt, baseline, gen)
    final_code = tuple(lc.code_of(speak))
    final_succ = lc.mutual_success(speak, listen)
    s090 = next((u for u, s in succ_curve if s >= 0.9), None)
    return snaps, final_code, final_succ, s090


def train_step(lc, speak, listen, opt, baseline, gen):
    import torch

    s_idx = torch.randint(0, lc.N_AGENTS, (lc.BATCH,), generator=gen)
    shift = torch.randint(1, lc.N_AGENTS, (lc.BATCH,), generator=gen)
    l_idx = (s_idx + shift) % lc.N_AGENTS
    m = torch.randint(0, lc.K, (lc.BATCH,), generator=gen)
    sp_dist = torch.distributions.Categorical(logits=speak[s_idx, m])
    sym = sp_dist.sample()
    li_dist = torch.distributions.Categorical(logits=listen[l_idx, sym])
    guess = li_dist.sample()
    r = (guess == m).float()
    adv = r - baseline
    new_baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
    loss = -(adv.detach() * (sp_dist.log_prob(sym)
                             + li_dist.log_prob(guess))).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return new_baseline


def continuation(seed_index, snap_u, j, snap):
    import torch
    import learn_convention as lc

    cont_seed = 90_000_000 + 97 * seed_index + 13 * snap_u + j
    torch.manual_seed(cont_seed)
    speak0, listen0, opt_state, baseline = snap
    speak = speak0.clone().requires_grad_(True)
    listen = listen0.clone().requires_grad_(True)
    opt = torch.optim.Adam([speak, listen], lr=lc.LR)
    opt.load_state_dict(copy.deepcopy(opt_state))
    gen = torch.Generator().manual_seed(cont_seed + 1)
    for _ in range(lc.UPDATES - snap_u):
        baseline = train_step(lc, speak, listen, opt, baseline, gen)
    succ = lc.mutual_success(speak, listen)
    code = tuple(lc.code_of(speak))
    return code if succ >= 0.8 else "unconverged"


def run_seed(seed_index):
    import torch

    torch.set_num_threads(2)
    seed = SEEDS[seed_index]
    snaps, final_code, final_succ, s090 = train_with_snapshots(seed)
    curve, labels_by_snap = [], {}
    for u in SNAPS:
        labels = [continuation(seed_index, u, j, snaps[u])
                  for j in range(M)]
        _, counts = np.unique([str(x) for x in labels], return_counts=True)
        p = counts / M
        reach = float(-(p * np.log2(p)).sum() / LOG2M)
        n_codes = len({x for x in labels if x != "unconverged"})
        curve.append(round(reach, 5))
        labels_by_snap[str(u)] = [str(x) for x in labels]
        print(f"seed {seed} u={u}: reach={reach:.3f} codes={n_codes}",
              flush=True)
    return {"seed": seed, "final_code": str(final_code),
            "final_success": round(final_succ, 5), "s090": s090,
            "reach_curve": curve, "labels": labels_by_snap}


def main() -> None:
    from scipy.stats import spearmanr

    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(run_seed, range(len(SEEDS))))

    vs1 = vs2 = vs3 = 0
    for row in rows:
        labels0 = row["labels"]["0"]
        n_codes0 = len({x for x in labels0 if x != "unconverged"})
        if row["reach_curve"][0] >= 0.75 and n_codes0 >= 4:
            vs1 += 1
        first_after = next((i for i, u in enumerate(SNAPS)
                            if row["s090"] is not None and u >= row["s090"]),
                           None)
        if first_after is not None:
            lab = row["labels"][str(SNAPS[first_after])]
            if all(x == row["final_code"] for x in lab):
                vs2 += 1
        rho = spearmanr(row["reach_curve"], SNAPS)[0]
        zero_run = 0
        irrevocable = True
        for v in row["reach_curve"]:
            if zero_run >= 2 and v > 0:
                irrevocable = False
            zero_run = zero_run + 1 if v == 0 else 0
        if rho <= -0.7 and irrevocable:
            vs3 += 1
        closure = next((u for u, v in zip(SNAPS, row["reach_curve"])
                        if v == 0), None)
        row["spearman"] = round(float(rho), 4)
        row["reach_closure_update"] = closure

    outcomes = {
        "VS1_open_at_zero": f"{vs1}/3", "VS1_pass": bool(vs1 == 3),
        "VS2_closed_after_capability": f"{vs2}/3",
        "VS2_pass": bool(vs2 == 3),
        "VS3_monotone_irrevocable": f"{vs3}/3", "VS3_pass": bool(vs3 == 3),
        "VS4_closure_vs_s090": [
            {"seed": r["seed"], "reach_closure": r["reach_closure_update"],
             "s090": r["s090"]} for r in rows],
    }
    (OUTPUTS / "reach_valid_signalling.json").write_text(json.dumps({
        "status": ("REACH-VALID gate VS; fresh seeds disjoint from the "
                   "published LEARN-CONVENTION seeds"),
        "config": {"seeds": SEEDS, "snaps": SNAPS, "m": M},
        "seeds": rows,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print("Wrote reach_valid_signalling.json")


if __name__ == "__main__":
    main()
