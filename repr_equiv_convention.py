"""REPR-EQUIV (convention): true representation battery.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Retrains
the five published convention seeds byte-identically (the behavioural
sampler R4 uses a dedicated generator so the training RNG stream is
untouched), computing seven representations of the convention openness
object at every checkpoint. Only the measurement changes.

  R1 population-mean speaker mapping entropy (published object);
  R2 mean per-agent speaker entropy (agent-level, permutation-invariant);
  R3 listener-side mapping entropy (role-dual representation);
  R4 behavioural estimate from 2,048 sampled triples per checkpoint;
  R5 P_ref = empirical checkpoint-0 behavioural entropy (not uniform);
  R6 probability truncation epsilon = 0.01 with renormalization;
  R7 coarse symbol binning 5 -> 3 ({0,1},{2,3},{4}).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ant_fine_onset import adjudicate
from learn_convention import (BATCH, EVAL_EVERY, GRID, K, LOG2K, N_AGENTS,
                              N_SEEDS, SEED, UPDATES)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
EPS_TRUNC = 0.01
N_BEHAV = 2048
BINS = [(0, 1), (2, 3), (4,)]
LOG2_3 = math.log2(3)
REPS = ("R1_population_mean", "R2_per_agent_mean", "R3_listener_dual",
        "R4_behavioural_2048", "R5_pref_checkpoint0", "R6_prob_truncation",
        "R7_symbol_binning_3")


def _ent(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log2(torch.clamp(p, min=1e-12))).sum(dim=-1)


def rep_curves_at(speak: torch.Tensor, listen: torch.Tensor,
                  bgen: torch.Generator) -> dict:
    """All seven raw (unnormalized-reference) readouts at one checkpoint."""
    with torch.no_grad():
        sp = torch.softmax(speak, dim=-1)                 # (N, m, s)
        li = torch.softmax(listen, dim=-1)                # (N, s, m)
        pbar = sp.mean(dim=0)                             # (m, s)
        out = {}
        out["R1_population_mean"] = float(_ent(pbar).mean() / LOG2K)
        out["R2_per_agent_mean"] = float(_ent(sp).mean() / LOG2K)
        out["R3_listener_dual"] = float(_ent(li.mean(dim=0)).mean() / LOG2K)
        # R4: sample (speaker, meaning) pairs, draw symbols behaviourally
        s_idx = torch.randint(0, N_AGENTS, (N_BEHAV,), generator=bgen)
        m_idx = torch.randint(0, K, (N_BEHAV,), generator=bgen)
        sym = torch.multinomial(sp[s_idx, m_idx], 1, generator=bgen)[:, 0]
        h_beh = []
        for m in range(K):
            sel = sym[m_idx == m]
            cnt = torch.bincount(sel, minlength=K).float()
            ph = cnt / torch.clamp(cnt.sum(), min=1.0)
            h_beh.append(float(_ent(ph)))
        out["R4_behavioural_2048"] = float(np.mean(h_beh) / LOG2K)
        # R6: truncated population-mean mapping
        pt = pbar.clone()
        pt[pt < EPS_TRUNC] = 0.0
        pt = pt / torch.clamp(pt.sum(dim=-1, keepdim=True), min=1e-12)
        out["R6_prob_truncation"] = float(_ent(pt).mean() / LOG2K)
        # R7: coarse symbol bins
        pb = torch.stack([pbar[:, list(b)].sum(dim=-1) for b in BINS], dim=-1)
        out["R7_symbol_binning_3"] = float(_ent(pb).mean() / math.log2(len(BINS)))
    return out


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    speak = torch.zeros((N_AGENTS, K, K), requires_grad=True)
    listen = torch.zeros((N_AGENTS, K, K), requires_grad=True)
    opt = torch.optim.Adam([speak, listen], lr=0.01)
    baseline = 0.0
    gen = torch.Generator().manual_seed(seed)
    bgen = torch.Generator().manual_seed(seed + 424_242)  # measurement only
    curves = {r: [] for r in REPS if r != "R5_pref_checkpoint0"}
    for u in range(UPDATES + 1):
        if u % EVAL_EVERY == 0:
            vals = rep_curves_at(speak, listen, bgen)
            for r, v in vals.items():
                curves[r].append(v)
        if u == UPDATES:
            break
        s_idx = torch.randint(0, N_AGENTS, (BATCH,), generator=gen)
        shift = torch.randint(1, N_AGENTS, (BATCH,), generator=gen)
        l_idx = (s_idx + shift) % N_AGENTS
        m = torch.randint(0, K, (BATCH,), generator=gen)
        sp_logits = speak[s_idx, m]
        sp_dist = torch.distributions.Categorical(logits=sp_logits)
        sym = sp_dist.sample()
        li_logits = listen[l_idx, sym]
        li_dist = torch.distributions.Categorical(logits=li_logits)
        guess = li_dist.sample()
        r = (guess == m).float()
        adv = r - baseline
        baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
        loss = -(adv.detach() * (sp_dist.log_prob(sym)
                                 + li_dist.log_prob(guess))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # R5: renormalize the behavioural curve by its checkpoint-0 value
    beh = np.array(curves["R4_behavioural_2048"])
    ref0 = max(beh[0], 1e-6)
    curves["R5_pref_checkpoint0"] = list(np.clip(beh / ref0, 0.0, 1.0))
    return curves


def main() -> None:
    torch.set_num_threads(4)
    per_seed = {}
    sums = {r: np.zeros(len(GRID)) for r in REPS}
    for i in range(N_SEEDS):
        curves = run_seed(SEED + i * 101)
        row = {}
        for r in REPS:
            y = np.array(curves[r])
            sums[r] += y
            adj = adjudicate(GRID, y * LOG2_3)
            h = adj.get("hinge", {})
            row[r] = {"b5_onset": adj["b5_onset"],
                      "t_star": h.get("t_star"),
                      "delta_bic": h.get("delta_bic"),
                      "curve": [round(float(v), 5) for v in y]}
        per_seed[str(i)] = row
        print(f"seed={i}: " + " ".join(
            f"{r.split('_')[0]}:B5={row[r]['b5_onset']},t*={row[r]['t_star']}"
            for r in REPS), flush=True)

    rep_cells = {}
    for r in REPS:
        mean_curve = sums[r] / N_SEEDS
        adj = adjudicate(GRID, mean_curve * LOG2_3)
        h = adj.get("hinge", {})
        rep_cells[r] = {"b5_onset": adj["b5_onset"],
                        "t_star": h.get("t_star"),
                        "delta_bic": h.get("delta_bic"),
                        "mean_curve": [round(float(v), 5)
                                       for v in mean_curve]}
        print(f"seed-mean {r}: B5={adj['b5_onset']} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')}", flush=True)

    onsets = [r for r in REPS if rep_cells[r]["b5_onset"]]
    tstars = [rep_cells[r]["t_star"] for r in onsets
              if rep_cells[r]["t_star"] is not None]
    span = float(GRID[-1] - GRID[0])
    trange = (max(tstars) - min(tstars)) / span if len(tstars) >= 2 else 0.0
    outcomes = {
        "n_representations": len(REPS),
        "n_onset_preserved": len(onsets),
        "onset_preservation_rate": round(len(onsets) / len(REPS), 4),
        "t_star_values": tstars,
        "t_star_range_frac_of_span": round(trange, 4),
        "RE2_conv_tstar_range_le_15pct": bool(trange <= 0.15),
        "breaking_representations": [r for r in REPS
                                     if not rep_cells[r]["b5_onset"]],
    }
    report = {
        "status": ("REPR-EQUIV convention battery; measurement-only "
                   "representation changes on byte-identical retrained "
                   "published seeds; registered before run"),
        "config": {"seeds": N_SEEDS, "seed0": SEED, "n_behav": N_BEHAV,
                   "eps_trunc": EPS_TRUNC, "bins": [list(b) for b in BINS],
                   "representations": list(REPS)},
        "representation_cells": rep_cells,
        "per_seed": per_seed,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "repr_equiv_convention.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
