"""OC-RING-REAL: within-episode realization commitment in the standard
Overcooked coordination_ring benchmark.

Preregistered in V2_ALIGNMENT_PREREGISTRATION.md (2026-08-05) before this
file was written. Uses ONLY the already-trained, stored checkpoints from
OC-RING / OC-RING-EXT; no new training, no parameter changes.

At the frozen-rule-selected mid-training checkpoint the population is
globally uncommitted (both circulation directions occur across episodes)
while each episode commits internally. We measure direction openness
within episodes by branched continuation rollouts and adjudicate the
per-seed median curve with the frozen B5 detector.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from ant_fine_onset import adjudicate
from overcooked_pilot import PolicyNet
from overcooked_ring_convention import winding_laps

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LOG2_3 = math.log2(3)

RING_SEEDS = (95_101, 95_202, 95_303, 95_606, 95_707, 95_808, 95_909, 96_010)
TAGS = {95_101: "ring95101", 95_202: "ring95202", 95_303: "ring95303",
        95_606: "ringx95606", 95_707: "ringx95707", 95_808: "ringx95808",
        95_909: "ringx95909", 96_010: "ringx96010"}

# frozen protocol constants (preregistered)
N_EP = 30
PROBES = tuple(range(0, 41, 2)) + tuple(range(45, 196, 5))
K_BRANCH = 12
C_STEPS = 50
LAPS_MIN_BRANCH = 0.25
P_LO, P_HI = 0.30, 0.70
N_COM_MIN = 20
HORIZON = 200


def select_checkpoint(seed: int) -> int | None:
    """Frozen rule: last checkpoint with p_ccw in [0.3, 0.7] and
    n_committed_episodes >= 20, from the stored formation curves."""
    orig = json.load(open(OUTPUTS / "overcooked_ring_convention.json"))
    ext = json.load(open(OUTPUTS / "oc_ring_ext.json"))
    rec = (orig["systems"]["ring"].get(str(seed))
           or ext["ext_seeds"][str(seed)])
    chosen = None
    for ck, row in zip(rec["grid"], rec["curves"]):
        if (P_LO <= row["p_ccw"] <= P_HI
                and row["n_committed_episodes"] >= N_COM_MIN):
            chosen = ck
    return chosen


def load_net(seed: int, ckpt: int) -> PolicyNet:
    path = OUTPUTS / f"overcooked_genesis_{TAGS[seed]}_s{seed}_{ckpt}.pt"
    net = PolicyNet()
    net.load_state_dict(torch.load(path, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    return net


def act(net: PolicyNet, env) -> list:
    obs = oc.featurize(env)
    with torch.no_grad():
        logits, _ = net(torch.tensor(np.stack(obs)))
        acts = torch.distributions.Categorical(logits=logits).sample()
    return [Action.ALL_ACTIONS[a] for a in acts.tolist()]


def branch_openness(net: PolicyNet, env, benv) -> float:
    """Direction openness from the current state via K branched
    continuations; RNG snapshot keeps the base episode unaffected."""
    rng_state = torch.get_rng_state()
    signs = []
    base_state = env.state
    for b in range(K_BRANCH):
        benv.state = base_state.deepcopy()
        benv.t = 0
        pos = [[], []]
        for _t in range(C_STEPS):
            for i, p in enumerate(benv.state.players):
                pos[i].append(p.position)
            _s, _r, done, _info = benv.step(act(net, benv))
            if done:
                break
        w = winding_laps(pos[0]) + winding_laps(pos[1])
        if abs(w) >= LAPS_MIN_BRANCH:
            signs.append(1 if w > 0 else -1)
    torch.set_rng_state(rng_state)
    n_com = len(signs)
    if n_com == 0:
        return 1.0
    p = (sum(1 for s in signs if s > 0) + 1) / (n_com + 2)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def measure_seed(net: PolicyNet, seed: int):
    env = oc.make_env("coordination_ring")
    benv = oc.make_env("coordination_ring")
    per_ep = []
    ep_dirs = []
    for ep in range(N_EP):
        torch.manual_seed(seed * 10_000 + ep)
        env.reset()
        probes = {}
        pos = [[], []]
        want = set(PROBES)
        for t in range(HORIZON):
            if t in want:
                probes[t] = branch_openness(net, env, benv)
            for i, p in enumerate(env.state.players):
                pos[i].append(p.position)
            _s, _r, done, _info = env.step(act(net, env))
            if done:
                break
        per_ep.append([probes[t] for t in PROBES])
        w = winding_laps(pos[0]) + winding_laps(pos[1])
        if abs(w) >= 0.5:
            ep_dirs.append(1 if w > 0 else -1)
        print(f"    ep {ep}: open0={per_ep[-1][0]:.3f} "
              f"openEnd={per_ep[-1][-1]:.3f} dir={ep_dirs[-1] if abs(w) >= 0.5 else 0}",
              flush=True)
    med = np.median(np.array(per_ep), axis=0)
    adj = adjudicate(np.array(PROBES, dtype=float), med * LOG2_3)
    h = adj.get("hinge", {})
    n_ccw = sum(1 for d in ep_dirs if d > 0)
    return {
        "median_curve": [round(float(v), 4) for v in med],
        "b5_onset": bool(adj.get("b5_onset")),
        "t_star": h.get("t_star"),
        "delta_bic": h.get("delta_bic"),
        "verdict": adj.get("verdict", "hinge_tested"),
        "drop": adj.get("drop"),
        "n_committed_eps": len(ep_dirs),
        "n_ccw_eps": n_ccw,
        "both_directions": bool(0 < n_ccw < len(ep_dirs)),
        "initial_openness": float(med[0]),
    }


def main() -> None:
    torch.set_num_threads(4)
    report = {"selected": {}, "untrained": {}, "final": {}}

    selections = {s: select_checkpoint(s) for s in RING_SEEDS}
    print("selected checkpoints:", selections, flush=True)

    for seed, ck in selections.items():
        if ck is None:
            report["selected"][str(seed)] = {"excluded": True}
            continue
        print(f"=== seed {seed} @ checkpoint {ck}", flush=True)
        r = measure_seed(load_net(seed, ck), seed)
        r["checkpoint"] = ck
        report["selected"][str(seed)] = r
        print(f"  -> onset={r['b5_onset']} t*={r['t_star']} "
              f"dBIC={r['delta_bic']} both_dirs={r['both_directions']}",
              flush=True)

    for seed in RING_SEEDS:
        print(f"=== seed {seed} untrained control", flush=True)
        torch.manual_seed(seed)
        net = PolicyNet()
        net.eval()
        report["untrained"][str(seed)] = measure_seed(net, seed)
        print(f"=== seed {seed} final-checkpoint control", flush=True)
        report["final"][str(seed)] = measure_seed(
            load_net(seed, 2_000_000), seed)

    sel = [r for r in report["selected"].values()
           if not r.get("excluded")]
    n_sel = len(sel)
    n_onset = sum(r["b5_onset"] for r in sel)
    n_both = sum(r["both_directions"] for r in sel)
    unt = list(report["untrained"].values())
    fin = list(report["final"].values())
    outcomes = {
        "OCRR1_ge_6of8_selectable": bool(n_sel >= 6),
        "n_selected": n_sel,
        "OCRR2_onset_ge_060_of_selected": bool(
            n_sel > 0 and n_onset / n_sel >= 0.60),
        "n_onset": n_onset,
        "OCRR3_both_directions_ge_060": bool(
            n_sel > 0 and n_both / n_sel >= 0.60),
        "n_both_directions": n_both,
        "OCRR4_untrained_zero_onset": bool(
            all(not r["b5_onset"] for r in unt)),
        "OCRR4_untrained_open_ge_07": bool(
            all(min(r["median_curve"]) >= 0.7 for r in unt)),
        "OCRR5_final_initial_lt_05_ge_6of8": bool(
            sum(r["initial_openness"] < 0.5 for r in fin) >= 6),
        "OCRR5_final_zero_onset": bool(
            all(not r["b5_onset"] for r in fin)),
    }
    out = OUTPUTS / "oc_ring_realization.json"
    out.write_text(json.dumps({
        "status": ("OC-RING-REAL within-episode realization commitment; "
                   "stored checkpoints only, frozen selection rule, "
                   "registered before run"),
        "config": {"n_ep": N_EP, "probes": list(PROBES),
                   "k_branch": K_BRANCH, "c_steps": C_STEPS,
                   "laps_min_branch": LAPS_MIN_BRANCH,
                   "selection": [P_LO, P_HI, N_COM_MIN],
                   "selections": selections},
        "systems": report,
        "registered_outcomes": outcomes}, indent=1), encoding="utf-8")
    print(json.dumps(outcomes, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
