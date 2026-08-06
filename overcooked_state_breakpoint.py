"""OC-STATE-BP: onset-type B5 in a real trained ML system, on the
current-state object.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Reuses the BP-FRESH dense-grid checkpoints (no retraining).
The object is the policy's joint action openness at a FIXED
reference state set -- a current-state object (V3.2) that decouples
policy commitment from state-visitation drift.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

np.Inf = np.inf
import torch

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_joint_collapse_curve import (LAYOUTS, N_ACTIONS,
                                             HORIZON)
from tri_c_breakpoint import hinge_linear
from kuramoto_breakpoint_r2 import truncate_at_saturation

OUTPUTS = Path(__file__).resolve().parent / "outputs"
DENSE_GRID = (40_000, 80_000, 120_000, 160_000, 240_000, 320_000,
              480_000, 640_000, 820_000, 1_000_000, 1_250_000,
              1_500_000, 1_750_000, 2_000_000)
SEEDS = (93_004, 93_005, 93_006)
REF_CKPTS = (40_000, 320_000, 820_000, 2_000_000)
N_REF_PER_LAYOUT = 2000
N_CAP_EPISODES = 30
GATE = 0.1
LOG2N = math.log2(N_ACTIONS)


def ckpt_path(seed: int, ck: int) -> Path:
    return OUTPUTS / f"overcooked_genesis_bpfresh_s{seed}_{ck}.pt"


def load_net(seed: int, ck: int) -> PolicyNet:
    net = PolicyNet()
    net.load_state_dict(torch.load(ckpt_path(seed, ck),
                                   weights_only=True,
                                   map_location="cpu"))
    net.eval()
    return net


def collect_reference_states(ref_seed: int
                             ) -> Dict[str, np.ndarray]:
    """Fixed (obs0, obs1) pairs per layout, pooled across REF_CKPTS
    rollouts of the reference seed."""
    ref = {}
    for li, layout in enumerate(LAYOUTS):
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for ck in REF_CKPTS:
            net = load_net(ref_seed, ck)
            policy = oc.TeamPolicy("net", net=net)
            env = oc.make_env(layout)
            rng = random.Random(ref_seed + li * 31 + ck)
            env.reset()
            for t in range(HORIZON * 6):
                obs = oc.featurize(env)
                pairs.append((obs[0].copy(), obs[1].copy()))
                torch.manual_seed(ref_seed + t)
                actions = policy.actions(env, obs, rng)
                _s, _r, done, _i = env.step(actions)
                if done:
                    env.reset()
        rng2 = random.Random(1234 + li)
        idx = rng2.sample(range(len(pairs)),
                          min(N_REF_PER_LAYOUT, len(pairs)))
        o0 = np.stack([pairs[i][0] for i in idx])
        o1 = np.stack([pairs[i][1] for i in idx])
        ref[layout] = (o0, o1)
        print(f"  ref[{layout}]: {len(idx)} states "
              f"(pool {len(pairs)})", flush=True)
    return ref


def joint_openness(net: PolicyNet, ref: Dict) -> float:
    """Mean over reference states of normalized joint action
    entropy H(pi0)+H(pi1) (agents act independently given state)."""
    hs = []
    with torch.no_grad():
        for layout in LAYOUTS:
            o0, o1 = ref[layout]
            for arr in (o0, o1):
                logits, _v = net(torch.tensor(arr))
                p = torch.softmax(logits, dim=-1)
                h = -(p * torch.log2(p.clamp_min(1e-12))).sum(-1)
                hs.append(h.numpy())
    # each state contributes H0 + H1; normalize by 2 log2 N
    per_state = (hs[0] + hs[1]) if len(hs) == 2 else None
    all_states = np.concatenate([
        (hs[2 * k] + hs[2 * k + 1]) for k in range(len(LAYOUTS))])
    return float(all_states.mean() / (2 * LOG2N))


def capability(net: PolicyNet, seed: int) -> float:
    import overcooked_transition_certificate as otc  # noqa
    policy = oc.TeamPolicy("net", net=net)
    scores = []
    for li, layout in enumerate(LAYOUTS):
        env = oc.make_env(layout)
        rng = random.Random(seed + li * 9_991)
        for ep in range(N_CAP_EPISODES):
            env.reset()
            total = 0.0
            for t in range(HORIZON):
                obs = oc.featurize(env)
                torch.manual_seed(seed * 991 + li * 7 + ep * 13 + t)
                actions = policy.actions(env, obs, rng)
                _s, r, done, _i = env.step(actions)
                total += r
                if done:
                    break
            scores.append(total)
    return float(np.mean(scores))


def adjudicate(grid, openness: List[float]) -> Dict:
    x = np.array(grid, dtype=float)
    y = np.array(openness)
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        out["b5_onset"] = False
        return out
    xw, yw, t_sat = truncate_at_saturation(x, y)
    out["t_sat"] = t_sat
    out["window_points"] = len(yw)
    if len(yw) < 8:
        out["verdict"] = "window_too_short"
        out["b5_onset"] = False
        return out
    full = hinge_linear(xw, yw)
    span = xw[-1] - xw[0]
    thin, thin_ok = {}, True
    for parity in (0, 1):
        t = hinge_linear(xw[parity::2], yw[parity::2])
        ok = (t["delta_bic"] >= 2.0 and t["onset_type"]
              and abs(t["t_star"] - full["t_star"]) <= 0.10 * span)
        t["ok"] = bool(ok)
        thin_ok = thin_ok and ok
        thin[f"parity{parity}"] = t
    out.update({
        "hinge": full, "thinning": thin,
        "b5_onset": bool(full["delta_bic"] >= 10
                         and full["onset_type"] and thin_ok),
    })
    return out


def main() -> None:
    torch.set_num_threads(4)
    print("collecting primary reference states (seed 93004)...",
          flush=True)
    ref_primary = collect_reference_states(93_004)
    print("collecting robustness reference states (seed 93005)...",
          flush=True)
    ref_robust = collect_reference_states(93_005)

    seeds_out = {}
    for seed in SEEDS:
        op_primary, op_robust, caps = [], [], []
        for ck in DENSE_GRID:
            net = load_net(seed, ck)
            op_primary.append(joint_openness(net, ref_primary))
            op_robust.append(joint_openness(net, ref_robust))
            caps.append(capability(net, seed + ck // 40_000))
            print(f"seed {seed} ck {ck}: O_prim="
                  f"{op_primary[-1]:.4f} O_rob={op_robust[-1]:.4f} "
                  f"score={caps[-1]:.2f}", flush=True)
        adj_p = adjudicate(DENSE_GRID, op_primary)
        adj_r = adjudicate(DENSE_GRID, op_robust)
        cap_final = caps[-1]
        cap_cross = next((DENSE_GRID[i] for i, c in enumerate(caps)
                          if cap_final > 0 and c >= 0.9 * cap_final),
                         None)
        leads = bool(adj_p.get("b5_onset") and cap_cross is not None
                     and adj_p["hinge"]["t_star"] < cap_cross)
        seeds_out[str(seed)] = {
            "openness_primary": [round(v, 5) for v in op_primary],
            "openness_robust": [round(v, 5) for v in op_robust],
            "capability": [round(v, 3) for v in caps],
            "adj_primary": adj_p, "adj_robust": adj_r,
            "cap_090_cross": cap_cross, "leads_capability": leads,
        }
        h = adj_p.get("hinge", {})
        print(f"==> seed {seed}: b5={adj_p.get('b5_onset')} "
              f"t*={h.get('t_star')} dBIC={h.get('delta_bic')} "
              f"robust_b5={adj_r.get('b5_onset')} "
              f"cap_cross={cap_cross} leads={leads}", flush=True)

    onset = [s for s in seeds_out
             if seeds_out[s]["adj_primary"].get("b5_onset")]
    ocb1 = len(onset) >= 2
    ocb2 = all(seeds_out[s]["leads_capability"] for s in onset) \
        and bool(onset)
    ocb3 = all(seeds_out[s]["adj_robust"].get("b5_onset")
               for s in onset) and bool(onset)
    outcomes = {"OCB1_onset_real_ml": bool(ocb1),
                "OCB2_collapse_leads_capability": bool(ocb2),
                "OCB3_reference_robustness": bool(ocb3),
                "onset_seeds": onset}
    report = {"status": ("OC-STATE-BP onset-type B5 on the current-"
                         "state object in trained deep multi-agent "
                         "RL; registered before run; BP-FRESH "
                         "checkpoints reused; matured V3.1/V3.2 "
                         "detector"),
              "grid": list(DENSE_GRID), "seeds": seeds_out,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "overcooked_state_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
