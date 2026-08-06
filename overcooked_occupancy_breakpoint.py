"""OC-OCC-BP: onset-type B5 on the trajectory-occupancy possibility
space of a real trained ML system.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Definition-faithful current-state object: the joint macro-
configuration (held_0, held_1, sign(x0-x1)) pooled at fixed
in-episode phases across rollouts. Reuses BP-FRESH checkpoints; no
retraining. Matured V3.1/V3.2 detector.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

np.Inf = np.inf
import torch

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_joint_collapse_curve import LAYOUTS, HORIZON
from tri_c_breakpoint import hinge_linear
from kuramoto_breakpoint_r2 import truncate_at_saturation

OUTPUTS = Path(__file__).resolve().parent / "outputs"
DENSE_GRID = (40_000, 80_000, 120_000, 160_000, 240_000, 320_000,
              480_000, 640_000, 820_000, 1_000_000, 1_250_000,
              1_500_000, 1_750_000, 2_000_000)
SEEDS = (93_004, 93_005, 93_006)
N_EP = 40
PHASES = (20, 40, 60, 80, 100, 120, 140, 160, 180)
HELD = {None: 0, "onion": 1, "dish": 2, "soup": 3}
N_CONFIG = 4 * 4 * 3
GATE = 0.1


def ckpt_path(seed: int, ck: int) -> Path:
    return OUTPUTS / f"overcooked_genesis_bpfresh_s{seed}_{ck}.pt"


def held_id(player) -> int:
    obj = player.held_object
    name = obj.name if obj is not None else None
    return HELD.get(name, 0)


def config_id(env) -> int:
    p0, p1 = env.state.players[0], env.state.players[1]
    h0, h1 = held_id(p0), held_id(p1)
    sx = int(np.sign(p0.position[0] - p1.position[0])) + 1  # 0,1,2
    return (h0 * 4 + h1) * 3 + sx


def rollout_configs(net, seed: int):
    """Returns {phase: Counter(config)} pooled over layouts, plus
    mean sparse score."""
    policy = oc.TeamPolicy("net", net=net)
    by_phase: Dict[int, Counter] = {t: Counter() for t in PHASES}
    scores = []
    phase_set = set(PHASES)
    for li, layout in enumerate(LAYOUTS):
        env = oc.make_env(layout)
        rng = random.Random(seed + li * 8_888)
        for ep in range(N_EP):
            env.reset()
            total = 0.0
            for t in range(HORIZON):
                if t in phase_set:
                    by_phase[t][config_id(env)] += 1
                obs = oc.featurize(env)
                torch.manual_seed(seed * 733 + li * 17 + ep * 41 + t)
                actions = policy.actions(env, obs, rng)
                _s, r, done, _i = env.step(actions)
                total += r
                if done:
                    break
            scores.append(total)
    return by_phase, float(np.mean(scores))


def phase_openness(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    h = -sum((v / n) * math.log2(v / n)
             for v in counter.values() if v > 0)
    return h / math.log2(N_CONFIG)


def top2_mass(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    top = sorted(counter.values(), reverse=True)[:2]
    return sum(top) / n


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
    seeds_out = {}
    for seed in SEEDS:
        openness, caps, top2_first, top2_last = [], [], None, None
        for idx, ck in enumerate(DENSE_GRID):
            net = PolicyNet()
            net.load_state_dict(torch.load(ckpt_path(seed, ck),
                                           weights_only=True,
                                           map_location="cpu"))
            net.eval()
            by_phase, score = rollout_configs(net, seed + idx * 100)
            o = float(np.mean([phase_openness(by_phase[t])
                               for t in PHASES]))
            openness.append(o)
            caps.append(score)
            t2 = float(np.mean([top2_mass(by_phase[t])
                                for t in PHASES]))
            if idx == 0:
                top2_first = t2
            top2_last = t2
            print(f"seed {seed} ck {ck}: O={o:.4f} score={score:.2f} "
                  f"top2={t2:.3f}", flush=True)
        adj = adjudicate(DENSE_GRID, openness)
        cap_final = caps[-1]
        cap_cross = next((DENSE_GRID[i] for i, c in enumerate(caps)
                          if cap_final > 0 and c >= 0.9 * cap_final),
                         None)
        leads = bool(adj.get("b5_onset") and cap_cross is not None
                     and adj["hinge"]["t_star"] < cap_cross)
        seeds_out[str(seed)] = {
            "openness": [round(v, 5) for v in openness],
            "capability": [round(v, 3) for v in caps],
            "adj": adj, "cap_090_cross": cap_cross,
            "leads_capability": leads,
            "top2_mass_first": round(top2_first, 4),
            "top2_mass_last": round(top2_last, 4),
            "selectivity_ok": bool(top2_last >= 0.60
                                   and top2_first <= 0.35),
        }
        h = adj.get("hinge", {})
        print(f"==> seed {seed}: b5={adj.get('b5_onset')} "
              f"t*={h.get('t_star')} dBIC={h.get('delta_bic')} "
              f"drop={adj['drop']} cap_cross={cap_cross} "
              f"leads={leads} top2 {top2_first:.2f}->{top2_last:.2f}",
              flush=True)

    onset = [s for s in seeds_out
             if seeds_out[s]["adj"].get("b5_onset")]
    occ1 = len(onset) >= 2
    occ2 = all(seeds_out[s]["leads_capability"] for s in onset) \
        and bool(onset)
    occ3 = all(seeds_out[s]["selectivity_ok"] for s in onset) \
        and bool(onset)
    outcomes = {"OCC1_onset_real_ml": bool(occ1),
                "OCC2_collapse_leads_capability": bool(occ2),
                "OCC3_selectivity": bool(occ3),
                "onset_seeds": onset}
    report = {"status": ("OC-OCC-BP onset-type B5 on the trajectory-"
                         "occupancy possibility space of trained deep "
                         "multi-agent RL; registered before run; "
                         "BP-FRESH checkpoints reused"),
              "grid": list(DENSE_GRID), "phases": list(PHASES),
              "n_config": N_CONFIG, "seeds": seeds_out,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "overcooked_occupancy_breakpoint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
