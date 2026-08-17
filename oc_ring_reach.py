"""Protocol REACH: reachable-outcome openness in coordination_ring.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
2026-08-17 before the run). Launched only after the REACH-VALID gates
passed (VH1; fresh-seed VS1, VS2, VS3'). All 576 cells run with no
interim looks; outcomes reported as measured.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = (95_101, 95_202, 95_303, 95_606, 95_707, 95_808, 95_909, 96_010)
CKPTS = (100_000, 300_000, 500_000, 600_000, 700_000, 800_000,
         960_000, 1_200_000, 1_600_000)
M = 8
HORIZON = 2_000_000
OPEN_MIN = 0.95      # k in 3..5 of 8
CLOSED_MAX = 0.544   # k <= 1 or k >= 7
COMMIT_MARGIN = 0.3


def h2(k: int, m: int) -> float:
    p = k / m
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def run_cell(args):
    seed, ckpt, j = args
    import torch

    torch.set_num_threads(4)
    import oc_ring_intervention as oci
    from overcooked_pilot import PolicyNet
    from overcooked_ring_convention import eval_checkpoint

    path = OUTPUTS / f"overcooked_genesis_{oci.TAGS[seed]}_s{seed}_{ckpt}.pt"
    net = PolicyNet()
    net.load_state_dict(torch.load(path, weights_only=True,
                                   map_location="cpu"))
    rng_seed = 1_000_003 * seed + 31 * (ckpt // 1000) + j
    oci.resume_training(net, "coordination_ring", rng_seed, ckpt,
                        HORIZON - ckpt)
    tmp = OUTPUTS / f"oc_reach_{seed}_{ckpt}_{j}.pt"
    torch.save(net.state_dict(), tmp)
    ev = eval_checkpoint(tmp, "coordination_ring", seed)
    tmp.unlink()
    return {"seed": seed, "ckpt": ckpt, "j": j, "p_ccw": ev["p_ccw"],
            "soups": ev["mean_soups"],
            "label_ccw": bool(ev["p_ccw"] > 0.5),
            "committed": bool(abs(ev["p_ccw"] - 0.5) >= COMMIT_MARGIN)}


def t_beh(seed: int) -> int | None:
    """First grid checkpoint from which |p_ccw - 0.5| >= 0.3 holds at
    that and every later grid checkpoint of the stored formation
    record."""
    import oc_ring_intervention as oci

    rec = oci.formation_record(seed)
    pts = [(ck, row["p_ccw"]) for ck, row in zip(rec["grid"], rec["curves"])]
    for i, (ck, _p) in enumerate(pts):
        if all(abs(p - 0.5) >= COMMIT_MARGIN for _, p in pts[i:]):
            return ck
    return None


def main() -> None:
    from scipy.stats import spearmanr

    jobs = [(s, c, j) for s in SEEDS for c in CKPTS for j in range(M)]
    rows = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for r in ex.map(run_cell, jobs):
            rows.append(r)
            print(r, flush=True)

    per_seed = {}
    for s in SEEDS:
        curve = []
        for c in CKPTS:
            k = sum(r["label_ccw"] for r in rows
                    if r["seed"] == s and r["ckpt"] == c)
            curve.append(round(h2(k, M), 5))
        rho = spearmanr(curve, CKPTS)[0]
        open_idx = [i for i, v in enumerate(curve) if v >= OPEN_MIN]
        t_hi = CKPTS[open_idx[-1]] if open_idx else None
        t_lo = None
        if t_hi is not None:
            for i, c in enumerate(CKPTS):
                if c > t_hi and curve[i] <= CLOSED_MAX:
                    t_lo = c
                    break
        per_seed[str(s)] = {
            "reach_curve": curve,
            "k_ccw": [sum(r["label_ccw"] for r in rows
                          if r["seed"] == s and r["ckpt"] == c)
                      for c in CKPTS],
            "spearman": round(float(rho), 4) if rho == rho else None,
            "t_hi": t_hi, "t_lo": t_lo, "t_beh": t_beh(s),
            "reach_960k": curve[CKPTS.index(960_000)],
        }

    from ant_fine_onset import adjudicate

    for s in SEEDS:
        v = per_seed[str(s)]
        try:
            adj = adjudicate(list(CKPTS), np.array(v["reach_curve"]))
            v["RE5_detector_verdict"] = {
                "b5_onset": adj.get("b5_onset"),
                "delta_bic": adj.get("hinge", {}).get("delta_bic"),
                "t_star": adj.get("hinge", {}).get("t_star"),
            }
        except Exception as e:  # grid below validated range; descriptive
            v["RE5_detector_verdict"] = {"error": str(e)}

    n_mono = sum(1 for v in per_seed.values()
                 if v["spearman"] is not None and v["spearman"] <= -0.7)
    both = [v for v in per_seed.values()
            if v["t_hi"] is not None and v["t_lo"] is not None]
    sharp = sum(1 for v in both if v["t_lo"] - v["t_hi"] <= 500_000)
    closed_start = [s for s, v in per_seed.items() if v["t_hi"] is None]
    re3 = sum(1 for v in per_seed.values()
              if v["t_lo"] is not None and v["t_beh"] is not None
              and v["t_lo"] <= v["t_beh"])
    re4 = sum(1 for v in per_seed.values()
              if v["reach_960k"] <= CLOSED_MAX)

    outcomes = {
        "RE1_monotone": f"{n_mono}/8", "RE1_pass": bool(n_mono >= 6),
        "RE2_sharp_closure": f"{sharp}/{len(both)}",
        "RE2_closed_from_start_seeds": closed_start,
        "RE2_pass": bool(len(both) > 0
                         and sharp >= math.ceil(0.75 * len(both))),
        "RE3_closure_precedes_behaviour": f"{re3}/8",
        "RE3_pass": bool(re3 >= 6),
        "RE4_960k_closed_all": f"{re4}/8", "RE4_pass": bool(re4 == 8),
    }
    (OUTPUTS / "oc_ring_reach.json").write_text(json.dumps({
        "status": ("Protocol REACH confirmatory grid; launched after "
                   "REACH-VALID gates passed; no interim looks"),
        "config": {"seeds": SEEDS, "ckpts": CKPTS, "m": M,
                   "horizon": HORIZON, "open_min": OPEN_MIN,
                   "closed_max": CLOSED_MAX},
        "runs": rows,
        "per_seed": per_seed,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print("Wrote oc_ring_reach.json")


if __name__ == "__main__":
    main()
