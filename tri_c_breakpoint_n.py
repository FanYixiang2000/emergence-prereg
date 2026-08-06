"""TRI-C-BP-N: ten-seed robustness of the learned high-order
breakpoint.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Identical measurement to TRI-C-BP under the matured detector
contract (V3.1 effect-size gate, saturation truncation, RE-2
thinning bar).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tri_c_breakpoint import run_seed, GRID, N_UPDATES, hinge_linear
from kuramoto_breakpoint_r2 import truncate_at_saturation

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = tuple(range(95_331, 95_341))
GATE = 0.1


def adjudicate(curve) -> dict:
    x = np.array(GRID, dtype=float)
    y = np.array([curve[str(u)]["openness"] for u in GRID])
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        return out
    xw, yw, t_sat = truncate_at_saturation(x, y)
    out["t_sat"] = t_sat
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
    r3_cross = next((u for u in GRID if curve[str(u)]["r3"] >= 0.9),
                    None)
    out.update({
        "hinge": full, "thinning": thin, "r3_090_cross": r3_cross,
        "onset_pass": bool(full["delta_bic"] >= 10
                           and full["onset_type"] and thin_ok),
        "t_star_before_r3": bool(r3_cross is not None
                                 and full["t_star"] < r3_cross),
    })
    return out


def main() -> None:
    import torch
    torch.set_num_threads(4)
    per_seed = {}
    for seed in SEEDS:
        curve = run_seed(seed)
        adj = adjudicate(curve)
        adj["r_total_final"] = curve[str(N_UPDATES)]["r_total"]
        per_seed[str(seed)] = {"verdict": adj, "curve": curve}
        h = adj.get("hinge", {})
        print(f"seed {seed}: r={adj['r_total_final']} "
              f"gate={adj['gate_passed']} "
              f"dBIC={h.get('delta_bic')} t*={h.get('t_star')} "
              f"onset_pass={adj.get('onset_pass')} "
              f"lead={adj.get('t_star_before_r3')}", flush=True)

    learning = [s for s in per_seed
                if per_seed[s]["verdict"]["r_total_final"] >= 2.7]
    n1 = len(learning)
    onset = [s for s in learning
             if per_seed[s]["verdict"].get("onset_pass", False)]
    n2_frac = len(onset) / max(n1, 1)
    n3 = all(per_seed[s]["verdict"].get("t_star_before_r3", False)
             for s in onset) if onset else False

    outcomes = {"TRICBPN1_learning": bool(n1 >= 9),
                "TRICBPN2_onset_frac": round(n2_frac, 3),
                "TRICBPN2_pass": bool(n2_frac >= 0.9),
                "TRICBPN3_lead": bool(n3),
                "n_learning": n1, "n_onset": len(onset)}
    report = {"status": ("TRI-C-BP-N ten-seed robustness; registered "
                         "before run; matured detector contract"),
              "grid": GRID,
              "seeds": {s: v["verdict"] for s, v in per_seed.items()},
              "curves": {s: v["curve"] for s, v in per_seed.items()},
              "registered_outcomes": outcomes}
    out = OUTPUTS / "tri_c_breakpoint_n.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
