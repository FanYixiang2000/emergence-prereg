"""KUR-BP-R2: Kuramoto breakpoint with the saturation-truncated
window.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Window rule: analysis ends at t_sat, the first grid point
where openness comes within 5% of total drop of its final value.
Otherwise identical to KUR-BP-R. Fresh seeds 81021-81023.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kuramoto_breakpoint import simulate, hinge, K_SUPER, K_SUB, \
    MEASURE_EVERY, DT

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = (81_021, 81_022, 81_023)
GATE = 0.1


def truncate_at_saturation(x: np.ndarray, y: np.ndarray):
    drop = y[0] - y[-1]
    if drop <= 0:
        return x, y, None
    thresh = y[-1] + 0.05 * drop
    for i, v in enumerate(y):
        if v <= thresh:
            end = max(i + 1, 5)  # keep a minimally fittable window
            return x[:end], y[:end], float(x[min(i, len(x) - 1)])
    return x, y, None


def adjudicate(res) -> dict:
    x = np.array(res["grid"])
    y = np.array(res["openness"])
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        return out
    xw, yw, t_sat = truncate_at_saturation(x, y)
    out["t_sat"] = t_sat
    out["window_points"] = len(yw)
    full = hinge(xw, yw)
    step2 = 2 * MEASURE_EVERY * DT
    thin, thin_ok = {}, True
    for parity in (0, 1):
        t = hinge(xw[parity::2], yw[parity::2])
        ok = (t["delta_bic"] >= 2.0 and t["onset_type"]
              and abs(t["t_star"] - full["t_star"]) <= step2)
        t["ok"] = bool(ok)
        thin_ok = thin_ok and ok
        thin[f"parity{parity}"] = t
    r_final = res["r"][-1]
    t_r90 = next((g for g, r in zip(res["grid"], res["r"])
                  if r >= 0.9 * r_final), None)
    out.update({
        "hinge": full, "thinning": thin, "t_r90": t_r90,
        "onset_pass": bool(full["delta_bic"] >= 10
                           and full["onset_type"] and thin_ok
                           and t_r90 is not None
                           and full["t_star"] < t_r90),
    })
    return out


def main() -> None:
    per_seed = {}
    for seed in SEEDS:
        row = {}
        for name, k in (("super", K_SUPER), ("sub", K_SUB)):
            res = simulate(k, seed)
            adj = adjudicate(res)
            adj["final_ladder"] = res["final_ladder"]
            row[name] = adj
            print(f"seed {seed} {name}: drop={adj['drop']} "
                  f"gate={adj['gate_passed']} "
                  f"t_sat={adj.get('t_sat')} "
                  f"{adj.get('hinge', adj.get('verdict'))}", flush=True)
        per_seed[str(seed)] = row

    kurr1 = all(per_seed[str(s)]["super"].get("onset_pass", False)
                for s in SEEDS)
    kurr2 = all(not per_seed[str(s)]["sub"]["gate_passed"]
                for s in SEEDS)

    def rel_ok(lad):
        return ((lad["C_pair"] + lad["C_high"]) >= 0.8 * lad["C_total"]
                and lad["C_individual"] <= 0.1 * lad["C_total"])

    kurr3 = all(rel_ok(per_seed[str(s)]["super"]["final_ladder"])
                for s in SEEDS)

    outcomes = {"KURR2_1_onset_3of3": bool(kurr1),
                "KURR2_2_subcritical_gated_null_3of3": bool(kurr2),
                "KURR2_3_relational_carrier_3of3": bool(kurr3)}
    report = {"status": ("KUR-BP-R2 saturation-truncated window; "
                         "registered before run; fresh seeds"),
              "gate": GATE, "seeds": per_seed,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "kuramoto_breakpoint_r2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
