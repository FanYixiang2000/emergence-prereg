"""KUR-BP-R: Kuramoto breakpoint under the amended detector
contract.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Amendments: effect-size gate (openness drop >= 0.1 required
for hinge testing) and RE-2's thinning bar (Delta-BIC >= 2, onset
preserved, t* shift <= 2 coarse steps). Three fresh seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kuramoto_breakpoint import simulate, hinge, K_SUPER, K_SUB, \
    MEASURE_EVERY, DT

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = (81_011, 81_012, 81_013)
GATE = 0.1


def adjudicate(res) -> dict:
    x = np.array(res["grid"])
    y = np.array(res["openness"])
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        return out
    full = hinge(x, y)
    step2 = 2 * MEASURE_EVERY * DT
    thin_ok = True
    thin = {}
    for parity in (0, 1):
        t = hinge(x[parity::2], y[parity::2])
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

    outcomes = {"KURR1_onset_3of3": bool(kurr1),
                "KURR2_subcritical_gated_null_3of3": bool(kurr2),
                "KURR3_relational_carrier_3of3": bool(kurr3)}
    report = {"status": ("KUR-BP-R amended detector contract; "
                         "registered before run; fresh seeds"),
              "gate": GATE, "seeds": per_seed,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "kuramoto_breakpoint_r.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
