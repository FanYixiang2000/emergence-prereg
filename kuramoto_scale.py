"""KUR-SCALE: breakpoint time vs distance from criticality.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Sweeps coupling K across the transition; detector identical
to KUR-BP-R2 (gate, saturation truncation, thinning). Two fresh
seeds per K.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kuramoto_breakpoint import simulate
from kuramoto_breakpoint_r2 import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
KS = (0.9, 1.1, 1.5, 2.0, 2.5)
SEEDS = (82_001, 82_002)


def main() -> None:
    per_k = {}
    for k in KS:
        rows = {}
        for seed in SEEDS:
            res = simulate(k, seed)
            adj = adjudicate(res)
            rows[str(seed)] = adj
            h = adj.get("hinge", {})
            print(f"K={k} seed {seed}: drop={adj['drop']} "
                  f"gate={adj['gate_passed']} "
                  f"onset_pass={adj.get('onset_pass')} "
                  f"t*={h.get('t_star')} "
                  f"slope_after={h.get('slope_after')}", flush=True)
        per_k[str(k)] = rows

    def seed_pass(adj):
        return adj.get("onset_pass", False)

    def mean_of(kk, field):
        vals = [abs(per_k[str(kk)][str(s)]["hinge"][field])
                for s in SEEDS
                if seed_pass(per_k[str(kk)][str(s)])]
        return float(np.mean(vals)) if vals else None

    ks1 = all(all(seed_pass(per_k[str(k)][str(s)]) for s in SEEDS)
              for k in KS if k >= 1.1)
    k09 = per_k["0.9"]
    k09_gated_null = all(not k09[str(s)]["gate_passed"] for s in SEEDS)
    higher_tstars = [mean_of(k, "t_star") for k in KS if k >= 1.1]
    higher_tstars = [t for t in higher_tstars if t is not None]
    k09_tstars = [k09[str(s)]["hinge"]["t_star"] for s in SEEDS
                  if seed_pass(k09[str(s)])]
    k09_slower = bool(k09_tstars
                      and min(k09_tstars) > max(higher_tstars
                                                or [float("-inf")]))
    ks1_k09 = bool(k09_gated_null or k09_slower)

    passing = [k for k in KS
               if all(seed_pass(per_k[str(k)][str(s)]) for s in SEEDS)]
    tmeans = [mean_of(k, "t_star") for k in passing]
    smeans = [mean_of(k, "slope_after") for k in passing]
    ks2 = bool(len(passing) >= 2
               and all(a > b for a, b in zip(tmeans, tmeans[1:])))
    ks3 = bool(len(passing) >= 2
               and all(a < b for a, b in zip(smeans, smeans[1:])))

    outcomes = {"KS1_existence": bool(ks1),
                "KS1_nearcritical_consistent": ks1_k09,
                "KS2_critical_slowing": ks2,
                "KS3_sharpness": ks3,
                "passing_K": passing,
                "mean_t_star": tmeans,
                "mean_post_slope": [round(s, 6) for s in smeans
                                    if s is not None]}
    report = {"status": ("KUR-SCALE breakpoint time vs distance from "
                         "criticality; registered before run; "
                         "KUR-BP-R2 detector contract"),
              "config": {"Ks": KS, "seeds": SEEDS},
              "per_K": per_k,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "kuramoto_scale.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
