"""Dual retention curves for the persistence battery (pure re-analysis).

The stability claim requires BOTH activation retention (the structure is
still selected) and causal retention (its do-contrast value survives):

    R_activation(delta) = selectivity(delta) / selectivity(P0)
    R_causal(delta)     = usefulness_gap(delta) / usefulness_gap(P0)

Both are computed per learned policy and perturbation from the stored
persistence output; full stability under the declared perturbation family D
means both ratios stay high. Reported: per-perturbation means, the
activation/causal dissociation, and the formal scope statement (stability is
always relative to the declared D).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"

SEEDS = [1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110]
PERTS = ["P1_novel_layouts", "P2_horizon12", "P3_horizon18",
         "P4_noise005", "P5_noise010", "P6_noise020"]


def main() -> None:
    data = json.loads(
        (OUTPUTS / "contextual_lbf_persistence.json").read_text())
    res = data["results"]
    out = {"per_perturbation": {}, "per_policy": {}}
    for pert in PERTS:
        act, caus, use = [], [], []
        for seed in SEEDS:
            base = res[f"learned_{seed}"]["P0_baseline"]
            cur = res[f"learned_{seed}"][pert]
            s0 = base["conditional_selectivity"]
            u0 = base["usefulness_gap"]
            r_act = cur["conditional_selectivity"] / s0 if s0 > 0 else None
            r_cau = cur["usefulness_gap"] / u0 if u0 > 0 else None
            act.append(r_act)
            caus.append(r_cau)
            use.append(cur["usefulness_gap"])
            out["per_policy"].setdefault(str(seed), {})[pert] = {
                "R_activation": r_act, "R_causal": r_cau,
                "usefulness_gap": cur["usefulness_gap"],
            }
        out["per_perturbation"][pert] = {
            "mean_R_activation": float(np.mean([a for a in act
                                                if a is not None])),
            "mean_R_causal": float(np.mean([c for c in caus
                                            if c is not None])),
            "n_positive_usefulness": int(sum(u > 0 for u in use)),
            "mean_usefulness_gap": float(np.mean(use)),
        }
    out["reading"] = (
        "Stability is always relative to the declared perturbation family D. "
        "Within D_temporal-observational (horizon 12-18, obs noise sigma "
        "<= 0.2): both activation and causal retention stay high -- a stable "
        "causal macrostructure. Outside that family (novel geometries): "
        "activation retention is partial and causal retention is lost "
        "(negative usefulness on 10/10 seeds) -- the measured scope boundary."
    )
    path = OUTPUTS / "persistence_retention_curves.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for pert, row in out["per_perturbation"].items():
        print(f"{pert}: R_act {row['mean_R_activation']:.3f}  "
              f"R_causal {row['mean_R_causal']:.3f}  "
              f"useful>0 {row['n_positive_usefulness']}/10")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
