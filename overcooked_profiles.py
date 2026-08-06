"""Continuous emergence profiles for the Overcooked-AI round-1 systems.

Read-only assembly (same discipline as profile_existing_systems.py):
recompute the declared continuous record from the stored round-1
confirmation JSONs, no re-evaluation. Declared before computing:
sigma_V = 100 sparse-reward units (natural team scores in the pilots
ranged 130-440 per episode pair); basins |B| = 6.

Declared expectations (read-only, descriptive):
    OP-1 every accepted learned seed has E_adapt > 0;
    OP-2 every control has E_adapt = 0 (twin/untrained via Q = 0 or
         V <= 0; scripted/clone via Q = 0);
    OP-3 the four selectivity-rejected learned seeds have E_struct
         below every accepted learned seed (the continuous record
         ranks the rejected seeds below the accepted ones without
         seeing the verdict).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import emergence_profile as ep

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = list(range(77001, 77013))
SIGMA_V = 100.0
N_BASINS = 6
SYSTEMS = ("learned", "initial_twin", "scripted_roles", "bc_clone",
           "untrained_other")


def profile_for(m: dict, acquisition: float, twin_m: dict) -> dict:
    p = ep.potential_norm(m["potential_bits"], N_BASINS)
    s = min(1.0, m["conditional_selectivity"])
    mm = ep.magnitude_norm(m["specificity_js_bits"])
    v = ep.value_signed(m["usefulness_gap"], SIGMA_V)
    q = ep.acquisition_norm(mm, ep.magnitude_norm(
        twin_m["specificity_js_bits"]), s,
        min(1.0, twin_m["conditional_selectivity"])) \
        if acquisition > 0 else 0.0
    es = ep.e_struct(p, s, mm)
    return {"P": p, "S": s, "M": mm, "V": v, "Q": q,
            "E_struct": es, "E_adapt": ep.e_adapt(es, q, v)}


def main() -> None:
    rows = {}
    for seed in SEEDS:
        data = json.loads(
            (OUTPUTS / f"overcooked_confirm_s{seed}.json").read_text())
        entry = data["seeds"][str(seed)]
        twin_m = entry["initial_twin"]["metrics"]
        rows[seed] = {}
        for name in SYSTEMS:
            sys = entry[name]
            rows[seed][name] = profile_for(
                sys["metrics"], sys.get("acquisition", 0.0), twin_m)
            rows[seed][name]["verdict"] = sys["verdict"]["emergent"]

    accepted = [s for s in SEEDS if rows[s]["learned"]["verdict"]]
    rejected = [s for s in SEEDS if not rows[s]["learned"]["verdict"]]
    op1 = all(rows[s]["learned"]["E_adapt"] > 0 for s in accepted)
    op2 = all(rows[s][c]["E_adapt"] == 0.0
              for s in SEEDS for c in SYSTEMS[1:])
    min_acc = min(rows[s]["learned"]["E_struct"] for s in accepted)
    max_rej = max(rows[s]["learned"]["E_struct"] for s in rejected)
    op3 = max_rej < min_acc

    report = {
        "status": ("read-only continuous profiles for Overcooked "
                   "round-1; sigma_V and expectations declared in "
                   "docstring"),
        "sigma_V": SIGMA_V,
        "profiles": {str(s): rows[s] for s in SEEDS},
        "declared_outcomes": {
            "OP1_accepted_learned_E_adapt_positive":
                f"{sum(rows[s]['learned']['E_adapt'] > 0 for s in accepted)}"
                f"/{len(accepted)} -> {op1}",
            "OP2_all_controls_E_adapt_zero": op2,
            "OP3_rejected_below_accepted_E_struct":
                f"{op3} (max rejected {max_rej:.3f} vs min accepted "
                f"{min_acc:.3f})",
        },
    }
    out = OUTPUTS / "overcooked_profiles.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["declared_outcomes"], indent=1))
    for s in SEEDS:
        r = rows[s]["learned"]
        print(f"{s} learned E_struct {r['E_struct']:.3f} "
              f"E_adapt {r['E_adapt']:+.3f} verdict {r['verdict']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
