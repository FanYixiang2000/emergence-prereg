"""Delta-M mechanism-null band (NB in the V2 alignment prereg).

Pure aggregation of STORED outputs -- no new rollouts, nothing
modified. The null band for the macro-gain M is defined by mechanisms
that cannot carry an endogenous-formation claim (scripted roles,
context-marginal replay, the smoke scripted null and the untrained
initial policy). Learned systems must exceed the band's upper edge
for any M-based claim to count.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def load(name: str):
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def main() -> None:
    comparison = load("overcooked_genesis_comparison_pilot.json")
    sysrows = comparison["systems"]

    null_sources = {
        "comparison_scripted_roles": sysrows["scripted_roles"]["M_score_gain"],
        "comparison_context_marginal":
            sysrows["context_marginal"]["M_score_gain"],
    }
    smoke_scripted = load("overcooked_transition_certificate_smoke_scripted.json")
    smoke_initial = load("overcooked_transition_certificate_smoke_initial.json")
    for tag, blob in (("smoke_scripted", smoke_scripted),
                      ("smoke_initial", smoke_initial)):
        for key in ("M_score_gain", "M"):
            if key in blob:
                null_sources[tag] = blob[key]
                break
        else:
            for v in blob.values():
                if isinstance(v, dict) and "M_score_gain" in v:
                    null_sources[tag] = v["M_score_gain"]
                    break

    band = [min(null_sources.values()), max(null_sources.values())]

    learned = {
        "comparison_learned_2M": sysrows["learned"]["M_score_gain"],
        "comparison_bc_clone": sysrows["bc_clone_of_learned"]["M_score_gain"],
    }
    for seed in (93001, 93002, 93003):
        g = load(f"overcooked_genesis_curve_curve_s{seed}.json")
        learned[f"genesis_s{seed}_2M"] = g["curve"]["2000000"]["M_score_gain"]

    above = {k: v > band[1] for k, v in learned.items()}
    report = {
        "status": ("Delta-M mechanism-null band; registered in "
                   "V2_ALIGNMENT_PREREGISTRATION.md (NB); aggregation of "
                   "stored outputs only, nothing rerun or modified"),
        "null_sources_M": null_sources,
        "null_band_M": band,
        "learned_M": learned,
        "learned_above_band": above,
        "registered_outcomes": {
            "NB1_learned_M_exceeds_null_band": all(above.values()),
        },
        "reading": ("M is never reported alone: a scripted mechanism "
                    "already yields M at the band's upper edge from "
                    "desynchronization alone, so only M above the band "
                    "plus a positive G certificate carries any "
                    "endogenous-formation claim."),
    }
    out = OUTPUTS / "delta_m_null_band.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
