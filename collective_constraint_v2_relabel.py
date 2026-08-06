"""v2 source-typed relabel of the four-mechanism battery (RL).

Registered in V2_ALIGNMENT_PREREGISTRATION.md. Reads the STORED
collective_constraint.json (untouched) and emits a NEW file that
reinterprets the same numbers under definition v2.0: the interaction
cut is a source decomposer, not an accept/reject gate. RL-1 verifies
the copied numeric fields are preserved bit-for-bit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"

V2_TYPES = {
    "central_script": {
        "v2_type": "externally-specified organization",
        "emergence_channel": None,
        "boundary_violated": "B3 (macro-regime hard-coded by an "
                             "external controller, by construction)",
        "v2_verdict": "not emergence (same verdict as v1, new reason: "
                      "boundary B3, not the absence of G)",
    },
    "common_cause": {
        "v2_type": "environment-mediated coordination",
        "emergence_channel": "C_env",
        "boundary_violated": "none intrinsic; in THIS authored battery "
                             "the E-following rule is itself hard-coded "
                             "(B3), so the row validates the instrument "
                             "rather than certifying a wild emergence "
                             "claim",
        "v2_verdict": "emergence TYPE label: environment-mediated; "
                      "v1's blanket 'reject' is retired",
    },
    "independent_coincidence": {
        "v2_type": "transient parallel alignment",
        "emergence_channel": "C_individual (transient)",
        "boundary_violated": "B4 (persistence fails: R = 0.30)",
        "v2_verdict": "not emergence (fails persistence under both "
                      "versions)",
    },
    "local_feedback": {
        "v2_type": "interaction-generated collective constraint",
        "emergence_channel": "C_pair / C_high (relational)",
        "boundary_violated": "none",
        "v2_verdict": "emergence TYPE label: relational/higher-order "
                      "(v1 accept preserved, now as the strongest "
                      "channel rather than the only admissible one)",
    },
}

COPIED_FIELDS = ("H_real", "H_broken", "C_constraint", "G_reorganization",
                 "M_endogenous_macro_gain", "N_irreducibility_given_env",
                 "R_persistence", "P_Z_real", "P_Z_broken",
                 "micro_down_macro_up")


def field_hash(mechanisms: dict) -> str:
    payload = json.dumps(
        {m: {k: mechanisms[m][k] for k in COPIED_FIELDS}
         for m in sorted(mechanisms)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    src = json.loads((OUTPUTS / "collective_constraint.json")
                     .read_text(encoding="utf-8"))
    mech = src["mechanisms"]
    rows = {}
    for name, row in mech.items():
        rows[name] = {k: row[k] for k in COPIED_FIELDS}
        rows[name]["v1_accept"] = row["accept"]
        rows[name].update(V2_TYPES[name])

    h = field_hash(mech)
    h_new = field_hash(rows)
    report = {
        "status": ("v2 source-typed relabel of the frozen four-mechanism "
                   "battery; registered in V2_ALIGNMENT_PREREGISTRATION.md "
                   "(RL); pure reinterpretation -- the v1 file is "
                   "unmodified and remains the historical record"),
        "v1_source_file": "collective_constraint.json (untouched)",
        "definition": "EMERGENCE_DEFINITION_V2.md",
        "mechanisms": rows,
        "copied_fields_sha256_v1": h,
        "copied_fields_sha256_v2": h_new,
        "registered_outcomes": {
            "RL1_numeric_fields_preserved_bit_for_bit": h == h_new,
        },
        "reading": ("v1 asked 'is it emergence? (only local feedback)'; "
                    "v2 asks 'which channel carries the collapse, and do "
                    "the boundary conditions B1-B4 hold?'. The same "
                    "numbers answer both questions; only the ontology "
                    "changed, which is why no rerun is needed or "
                    "permitted."),
    }
    out = OUTPUTS / "collective_constraint_v2_typology.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
