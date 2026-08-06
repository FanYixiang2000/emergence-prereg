"""E5 baseline race on the analytic matched confound.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (wave 3). Imports the
frozen battery's mechanism definitions and computes purely
observational baselines; shows they CANNOT separate mechanisms that
share one joint distribution by construction, while the stored
cut-based certificate does. No stored file is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

from collective_constraint import H, mechanism, p_Z

OUTPUTS = Path(__file__).resolve().parent / "outputs"
MATCHED = ("central_script", "common_cause", "local_feedback")


def observational_profile(p) -> dict:
    h = {i: H(p, idx=(i,)) for i in range(3)}
    h_pair = {(i, j): H(p, idx=(i, j))
              for i, j in ((0, 1), (0, 2), (1, 2))}
    h_joint = H(p)
    tc = sum(h.values()) - h_joint
    mi = {f"MI_{i}{j}": h[i] + h[j] - h_pair[(i, j)]
          for i, j in h_pair}
    # O-information for n=3: H(X) + sum_i [H(X_i) - H(X_{-i})]
    others = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    o_info = h_joint + sum(h[i] - h_pair[others[i]] for i in range(3))
    return {
        "H_joint": round(h_joint, 9),
        "marginal_entropies": [round(h[i], 9) for i in range(3)],
        "total_correlation": round(tc, 9),
        **{k: round(v, 9) for k, v in mi.items()},
        "O_information": round(o_info, 9),
        "P_macro_success": round(p_Z(p), 9),
    }


def main() -> None:
    rows = {}
    for kind in MATCHED:
        real, _broken, _je = mechanism(kind)
        rows[kind] = observational_profile(real)

    keys = list(rows[MATCHED[0]].keys())
    identical = {}
    for k in keys:
        vals = []
        for kind in MATCHED:
            v = rows[kind][k]
            vals.append(tuple(v) if isinstance(v, list) else v)
        identical[k] = all(
            (abs(a - b) < 1e-9 if not isinstance(a, tuple)
             else all(abs(x - y) < 1e-9 for x, y in zip(a, b)))
            for a, b in zip(vals, vals[1:]))

    stored = json.loads((OUTPUTS / "collective_constraint.json")
                        .read_text(encoding="utf-8"))
    g_cert = {kind: stored["mechanisms"][kind]["G_reorganization"]
              for kind in MATCHED}
    outcomes = {
        "E5_1_all_observational_baselines_identical":
            all(identical.values()),
        "E5_2_cut_certificate_separates":
            (g_cert["central_script"] == 0.0
             and g_cert["common_cause"] == 0.0
             and g_cert["local_feedback"] > 0.4),
    }
    report = {
        "status": ("baseline race on the analytic matched confound; "
                   "registered in V2_ALIGNMENT_PREREGISTRATION.md "
                   "wave 3 (E5); mechanism definitions imported from "
                   "the frozen battery, stored outputs untouched"),
        "observational_profiles": rows,
        "identical_across_mechanisms": identical,
        "stored_certificate_G": g_cert,
        "registered_outcomes": outcomes,
        "reading": ("Reward, marginal entropies, MI, total "
                    "correlation and O-information are constant "
                    "across a prewired script, a common environment "
                    "driver and genuine local feedback, because the "
                    "three share one joint distribution by "
                    "construction. Only the mechanism-matched cut "
                    "separates them. This is the R7 answer: the "
                    "certificate knows something no observational "
                    "baseline can know."),
    }
    out = OUTPUTS / "baseline_race_matched_confound.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**{k: rows[k]["total_correlation"] for k in rows},
                      **outcomes}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
