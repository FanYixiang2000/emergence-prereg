"""E3C confirmatory analysis (registered plan, V2 alignment prereg).

Loads the 25 stored E3C run outputs and computes, per condition,
seed means of M, C_relational, G and score, plus exact permutation
tests for the three registered one-sided comparisons. Aggregation
only; nothing rerun, nothing modified.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
CONDITIONS = ("none", "early", "commit", "late", "random")
SEEDS = (93201, 93202, 93203, 93204, 93205)


def exact_perm_p(a, b) -> float:
    """One-sided exact permutation p for mean(a) < mean(b)."""
    pool = list(a) + list(b)
    observed = np.mean(a) - np.mean(b)
    n = len(a)
    count = 0
    total = 0
    for idx in combinations(range(len(pool)), n):
        sel = [pool[i] for i in idx]
        rest = [pool[i] for i in range(len(pool)) if i not in idx]
        if np.mean(sel) - np.mean(rest) <= observed + 1e-12:
            count += 1
        total += 1
    return count / total


def main() -> None:
    rows = {}
    missing = []
    for c in CONDITIONS:
        rows[c] = []
        for s in SEEDS:
            p = OUTPUTS / f"overcooked_e3c_{c}_s{s}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            rows[c].append({
                "seed": s,
                "window": r["window"],
                "M": r["certificate_2M"]["M_score_gain"],
                "G": r["certificate_2M"]["G_js_bits"],
                "score": r["certificate_2M"]["real_score"],
                "C_rel": r["joint_ladder_2M"]["C_relational"],
            })
    if missing:
        print(f"MISSING {len(missing)} runs: {missing}")
        return

    means = {c: {k: float(np.mean([x[k] for x in rows[c]]))
                 for k in ("M", "G", "score", "C_rel")}
             for c in CONDITIONS}
    for c in CONDITIONS:
        print(f"{c:8s} M={means[c]['M']:+7.2f} "
              f"C_rel={means[c]['C_rel']:.4f} "
              f"G={means[c]['G']:.4f} score={means[c]['score']:6.1f}")

    def vals(c, k):
        return [x[k] for x in rows[c]]

    tests = {
        "p_M_commit_lt_random": exact_perm_p(vals("commit", "M"),
                                             vals("random", "M")),
        "p_M_commit_lt_none": exact_perm_p(vals("commit", "M"),
                                           vals("none", "M")),
        "p_Crel_commit_lt_random": exact_perm_p(vals("commit", "C_rel"),
                                                vals("random", "C_rel")),
    }
    outcomes = {
        "E3C1_commit_lowest_mean_M":
            means["commit"]["M"] == min(m["M"] for m in means.values()),
        "E3C2_perm_p_M_commit_lt_random_sig":
            tests["p_M_commit_lt_random"] < 0.05,
        "E3C3_commit_lowest_mean_Crel":
            means["commit"]["C_rel"] == min(m["C_rel"]
                                            for m in means.values()),
        "E3C4_early_score_above_none_MAY_MISS":
            means["early"]["score"] > means["none"]["score"],
    }
    report = {
        "status": ("E3C confirmatory analysis; registered plan in "
                   "V2_ALIGNMENT_PREREGISTRATION.md; aggregation of the "
                   "25 stored run outputs only"),
        "per_run": rows,
        "seed_means": means,
        "exact_permutation_tests_one_sided": tests,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "overcooked_e3c_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**tests, **outcomes}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
