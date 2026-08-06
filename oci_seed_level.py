"""OC-RING-INT seed-level reanalysis (post-hoc, labeled as such).

Reanalyzes the recorded oc_ring_intervention.json at seed level, the
independent experimental unit, with the frozen endpoint definitions.
No new runs. See the post-hoc addendum in V2_ALIGNMENT_PREREGISTRATION.md.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from math import comb
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def paired_sign_flip_p(a: list, b: list) -> float:
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    obs = sum(diffs)
    count = total = 0
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        total += 1
        if sum(s * d for s, d in zip(signs, diffs)) >= obs:
            count += 1
    return count / total


def mcnemar_exact_p(a: list, b: list) -> tuple[int, int, float]:
    n_ab = sum(1 for x, y in zip(a, b) if x and not y)
    n_ba = sum(1 for x, y in zip(a, b) if y and not x)
    nd = n_ab + n_ba
    p = (sum(comb(nd, k) for k in range(n_ab, nd + 1)) / 2 ** nd
         if nd else 1.0)
    return n_ab, n_ba, p


def main() -> None:
    d = json.loads((OUTPUTS / "oc_ring_intervention.json").read_text())
    by = defaultdict(lambda: defaultdict(list))
    for r in d["runs"]:
        by[r["seed"]][r["time"]].append(r)
    seeds = sorted(by.keys())

    def seed_any(t, pred):
        return [any(pred(r) for r in by[s].get(t, [])) for s in seeds]

    moved = lambda r: r["outcome"] != "held"
    flip = lambda r: r["outcome"] == "flip"

    open_m, late_m = seed_any("open", moved), seed_any("late", moved)
    open_f, late_f = seed_any("open", flip), seed_any("late", flip)
    early_m = seed_any("early", moved)

    b_m, c_m, mc_m = mcnemar_exact_p(open_m, late_m)
    b_f, c_f, mc_f = mcnemar_exact_p(open_f, late_f)

    report = {
        "status": ("OC-RING-INT seed-level reanalysis; POST-HOC (recorded "
                   "after the preregistered run-level outcome); frozen "
                   "endpoint definitions; no new runs"),
        "n_seeds": len(seeds),
        "moved": {
            "open_seeds": sum(open_m), "late_seeds": sum(late_m),
            "early_seeds": sum(early_m),
            "sign_flip_p": paired_sign_flip_p(open_m, late_m),
            "mcnemar_discordant": [b_m, c_m], "mcnemar_p": mc_m,
        },
        "strict_flip": {
            "open_seeds": sum(open_f), "late_seeds": sum(late_f),
            "sign_flip_p": paired_sign_flip_p(open_f, late_f),
            "mcnemar_discordant": [b_f, c_f], "mcnemar_p": mc_f,
            "note": "not significant at seed level; descriptive only",
        },
        "run_level_counts": {
            t: {o: sum(1 for r in d["runs"]
                       if r["time"] == t and r["outcome"] == o)
                for o in ("held", "flip", "uncommitted")}
            for t in ("early", "open", "late")
        },
    }
    out = OUTPUTS / "oci_seed_level.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
