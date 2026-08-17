"""REACH-VALID-2: corrected monotonicity clause on fresh seeds.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Same recipe, grid, m and label
definition as reach_valid_signalling.py; fresh seeds at seed_index
3..5; VS3 replaced by VS3' (non-increase within a 0.15 one-label
quantum up to the first zero, plus irrevocability after two
consecutive zeros). Original VS3 Spearman reported verbatim, no bar.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import reach_valid_signalling as base

base.SEEDS = (None, None, None, 717_304, 717_405, 717_506)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SNAPS = base.SNAPS
M = base.M
TOL = 0.15


def run_idx(i):
    import reach_valid_signalling as b

    b.SEEDS = (None, None, None, 717_304, 717_405, 717_506)
    return b.run_seed(i)


def vs3_prime(curve):
    first_zero = next((i for i, v in enumerate(curve) if v == 0),
                      len(curve) - 1)
    for i in range(first_zero):
        if curve[i + 1] - curve[i] > TOL:
            return False
    zero_run = 0
    for v in curve:
        if zero_run >= 2 and v > 0:
            return False
        zero_run = zero_run + 1 if v == 0 else 0
    return True


def main() -> None:
    from scipy.stats import spearmanr

    with ProcessPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(run_idx, (3, 4, 5)))

    vs1 = vs2 = vs3p = 0
    for row in rows:
        labels0 = row["labels"]["0"]
        n_codes0 = len({x for x in labels0 if x != "unconverged"})
        if row["reach_curve"][0] >= 0.75 and n_codes0 >= 4:
            vs1 += 1
        first_after = next((i for i, u in enumerate(SNAPS)
                            if row["s090"] is not None and u >= row["s090"]),
                           None)
        if first_after is not None:
            lab = row["labels"][str(SNAPS[first_after])]
            if all(x == row["final_code"] for x in lab):
                vs2 += 1
        if vs3_prime(row["reach_curve"]):
            vs3p += 1
        row["spearman_verbatim"] = round(float(
            spearmanr(row["reach_curve"], SNAPS)[0]), 4)
        row["reach_closure_update"] = next(
            (u for u, v in zip(SNAPS, row["reach_curve"]) if v == 0), None)

    outcomes = {
        "VS1_open_at_zero": f"{vs1}/3", "VS1_pass": bool(vs1 == 3),
        "VS2_closed_after_capability": f"{vs2}/3",
        "VS2_pass": bool(vs2 == 3),
        "VS3prime_monotone_irrevocable": f"{vs3p}/3",
        "VS3prime_pass": bool(vs3p == 3),
        "original_VS3_spearman_verbatim": [r["spearman_verbatim"]
                                           for r in rows],
        "closure_vs_s090": [
            {"seed": r["seed"], "reach_closure": r["reach_closure_update"],
             "s090": r["s090"]} for r in rows],
        "launch_full_reach": bool(vs1 == 3 and vs2 == 3 and vs3p == 3),
    }
    (OUTPUTS / "reach_valid_signalling2.json").write_text(json.dumps({
        "status": ("REACH-VALID-2 fresh-seed re-validation with the "
                   "corrected VS3' clause; original VS3 reported "
                   "verbatim with no bar"),
        "config": {"seeds": base.SEEDS[3:], "snaps": SNAPS, "m": M,
                   "tol": TOL},
        "seeds": rows,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print("Wrote reach_valid_signalling2.json")


if __name__ == "__main__":
    main()
