"""STANCE-STAT-UNIT: seed-level statistics for the stance races.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Both arms are deterministic reruns of
the LEARN-STANCE-STICKY module (control arm sets STICK_P = 1.0), same
seeds and intervention grid, so the episode streams are identical to
the published races; the only new content is seed-level aggregation.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
NAMES = ("open", "absx", "absv", "tau")
SIGNS = {"open": 1.0, "absx": -1.0, "absv": -1.0, "tau": -1.0}
N_BOOT = 10_000
BOOT_SEED = 0


def run_seed_arm(args):
    arm, i = args
    import torch
    torch.set_num_threads(4)
    import learn_stance_sticky as base
    from learn_transport_eq_utility import auc

    if arm == "control":
        base.STICK_P = 1.0
    seed = base.SEED + i * 101
    policy = base.run_seed(seed)
    data = {n: [] for n in NAMES + ("switch",)}
    for tau in base.TAUS:
        row = base.intervention_eval(policy, tau, seed=seed + tau)
        for n in ("switch", "open", "absx", "absv"):
            data[n].extend(row[n].tolist())
        data["tau"].extend(row["tau_arr"].tolist())
    sw = np.array(data["switch"])
    aucs = {n: round(auc(SIGNS[n] * np.array(data[n]), sw), 5)
            for n in NAMES}
    print(f"{arm} seed {i}: n={len(sw)} " +
          " ".join(f"{n}={aucs[n]}" for n in NAMES), flush=True)
    return arm, i, {n: np.array(data[n]) for n in NAMES + ("switch",)}, aucs


def main() -> None:
    from learn_transport_eq_utility import auc
    import learn_stance_sticky as base

    jobs = [(arm, i) for arm in ("sticky", "control")
            for i in range(base.N_SEEDS)]
    arrays = {"sticky": [None] * base.N_SEEDS,
              "control": [None] * base.N_SEEDS}
    per_seed = {"sticky": [None] * base.N_SEEDS,
                "control": [None] * base.N_SEEDS}
    with ProcessPoolExecutor(max_workers=10) as ex:
        for arm, i, arrs, aucs in ex.map(run_seed_arm, jobs):
            arrays[arm][i] = arrs
            per_seed[arm][i] = aucs

    rng = np.random.default_rng(BOOT_SEED)
    boot_ci = {}
    loo = {}
    n = base.N_SEEDS
    for arm in ("sticky", "control"):
        boot = {k: [] for k in ("open", "absx", "absv")}
        for _ in range(N_BOOT):
            pick = rng.integers(0, n, n)
            sw = np.concatenate([arrays[arm][j]["switch"] for j in pick])
            for k in boot:
                vals = np.concatenate([arrays[arm][j][k] for j in pick])
                boot[k].append(auc(SIGNS[k] * vals, sw))
        boot_ci[arm] = {k: [round(float(np.percentile(v, 2.5)), 5),
                            round(float(np.percentile(v, 97.5)), 5)]
                        for k, v in boot.items()}
        vals = []
        for drop in range(n):
            keep = [j for j in range(n) if j != drop]
            sw = np.concatenate([arrays[arm][j]["switch"] for j in keep])
            v = np.concatenate([arrays[arm][j]["open"] for j in keep])
            vals.append(round(auc(v, sw), 5))
        loo[arm] = vals

    ssu1 = sum(a["open"] > a["absx"] for a in per_seed["sticky"])
    ssu2 = sum(a["absv"] > a["open"] for a in per_seed["control"])
    outcomes = {
        "per_seed_auc": per_seed,
        "boot_ci_95_seed_cluster": boot_ci,
        "leave_one_seed_out_open": loo,
        "SSU1_sticky_open_beats_absx_seeds": f"{ssu1}/{n}",
        "SSU1_pass": bool(ssu1 >= 4),
        "SSU2_control_absv_beats_open_seeds": f"{ssu2}/{n}",
        "SSU2_pass": bool(ssu2 >= 4),
    }
    out = OUTPUTS / "learn_stance_stat_unit.json"
    out.write_text(json.dumps({
        "status": ("STANCE-STAT-UNIT seed-level statistics; episode "
                   "streams identical to LEARN-STANCE-STICKY/CONTROL; "
                   "analysis addendum frozen before run"),
        "config": {"n_seeds": n, "n_boot": N_BOOT, "boot_seed": BOOT_SEED},
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
