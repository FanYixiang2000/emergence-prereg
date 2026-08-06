"""LEARN-STANCE-CONTROL: parameter-matched separability control.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Reuses
the LEARN-STANCE-STICKY module wholesale (same horizon, taus, flip
count, hyperparameters, seeds) with a single change: STICK_P = 1.0, so
lean actions take effect immediately and there is no inertial
consolidation phase. Tests whether the separability advantage of
openness disappears under otherwise identical parameters.
"""

from __future__ import annotations

import json

import numpy as np

import learn_stance_sticky as base
from learn_transport_eq_utility import auc, rank_corr

base.STICK_P = 1.0  # the single manipulated parameter

OUT = base.OUTPUTS / "learn_stance_control.json"


def main() -> None:
    rows = {}
    pool = {k: [] for k in ("switch", "open", "absx", "absv", "tau")}
    for i in range(base.N_SEEDS):
        policy = base.run_seed(base.SEED + i * 101)
        ev = base.eval_policy(policy)
        per_tau = {}
        for tau in base.TAUS:
            row = base.intervention_eval(policy, tau,
                                         seed=base.SEED + i * 101 + tau)
            for k in ("switch", "open", "absx", "absv"):
                pool[k].extend(row[k].tolist())
            pool["tau"].extend(row["tau_arr"].tolist())
            per_tau[str(tau)] = {"switch_rate": round(row["switch_rate"], 5),
                                 "success": round(row["success"], 5)}
        rows[str(i)] = {
            "final_success": round(ev["success"], 5),
            "frac_right": ev["frac_right"],
            "episode_adj": ev["episode_adj"],
            "stance_entropy_curve": ev["stance_entropy_curve"],
            "ladder": ev["ladder"],
            "per_tau": per_tau,
        }
        lad_last = ev["ladder"][str(base.LADDER_TIMES[-1])]
        print(f"seed={i}: succ={rows[str(i)]['final_success']} "
              f"fracR={ev['frac_right']} H1_end={lad_last['H1']} "
              f"TC_end={lad_last['TC']} "
              f"switch@2={per_tau['2']['switch_rate']} "
              f"switch@30={per_tau['30']['switch_rate']}", flush=True)

    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    switch = np.array(pool["switch"])
    race = {}
    for name, sign in (("open", 1.0), ("absx", -1.0), ("absv", -1.0),
                       ("tau", -1.0)):
        vals = sign * np.array(pool[name])
        race[name] = {"rank_corr": round(rank_corr(vals, switch), 5),
                      "auc": round(auc(vals, switch), 5)}
    lad_ok = [
        r for r in learned
        if (r["ladder"][str(base.LADDER_TIMES[-1])]["H1"] >= 0.7
            and r["ladder"][str(base.LADDER_TIMES[-1])]["TC"] >= 3.0)
    ]
    outcomes = {
        "LSC1_learnability": bool(len(learned) >= 4),
        "LSC2_separability_reversal": bool(
            race["open"]["auc"] < race["absx"]["auc"]),
        "LSC3_relational_collapse": bool(
            learned and len(lad_ok) >= max(3, len(learned) - 1)),
        "n_learned": len(learned),
        "baseline_race": race,
    }
    report = {
        "status": "LEARN-STANCE-CONTROL parameter-matched control (STICK_P=1.0); preregistered",
        "config": {"N_agents": base.N_AGENTS, "force_min": base.FORCE_MIN,
                   "goal": base.GOAL, "max_steps": base.MAX_STEPS,
                   "seeds": base.N_SEEDS, "updates": base.UPDATES,
                   "batch": base.BATCH, "stick_p": base.STICK_P,
                   "taus": base.TAUS, "flip_count": base.FLIP_COUNT,
                   "ladder_times": base.LADDER_TIMES},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
