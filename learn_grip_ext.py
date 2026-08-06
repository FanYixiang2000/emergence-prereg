"""LEARN-GRIP-EXT: flagship seed expansion to 10 seeds.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Trains
seeds 5-9 of the grip flagship with the identical code path, then
adjudicates the frozen LGT-B side-openness clauses over all 10 seeds
(the original five are loaded from learn_grip_transport.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ant_fine_onset import adjudicate
from learn_grip_transport import SEED, run_seed

OUTPUTS = Path(__file__).resolve().parent / "outputs"
NEW_SEED_OFFSETS = (5, 6, 7, 8, 9)


def side_openness_adj(seed_row: dict) -> dict:
    curve = np.array(seed_row["side_openness_curve"], dtype=float)
    adj = adjudicate(range(len(curve)), curve)
    plateau = 0
    for val in curve:
        if val >= 0.8:
            plateau += 1
        else:
            break
    return {
        "final_success": seed_row["final_success"],
        "final_side_mean": seed_row["final_side_mean"],
        "plateau_len": plateau,
        "side_openness_final": round(float(curve[-1]), 5),
        "adj": adj,
    }


def main() -> None:
    new_rows = {}
    for off in NEW_SEED_OFFSETS:
        row = run_seed(SEED + off * 101)
        new_rows[str(off)] = row
        h = row["episode_adj"].get("hinge", {})
        print(f"seed={off}: success={row['final_success']} "
              f"side_mean={row['final_side_mean']} "
              f"plateau={row['side_openness_plateau_len']}", flush=True)

    original = json.loads(
        (OUTPUTS / "learn_grip_transport.json").read_text())["seeds"]
    all_rows = {**original, **new_rows}
    adjudicated = {k: side_openness_adj(r) for k, r in all_rows.items()}
    for k in sorted(adjudicated, key=int):
        a = adjudicated[k]
        h = a["adj"].get("hinge", {})
        print(f"seed={k}: succ={a['final_success']} B5={a['adj']['b5_onset']} "
              f"dBIC={h.get('delta_bic')} t*={h.get('t_star')} "
              f"plateau={a['plateau_len']}", flush=True)

    learned = [a for a in adjudicated.values() if a["final_success"] >= 0.8]
    outcomes = {
        "LGTX1_learnability": bool(len(learned) >= 8),
        "LGTX2_b5_reproducibility": bool(
            sum(a["adj"]["b5_onset"] for a in learned) >= 8),
        "LGTX3_plateau_collapse_shape": bool(
            learned and all(
                a["plateau_len"] >= 5 and a["side_openness_final"] <= 0.3
                for a in learned)),
        "LGTX4_symmetry": bool(
            learned and all(
                abs(a["final_side_mean"]) <= 0.4 for a in learned)),
        "n_total": len(adjudicated),
        "n_learned": len(learned),
        "b5_count": sum(a["adj"]["b5_onset"] for a in learned),
        "t_stars": sorted(
            a["adj"].get("hinge", {}).get("t_star")
            for a in learned if a["adj"].get("hinge")),
    }
    report = {
        "status": "LEARN-GRIP-EXT 10-seed flagship adjudication; preregistered",
        "new_seeds": new_rows,
        "adjudicated_all": adjudicated,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_ext.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
