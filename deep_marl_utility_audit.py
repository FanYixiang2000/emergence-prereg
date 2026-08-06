"""Retrospective utility audit for the existing deep MARL probe.

Declared in V2_ALIGNMENT_PREREGISTRATION.md before running this audit,
but uses already existing MPE simple_spread data. This is exploratory
design evidence, not confirmatory validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = (11, 22, 33)


def rank_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    rows = []
    per_seed = {}
    for seed in SEEDS:
        data = json.loads((OUTPUTS / f"deep_marl_collapse_mappo_seed{seed}.json").read_text())
        eps = data["conditions"][f"trained_seed{seed}"]["episodes"]
        seed_rows = []
        for e in eps:
            row = {
                "seed": seed,
                "episode": e["episode"],
                "early_potential_bits": e["early_potential_bits"],
                "commit_collapse_bits": e["commit_collapse_bits"],
                "commit_step": e["commit_step"],
                "do_gap": e["p_win_do_commit"] - e["p_win_do_block"],
                "abs_do_gap": abs(e["p_win_do_commit"] - e["p_win_do_block"]),
                "do_assignment_js_bits": e.get("do_assignment_js_bits", 0.0),
            }
            rows.append(row)
            seed_rows.append(row)
        per_seed[str(seed)] = {
            "rank_commit_vs_assignment_js": round(rank_corr(
                [r["commit_collapse_bits"] for r in seed_rows],
                [r["do_assignment_js_bits"] for r in seed_rows]), 5),
            "rank_commit_vs_abs_gap": round(rank_corr(
                [r["commit_collapse_bits"] for r in seed_rows],
                [r["abs_do_gap"] for r in seed_rows]), 5),
            "rank_early_vs_assignment_js": round(rank_corr(
                [r["early_potential_bits"] for r in seed_rows],
                [r["do_assignment_js_bits"] for r in seed_rows]), 5),
        }

    pooled = {
        "rank_commit_vs_assignment_js": round(rank_corr(
            [r["commit_collapse_bits"] for r in rows],
            [r["do_assignment_js_bits"] for r in rows]), 5),
        "rank_commit_vs_abs_gap": round(rank_corr(
            [r["commit_collapse_bits"] for r in rows],
            [r["abs_do_gap"] for r in rows]), 5),
        "rank_early_vs_assignment_js": round(rank_corr(
            [r["early_potential_bits"] for r in rows],
            [r["do_assignment_js_bits"] for r in rows]), 5),
        "median_do_gap": round(float(np.median([r["do_gap"] for r in rows])), 5),
        "median_abs_do_gap": round(float(np.median([r["abs_do_gap"] for r in rows])), 5),
        "n": len(rows),
    }
    outcomes = {
        "DMA1_commit_predicts_assignment_js": pooled["rank_commit_vs_assignment_js"] > 0.1,
        "DMA2_commit_predicts_abs_gap": pooled["rank_commit_vs_abs_gap"] > 0.1,
        "DMA3_early_predicts_assignment_js": pooled["rank_early_vs_assignment_js"] > 0.1,
    }
    report = {
        "status": "DEEP-MARL-UTILITY-AUDIT retrospective, not confirmatory",
        "pooled": pooled,
        "per_seed": per_seed,
        "registered_audit_outcomes": outcomes,
    }
    out = OUTPUTS / "deep_marl_utility_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
