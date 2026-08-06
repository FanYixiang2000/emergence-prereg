"""Convergent validity of the continuous record: component -> matching
endpoint, on fresh seeds, frozen before running.

The registered predictive-validity battery (PV-1..3, retained misses)
asked one component to predict one global label and failed. The mature
question is convergent validity: does each early dimension predict its
OWN matching future endpoint, and does the record as a whole add
predictive content over early performance alone?

Population: four outcome-variable 2-D cells (disclosed choice, same
cells as predictive_validity.py), five FRESH seeds each (base 9800;
the 9600-series seeds of the earlier battery are excluded). Snapshot
at 25% of training; final endpoints measured with the frozen 2-D
protocol.

Early predictors (25% snapshot): M_early (do-law JS), U_early
(usefulness gap), S_early (trigger separation), perf_early (natural
return).

Matching endpoints (final policy): M_final (do-law JS), U_final
(usefulness gap), S_final (trigger separation).

Registered predictions (frozen before running; Spearman over the 20
seeds, matched vs cross):

    CV-1 (matched beats crossed, M) rho(M_early, M_final) exceeds
         rho(perf_early, M_final).
    CV-2 (matched beats crossed, U) rho(U_early, U_final) exceeds
         rho(perf_early, U_final).
    CV-3 (incremental content) OLS of U_final on perf_early alone vs
         on perf_early + (M_early, S_early, U_early): the record adds
         incremental predictive content, leave-one-out R^2 increases.
         No refitting per cell; one pooled model.

Misses are retained.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from predictive_validity import (
    CELLS,
    train_with_snapshot,
    early_predictors,
    final_outcome,
    spearman,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEED_BASE = 9800
N_SEEDS = 5


def loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        Xi = np.column_stack([np.ones(mask.sum()), X[mask]])
        beta, *_ = np.linalg.lstsq(Xi, y[mask], rcond=None)
        preds[i] = np.concatenate([[1.0], X[i]]) @ beta
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    rows = []
    for g, w in CELLS:
        for k in range(N_SEEDS):
            seed = SEED_BASE + int(g) * 100 + int(w * 100) + k
            print(f"convergent validity: cell ({g},{w}) seed {seed}",
                  flush=True)
            snapshot, final_table = train_with_snapshot(g, w, seed)
            early = early_predictors(snapshot, g, seed + 7)
            fin = final_outcome(final_table, g, w, seed + 13)
            rows.append({
                "cell": f"{g}|{w}", "seed": seed,
                "M_early": early["M_early"],
                "U_early": early["U_early"],
                "perf_early": early["perf_early"],
                "M_final": fin["specificity_js"],
                "U_final": fin["usefulness_gap"],
                "S_final": fin["selectivity_tension"],
                "accepted": fin["accepted"],
            })
            print(f"  M {early['M_early']:.3f}->{fin['specificity_js']:.3f}"
                  f"  U {early['U_early']:+.2f}->"
                  f"{fin['usefulness_gap']:+.2f}", flush=True)

    def col(name):
        return [r[name] for r in rows]

    rho = {
        "M_early->M_final": spearman(col("M_early"), col("M_final")),
        "perf_early->M_final": spearman(col("perf_early"), col("M_final")),
        "U_early->U_final": spearman(col("U_early"), col("U_final")),
        "perf_early->U_final": spearman(col("perf_early"), col("U_final")),
        "M_early->U_final": spearman(col("M_early"), col("U_final")),
    }

    y = np.array(col("U_final"))
    r2_perf = loo_r2(np.array(col("perf_early")).reshape(-1, 1), y)
    r2_full = loo_r2(np.column_stack(
        [col("perf_early"), col("M_early"), col("U_early")]), y)

    report = {
        "status": ("convergent validity, component -> matching "
                   "endpoint; CV-1..CV-3 frozen in the docstring; "
                   "fresh 9800-series seeds"),
        "rows": rows,
        "n": len(rows),
        "spearman": rho,
        "loo_r2_U_final": {"perf_only": r2_perf,
                           "perf_plus_record": r2_full},
        "registered_outcomes": {
            "CV1_M_matched_beats_crossed": bool(
                rho["M_early->M_final"] > rho["perf_early->M_final"]),
            "CV2_U_matched_beats_crossed": bool(
                rho["U_early->U_final"] > rho["perf_early->U_final"]),
            "CV3_record_adds_loo_r2": bool(r2_full > r2_perf),
        },
    }
    out = OUTPUTS / "convergent_validity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(rho, indent=1))
    print(json.dumps(report["registered_outcomes"], indent=1))
    print(f"LOO R2: perf {r2_perf:.3f} vs perf+record {r2_full:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
