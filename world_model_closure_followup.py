"""Disclosed follow-up to the retained WM-3 miss (coverage-augmented rule).

The registered WM-3 decidability rule used ensemble disagreement as the
error proxy and FAILED: with very scarce data (K=200) all five world
models share the same bias -- they are wrong about the simulator in the
same way -- so their mutual disagreement is tiny while their error is
large. Ensemble spread measures variance, not shared bias.

The stored output already contains a bias diagnostic that needs no
simulator access: model COVERAGE, the fraction of probe rollouts that
hit a state-action pair never seen in the training data
(`incomplete_rollouts`). This follow-up evaluates the coverage-augmented
rule, declared here after the miss and computed read-only from the
stored output:

    decidable' = (registered margin rule) AND
                 (incomplete-rollout fraction < 0.10).

Follow-up questions (declared before computing):
    F-1  does the augmented rule catch every one of the 15 mismatches
         (no silent wrong verdict at any K)?
    F-2  does it stay permissive where models are right (no abstention
         at K = 20000)?
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_ROLLOUTS = 900  # 300 probe episodes x 3 intervention kinds
COVERAGE_CUT = 0.10


def main() -> None:
    data = json.loads((OUTPUTS / "world_model_closure.json").read_text())
    caught = 0
    mismatches = 0
    silent_wrong = []
    abstain_at_max_k = 0
    for pseed, entry in data["policy_seeds"].items():
        sim_verdict = entry["simulator"]["verdict"]
        for K, cell in entry["K"].items():
            for i, row in enumerate(cell["models"]):
                incomplete_frac = (row["metrics"]["incomplete_rollouts"]
                                   / N_ROLLOUTS)
                augmented_abstain = (row["abstain"]
                                     or incomplete_frac >= COVERAGE_CUT)
                if row["verdict"] != sim_verdict:
                    mismatches += 1
                    if augmented_abstain:
                        caught += 1
                    else:
                        silent_wrong.append((pseed, K, i, incomplete_frac))
                if K == "20000" and augmented_abstain:
                    abstain_at_max_k += 1
    report = {
        "status": ("disclosed follow-up to the retained WM-3 miss; "
                   "coverage-augmented decidability rule computed "
                   "read-only from the stored output"),
        "rule": ("registered margin rule AND incomplete-rollout fraction "
                 f"< {COVERAGE_CUT}"),
        "mismatches_total": mismatches,
        "mismatches_caught_by_augmented_rule": caught,
        "silent_wrong_verdicts_remaining": silent_wrong,
        "abstentions_at_K20000": abstain_at_max_k,
        "F1_all_mismatches_caught": caught == mismatches,
        "F2_no_abstention_at_K20000": abstain_at_max_k == 0,
        "reading": (
            "Ensemble spread detects variance but not shared bias; model "
            "coverage detects the shared-bias failure mode directly and "
            "is available without simulator access. The registered WM-3 "
            "miss is retained; the augmented rule is a disclosed "
            "follow-up, to be frozen prospectively in the next domain."
        ),
    }
    out = OUTPUTS / "world_model_closure_followup.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("reading", "status")}, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
