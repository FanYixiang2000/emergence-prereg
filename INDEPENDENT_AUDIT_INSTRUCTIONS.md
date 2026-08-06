# Independent audit instructions (for a non-author auditor)

Purpose: internal hashes, the 85-check consistency audit, the manifest
and the ledger prove that the project's files agree with each other.
They cannot prove the implementation is free of a shared bug. This
document makes a fully independent audit turnkey for someone who did
not write the formulas or the code. Estimated effort: 1-2 days.

## Who qualifies

Anyone who has not edited this repository's analysis code: a lab
colleague, a collaborator from another group, or a hired research
engineer. The auditor should NOT discuss implementation details with
the authors before finishing step 5.

## Step 1: clean-environment reproduction (half a day, mostly waiting)

1. Fresh machine or container (Python 3.10+). Clone the repository.
2. `pip install -r requirements.txt` (versions pinned).
3. `make audit` -- must print `85/85 checks passed` and regenerate
   `manifest.json` with hashes identical to `FINAL_FREEZE.md`.
4. `make figures && make paper` -- every figure and the PDF must
   regenerate from stored outputs alone (no training, no downloads).

## Step 2: random spot-check of manuscript numbers (2-3 hours)

1. Pick 20 numeric claims from `paper/main.tex` at random (a claim =
   any number with a stated source experiment).
2. For each, trace it to its JSON in `outputs/` via
   `CLAIM_EVIDENCE_MAP.md` and confirm the value independently (read
   the JSON yourself; do not use `verify_manuscript_numbers.py`).
3. Record any mismatch verbatim.

## Step 3: independent reimplementation of the core quantity (2-4 hours)

Without reading `within_episode_collapse_probe.py`, implement from the
Methods text alone:

1. the do-law JS divergence between do-commit and do-block future-basin
   distributions (Methods, "specificity");
2. the interventional identity I(A;B) of Proposition B.

Run both on `outputs/bridge_identity_verification.json`'s stored
random-system parameters (or regenerate 1,000 random systems from the
stated recipe) and confirm: the identity gap is < 1e-12 and your JS
values match the reference evaluator on ten stored CLBF episodes
(inputs in `outputs/contextual_lbf_confirmation.json`, field
`systems.*.metrics`).

## Step 4: manual trajectory inspection (1-2 hours)

1. Re-run `python overcooked_criterion.py --demo-episode` (renders 3
   episodes per condition as text).
2. Verify by eye: do-block actually prevents the trigger; do-commit
   actually forces it; the basin label matches what happened in the
   episode; the team score matches the sparse reward printed by the
   unmodified benchmark.

## Step 5: audit statement

Write one page: environment, steps completed, mismatches found (or
"none"), and whether the auditor endorses the statement "the reported
numbers regenerate from the stored outputs and the core quantities are
independently reproducible." The statement, with the auditor's name
and date, is included in the submission package as
`INDEPENDENT_AUDIT_STATEMENT.md`.

## What the authors must NOT do

- Do not pre-brief the auditor on known pitfalls.
- Do not fix anything mid-audit; mismatches are recorded first,
  repaired after, and the repair is disclosed in the statement.
