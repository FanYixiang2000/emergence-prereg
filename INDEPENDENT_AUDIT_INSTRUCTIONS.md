# Independent audit instructions (for a non-author auditor)

Purpose: internal hashes, the number audit, the manifest and the
preregistration ledger prove that the project's files agree with each
other. They cannot prove the implementation is free of a shared bug.
This document makes a fully independent audit turnkey for someone who
did not write the formulas or the code. Estimated effort: 1-2 days.

## Who qualifies

Anyone who has not edited this repository's analysis code: a lab
colleague, a collaborator from another group, or a hired research
engineer. The auditor should NOT discuss implementation details with
the authors before finishing step 5.

## Step 1: clean-environment reproduction (half a day, mostly waiting)

1. Fresh machine or container (Python 3.10+). Clone the repository.
2. `pip install -r requirements-lock.txt` (pinned versions).
3. `make audit` -- `verify_paper_numbers.py` must report every check
   passed and `generate_manifest.py` must regenerate `manifest.json`.
4. `make figures && make paper` -- every data figure, every
   Supplementary Table and both PDFs must regenerate from stored
   outputs alone (no training, no downloads).
5. `make small-reproduction` -- the deterministic analysis layer must
   re-run from scratch and leave the audit passing.

## Step 2: random spot-check of manuscript numbers (2-3 hours)

1. Pick 20 numeric claims from `main.tex` at random (a claim = any
   number with a stated source experiment).
2. For each, trace it to its JSON in `outputs/` and confirm the value
   independently (read the JSON yourself; do not use
   `verify_paper_numbers.py`).
3. Record any mismatch verbatim.

## Step 3: independent reimplementation of the core quantities (3-5 hours)

Implement from the Methods text alone, without reading the repository's
analysis code:

1. **Openness and collapse.** Implement the normalized-entropy openness
   O_t and collapse C_t = 1 - O_t of a declared possibility space
   (Methods, "Openness and collapse"). Regenerate the 72 ground-truth
   factorial distributions from the generator recipe stated in Methods
   (or run `bench72_factorial.py` once to dump them) and confirm your
   O_t matches the stored `outputs/bench72_factorial.json` curves.
2. **Source ladder.** Implement the four-rung decomposition
   (environmental, individual, pairwise, higher-order; Methods,
   "Source decomposition") and confirm it assigns the registered
   dominant source in all 72 factorial cells, and that the four rungs
   sum to the total collapse (identity checked to numerical precision).
3. **Breakpoint detector.** Implement the two-regime piecewise fit,
   the Delta-BIC comparison against the one-regime fit, the effect-size
   gate, saturation truncation and parity thinning (Methods,
   "Breakpoint detection"). Run it on the stored collapse curves in
   `outputs/learn_grip_transport_b5.json` and
   `outputs/detector_validation.json`'s labelled curve families, and
   confirm breakpoint locations, gate decisions and the held-out
   power / false-positive counts reported in the manuscript.

## Step 4: manual trajectory inspection (1-2 hours)

1. Re-run `python barrier_xplay.py` (deterministic replay of stored
   policies) and verify the cross-play compatibility matrix it prints
   matches the manuscript's convention-barrier claim.
2. Load one stored Overcooked ring checkpoint pair
   (`outputs/overcooked_genesis_ring*_s*_*.pt`) with
   `overcooked_ring_convention.eval_checkpoint` and verify by eye that
   the reported circulation direction matches rolled-out behaviour on
   the unmodified public benchmark.

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
