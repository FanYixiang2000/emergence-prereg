# Evidence sufficiency and the stopping rule

Recorded: 2026-07-19. This document states why data collection STOPPED,
which is as much a part of scientific discipline as why it started.

## The stopping rule applied

A new experiment is justified only if (i) it tests a new core claim,
(ii) it excludes a credible and dangerous alternative explanation,
(iii) a different outcome would change the paper's conclusions, or
(iv) it would plausibly change an accept/reject decision. Experiments
that merely add certainty to already-supported claims are not run.

## The four closure questions and their closing evidence

1. **Does the phenomenon exist?** Closed: the controlled battery and
   the four full-criterion families (swarm, Contextual LBF,
   latent-context LM, Overcooked-AI) identify learned context-selective
   useful structure while rejecting every control (188/188 control
   verdicts across the four families).
2. **Why is it not something simpler?** Closed: fitted multivariate
   prior-signal baselines with equal freedom fail frozen transfer;
   adversarial and machine-discovered observers cannot manufacture or
   destroy the verdicts; matched-behaviour provenance separates
   structurally identical systems; the rollout audit separates policy
   stochasticity from openness.
3. **Is the measurement itself valid?** Closed: the six-knob
   ground-truth generator gives a diagonally dominant sensitivity
   matrix (zero GC-1 violations); eight record axioms are
   machine-verified; ablations admit each named counterexample.
4. **Does the account do anything beyond postdiction?** Closed: the
   1-D and 2-D phase boundaries were derived before training and
   matched; chess discovery was scored before labels; Overcooked round
   1 was externally timestamped before any seed and passed 5/5.

## Decisions taken under the rule

- **Overcooked round-2 held-out replication: NOT executed.** The
  drafted preregistration (`OVERCOOKED_ROUND2_PREREGISTRATION.md`) is
  retained as a draft, marked not-executed. Rationale: round 1 passed
  all five registered predictions; the exact-boundary 8/12 acceptance
  is a fact to report honestly, not a gap that a second sample would
  close -- no round-2 outcome would change the paper's claims (the
  framework does not predict that PPO learns context-selective
  conventions in every layout pair; it predicts what the verdict is
  when it does or does not). The layout-pair pilots (seeds 8951-8964)
  additionally showed that most held-out pairs train only one layout,
  so a replication there would measure training fragility, not the
  criterion. No confirmatory round-2 seed was ever launched; no
  round-2 verdict data exists to be selected on.
- **No further built domains.** The crowd-vote domain (added for the
  collective-control question) is the last; a fifth domain already
  duplicates the inferential role of the first four, and its results
  are reported in supplementary scope with its two retained misses.
- **No further profile-prediction batteries.** PV and CV answered the
  same question twice (early prediction near phase boundaries); both
  outcomes are retained; a third variant would not change the
  axis-specific conclusion.
- **No seed extensions anywhere.** Every family already has its
  registered seed count; adding seeds after seeing results is sample
  chasing regardless of outcome.

## Residual known limitations (reported, not repaired)

- Full six-component verdicts exist only at tabletop scale; frontier
  models get a velocity audit, not a verdict (stated in Discussion).
- Overcooked round-1 acceptance sits exactly at its registered line.
- One confirmation carries an external timestamp; the others are
  author-maintained hash-anchored records.
- The independent non-author audit requires a human collaborator
  (instructions are turnkey in INDEPENDENT_AUDIT_INSTRUCTIONS.md).

These are the limitations of the evidence, and the paper says so. The
core claims stand on the closed chain above; a reviewer pointing at a
residual limitation is pointing at declared scope, not at an
unsupported claim.
