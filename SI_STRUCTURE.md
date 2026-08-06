# Submission display-item plan: Extended Data vs Supplementary Information

NMI allows ~10 Extended Data items; we have 16 candidates. This plan
fixes the split and organizes the SI by REVIEWER QUESTION, not by run
order. Every SI item answers three sentences: Question / Result /
Implication.

## Main text (unchanged)

5 figures + 1 table: concept+walkthrough; battery+exact rivals+fair
baselines; three-domain confirmation; chess discovery; MARL seed-level
+ phase forecast; projection table.

## Extended Data (the 10 that survive)

| # | Current file | Why it stays |
|---|---|---|
| ED1 | ed_fig1 (battery components) | backs the decisive battery claim |
| ED2 | ed_fig3 (grokking bridge) | recovers known mechanism #1 |
| ED3 | ed_fig4 (induction heads) | recovers known mechanism #2 |
| ED4 | ed_fig5 (MultiBERTs) | public-model transfer, all 5 seeds |
| ED5 | ed_fig9 (Pythia scaling family) | scale family + 2.8B boundary |
| ED6 | ed_fig11 (public checkpoint detail) | integrity + double dissociation |
| ED7 | ed_fig12 (chess episode detail + robustness cells) | event-level decisive backing |
| ED8 | fig45 (Overcooked confirmation) | the externally timestamped result |
| ED9 | fig46 (generator calibration) | construct validity of the record |
| ED10 | ed_fig13 (dual observer) | observer-relativity, measured |

## Supplementary Information (indexed by reviewer question)

### SI-A. Reproducibility and integrity
- Makefile targets, manifest, FINAL_FREEZE, INVALID_DATA_REGISTRY,
  INDEPENDENT_AUDIT_INSTRUCTIONS.
- Q: can the numbers be regenerated and audited? R: 88/88 checks, 236
  hashed outputs, one-command rebuild. I: no reported number is
  unanchored.

### SI-B. Validity of the central measurements
- ed_fig2 (threshold plateaus), ed_fig7 (process-proxy robustness +
  hierarchical MARL), profile calibration/axioms detail, ranking
  stability, PV/CV batteries.
- Q: do the metrics measure their constructs and survive estimator
  choices? R: generator diagonal dominance, 8 axioms, bounded-burst
  transform, radius/thinning grids; retained misses GC-2, RS-2,
  PV-1..3, CV-2/3. I: constructs calibrated; predictive content
  axis-specific.

### SI-C. Alternative explanations
- ed_fig6 (persistence boundary), ed_fig10 (mechanism/embedding
  bridge), ed_fig14 (strength gradient), learned/cross-fitted basins,
  adversarial observer, rollout audit, contract ensemble, world-model
  closure, matched provenance, harmful emergence detail.
- Q: could the verdicts be observer artifacts, sampling noise,
  provenance flags or simulator access? R: each alternative has a
  targeted control with its numbers. I: the dangerous rivals are
  excluded, boundaries stated.

### SI-D. Extended generalization and failure cases
- Crowd-vote domain (full detail + two quarantined pilots), LBF/LM
  generalization audits, Pythia S8 frozen-but-unexecuted extension
  (network availability documented), Overcooked round-2 draft
  (not executed, stopping rule), the full 36-miss ledger.
- Q: where does it break? R: novel geometries, marker positions,
  grid resolution, boundary cells. I: scope is measured, not asserted.

## Rule applied

Anything essential to a main-text claim stays in main text or ED;
SI raises confidence and reproducibility only. No main-text conclusion
depends on SI-only evidence.
