# Analysis freeze: NMI manuscript candidate

Freeze time: 2026-07-17T10:27:42+08:00
Manuscript-freeze candidate refreshed: 2026-07-17T15:20+08:00
Reframing-revision amendment: 2026-07-17T18:30+08:00 (see below)

## Amendment (2026-07-19, public-environment round)

Reopened for one round to add: the externally timestamped Overcooked-AI
confirmation (OC-1..5, all passed; the project's first protocol with a
public timestamp preceding data collection), the continuous-profile
ranking-stability and predictive-validity audits (RS-2, PV-1..3
retained misses), and the corresponding manuscript sections. A silent
IDE file-reversion incident on 2026-07-19 was detected by the
verification scripts and repaired; all restored numbers were
re-verified (82/82). Canonical state: `FINAL_FREEZE.md`
(2026-07-19T14:00+08:00).

## Amendment (2026-07-17, reframing revision)

An external mock review identified load-bearing gaps that meet the
freeze's own reopening rule (a manuscript claim depending on missing or
insufficiently independent evidence). Three read-only-or-additive
audits were run; no frozen threshold, basin definition or stored output
was modified:

- `fair_baseline_comparison.py` (new): multivariate prior-signal
  baselines with matched degrees of freedom; frozen transfer to the
  fresh battery.
- `dual_observer_contracts.py` (new): second plausible observer
  contract on all 75 stored CLBF systems; two registered misses
  (DO-1/DO-3) retained.
- `chess_realized_outcome.py` (new): engine-free realized-outcome
  referee; one registered miss (RO-1) retained.

The manuscript was retitled and reframed (substrate + layered
structural/adaptive criteria), the figure hierarchy aligned with the
claim hierarchy (5 main figures + 1 main table), and the main text
compressed. Anchor hashes below refresh accordingly.

This file marks the transition from experiment-building to manuscript
construction. The current outputs are the analysis set for the Nature Machine
Intelligence submission candidate.

## Frozen scope

No further changes should be made to:

- the formal six-component full criterion;
- any frozen threshold;
- any basin definition;
- any published-checkpoint inclusion/exclusion rule;
- any reported registered failure;
- any output JSON/CSV used by the manuscript;
- any invalid/quarantined artifact or its exclusion reason.

From this point, allowed work is limited to:

- manuscript writing and tightening;
- figure layout/caption work from existing outputs;
- claim--evidence consistency checks;
- reproducibility packaging;
- external timestamp/archive preparation.

New experiments should only be started if a load-bearing manuscript claim is
found to depend on wrong, missing, contaminated or non-causal evidence. The
preferred response to overclaiming is to narrow the claim, not to add more
experiments.

## Truthfulness and provenance guarantees

All headline values in the manuscript are backed by stored outputs and scripts.
The current mechanical audit is:

- `verify_manuscript_numbers.py`: 72/72 headline checks pass.
- `manifest.json`: 35 claim records, 19 protocol hashes, 207 output hashes,
  no missing claim outputs.
- `INVALID_DATA_REGISTRY.md`: every invalid/quarantined artifact is listed
  with an exclusion reason and is not used in final statistics.

Known invalid outputs are preserved, not deleted, precisely to prevent
post-hoc laundering of mistakes.

## Freeze anchor hashes

```text
manifest.json
16d4a0caa36c9d7d1917e519c5309c086d70ba732a438ef342265827a53789a2

paper/main.pdf
8f92562b02f777b87b77ecdb61d3794c9912a1a903a185ec740362a56ea9bf74

paper/main.tex
625f2de5ff2089b690aa1821c7e94c637c9ce537c76da2f128457bdc485c8da6

PREDICTION_LEDGER.md
54e2cc782374ea7e00c0ae6c7f7d939a8b2c164f781cf9042849e74a4dd81e9e

INVALID_DATA_REGISTRY.md
ebb6a04fe971b89fed53f2d8f5adf66a7134257e1be149d17c2c8ae9cb074ebe

requirements-lock.txt
9da633597c81acbca8e4fd7383b0a8c447ebcf575323b6825bfc976650a22c0c
```

## Submission positioning

The manuscript should not claim a universal definition in the sense of
solving every historical use of "emergence." The defensible NMI claim is:

> A known trajectory-space information substrate fixes what emergence claims
> are about; two layered criteria (structural collapse + adaptive
> qualification) identify adaptive emergence acquired during learning under a
> declared observer contract. Structural verdicts survive a change of
> plausible contract; value verdicts are contract-relative by declaration and
> measured as such. Prior signatures, singly or in fitted multivariate
> combination with equal freedom, do not transfer; the layered protocol does,
> in three domains under frozen thresholds, with scoped process and event
> instruments transferring parts of the account to public checkpoint series
> and uncurated chess discovery.

This freeze intentionally retains failures:

- Pythia 2.8B S1/S7 burstiness/grid-resolution failures;
- Pythia larger-scale head-fact burstiness failures;
- CLBF persistence novel-layout boundary;
- sequence-model marker-position generalization failure;
- ordinary learner accepted by the process proxy (retained even though the
  later, separately frozen lower-order novelty test correctly rejects it);
- all earlier registered failures in `PREDICTION_LEDGER.md`.

These are scope-defining evidence, not defects to be repaired.
