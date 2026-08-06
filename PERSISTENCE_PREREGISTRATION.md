# Persistence of the acquired macro-structure (Contextual LBF)

Status: author-maintained protocol frozen BEFORE any perturbed evaluation
was run. Frozen: 2026-07-16. Uses only saved confirmation policies
(seeds 1101--1110) and reconstructed initialization twins; no retraining.

## Why this experiment

The theory distinguishes emergence from ordinary decision-making partly by
the *stability* of the collapsed macro-structure, but the frozen
six-component protocol measures the structure only on the confirmation
evaluation distribution. This experiment measures whether the acquired
structure (context-conditional food-0-first selection and its usefulness)
persists under declared perturbations, or is a knife-edge artifact of the
exact evaluation setup. Scripted controllers are expected to persist too:
persistence separates *structure* from one-shot choice; provenance
(endogeneity/acquisition) separates learned from scripted. Persistence is
reported as a post-confirmation extension of the domain, not a change to
the frozen six-component rule.

## Systems

- learned: the ten saved confirmation policies (seeds 1101--1110).
- initial twins: same-architecture nets reconstructed at each seed's
  initialization (nothing-to-lose reference).
- scripted: team_nearest, fixed_food0, fixed_food1 (prewired-robust
  reference).

## Declared perturbations (all evaluated with n_eval = 40 episodes per
context per condition; natural + do_non_trigger conditions only)

- P0 baseline: frozen layouts, horizon 15, fresh evaluation seed block
  (9,000,000 + seed*100,000) -- re-evaluation noise reference.
- P1 novel layouts: two new interior layouts per context, never used in
  training or confirmation, same geometric semantics (code asserts the
  context-consistent nearer-food identity).
- P2 horizon 12 and P3 horizon 18 (frozen layouts).
- P4/P5/P6 observation noise: i.i.d. Gaussian noise with sigma 0.05 /
  0.10 / 0.20 added to every observation component seen by neural
  policies (scripted controllers read the true state; their rows are
  reported but the noise contrast is about learned structure).

## Measured quantities per system x perturbation

- conditional selectivity (the acquired structure);
- usefulness gap (natural minus do_non_trigger discounted score);
- retention = perturbed selectivity / that system's P0 selectivity.

## Registered predictions

- PS1 (structure persists): for every perturbation except the strongest
  noise (P6), at least 8/10 learned policies retain selectivity
  >= 0.5 x their own P0 value.
- PS2 (usefulness persists): under the same perturbation set, at least
  8/10 learned policies keep a positive usefulness gap in at least 5 of
  the 6 settings each.
- PS3 (twins gain nothing): every initialization twin stays below the 0.5
  selectivity threshold under every perturbation.
- PS4 (graceful degradation, descriptive): mean learned selectivity is
  non-increasing in noise level (P0 >= P4 >= P5 >= P6).

## Failure handling

Failed predictions are recorded as registered failures with routes; no
threshold or perturbation may be changed after this freeze.

## Mechanics amendment (recorded before the reported run)

The first P1 attempt used novel-layout tuples that violated the benchmark's
lexicographic food-identity convention (`FoodIndex` sorts positions, so the
declared food0/food1 identities were swapped in two variants and the
context label inverted mechanically). This is a layout-specification bug of
the perturbation, not a property of any policy; the botched output is
preserved unmodified as `contextual_lbf_persistence_layoutbug.json`. The
layout-validity assertion now also checks the lexicographic convention and
non-ambiguity. No threshold, prediction, or measured system changed.

## Outcomes

(recorded 2026-07-16 after the corrected-layout run; nothing above edited;
`outputs/contextual_lbf_persistence.json`)

- PS1 **REGISTERED FAILURE via a single route**. Horizon and noise
  perturbations: 10/10 policies retain >= 50% of their own baseline
  selectivity under P2, P3, P4 and P5 (40/40 cells). Novel layouts (P1):
  only 5/10 reach the 50% retention bar. Mean selectivity roughly halves on
  unseen geometries (about 0.80 -> 0.29); two policies drop to zero, five
  remain at or above 0.45.
- PS2 PASS under the frozen code semantics (every policy positive in 6 of
  the 7 evaluated settings), with the failing setting the same for all ten
  policies: on novel layouts the usefulness gap is negative for every
  policy (-0.007 to -0.125). Structure and its value do not fully transfer
  to unseen geometries.
- PS3 PASS. All ten initialization twins stay below 0.5 selectivity under
  every perturbation (70/70 cells).
- PS4 **REGISTERED FAILURE, benign route**: mean learned selectivity is flat
  across noise (0.800 / 0.800 / 0.803 / 0.798) -- no measurable degradation
  up to sigma = 0.20, so strict monotonicity fails on +/-0.003 evaluation
  jitter. The structure is more noise-robust than predicted.

Reading: the acquired macro-structure is *stable* in the temporal and
observational directions (the persistence the non-triviality bridge
requires) and is never spontaneously present in twins, but its *spatial
generalization boundary* is measured and narrow: transfer to novel
geometries is partial for selectivity and negative for value. Stabilized
collapse is not unlimited generalization; the boundary is now a measured
property rather than an unstated assumption.
