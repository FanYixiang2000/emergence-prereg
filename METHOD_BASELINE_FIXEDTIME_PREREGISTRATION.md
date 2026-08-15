# Preregistration: method-baseline battery, Kuramoto seed extension, fixed-time Overcooked intervention

Frozen 2026-08-15, before any of the three scripts below was written or
run. Outcome sections are appended after each run and never edited
thereafter. Rival definitions, group assignments, decision rules and
pass thresholds in this document may not be changed after the first
run of the corresponding script.

---

## Protocol MB: method-baseline battery

Script: `bench_baselines.py` -> `outputs/method_baseline_battery.json`.

Question: do standard information-theoretic and change-point tools,
given their best a-priori shot on the same declared objects, reproduce
what the instrument reports? Each rival is evaluated on exactly the
data the instrument used; no rival is tuned after seeing results
(the only calibration allowed is the 5% false-positive calibration on
the flat family stated below, performed before evaluation on any other
family).

### Data (all stored or exactly regenerable)

1. The 72 ground-truth factorial cells and 5 pseudo-controls of
   BENCH-72 (`bench72_factorial.py`, exact enumeration).
2. The matched-confound construction of `collective_constraint.py`
   (CC-1): two generators with identical joint distributions and
   different couplings.
3. The frozen held-out detector benchmark of `detector_validation.py`
   (same families, same seed, same reference point).

### Rivals (frozen definitions)

- **R1 amplitude-only**: total collapse amplitude of the declared
  joint object, `M = 1 - O_final`; "emergent" iff `M >= 0.5`. No
  qualification conditions.
- **R2 distribution composite** (standard toolkit, best effort):
  from the same exact stage distributions compute marginal-entropy
  drop, total correlation `TC = sum_i H(X_i) - H(X)`, and mean
  pairwise mutual information. Source rule, fixed a priori:
  - "individual" if the drop in mean marginal entropy accounts for
    >= 50% of the total collapse and the TC rise is < 25% of the
    total collapse;
  - otherwise "pairwise" if the summed pairwise MI rise accounts for
    >= 50% of the TC rise;
  - otherwise "higher-order".
  The composite has no environment rung: without the contract's
  declared conditioning step, environmental and internal relational
  constraints are not separable by any function of the system joint
  distribution alone. Environment-driven cells are scored against
  whatever label the rule produces.
- **R3 change-point rivals** for onset timing, each with acceptance
  calibrated to a 5% false-positive rate on the flat family before
  any other family is evaluated:
  - Binseg with RBF cost (as already used for the external
    comparison), acceptance by calibrated gain threshold;
  - CUSUM on first differences, acceptance by calibrated threshold.
  Evaluated at the benchmark reference point for power (onset family)
  and false-positive rate (knee, gradual, s-curve, flat families).

### Registered outcomes

- **MB1**: the ladder's source classification on the 72 cells (stored:
  72/72) is strictly more accurate than the R2 composite's. Report
  both accuracies and the composite's per-source confusion counts.
- **MB2** (mechanistic prediction): the R2 composite misassigns the
  majority of the 24 environment-driven cells.
- **MB3** (exact): on the CC-1 matched pair, every distribution
  functional used by R1/R2 (joint entropy, marginals, TC, pairwise
  MI) is equal across the two generators to <= 1e-12, while the
  contract-level verdicts differ. This is a computation, not a
  sampled estimate.
- **MB4**: R1 accepts at least one BENCH-72 pseudo-control that the
  instrument rejects (prediction: revelation-only and/or
  metric-artifact).
- **MB5**: neither calibrated change-point rival simultaneously
  attains false-positive rate <= 0.05 on every negative family and
  power >= 0.80 on the onset family at the reference point. If a
  rival does attain both, this clause is recorded as a registered
  miss and reported in the manuscript with the same prominence as a
  pass.

---

## Protocol KUR-N10: Kuramoto seed extension (2 -> 10 seeds per coupling)

Script: `kuramoto_scale_n10.py` -> `outputs/kuramoto_scale_n10.json`.

Purpose: the stored KUR-SCALE sweep (`kuramoto_scale.json`) has two
seeds per coupling; the monotonicity claims deserve seed-level
uncertainty. Detector, simulation contract, couplings and adjudication
are byte-identical to KUR-SCALE (`kuramoto_breakpoint.simulate`,
`kuramoto_breakpoint_r2.adjudicate`, unchanged).

- Couplings: K in {0.9, 1.1, 1.5, 2.0, 2.5} (unchanged).
- Seeds: 82_001 .. 82_010 per coupling (the first two are the stored
  KUR-SCALE seeds, re-run identically; simulation is deterministic
  given the seed).
- Registered outcomes (evaluated on the 10-seed set):
  - **KN1**: every seed at K >= 1.1 passes onset detection; K = 0.9
    behaves consistently with near-critical slowing (gated null or
    latest onset), as in KUR-SCALE.
  - **KN2** (critical slowing): seed-mean breakpoint time t* is
    strictly decreasing in K across passing couplings; in addition,
    Spearman rank correlation between K and per-seed t* (all passing
    seeds pooled) is negative with exact p < 0.05.
  - **KN3** (sharpness): seed-mean post-breakpoint slope magnitude is
    strictly increasing in K across passing couplings; Spearman exact
    p < 0.05 as above.
  - **KN4**: report per-K 95% bootstrap confidence intervals (10,000
    resamples over seeds) for mean t* and mean post-slope; the
    manuscript's cited values are replaced by the 10-seed means.

---

## Protocol OC-RING-FIXT: fixed-time Overcooked intervention

Script: `oc_ring_fixed_time.py` -> `outputs/oc_ring_fixed_time.json`.

Purpose: OC-RING-INT compares perturbations at different training
times, so openness and training time are confounded (acknowledged in
the manuscript). This protocol perturbs every seed at the same
training step; only openness varies across seeds.

### Fixed-time rule (computed from stored formation records only)

T_FIX = the common-grid checkpoint that maximizes the cross-seed
variance of circulation openness, subject to every seed having >= 20
committed evaluation episodes at that checkpoint. Computed from
`overcooked_ring_convention.json` + `oc_ring_ext.json` before any new
run: **T_FIX = 960,000 steps** (grid of 100 checkpoints, 8 seeds).

Openness at T_FIX (stored values): 95101: 0.20, 95202: 0.95,
95303: 0.96, 95606: 0.20, 95707: 1.00, 95808: 0.34, 95909: 0.20,
96010: 0.20.

Group assignment, declared now: **open** = openness >= 0.5 at T_FIX
(95202, 95303, 95707); **committed** = openness < 0.5 (95101, 95606,
95808, 95909, 96010).

### Intervention (identical mechanics to OC-RING-INT)

Load the stored checkpoint at T_FIX, add unbiased Gaussian parameter
noise at scales {0.25, 0.5} (same generator-seed rule), resume
training for exactly 400,000 steps with byte-identical mechanics and
the original 2M-step anneal schedule, then evaluate with the same
`eval_checkpoint` and the same commitment margin 0.3. 8 seeds x 2
scales = 16 runs. Outcome classes as before: flip / held /
uncommitted; "moved" = flip or uncommitted.

### Registered outcomes

- **OCF1** (primary): seed-level one-sided Fisher exact test. A seed
  is "movable" if at least one of its two runs is moved. Prediction:
  movable seeds are concentrated in the open group. Pass if p < 0.05
  (with 3 open vs 5 committed seeds this requires a perfect 3/3 vs
  0/5 split, p = 1/56 = 0.018; anything weaker is a registered miss).
- **OCF2**: run-level one-sided Fisher exact test, open runs (6) vs
  committed runs (10), moved as success. Report p.
- **OCF3**: AUC of openness-at-T_FIX for the moved outcome across all
  16 runs >= 0.70.
- **OCF4** (descriptive, registered): strict direction flips occur
  only in open-group runs.
- **OCF5** (maturity control): evaluate the unperturbed stored
  checkpoint at T_FIX for every seed (deterministic evaluation, mean
  soups); report whether task performance at T_FIX predicts
  movability as well as openness does (AUC comparison, descriptive).
  Prediction: openness predicts movability; matched training time
  removes the time confound by construction.

### Interpretation rules

Because all 16 runs share the same training step, any difference in
movability between groups cannot be attributed to training time. If
OCF1 passes, the manuscript's controllability claim is upgraded from
"time-confounded comparison plus grip-domain fixed-time controls" to
"fixed-time comparison in the standard benchmark itself". If OCF1
fails, the result is reported as a registered miss alongside the
existing time-confound caveat.

---

# Registered outcomes (appended after the runs; nothing above edited)

## MB outcomes (2026-08-15, outputs/method_baseline_battery.json)

- MB1 PASS: ladder 72/72 (1.00) vs composite 54/72 (0.75).
- MB2 PASS, with a bookkeeping correction: the factorial has 18
  environment-driven cells (72/4), not 24 as miswritten above; the
  composite misassigned 18/18 (all to "pairwise", the common-cause
  signature the toolkit cannot separate without the contract's
  declared conditioning).
- MB3 PASS: max difference of joint entropy, TC and pairwise MI
  across central_script / common_cause / local_feedback = 0.0
  (exactly, by construction); contract verdicts differ
  (local_feedback accepted, both external mechanisms rejected).
- MB4 PASS: R1 accepts external_mask and external_overwrite, both of
  which the instrument rejects as exogenous. The parenthetical
  prediction above named the revelation/metric controls; those have
  zero entropy amplitude and are rejected by R1 as well, so the named
  detail was wrong even though the clause passed. R1 also accepts
  0/72 true positives (an uncalibrated absolute threshold has no
  operating point on this family).
- MB5 PASS: binseg-RBF-gain power 1.00 but FPR 1.00 on knee and 1.00
  on gradual; CUSUM power 1.00 but FPR 1.00 on knee. The instrument's
  stored reference rates: power 1.00, FPR 0.00 on knee/gradual/flat.

## KUR-N10 outcomes (2026-08-15, outputs/kuramoto_scale_n10.json)

- KN1 PASS: 50/50 onset passes; K=0.9 consistent via the
  latest-onset branch (mean t* 6.64, later than every higher-K mean).
- KN2 PASS: strict monotone mean t* 6.64 -> 5.50 -> 3.24 -> 2.38 ->
  1.80; Spearman rho = -0.99, permutation p = 1e-5 (100,000
  permutations; the clause said "exact p", implemented as a seeded
  permutation test).
- KN3 PASS: strict monotone mean post-slope 0.030 -> 0.065 -> 0.100
  -> 0.151 -> 0.197; rho = 0.98, permutation p = 1e-5.
- KN4: per-K bootstrap CIs recorded in the output; manuscript numbers
  updated to the 10-seed means.

## OC-RING-FIXT outcomes (2026-08-15, outputs/oc_ring_fixed_time.json)

- OCF1 REGISTERED MISS: movable 0/3 open seeds vs 0/5 committed
  seeds; Fisher p = 1.0. No run moved (0/16).
- OCF2: 0/6 vs 0/10, p = 1.0.
- OCF3 FAIL (vacuous): no positive class; AUC undefined.
- OCF4: vacuously satisfied (no strict flips anywhere).
- OCF5: AUC undefined for the same reason. Open seeds had lower task
  performance at T_FIX (0.27-0.53 soups) than committed seeds
  (0.53-1.9), so maturity does not explain the null.
- Interpretation recorded at outcome time: every perturbed
  continuation, including the three behaviourally open seeds
  (openness 0.95-1.00 at 960k), re-converged to its seed's eventual
  direction. Together with OC-RING-INT (strict flips occurred only at
  per-seed late-open checkpoints, 1.20-1.34M steps, under
  sparse-only reward), this says the redirectable window is set by
  the seed's own commitment dynamics, not by behavioural openness
  read at a fixed calendar step: direction is parameter-committed
  before it is behaviourally visible. The manuscript reports this
  miss and the refined interpretation explicitly.
