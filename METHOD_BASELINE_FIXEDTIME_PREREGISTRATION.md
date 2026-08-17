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

---

# Protocol RDC: discovered-regime controllability (frozen 2026-08-15, appended after the three protocols above were complete; nothing above edited)

Script: `learn_grip_discovery_utility.py` ->
`outputs/learn_grip_discovery_utility.json`.

Question: the regime-discovery audit showed that a fixed k-means
recipe recovers a two-cluster regime variable in the grip system
(k = 2 in 5/5 seeds, the left/right side split) but that the
discovered openness curve's temporal shape fails the onset gate.
The construct-validity question that matters is functional: with the
analyst removed, does the machine-discovered regime variable still
carry the controllability information? This protocol races the
discovered openness against the declared side-openness on the same
intervention outcomes.

### Design (all components frozen; no new tuning anywhere)

- Grip seeds, training, intervention grid (taus 5, 10, 14, 16, 18,
  20, 24, 30), kick parameters and evaluation batch are byte-identical
  to LEARN-GRIP-UTILITY.
- Discovery recipe is byte-identical to the REGIME-DISCOVERY audit's
  grip arm: 2,048 fresh rollouts per seed, k-means on the raw
  80-step position traces, k chosen by silhouette over 2..8 with
  cluster seed 0.
- Discovered openness of an intervention episode at time tau: fit the
  25-NN classifier on the recipe episodes' states (x, v, att) at step
  tau with their trace-cluster labels; the predictor is the
  normalized entropy of the class probabilities at the episode's own
  pre-kick state.
- Additional simple baseline: policy action entropy, the normalized
  entropy of the policy's action distribution at the pre-kick state.
- Predictors raced on pooled episodes (5 seeds x 8 taus x 2,048):
  discovered openness, declared side-openness, policy action entropy,
  |x|, |v|, att, tau. Outcome = the kick switches the final side.

### Registered outcomes

- RDC1: pooled AUC of discovered openness >= 0.80.
- RDC2: at every tau where both outcome classes have >= 20 episodes,
  fixed-time AUC of discovered openness >= 0.80.
- RDC3: discovered openness beats the time baseline
  (AUC(disc) > AUC(tau)).
- RDC4 (registered prediction): declared side-openness beats raw
  policy action entropy (AUC(side) > AUC(entropy)); the discovered
  openness vs policy entropy comparison is reported descriptively.

### Interpretation rules

If RDC1-3 pass, the controllability conclusion survives removal of
the analyst-declared representation: the machine-discovered regime
variable predicts, at matched intervention times, whether the
outcome can still be steered. A failure is reported as a registered
miss and the construct-validity limitation stands as currently
written in the manuscript.

### Outcomes (appended 2026-08-15 after the run; nothing above edited)

Recorded in `outputs/learn_grip_discovery_utility.json`.

- Discovery: k = 2 in 5/5 seeds (silhouette over 2..8), as in the
  regime-discovery audit. The recipe recovers the regime cardinality
  with no analyst input.
- RDC1: FAIL. Pooled AUC of discovered openness 0.750 (< 0.80).
- RDC2: FAIL. Both outcome classes have >= 20 episodes at taus 18,
  20, 24, 30; fixed-time discovered-openness AUCs are 0.751, 0.729,
  0.925, 0.834 (2/4 above 0.80). Declared side-openness at the same
  taus: 0.986, 0.979, 0.981, 0.993.
- RDC3: FAIL. AUC(disc) 0.750 < AUC(tau) 0.941. The pooled race is
  dominated by intervention time itself (all kicks before tau 18
  switch the side), which compresses pooled AUCs for every
  state-based predictor; the informative comparison is the
  fixed-time one in RDC2.
- RDC4: PASS. AUC(side) 0.996 > AUC(entropy) 0.621.
- Full pooled race (rank corr / AUC): disc_open 0.36/0.750,
  side_open 0.61/0.996, pol_ent 0.09/0.621, |x| 0.60/0.999,
  |v| 0.60/0.996, att 0.03/0.482, tau 0.42/0.941.

Interpretation under the registered rules: RDC1-3 are a registered
miss; the construct-validity limitation stands as written. The
descriptive content is reported alongside the miss: the analyst-free
recipe recovers the two-regime structure in every seed and its
openness carries a real but weaker share of the controllability
signal at matched times (0.73-0.93 versus 0.98-0.99 declared),
while raw policy entropy carries almost none (0.62). In this
low-dimensional system the physical magnitudes |x| and |v| match the
declared object (0.999, 0.996), as expected where side-openness is
nearly a function of the state; the declared object's value is that
it generalizes to systems with no such privileged coordinates.

# Analysis addendum STAT-UNIT: seed-level statistics for the intervention races (frozen 2026-08-16 before the run; nothing above edited)

Script: `learn_grip_stat_unit.py` -> `outputs/learn_grip_stat_unit.json`.

Motivation: the grip intervention races pool 81,920 episodes that are
nested within 5 independently trained seeds. Episode counts measure
precision, not replication; the training seed is the independent
unit. This addendum recomputes the published AUCs at seed level with
no new data collection: training, discovery, intervention grid, kick
and predictor definitions are byte-identical to LEARN-GRIP-UTILITY
and RDC (same seeds, same generators), so the episode streams are
identical reruns.

### Frozen analysis

- Per-seed pooled AUC of every raced predictor (side_open, disc_open,
  pol_ent, absx, absv, att, tau), computed within each seed across
  its 8 x 2,048 episodes.
- Seed-cluster bootstrap: 10,000 resamples of the 5 seeds with
  replacement, pooled AUC recomputed on the concatenated episodes of
  each resample; report the 2.5/97.5 percentiles.
- Leave-one-seed-out pooled AUC range for side_open and disc_open.
- Per-seed fixed-time AUCs at each tau where both outcome classes
  have >= 20 episodes within that seed.

### Registered outcomes

- SU1: side_open per-seed pooled AUC exceeds pol_ent per-seed pooled
  AUC in 5/5 seeds.
- SU2: the seed-cluster bootstrap 95% CI for the pooled side_open
  AUC lies entirely above 0.95.
- SU3 (descriptive, no bar): per-seed and leave-one-seed-out spreads
  for disc_open are reported as measured.

### Outcomes (appended 2026-08-16 after the run; nothing above edited)

- Discovery recovered k = 2 in 5/5 seeds.
- Per-seed pooled AUC: side_open 0.99943-0.99999, disc_open
  0.73362-0.85248, pol_ent 0.60492-0.64342.
- Seed-cluster bootstrap 95% CI: side_open [0.9932, 0.9998],
  disc_open [0.7301, 0.8509], pol_ent [0.6058, 0.6357].
- Leave-one-seed-out pooled AUC: side_open 0.99532-0.99849,
  disc_open 0.78065-0.84613.
- SU1 PASS (side_open beats pol_ent in 5/5 seeds).
- SU2 PASS (side_open CI lower bound 0.9932 > 0.95).
- SU3 reported as measured (`outputs/learn_grip_stat_unit.json`).

# Protocol OC-CC: counter_circuit convention study (NOT FROZEN; stopped at the pilot gate, 2026-08-16)

Intent: a second official Overcooked layout with two mirror-equivalent
circulation conventions (counter_circuit, 9x5, central counter block,
ring centre (4,2)), to replicate the coordination-ring formation
result under a protocol frozen in advance. Following the MPE
precedent, a competence pilot gates the confirmatory run: if pilot
seeds cannot deliver soups, the layout fails the competence
precondition and no confirmatory protocol is frozen.

Pilot outcomes (scripts oc_cc_pilot.py, oc_cc_pilot2.py; outputs
oc_cc_pilot.json, oc_cc_pilot2.json; pilot seeds 97001-97003,
excluded from any future confirmatory set):

- Round 1: 4M steps, the recipe's standard 0.6-horizon shaping
  anneal. Zero soups at every checkpoint. A shaped-reward circulation
  habit commits early (p_ccw ~= 0.03 from 400k) and dissolves after
  shaping expires (~3.0-3.2M), ending fully open.
- Round 2, arm A: 8M steps, 0.9-horizon anneal. Zero soups; committed
  circulation (p_ccw ~= 0.97) until shaping expires (~7.2M), then
  dissolves.
- Round 2, arm B: 6M steps, shaping never annealed. Zero soups;
  circulation habit persists under permanent shaping.

Decision under the gate: the training recipe (the paper's fixed
2-layer MLP on featurized states, PPO self-play, mechanics identical
to the ring study) does not reach task competence on counter_circuit
at up to 8M steps under three disclosed shaping schedules, so the
confirmatory convention study was not run. Recorded as a pilot-stage
competence failure with no theory claim either way. Per the frozen
decision rule adopted from the reviewer-response plan, no further
Overcooked layouts will be piloted for this manuscript: the
circulation habit without realized task value also illustrates why
the certificate requires competence before adjudicating a
convention.

# Protocol REACH: reachable-outcome openness in coordination_ring (frozen 2026-08-17 before the run; nothing above edited)

Script: `oc_ring_reach.py` -> `outputs/oc_ring_reach.json`.
Timing/sanity pilot: `oc_ring_reach_pilot.py` -> `outputs/oc_ring_reach_pilot.json`.

Motivation. The ring formation curves are non-monotone in behavioural
openness, which is why the frozen monotone detector certifies onset
in 1/8 seeds, and the OC-RING-FIXT null showed that behaviourally
open seeds at 960k were not movable. Both facts suggest the
behavioural object is a distorted proxy at formation level. The
possibility space named by the framework is the set of futures still
reachable; this protocol measures it directly.

Object. For seed s and stored checkpoint c, REACH-openness is the
normalized binary entropy H2(k/m) of the final circulation direction
across m independent continuations of the original training recipe
from checkpoint c to the full 2,000,000-step horizon
(`resume_training`, byte-identical mechanics, no perturbation), where
k of m continuations end counterclockwise. Final direction of a
continuation: p_ccw > 0.5 under the standard 30-episode
`eval_checkpoint`; the commit-margin flag (|p_ccw - 0.5| >= 0.3) is
recorded descriptively.

Frozen design.

- Seeds: the 8 confirmatory ring seeds (95101...96010).
- Checkpoint grid (9 points): 100k, 300k, 500k, 600k, 700k, 800k,
  960k, 1200k, 1600k.
- m = 8 continuations per (seed, checkpoint); continuation rng seed
  = 1000003 * seed + 31 * (checkpoint // 1000) + j, j = 0..7.
- No interim looks; all 576 cells run; outcomes reported as
  measured.
- Granularity-aware thresholds (m = 8): a grid point is OPEN if
  openness >= 0.95 (k in 3..5) and CLOSED if openness <= 0.544
  (k <= 1 or k >= 7). t_hi(s) = last OPEN grid step; t_lo(s) = first
  CLOSED grid step after t_hi(s).
- Behavioural commitment step t_beh(s): first grid checkpoint from
  which |p_ccw - 0.5| >= 0.3 holds at that and every later grid
  checkpoint in the stored formation record.

Registered outcomes.

- RE1 (monotone object): Spearman rho(openness, step) <= -0.7 in
  >= 6/8 seeds.
- RE2 (sharp closure): t_lo - t_hi <= 500k steps in >= 6/8 of the
  seeds where both are defined; seeds with no OPEN point are
  reported as closed-from-start and excluded from the RE2
  denominator.
- RE3 (closure precedes behavioural commitment): t_lo <= t_beh in
  >= 6/8 seeds.
- RE4 (explains the OC-RING-FIXT null): REACH-openness at 960k is
  CLOSED (<= 0.544) in 8/8 seeds, including the three behaviourally
  open seeds.
- RE5 (descriptive, no bar): the frozen breakpoint detector's
  verdict on each 9-point REACH curve is recorded; the grid density
  is below the detector's validated operating range, so no pass bar
  is attached.

Pilot clause. Two continuations (seed 95101, checkpoint 500k,
j = 900, 901) verify that the recipe completes and measure wall
time. Pilot runs are excluded from the confirmatory cells; the only
parameter the pilot may adjust is worker parallelism.

### Pilot outcome (appended 2026-08-17 after the pilot; nothing above edited)

Both pilot continuations completed (wall 910/926 s at 4 threads
each); both ended committed counterclockwise (p_ccw 0.9688) with
task competence retained (2.2-2.4 soups). Timing implies the full
576-cell grid costs roughly 10-12 h at 32 parallel workers. No
protocol parameter was changed.

# Analysis addendum STANCE-STAT-UNIT: seed-level statistics for the stance races (frozen 2026-08-17 before the run; nothing above edited)

Script: `learn_stance_stat_unit.py` -> `outputs/learn_stance_stat_unit.json`.

Motivation: the LEARN-STANCE-STICKY and LEARN-STANCE-CONTROL races
(Fig. 5c) pool episodes across 5 seeds per arm. As with STAT-UNIT,
this addendum recomputes the published AUCs at seed level with no
new data collection: both arms are deterministic reruns of the same
module (control arm sets STICK_P = 1.0), same seeds, same
intervention grid, same predictor definitions and signs.

Frozen analysis: per-seed pooled AUC of open/absx/absv/tau within
each arm; seed-cluster bootstrap (10,000 resamples, seed 0) for
open, absx, absv per arm; leave-one-seed-out pooled AUC for open in
both arms.

Registered outcomes.

- SSU1 (sticky ordering per seed): AUC(open) > AUC(absx) in >= 4/5
  sticky seeds.
- SSU2 (control reversal per seed): AUC(absv) > AUC(open) in >= 4/5
  control seeds.
- SSU3 (descriptive): CIs and leave-one-seed-out ranges reported as
  measured.

### Outcomes (appended 2026-08-17 after the run; nothing above edited)

Recorded in `outputs/learn_stance_stat_unit.json`.

- SSU1 FAIL: openness beats |x| in 0/5 sticky seeds. Within every
  sticky seed |x| leads: 0.94585-0.95644 against openness
  0.90996-0.93125. Seed-cluster bootstrap 95% CI: openness
  [0.9158, 0.9276], |x| [0.8932, 0.9466] (overlapping).
- SSU2 PASS: |v| beats openness in 5/5 control seeds.
- SSU3: leave-one-seed-out openness 0.9188-0.9239 (sticky),
  0.7841-0.8272 (control).
- Interpretation under the registered rules: the pooled sticky-arm
  ordering (openness 0.886 > |x| 0.849) is an aggregation effect
  that does not persist at the seed level. The main-text claim that
  openness outperforms the physical order parameter when a hidden
  consolidation phase exists is withdrawn and replaced by the
  narrower seed-level statement; the control-arm reversal stands.
  No rescue analysis was or will be run.

# Protocol LLM-CONV: in-context convention commitment in an LLM population (pilot gate frozen 2026-08-17 before the pilot; nothing above edited)

Pilot script: `llm_conv_pilot.py` -> `outputs/llm_conv_pilot.json`.

Intent: a realization-level convention system in a modern learned
model, structurally parallel to the Lewis signalling population: a
group of N = 4 instances of a fixed open-weight LLM
(Qwen2.5-7B-Instruct, local weights, temperature 0.8, top_p 0.95)
repeatedly and simultaneously picks one of 5 arbitrary symbols
(zib, kem, rop, dax, fen), seeing only the group's choice history in
context; any unanimous symbol is an equally valid convention. If the
pilot gate passes, a confirmatory protocol (branch-resampled
possibility-space openness across rounds, plus a deviant-message
intervention raced against openness) will be frozen as a separate
addendum BEFORE any confirmatory conversation is run.

Pilot gate (8 conversations, seeds 0..7, 30 rounds each):

- G1 (competence): >= 6/8 conversations reach unanimity sustained
  for 3 consecutive rounds within 30 rounds.
- G2 (symmetry breaking, not prompt bias): among converged
  conversations, >= 2 distinct final symbols are adopted.

If either gate fails, the line is stopped and recorded as a
competence/bias failure, as with MPE and counter_circuit; pilot
conversations are excluded from any confirmatory set.

### Outcomes (appended 2026-08-17 after the pilot; nothing above edited)

Recorded in `outputs/llm_conv_pilot.json`.

- G1 PASS: 8/8 conversations reach sustained unanimity, all within
  3 rounds, with zero parse failures.
- G2 FAIL: all 8 conversations converge to the same symbol ("zib",
  the first item of the symbol list). The convergence is driven by
  the lexical prior of the shared model, not by history-dependent
  symmetry breaking; there is no seed-dependent convention to
  measure.
- Decision under the gate: the LLM-CONV line is stopped and recorded
  as a bias failure. No confirmatory protocol is frozen, no prompt
  engineering or symbol-set search will be run for this manuscript.

# Addendum REACH-VALID: estimator validity gates (frozen 2026-08-17 before any gate run; nothing above edited)

Scripts: `reach_valid_ring.py` -> `outputs/reach_valid_ring.json`;
`reach_valid_signalling.py` -> `outputs/reach_valid_signalling.json`.

Derivation lineage (stated before any gate or confirmatory result is
seen). The manuscript defines the possibility space as the effective
joint distribution over the state-action-trajectory support still
open to the collective. At formation level the trajectory support is
the set of training futures, so the object's direct estimator is the
distribution over final regimes across fresh stochastic continuations
of the unchanged recipe: REACH as defined in Protocol REACH.
Behavioural openness is the within-checkpoint proxy for this object.
REACH is therefore not a new quantity introduced after the
OC-RING-FIXT miss; it is the trajectory-level instance of the
manuscript's own definition, and these gates test the estimator, not
the attractiveness of its Overcooked result. The gates are decided on
one ring seed's pilot cells and on a tractable system whose
ground-truth openness is known, before any confirmatory REACH cell is
run.

### Gate VH: continuation-horizon convergence (ring)

Seed 95101 (pilot seed of Protocol REACH), checkpoints 500k and
960k, continuation indices j = 902..909 (disjoint from confirmatory
j = 0..7 and pilot j = 900, 901; all gate runs are excluded from
confirmatory cells). Each continuation is evaluated at horizons
1.2M (descriptive), 1.6M and 2.0M by running `resume_training` from
the checkpoint to each horizon with the SAME continuation rng seed
(1_000_003 * seed + 31 * (ckpt // 1000) + j), so the shorter-horizon
runs are prefixes of the longer one when the pipeline is
deterministic. A determinism check (one cell run twice with
identical seeds; identical final p_ccw required) is recorded first.

- VH1: the direction label (p_ccw > 0.5) at 1.6M equals the label at
  2.0M in >= 7/8 continuations at each checkpoint.
- Fallback reading, frozen now: if the determinism check fails, VH1
  is evaluated at distribution level instead
  (|k_ccw(1.6M) - k_ccw(2.0M)| <= 1 at each checkpoint).

### Gate VS: tractable-system sanity (Lewis signalling)

Fresh seeds 717001, 717102, 717203 (disjoint from the published
LEARN-CONVENTION seeds). For each seed, the published recipe
(`learn_convention.py`, unchanged constants) is trained once with
parameter + Adam-state + baseline snapshots at updates
{0, 100, ..., 1000, 1500, 2000}. From every snapshot, m = 8
continuations run to the 4000-update horizon with fresh sampling
randomness (continuation seed = 90_000_000 + 97 * seed_index +
13 * snapshot + j, applied to torch.manual_seed and to the pairing
generator). Final label of a continuation: the population majority
code (tuple) if final mutual success >= 0.8, else "unconverged".
REACH-openness of a snapshot = H(empirical label distribution) /
log2(8). Ground truth in this system: all 120 codes are equivalent
and reachable at update 0; after capability (success 0.9) the code
is absorbing.

- VS1 (open where ground truth is open): at update 0, REACH >= 0.75
  and >= 4 distinct converged codes, in 3/3 seeds.
- VS2 (closed after capability): at the first snapshot at or after
  the base run's success-0.9 crossing, 8/8 continuations end at the
  base run's own final code (REACH = 0), in 3/3 seeds.
- VS3 (monotone and irrevocable): Spearman rho(REACH, update)
  <= -0.7, and once REACH = 0 at two consecutive snapshots it stays
  0 at every later snapshot, in 3/3 seeds.
- VS4 (descriptive, no bar): the REACH closure update compared with
  the behavioural breakpoint t* and the capability crossing of the
  base run.

### Decision rule (frozen)

The confirmatory 576-cell REACH grid of Protocol REACH is launched
only if VH1 and VS1-VS3 all pass. If any gate fails, the REACH line
stops with no modification (no horizon change, no entropy-estimator
change, no detector change, no re-run), the gate outcome is reported
as an estimator-validation failure, and the manuscript's existing
conclusions stand unchanged.

### Outcomes (appended 2026-08-17 after the gate runs; nothing above edited)

Recorded in `outputs/reach_valid_ring.json` and
`outputs/reach_valid_signalling.json`.

- VH determinism check PASS (identical p_ccw on the duplicated
  cell); primary reading used.
- VH1 PASS: label(1.6M) = label(2.0M) in 8/8 continuations at both
  checkpoints (all counterclockwise at both horizons).
- VS1 PASS 3/3: REACH at update 0 is 0.917-1.0 with 7-8 distinct
  converged codes.
- VS2 PASS 3/3: at the first snapshot after the capability crossing,
  8/8 continuations end at the base run's own code (REACH = 0).
- VS3 FAIL 1/3 (rho -0.8187, -0.629, -0.629 against the -0.7 bar).
  Diagnosis, verifiable analytically: every measured curve is
  non-increasing at every consecutive pair and never reopens after
  zero; the two missing seeds close by the third snapshot, leaving
  11 of 13 snapshots tied at zero, and the tie-corrected Spearman
  magnitude for ANY such curve is capped at
  72/sqrt(182*72) = 0.629 -- both seeds measured exactly this
  ceiling. The registered statistic cannot reach 0.7 for curves that
  close early, i.e., it penalizes fast irrevocable closure, the very
  behaviour the gate exists to reward. This is a mis-calibrated
  clause of the same kind as the barrier-threshold entry in
  Supplementary Note 1, and is reported verbatim.
- VS4 (descriptive): REACH closure precedes the capability crossing
  in 3/3 seeds (400 vs 800; 200 vs 650; 200 vs 700 updates).

# Amendment REACH-VALID-2: corrected monotonicity clause, fresh-seed re-validation (frozen 2026-08-17 before the run; nothing above edited)

Script: `reach_valid_signalling2.py` ->
`outputs/reach_valid_signalling2.json`.

Following the detector-amendment precedent (original miss reported
verbatim; amended clause registered; re-validated on FRESH data
before any confirmatory use), the mis-calibrated VS3 statistic is
replaced and the whole VS gate re-run on three fresh seeds. Nothing
else changes: recipe, snapshot grid, m = 8, label definition,
continuation-seed formula (with seed_index values 3, 4, 5) are those
of Addendum REACH-VALID.

- Fresh seeds: 717304, 717405, 717506 (disjoint from all previous).
- VS3' (amended monotonicity-irrevocability): at every consecutive
  snapshot pair up to and including the first zero, the REACH curve
  does not increase by more than 0.15 (the approximate one-label
  quantum at m = 8); and once the curve is 0 at two consecutive
  snapshots it stays 0 at every later snapshot. Required in 3/3
  fresh seeds.
- VS1 and VS2 are re-required on the fresh seeds with unchanged
  bars. The original VS3 Spearman value is also computed and
  reported verbatim with no bar attached.
- Decision rule: the confirmatory 576-cell REACH grid is launched
  only if fresh-seed VS1, VS2 and VS3' all pass (VH1 has already
  passed). If any fails, the REACH line stops permanently under the
  original stopping conditions.
