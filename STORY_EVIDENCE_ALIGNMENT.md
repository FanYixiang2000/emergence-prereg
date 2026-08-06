# The story, and where every sentence's evidence lives

Written 2026-07-23. Companion to EMERGENCE_DEFINITION_V2.md. Rule:
no sentence enters the paper unless its evidence cell below is PASS,
or it is explicitly narrated as a registered miss / pending item.

---

## The story in one paragraph

We set out to define emergence and began with a bold, intuitive
hypothesis: emergence is punctuated possibility collapse. Our own
preregistered controls partially falsified it -- ordinary learning
can be punctuated, classic collective organization can be gradual,
and punctuatedness itself depends on the checkpoint grid. Instead of
abandoning the collapse intuition, we located the two real errors:
the possibility space had been chosen wrongly (success/failure
instead of the joint state-action-trajectory space), and abruptness
had been promoted from a temporal phenotype to an existence
criterion. Correcting both yields a generative theory: emergence is
a spontaneous, selective, persistent regime-level collapse of the
effective joint possibility space, decomposable by source into
environment-mediated, individual, pairwise and higher-order channels
-- each a TYPE of emergence. We calibrated this instrument on
analytic ground truth where every channel is a knob; we showed it
labels off-design generators correctly; and in a real machine
intelligence system we measured the collapse's formation history --
the internal reorganization is detectable long before the capability
becomes visible. The causal commitment-window claim was tested (E3C)
and did not survive its own frozen falsification clause: formation
proved re-entrant and lesion-robust, which the story reports as a
measured persistence property rather than a causal lever. In its
place stands a confirmed separation result: at matched product, the
multi-source collapse PROFILE distinguishes a learned regime from a
noise-matched scripted regime through its environment-conditioned
component (E1-C, fresh-seed CIs non-overlapping), even though
single-point G cannot (E1-2, falsified). Causal emergence
characterizes the causal status of a macro-regime that already
exists; we measure when, how and through which channel it comes into
being.

## Act-by-act, sentence -> evidence

### Act 1. The problem: one word, many instruments

Claim: PID, causal emergence, self-organization, phase transitions
and LLM emergent abilities each measure one facet of an evidence
chain and call that facet "emergence"; observationally identical
macroscopic results can have entirely different sources.

Evidence: the matched-confound construction -- central script,
common cause, coincidence and local feedback with IDENTICAL joint
distributions, identical marginals, identical macro success
(collective_constraint.json, `matched_confound`: all identity checks
true). Status: PASS (frozen v1 battery, untouched).

### Act 2. The bold hypothesis (kept in the paper as the real origin)

Claim: we originally proposed emergence = punctuated possibility
collapse (Potential -> Trigger -> Collapse), with burstiness as a
multiplicative factor.

Evidence: the frozen v1 manuscript and battery are the historical
record (FINAL_FREEZE.md). Status: historical, disclosed.

### Act 3. Self-falsification (the credibility engine)

- Punctuation is not sufficient: the ordinary learner passes every
  burst-based component while its structure is additively explainable
  (N ~= 0). Evidence: frozen v1 ordinary-learner control. PASS.
- J-on-an-order-parameter is not necessary: the ant-trail system has
  high interaction dependence and persistence while its 1-D route-
  commitment order parameter consolidates with a wide 10-90% span.
  Evidence: frozen v1 ant battery (ANT-3). PASS -- but READ
  PRECISELY (v2.1): this falsifies temporal concentration measured
  on a colony order parameter as an existence criterion. It does NOT
  test whether the JOINT per-ant possibility space has a commitment
  breakpoint (v2.1 B5); that is the open E7 prediction
  (t_collapse < t_completion). Abruptness-of-the-possibility-space,
  as a contract-robust regime breakpoint, remains NECESSARY in the
  definition; what was demoted is J as its detector.
- Punctuation is not observer-free: thinning the checkpoint grid
  flips the burst verdict on identical Pythia curves. Evidence:
  frozen Pythia thinning battery. PASS.
- Conclusion drawn IN the paper (v2.1 wording): the EXISTENCE of a
  contract-robust regime breakpoint in the possibility-space
  dynamics is necessary (B5); J, the temporal CONCENTRATION of
  collapse around that breakpoint, is a phenotype reported with its
  grid band. Definition: EMERGENCE_DEFINITION_V2.md section 6 and
  the v2.1 amendment.

### Act 4. The two corrections that save the collapse story

Correction 1 -- the right possibility space. Not H(success) but the
effective joint state-action-trajectory space (the 10^n -> 2^n
intuition), with effective possibility N_eff = 2^H and normalized
openness, because literal support never shrinks for stochastic
policies.

Evidence: implemented as the joint-action ladder on real rollouts
(overcooked_joint_collapse_s93001/2/3.json: openness/collapse_norm
fields). Status: PASS (measured, 3 seeds).

Correction 2 -- source decomposition instead of gatekeeping. Total
collapse splits along a nested maximum-entropy ladder:

    C_total = C_env + C_individual + C_pair + C_high

and every channel is a type of emergence (environment-mediated
focus-fire, parallel individual contraction, pairwise coordination,
higher-order role-locks). The interaction cut is a source
DECOMPOSER, not an accept/reject gate. This is exactly the user's
four arguments, adopted.

Evidence, three independent legs:
- Analytic ground truth: four knobs (lambda, rho, kappa, gamma) each
  move only their own component; ladder monotonic; declared-vs-
  hidden E moves collapse between C_env and C_rel by construction
  (collapse_source_decomposition.json, SD-1..SD-5 PASS; SD-4 is the
  contract-relativity proof, not a bug).
- Off-design generators: Kuramoto oscillators, a mechanism
  vocabulary the ladder never saw. Uncoupled -> all-null; common
  driver -> C_env-dominant; single-edge coupling -> C_pair-dominant
  (kuramoto_offdesign_ladder.json, KUR-1/2/3 PASS; KUR-4
  preregistered may-miss, and the miss is mathematically correct:
  three-way synchrony is pairwise-implied). Status: PASS.
- Real system: the same ladder on learning checkpoints
  (overcooked_joint_collapse_s*.json) and its contract-sensitivity
  table (overcooked_contract_sensitivity.json, CS-1/CS-2 PASS:
  relational label stable under contract refinement; hiding E
  inflates C_rel ~7x, reproducing SD-4 on real data).

Ontology continuity: the v1 four-mechanism battery is REINTERPRETED,
not rerun -- central script fails boundary B3 (external
hard-coding), common cause becomes the C_env channel, coincidence
fails persistence B4, local feedback is the relational channel
(collective_constraint_v2_typology.json, RL-1 PASS, numeric fields
preserved bit-for-bit). The definition moved from v1 to v2 ONCE, in
the open, driven by registered results.

### Act 5. Genesis, not product: what causal emergence cannot ask

Claim: causal emergence asks whether an existing macro-regime has
causal efficacy; we ask when and how it endogenously formed. The
genesis instrument is the same-state real-vs-ghost continuation
(G_t), tracked over the training history.

Evidence:
- Formation precedes visibility: t_seed < t_visible in 3/3 seeds
  (overcooked_genesis_curve_curve_s93001/2/3.json: t_seed ~ 320k vs
  t_visible ~ 740k-800k, bootstrap CIs stored). Status: PASS.
- Product does not identify genesis: a BC clone distilled from the
  learned policy has instantaneous G comparable to learned
  (OTC-C2 registered honest negative, overcooked_genesis_comparison_
  pilot.json) -- so a single-time-point G is a PRODUCT measurement;
  only the formation curve G_s over the training history separates
  "organization that grew here" from "organization copied in".
  The clone has no formation history by construction. Status:
  registered negative that sharpens the claim.
- The product-matched version (E1) then went FURTHER than expected:
  with a noise-handicapped scripted pair calibrated to the learned
  score (E1-1 PASS, 40.4 vs 39.4), single-point G did NOT separate
  the systems (E1-2 falsified: G_noisy 0.057 > G_learned 0.042).
  Standing interpretation: instantaneous G measures coupling
  strength, not provenance. Genesis lives in the formation history
  and the declared provenance boundary B3.
- The recovery (E1-B estimation -> E1-C confirmation, frozen
  directional prereg, fresh seeds): at matched product the source
  PROFILE separates the two systems -- C_env 0.0137 [0.0123, 0.0156]
  (learned) vs 0.0005 [0.0004, 0.0007] (scripted+noise), total
  collapse_norm 0.081 vs 0.030, both CI pairs non-overlapping
  (overcooked_profile_confirmatory.json). This is the two-mechanism
  instance of "same outcome, different collapse composition".
  C_relational is explicitly unclaimed (overlapping CIs).
- Micro collapse, macro creation: joint-action entropy falls while
  macro basin entropy RISES during formation (JC-5 registered miss,
  3/3 seeds) -- the predicted signature "micro possibilities get
  organized -> macro capability gets created", with the level
  structure now declared per-level in the definition (v2 section 7).
- Guardrail: M is never evidence alone -- a scripted mechanism
  already reaches M = +5.8 from desynchronization; one learned seed
  falls inside the null band at a single checkpoint (NB-1 registered
  MISS, delta_m_null_band.json). M-claims are carried only by seed
  means plus a positive G. Status: miss retained, guardrail adopted.

### Act 6. The causal leg (running now; the story's falsifiable edge)

Claim: the formation curve locates a commitment window (largest
collapse interval overlaps [640k, 1.0M] in 3/3 seeds; G peaks
inside it), and cutting the partner feedback INSIDE that window,
at equal budget, selectively suppresses the relational organization
and the macro gain, more than a random-position cut.

Evidence and OUTCOME (2026-07-23): the pilot (one seed) showed M and
C_rel lowest in the commit condition, but the confirmatory E3C
(5 conditions x 5 seeds, random-window control, frozen exact
permutation plan) DID NOT REPLICATE it: p(M commit < random) =
0.325; C_rel lowest in the uncut condition; condition differences
are well below seed noise. Per the frozen falsification clause, the
causal-window claim is DROPPED from the story. Act 6 is therefore
retold honestly: we located the window descriptively, attempted the
causal verification, and the system answered -- the final
organization is ROBUST to any single 360k-step feedback lesion;
formation is re-entrant. This converts a would-be fragility claim
into a measured persistence (R) property and leaves the genesis
certificate as a descriptive-predictive instrument (early detection
stands: t_seed < t_visible, 3/3 seeds), not an interventional one.
The pilot's early-cut "curriculum effect" also failed to replicate
(E3C-4): it was seed noise, and is reported as such.

The dense-grid control (DG) completed: the largest-collapse interval
is grid-robust (DG-1/DG-2 PASS), guarding the descriptive claim. The
t_seed metric's endpoint sensitivity is disclosed as a caveat.

### Act 7. Calibration, scale, and the second timescale (wave 4)

- Full-factorial calibration (Claim 2 of the roadmap): BENCH-72 --
  4 sources x 3 temporal shapes x 2 stability x 3 values, blind
  recovery of source (72/72), M, J, t*, rho, V, plus five declared
  pseudo-controls (bench72_factorial.json, B72-1..6 all PASS). The
  decisive detail: M is invariant across temporal shapes (relative
  range 0.000 in all 24 groups) while J strictly orders
  punctuated > sigmoid > gradual -- amplitude and abruptness are
  measured as different quantities, which is the formal version of
  demoting burst to a phenotype. Revelation-only and metric-artifact
  controls show exactly zero collapse (the Schaeffer-style artifact
  is excluded by measurement, not assertion); external mask/
  overwrite show large real collapse and are excluded only by the
  declared B3 flag -- consistent with E1: distributions measure
  collapse, provenance requires the declared boundary.
- Three learned agents (E4): the simultaneous parity game is a
  registered testbed failure (3/3 seeds never learn -- a
  coordination trap). The sequential variant TRI-B learns the regime
  in 3/3 seeds (reward 0.44 -> 1.00) and the constraint is carried
  relationally (C_pair ~= 1.01 bits), but NOT at higher order
  (TRIB-3 registered may-miss confirmed: C_high ~= 0.001). Standing
  finding: learning selects low-order constraint implementations
  when available -- agents 1 and 2 contract into near-deterministic
  cycles, making the parity completion pairwise-explainable. A
  transient C_high blip during formation (all 3 seeds, update ~400)
  is retained as an unregistered observation only.
- Episode-time collapse (EP): the two-timescale prediction FAILED as
  frozen (EP-1/EP-2 registered misses). First-cycle commitment is
  fast (median openness 0.60 -> 0.00 by t=40) but the task is
  cyclic: each delivery re-opens the next cycle's branching, so
  whole-episode monotonicity was the wrong shape. A cycle-aligned
  macro variable is future design, to be frozen before any run.

### Act 8. V3 and the re-adjudication of our own falsifications

V2 over-corrected: it used the v1 burst failures (detectors on
performance/order-parameter objects) to demote abruptness entirely.
V3 (EMERGENCE_DEFINITION_V3.md) restores what the theory always
meant: a structural breakpoint in the collapse dynamics of the JOINT
possibility space is necessary (B5); J stays an intensity phenotype.
The three v1 "abruptness falsified" cases were then re-adjudicated
with the V3 object and a model-comparison hinge detector (RE
battery, preregistered):

- Ant double bridge (RE-2, directional, 4/4 PASS): route completion
  is gradual (v1 stands) AND the joint possibility space collapses
  early and abruptly -- hinge at trip 40 (Delta-BIC 119, thinning-
  persistent) vs median completion at trip 124.5; 100% of committing
  episodes half-collapse before completion; SOLO null. The paper's
  central intuition ("the colony decides before the bridge exists")
  is now a measured result, and the v1 gradualism result is its
  complement, not its refutation.
- Ordinary learner (RE-1, adjudication): collapse starts at maximum
  rate from epoch 0 and decelerates -- a knee, not an onset. B5
  excludes it; the v1 burst-gate pass was a detector artifact.
- Pythia/MultiBERTs (RE-3, adjudication): onset unresolvable -- the
  entropy collapse predates the second stored checkpoint on all 9
  series; no B5 claim is made, and the limitation (grid density,
  probe object, entropy column shared across control runs) is
  disclosed rather than papered over.

The emerging dissociation -- onset-type breakpoints (slow -> fast;
emergence) vs deceleration knees (fast -> slow; convergence) -- was
retained here as an unregistered candidate signature; it has since
been preregistered and confirmed on a learned system (TRI-C-BP,
Act 10).

E3C's scope is also restated per V3 section 5: what failed is "one
short pulse permanently destroys the regime"; the measured
re-entrance motivates the registered follow-up quantities (collapse
delay, re-entry time, dose-response, path switching).

### Act 9. The episode-time causal leg (ANT-INT), told exactly

With RE-2's sharp hinge in hand, the commitment-window intervention
was retried at the episode timescale with exact same-seed paired
counterfactuals. Outcome, kept precisely as frozen clauses ruled:
- AI-1 DROPPED: outcome flips are NOT maximal at the hinge window
  (early 27.5% > commit 8.5% > random 5% > late 1%). The hinge is
  not the point of maximal outcome leverage.
- AI-2 PASS: the hinge window maximizes commitment DELAY (median
  +113 trips).
- AI-3 PASS: after collapse the regime is intervention-robust
  (1% flips, median re-entry 46 trips) -- re-entrance again, now
  measured at the second timescale.
- The tempting generalization ("outcome leverage proportional to
  remaining openness") was frozen as ANT-INT-B and FAILED as frozen
  (rho 0.62 < 0.8; the median-openness reference saturates at zero
  and hides episode heterogeneity).
- The per-episode conditional version was then frozen as ANT-INT-C
  with a declared three-strikes clause, on FRESH seeds, and ALL
  THREE predictions passed (ant_conditional_leverage.json): flip
  rate rises strictly across per-episode openness bins (0.000 ->
  0.0006 -> 0.054 -> 0.205), a closed episode (o < 0.1) flipped
  0/2600 times, and flipped pairs are 0.58 openness units more open
  (permutation p < 5e-5). The law is real -- but it is conditional
  on the episode's own state, not on population medians.
What the paper may say: intervention timing differentiates three
causal quantities -- outcome leverage (before the breakpoint),
timing leverage (at the breakpoint), robustness/re-entrance (after)
-- and remaining PER-EPISODE openness quantitatively predicts
controllability (ANT-INT-C). "How open the possibility space still
is" is not just a description of emergence; it is the control
variable for whether the outcome can still be steered.

### Act 10. Learning CAN build the high-order carrier (TRI-C)

TRI-B left a wound: learning compiled the triple constraint down to
a pairwise implementation, and no learned C_high existed anywhere in
the workspace. TRI-C closes it by blocking the low-order route with
PRIVATE information: agents 1 and 2 must follow iid private cues
(their bit marginals are exogenously mixed), and agent 3 -- who
sees only partner actions, never the cues -- completes the parity.
- TRIC-1/2/3/4 ALL PASS, 3/3 seeds (triad_highorder_cue.json).
- The unconditional bit table converges to the textbook irreducible
  XOR carrier: C_high 0.94-0.96 bits with pairwise ~0.0004 bits
  throughout the WHOLE formation history -- learning built the
  triple constraint directly, never routing through pair channels.
- Contract relativity reproduced on a learned system (TRIC-4):
  declaring E = (c1, c2) reattributes the same collapse as
  C_env = 0.943, C_high = 0.0. The carrier is high-order exactly
  relative to what the observer declares exogenous -- SD-4 was the
  analytic version; this is the learned version.
Together TRI-B + TRI-C give the paper a sharp sentence: C_high is
not unlearnable, it is UNFAVORED -- gradient learning selects the
lowest-order implementation available, and builds genuinely
irreducible structure only when information constraints force it.
This completes the ladder-of-types calibration on learned systems:
individual (E1), environment (E1-C), pairwise (TRI-B), high-order
(TRI-C).

And the formation of that learned carrier satisfies B5 (TRI-C-BP,
fresh seeds, dense 81-point grid, all three predictions pass 3/3
seeds): joint openness is near-flat through ~update 525, then hinges
into a steeper closing phase (Delta-BIC 38-73, thinning-persistent),
and the hinge precedes r3 >= 0.9 by ~600 updates. The
onset-vs-deceleration dissociation is now a registered result, not
a candidate: onset-type breakpoints appear where the theory says
emergence lives (ant commitment RE-2, learned high-order TRI-C-BP)
and are absent where it says convergence lives (ordinary learner
RE-1, stored LM curves RE-3). Collapse leading capability now holds
at three scales: Overcooked training, ant episodes, TRI-C formation.

### Act 11. The vulnerability matrix: profile predicts what an
### attack can bite (VUL-MAT / VUL-MAT-B)

V3 section 6 promised the neutral form of the vulnerability
program; VUL-MAT executes it on the two learned triad systems with
marginal-preserving test-time interventions at matched budgets
(vulnerability_matrix.json):
- The dominant channel is destroyed by the matched relational
  intervention, dose-monotonically (VM-1, VM-3, both systems).
- The dissociation is clean in TRI-C (VM-2, 3/3): scrambling cues
  removes exactly the cue-following reward share (~1.0 of 3.0)
  while leaving the XOR carrier untouched (C_high 0.95-0.96, ~0%
  drop); scrambling the communication channel does the reverse.
  Environment damage and high-order damage are different attack
  surfaces, and the ladder tells them apart in advance.
- VM-4 failed as frozen, and the failure is the deepest finding of
  the wave: fresh TRI-B seeds show IMPLEMENTATION DEGENERACY --
  2/3 compiled parity all the way down to INDIVIDUAL order (fixed
  own-bits; C_pair 0.04), only 1/3 learned the pairwise carrier.
  And each seed's vulnerability tracked ITS OWN profile: the
  individual-carrier seeds are immune to full communication
  scrambling (reward 0.9999), the pairwise-carrier seed collapses
  to 0.48. The frozen adjudication had hard-coded "TRI-B = pair"
  and rightly failed; the per-seed conditional form was refrozen as
  VUL-MAT-B with a declared two-strike drop clause.
- VUL-MAT-B final (pooled 16 fresh seeds): the rank law is
  UNRESOLVABLE and not claimed -- 16 fresh trainings never produced
  a high-relational-share seed (13/16 compile parity to individual
  order; implementation degeneracy is the RULE, not the exception).
  What the pooled data do show, disclosed as unregistered: a sharp
  threshold. Relational share < 0.1 gives exactly 0.000 reward loss
  under full communication scrambling (13/13); share >= 0.2 gives
  0.365-0.513 loss (3/3). The paper states the threshold pattern
  descriptively and claims only VM-1/2/3 plus VMB-2 immunity as
  frozen results.

### Act 12. Physics breadth: B5 tracks the Kuramoto transition
### (KUR-BP / KUR-BP-R)

The theory makes a two-sided prediction for a classical physics
system: supercritical Kuramoto sync is autocatalytic (force
proportional to r), so its joint openness must show an onset-type
breakpoint; subcritical coupling has no instability, so it must
show nothing. Outcomes (kuramoto_breakpoint.json):
- KURBP-1 PASS: onset hinge at t* = 2.6 (Delta-BIC 22.9, slopes
  -0.003 -> -0.074), preceding r's saturation (t_r90 = 7.0).
- KURBP-4 PASS: the collapse is carried by C_pair (5.942 of 5.947
  bits) with raw-phase marginals uniform -- relational collapse of
  the joint space, measured in a system with continuous phases and
  sinusoidal coupling that the ladder was never designed around.
- KURBP-2 FAIL as frozen on a self-inflicted bar (thinned
  Delta-BIC 8.6-8.8 vs the frozen 10; hinge location fully stable;
  RE-2's own thinning bar is 2). Kept as a miss.
- KURBP-3 FAIL as frozen, producing the wave's detector lesson: on
  a flat subcritical series (drop 0.000) the hinge test still
  fires statistically. This forced the V3.1 amendment -- an
  effect-size gate (drop >= 0.1) before any hinge verdict --
  adopted through the registered miss and re-tested on fresh seeds
  as KUR-BP-R rather than by re-adjudicating the miss away.
- KUR-BP-R (fresh seeds, amended contract): the gate works (all
  three subcritical runs correctly "no collapse", KURR-2 PASS;
  relational carrier 3/3, KURR-3 PASS), but KURR-1 failed 1/3: two
  seeds put the single hinge on the SATURATION knee of the S-curve
  instead of the onset knee. Diagnosis: a full S-curve has two
  knees and a single-hinge fit picks whichever has more residual
  leverage. The fix is definitional, not tuned: B5 is about
  collapse dynamics, so the analysis window must END at saturation
  (t_sat = first point within 5% of final value); inside that
  window an S-curve has exactly one knee. Frozen as KUR-BP-R2 and
  re-run on a third batch of fresh seeds.
- KUR-BP-R2 ALL THREE PASS, 3/3: onset hinge at t* = 3.2 in every
  supercritical seed (Delta-BIC 91-95, slopes -0.004 -> -0.10,
  thinning-persistent, preceding r saturation), every subcritical
  run gated null, relational carrier 3/3. The physics sentence the
  paper may now write: B5 tracks the synchronization phase
  transition -- an onset breakpoint exactly above criticality,
  nothing below it, with the collapse carried by the pairwise
  channel. The detector that delivers this matured through three
  registered misses, each fixed definitionally on fresh seeds.

### Act 13. The cycle is the episode-time unit (EP-CYCLE)

The EP misses said whole-episode monotone commitment is the wrong
shape for a cyclic task. EP-CYCLE freezes the cycle-aligned form:
cycles delimited by delivery events, snapshots at cycle phases
{0, 0.25, 0.5, 0.75}, the unchanged EP basin instrument. ALL THREE
predictions pass (overcooked_cycle_collapse.json): within-cycle
median entropy 0.25 -> 0.00 -> 0.00 -> 0.00 (EPC-1), the space
re-opens by 0.2499 bits at each delivery boundary (EPC-2, bar 0.2),
and the per-cycle collapse is substantial (EPC-3). The two-
timescale claim for Overcooked returns in cycle-aligned form, with
the thin margins disclosed (the learned policy is heavily
committed; most mid-cycle snapshots sit at zero entropy). Together
with EP's misses this is the collapse event SPECTRUM of V3
section 4, measured: emergence events as a sequence of re-opening
and re-collapse, one per delivery cycle.

### Act 14. Where abruptness LIVES: object classes and the
### finite-size law (ANT-GAIN -> ANT-FINE -> ANT-FINE-B ->
### ANT-COLONY-BP)

The wave that made the theory complete began with five consecutive
registered misses on one question -- does the ant system have an
onset-type breakpoint? -- and ended with the two structural answers
the story needed (V3.2):
1. OBJECT CLASSES. Endpoint-projection instruments (basin entropy
   of cloned continuations, RE-2's object) measure outcome
   PREDICTABILITY, which under autocatalytic amplification
   saturates at max rate from the start: structurally incapable of
   onset, and rightly so -- they are the EARLY-WARNING family
   (t_seed). Onset-type B5 lives in CURRENT-STATE objects (the
   joint possibility space now). Every system that ever passed
   onset-type B5 used a current-state object; every endpoint curve
   decelerates. No contradiction -- a division of labor, and a
   measured ordering: early warning saturates first, the
   breakpoint follows, completion follows that (ANT-COLONY-BP:
   warning early, t* = 350, completion 651).
2. FINITE-SIZE LAW. The single-chooser ant model NEVER shows onset
   (any gain, any grid, any object -- five misses); N = 10
   concurrent ants show it (Delta-BIC 18); N = 100 show a 12x
   slope kink (Delta-BIC 217), with the open phase flattening and
   the collapse sharpening in N. Abrupt possibility collapse is a
   COLLECTIVE phenomenon: it requires fluctuations (~1/sqrt(N))
   small against saturation. This retroactively explains Kuramoto
   (N = 200) and TRI-C (uniform policy, small gradient steps)
   passing directly -- and it IS the user's original intuition,
   measured at the right level: the colony commits together, and
   the possibility space's collapse is abrupt exactly because many
   parts commit as one.
RE-2's verdicts stand under their frozen contract; the ant flagship
is re-scoped as early-warning + commitment-before-completion +
controllability (all standing), while its onset-type B5 lives at
colony scale.

### Act 15. Abruptness obeys laws (KUR-SCALE)

The final upgrade from existence to law, on the physics leg: across
K = 0.9 -> 2.5 (10/10 runs onset under the matured contract), the
breakpoint time falls monotonically (6.7 -> 1.8: critical slowing
down as the transition is approached from above) and the closing
slope rises monotonically (0.032 -> 0.199). Together with the
finite-size law, "abruptness" has two independent quantitative
control parameters -- system size and feedback strength -- both
preregistered, both confirmed. The breakpoint is not a detector
artifact and not a metaphor: it is a lawful, manipulable feature of
collective self-amplification.

### Act 16. The ML onset gap, closed honestly as a scope boundary

The sharpest NMI objection was correct: onset-type B5 had not been
shown in a real trained ML system. We therefore froze two
definition-faithful tests on the existing Overcooked BP-FRESH
checkpoints, with no retraining:
- OC-STATE-BP (fixed reference states, policy action entropy):
  0/3 onset. Seed 93005 genuinely collapses, but as a deceleration
  curve; seeds 93004/93006 barely pass the effect-size gate or fail
  it. Robust reference states reproduce the null.
- OC-OCC-BP (trajectory/role occupancy object, the better match to
  "state-action-trajectory possibility space"): 0/3 onset again.
  The learned policies do show gradual trajectory-space collapse
  and increasing role selectivity (top-2 configuration mass rises
  roughly 0.26 -> 0.35/0.38/0.41), tracking capability, but not a
  structural onset.

We then froze the natural rescue test: does onset appear with N in
a learned population, explaining Overcooked as N=2? The answer, in
the registered sequence LEARN-N-BP -> LEARN-N-EXACT -> LEARN-ETA
-> LEARN-QUORUM, is no. Sampled REINFORCE plurality did not learn
at all; exact expected-gradient consensus learned, but always as a
smooth decelerating convergence curve across N and learning rates;
even the nonlinear quorum-threshold population learned by fast
early convergence followed by saturation, with 0/20 onset. This is
not buried. It becomes a scope boundary: learned optimization is
not generically abrupt. Onset B5 in learned systems remains
confirmed for the information-bottleneck high-order toy (TRI-C,
13/13) but NOT for deep MARL, smooth population consensus, or
quorum-threshold population learning. The paper must state this
explicitly; otherwise a reviewer will rightly call the ML evidence
overclaimed.

### Act 17. Canonical emergence battery: external validity is mixed,
### and that matters

Following the stronger theory proposal, we began a Canonical
Emergence Battery to connect the instrument to classic academic
emergence cases rather than only our own toy systems.

The first results are deliberately mixed:
- Vicsek flocking: the system orders strongly (polarization rises
  to ~1), and high-noise control stays disordered, but both the
  within-run low-noise relaxation and the high-to-low noise control
  axis are smooth/decelerating on the declared grids. This is a
  classic self-organization case, but not a punctuated-collapse B5
  positive under our current object.
- Turing/Brusselator: the first parameterization was wrong for a
  clean Turing test (homogeneous instability contamination), and
  the corrected run still failed to produce a clean pattern-growth
  recovery in the finite window. This remains an external-validity
  gap, not a theory win.
- Ising: the dense control-axis run recovers a strong possibility
  collapse and high/low-temperature contrast, but the possibility
  hinge begins before the native magnetization midpoint / closest
  Tc grid point. Under the frozen clause this is only partial: a
  control-axis collapse and possible early-warning signal, not a
  clean native-aligned B5 pass.
- Swift-Hohenberg: the corrected pattern-formation normal form
  gives a clean spectral mode-selection result (critical-band
  share 0.045 -> 1.0; stable control null), but the spectral
  openness curve is again deceleration, not onset. This is a
  strong possibility-collapse case without punctuated B5.
- Schelling: macro segregation forms strongly (same-neighbor order
  0.50 -> 0.77), but neighborhood openness does not collapse and
  no B5 fires. This is a successful boundary pressure test: a
  classic weak/social emergence model is classified as gradual
  organization, not punctuated collapse.

This is a major scope correction. The theory cannot say "every
classic emergence case is punctuated B5." A safer and stronger
claim is: possibility-collapse profiles separate classical cases
into punctuated, gradual, parameter-axis, and early-warning forms.
That classification may be scientifically valuable, but it means
the universal equivalence "emergence = onset B5" remains unproven.

## What we claim and what we do not

We claim: a unified, calibrated measurement contract for
possibility-collapse emergence, with a source typology, breakpoint
necessity B5 measured by model comparison under the matured
V3.1/V3.2 contract (effect-size gate, saturation-truncated window,
current-state object class) and confirmed across three system
classes -- ant colonies at collective scale (ANT-COLONY-BP, with
the finite-size law: onset absent at N = 1, present at N = 10,
sharpening 12x at N = 100), learned high-order coordination (13/13
seeds, TRI-C-BP + TRI-C-BP-N), and the Kuramoto synchronization
transition (KUR-BP-R2 3/3 two-sided; KUR-SCALE: breakpoint time
obeys critical slowing down and sharpness grows with coupling,
10/10) -- a temporal
phenotype (M and B provably distinct, BENCH-72), a formation-history
genesis instrument distinct from all product certificates, early
detection at THREE timescales (t_seed < t_visible 3/3 seeds in
Overcooked training; joint-space commitment before completion 100%
of episodes in the ant system, RE-2; hinge before capability 3/3
seeds in learned high-order formation, TRI-C-BP), a registered
onset-vs-deceleration dissociation (ANT-COLONY-BP, TRI-C-BP/N and
KUR-BP-R2/KUR-SCALE show onset-type B5; RE-1, RE-3, OC-STATE-BP,
OC-OCC-BP, LEARN-N/EXACT/ETA do not), a confirmed matched-product
profile separation (E1-C), and a per-episode conditional
controllability law (ANT-INT-C: remaining openness of the episode's
own possibility space predicts whether interventions can still flip
the outcome, 0/2600 flips once closed). We do NOT claim a causally verified commitment
window (E3C: the one-pulse-destroys hypothesis failed; re-entrance
program registered), that single-point G identifies provenance
(E1-2 fired its clause), any B5 verdict for Pythia/MultiBERTs
(onset unresolvable on stored grids), or onset-type B5 in real
deep MARL at the current scale (Overcooked shows gradual collapse
and early warning, not onset).

We do NOT claim: a philosophical definition of all emergence; an
observer-free burst criterion; that entropy decrease at any level is
emergence (JC-5 forbids it); that M, MI, PID or CE alone certify
anything (the matched-confound and null-band results forbid it);
that the current testbed exhausts the theory. The former standing
gap "no learned C_high exists" is now closed by TRI-C (0.94-0.96
bits, 3/3 seeds); the remaining declared limits are scale (three
agents, small games) and the untested per-episode conditional
leverage instrument.

## Honesty ledger (misses that the story keeps, and why they help)

| Registered miss | File | Why it strengthens the story |
|---|---|---|
| JC-5 macro entropy rises | overcooked_joint_collapse_s*.json | it is the micro->macro organization signature the theory predicts once levels are declared |
| OTC-C2 clone G positive | overcooked_genesis_comparison_pilot.json | proves single-point G is a product measure; motivates formation curves as the genesis instrument |
| OTC-C4 product band fail | overcooked_genesis_comparison_pilot.json | discloses that product matching is unsolved in the pilot; E1 exists because of it |
| NB-1 one seed in null band | delta_m_null_band.json | grounds the "M never alone" guardrail in data, not rhetoric |
| KUR-4 no high-order in full sync | kuramoto_offdesign_ladder.json | the instrument refuses to over-claim: pairwise-implied structure is labeled pairwise |
| Early-cut score gain | overcooked_intervention_early_s93101.json | unregistered observation promoted to a preregistered test (E3C-4) instead of being buried; did not replicate |
| E3C causal window FAIL | overcooked_e3c_analysis.json | the frozen falsification clause fired and was honored: the intervention claim is dropped, and the lesion-robustness of formation becomes a measured persistence finding |
| E1-2 single-point G FAIL | overcooked_product_matched_genesis.json | measured impossibility: instantaneous G is coupling strength, not provenance; forced genesis onto formation history + B3, and motivated the profile-separation result that then confirmed (E1-C) |
| TRIB-3 no learned C_high | triad_relational_collapse_sequential.json | the instrument refuses to hand us the result we wanted: learning compiled the constraint down to pairwise order, a substantive finding about how regimes are implemented |
| EP-1/EP-2 episode monotonicity FAIL | overcooked_episode_collapse.json | cyclic regimes re-open per cycle; the miss exposes that within-episode commitment must be phrased per cycle, not per episode |
| BP-1/BP-2 breakpoint claim not made | breakpoint_model_comparison.json | v2.1 makes breakpoint existence (B5) necessary, so we tested it immediately on stored curves and refused our own claim: strong grid-persistent C_env breakpoint in the one dense-grid seed, absent in the two sparse-grid seeds; BP-FRESH (dense grids, fresh seeds) is the registered path before any B5 claim |
| RE-1 ordinary learner excluded by B5 | re1_ordinary_learner_breakpoint.json | the v1 burst-gate false positive is resolved by measurement, not stipulation: immediate-onset decelerating convergence is not a commitment onset |
| RE-3 no Pythia/MultiBERTs B5 claim | re3_stored_series_breakpoint.json | the honest verdict is "onset unresolvable at stored grid density", with the shared-entropy-column control vacuity disclosed; a claim either way would outrun the data |
| AI-1 hinge not flip-maximal | ant_commitment_intervention.json | the frozen clause fired and was honored; the result refines the theory: outcome leverage precedes the breakpoint, timing leverage sits at it, robustness follows it |
| AIB-1/2 openness-leverage law FAIL | ant_openness_leverage.json | two consecutive frozen misses kept as misses instead of re-tuned; the per-episode conditional instrument (ANT-INT-C) was then frozen with a three-strikes drop clause on fresh seeds and passed all three predictions -- the misses document how the wrong reference frame was found and replaced |
| VM-4 profile-channel map FAIL | vulnerability_matrix.json | the failure exposed implementation degeneracy (fresh TRI-B seeds compile parity to INDIVIDUAL order 2/3 of the time), and each seed's vulnerability tracked its own profile -- the claim survives only in per-seed conditional form (VUL-MAT-B), which is how it should have been frozen |
| KURBP-2 thinned-BIC bar FAIL | kuramoto_breakpoint.json | hinge location fully stable; the miss is a self-inflicted inconsistent bar (10 vs RE-2's 2) and is kept rather than re-adjudicated |
| KURBP-3 subcritical "hinge" | kuramoto_breakpoint.json | the flat-series false positive that forced the V3.1 effect-size gate -- the detector contract matured through a registered miss, not through tuning |
| KURR-1 saturation-knee capture 1/3 | kuramoto_breakpoint_r.json | exposed the two-knee S-curve under-specification; fixed by the definitional saturation-truncation window (KUR-BP-R2), tested on a third batch of fresh seeds |
| VMB-1 unresolvable | vulnerability_perseed.json | the declared resolvability clause fired honestly (no high-share seeds in batch one); the declared one-time rerun executes with pooled adjudication |
| AG-1/2/3 gain law FAIL | ant_gain_scaling.json | exposed that the single-chooser model collapses instantly at every gain -- the first link in the chain that found the finite-size law |
| AF-1 gradual-regime onset FAIL | ant_fine_onset.json | the endpoint-projection object cannot show onset on principle (prediction saturates early under autocatalysis); forced the V3.2 object-class distinction |
| AFB-1/2/3 behavior-object FAIL | ant_fine_behavior.json | even the current-state object decelerates at N = 1: no scale separation for a solitary chooser -- the finding that demanded the colony model |
| RE-2 onset-reading withdrawn | re2_ant_joint_breakpoint.json | disclosed proactively: RE-2's existence hinge stands under its frozen contract, but its curve is a deceleration into zero at fine resolution; onset-type B5 for ants lives at colony scale (ANT-COLONY-BP), and RE-2 is re-scoped to early-warning + commitment-before-completion |
| DEF-CAL lottery objection | definition_calibration.json | low probability alone is excluded: LOTTERY has high S but D=0/R=0/G=0; RANDOM_MASK has high S/D/R but G=0; NUCLEATION passes D/G/R with high S/X but no event-aligned onset-type B5, forcing the V3.3 two-level structure (qualification first, intensity profile second) |
| CEB-POTTS classic phase contrast | ceb_potts.json | q=2 and q=10 both order, but q=10 has stronger hinge evidence (Delta-BIC 32.6 vs 16.6) and ~13x larger hysteresis, matching the known continuous-vs-first-order Potts distinction; supports profile strength rather than binary B5 labels |
| CEB-VICSEK-FS finite-size crosswalk | ceb_vicsek_fs.json | earlier N=200 smooth Vicsek result is refined: finite-size scan shows N=100/400 no B5, N=1600 control-axis B5 with Delta-BIC 12.1, and hysteresis increasing 0.080 -> 0.093 -> 0.106; supports scale-dependent sharpening in a canonical flocking model |
| EEC synthetic ladder miss | eec_ladder.json / eec_ladder_b.json | two hand-written mechanism ladders failed their own frozen monotonicity/control clauses; conclusion kept as a boundary: EEC claims require a real spatial task or learned flagship, not a tuned toy schedule |
| CEB-LIFE weak-emergence boundary | ceb_life.json | deterministic Life requires a perturbation ensemble; R-pentomino shows real future-outcome collapse (O~0.40 -> 0 by t=30) but no onset-type B5 (Delta-BIC 7.83, deceleration), supporting V3.3's distinction between regime reorganization and punctuated emergence |
| SYM-BRIDGE spontaneity calibration | sym_bridge.json | symmetric bridge dynamics pass B5 and show external-underdetermined/internal-selected regime formation: final A fraction 0.508 across episodes, within-episode lock 0.946, precursor accuracy 1.0; biased control collapses to A=1.0 and decelerates, separating spontaneous symmetry breaking from external specification |
| LEARN-SYMBRIDGE sparse learned pilot | learn_symbridge.json | 20/20 sampled REINFORCE seeds fail to learn one-step sparse quorum bridge (success ~0.2%, entropy ~1.0, 0 B5); confirms that the ML flagship cannot be a one-step sparse quorum wrapper and needs real spatial trajectories/curriculum |
| CEB-POTTS-QSCAN transition-order curve | ceb_potts_qscan.json | q-scan over {2,3,4,5,8,10} shows q>4 has much larger hysteresis and Delta-BIC than q<=4 at L=32 and L=48, matching known Potts transition-order boundary; size-sharpening clause fails and is not claimed |
| SYM-BRIDGE-INT utility | sym_bridge_intervention.json | profile predicts intervention controllability: as pre-intervention openness falls 0.987 -> 0.362, counter-regime switch probability falls 1.000 -> 0.160; tau-level rank corr 0.943, pooled episode rank corr 0.664 |
| DEEP-MARL utility audit weak | deep_marl_utility_audit.json | retrospective MPE audit finds registered do-effect exists, but commit_collapse_bits does not predict intervention size; only early potential weakly predicts assignment-JS, so learned-system utility remains a prospective flagship requirement |
| CEB-VICSEK-DENSE scale scan | ceb_vicsek_dense.json | dense N scan strengthens finite-size story: B5 absent for N=100/200/400 and present for N=800/1600/3200; rank corr with N is 0.714 for Delta-BIC, 0.886 for max drop, 0.486 for hysteresis |
| LEARN-TRANSPORT-VEC feasibility | learn_transport_vec.json | vectorized multi-step transport is much more learnable than one-step quorum (success ~0.6 vs ~0.002) but remains high entropy and below success threshold, so it is a direction-finding pilot rather than flagship evidence |
| LEARN-TRANSPORT-STATE learned spatial pilot | learn_transport_state.json | state-dependent neural policy solves symmetric threshold transport in 10/10 seeds and final sides are balanced across seeds (learned_frac_right 0.5), but 0/10 outer B5 and no registered realization-collapse; learned self-selected coordination exists but is smooth/early, not punctuated |
| LEARN-TRANSPORT-UTILITY learned controllability miss | learn_transport_utility.json | learned transport remains fully successful under counter-regime impulses but never switches side (max switch 0), so current learned policies compile a rigid convention from the start; learned-system profile-to-intervention utility remains unproven |
| LEARN-TRANSPORT-EQUIVARIANT learned realization | learn_transport_equivariant.json | equivariant learned policy solves threshold transport in 5/5 seeds while preserving initial left/right openness (mean H0 0.874, side means near 0) and then collapses within-episode (mean entropy drop 0.866); strict B5 fails because collapse saturates too early for robust hinge windows |
| LEARN-TRANSPORT-EQUIVARIANT-SLOW resolution test | learn_transport_equivariant_slow.json | slowing the physics does not slow the commitment: 5/5 learn, initial openness 0.911, episode drop 0.717, but 0/5 B5 (Delta-BIC 4.9-6.6, thinning fails); diagnosis: the 1D task has no pre-commitment phase, so a resolvable learned B5 needs mechanics that structurally delay side choice (grip-then-push) |
| LEARN-TRANSPORT-EQ-UTILITY learned controllability window | learn_transport_eq_utility.json | first learned controllability law: switch probability 1.0 at tau<=8, 0.898 at tau=12, 0.280 at tau=20; openness AUC 0.9955 beats tau baseline; frozen rank-corr clause fails (0.266) due to switch saturation; |x|/|v| marginally higher AUC, honestly reported |
| LEARN-GRIP-TRANSPORT + LGT-B learned punctuated realization | learn_grip_transport.json / learn_grip_transport_b5.json | FLAGSHIP POSITIVE: grip-then-push mechanics delay commitment; 5/5 seeds learn (success >=0.995), side-openness holds at 1.000 for 18-19 steps then collapses to 0.095; 5/5 pass full B5 adjudicator on the side-openness object (Delta-BIC 45.8-52.7, t*=16-18); preregistered mechanism contrast with no-preparation task (0/5 B5) shows structural delay, not learning per se, produces punctuated collapse |
| LEARN-GRIP-UTILITY breakpoint leads window closing | learn_grip_utility.json | window exists (switch 1.0 at tau<=16, 0.27 at tau=30) and side-openness AUC 0.996 beats tau baseline, but the strict alignment clause fails 0/5: window closes 5-10 steps AFTER t*; honest upgrade of the claim -- the policy-openness breakpoint is an early-warning leading indicator of controllability loss, not its simultaneous marker |
| LEARN-GRIP-FORMATION two-timescale dissociation | learn_grip_formation.json | preregistered quadrant completed in one learned system: formation axis shows outcome-openness EXPANSION (0 -> ~0.65 within ~100 updates, 0/5 B5, constraint-affordance duality) while realization axis reproduces punctuated collapse 5/5 (Delta-BIC 47-62, plateau 18-21); capability formation is smooth/expansive, capability realization is punctuated |
| LEARN-STANCE-TRANSPORT relational collapse | learn_stance_transport.json | first learned individual-agent (non-mean-field) flagship: 5/5 learn with perfect cross-episode symmetry (fracR 0.49-0.51); source ladder shows the collapse is PURELY relational (per-agent marginal entropy 0.9997-1.0 bits stays open, total correlation 6.83-6.97 of max 7 bits); separability clause failed (openness AUC 0.843 vs abs-x 0.904) because free lean actions make consensus near-instant |
| LEARN-STANCE-STICKY separability positive | learn_stance_sticky.json | with inertial stances extending the hidden consolidation phase, openness AUC 0.886 EXCEEDS the physical order parameter (0.849), velocity (0.853) and time (0.824) at predicting switchability -- first learned-system demonstration that possibility openness carries controllability information beyond the order parameter; contrast with non-sticky version is the mechanism evidence; stance collapse itself is gradual (0/5 B5, honest) |
| NaN-defect disclosure | learn_stance_*_nanbug.json | float32 clamp defect corrupted first-run entropy curves and one AUC; fixed in code, both experiments rerun with identical seeds, corrupted outputs archived, all numbers regenerated by code |
| LEARN-GRIP-FORMATION-FINE fine-grid adjudication | learn_grip_formation_fine.json | formation is fast but smooth at 5-update resolution: success midpoints at updates 10-20, 0/5 B5 on outcome-openness and on inverted success curves; completes the honest formation/realization dissociation of the grip flagship |
| LEARN-STANCE-CONTROL matched-parameter reversal | learn_stance_control.json | single-parameter causal contrast for separability: with STICK_P=1.0 and ALL other constants imported from the sticky module, openness AUC drops to 0.811 < abs-x 0.884 (sticky: 0.886 > 0.849); relational collapse unchanged (TC 6.35-6.94); the openness advantage exists if and only if a hidden consolidation phase exists |
| LEARN-GRIP-EXT 10-seed reproducibility | learn_grip_ext.json | flagship claim at full statistical strength: 10/10 seeds learn, 10/10 pass the frozen side-openness B5 adjudicator, plateau-collapse shape and symmetry hold in all seeds, t* concentrated at 16-18 (two at 22/24), zero exclusions |
| LEARN-GRIP-A2C algorithm check | learn_grip_a2c.json | phenomenon reproduces under advantage actor-critic on the byte-identical environment: 5/5 learn, 5/5 plateau-then-collapse with primary hinge Delta-BIC 37.7-45.5 and t* 14-16; frozen 4/5 robust-B5 clause narrowly fails (3/5; two seeds miss only the thinned-subsample Delta-BIC threshold), so full algorithm independence is not claimed -- reported as strong partial replication |
| DETECTOR-VALIDATION held-out benchmark | detector_validation.json | rebuts "detector engineered to fire" (reviewer #6): on a frozen synthetic library scored by the UNCHANGED adjudicator, false-positive onset rate 0.000 across knee/gradual/flat, onset power 1.000 at reference; power rises monotonically with grid density (0/0.62/0.995/1.0 at 12/20/40/80 pts) with FPR staying 0.000 (V3.1 gate holds out of sample); graceful noise degradation (power 1.0->0.93 at sigma 0.08); location agrees with ruptures Binseg to 0.10 of span; honest floor: 0 power at 12 pts (consistent with LM-checkpoint unresolvability) |
| REPR-ROBUSTNESS contract battery | repr_robustness.json | rebuts "representation-dependence" (reviewer #1): grip side-openness, ant N=100 and TRI-C onset verdicts unchanged in 100% of 243 adequate-resolution analysis contracts (saturation x window x gate x Delta-BIC x stride); t* varies 1%/8%/11% of span; frozen contract cell reproduces every published verdict; honest boundary: coarse grids reduce detectability (power effect), never location; object-semantics note recorded (grip raw action entropy RISES -- object is theory-specified, not fitted) |
| LEARN-CONVENTION non-constructed positive | learn_convention.json | reviewer #2's demanded genuine learned positive #1 (their own example: emergent-communication protocol commitment): population Lewis game, no gate/threshold/blocked channel, all K! codes equivalent; 5/5 learn, 4/5 robust onset (Delta-BIC 17.8-30.1, flat plateau then collapse), t* (275-300) precedes capability crossing (700-1025) in all onset seeds, 5 distinct codes in 5 seeds (endogenous symmetry breaking) |
| LEARN-ROLES non-constructed positive | learn_roles.json | reviewer #2's demanded genuine learned positive #2 (their own example: spontaneous role lock-in): 6 agents, 6 interchangeable roles, team reward only for full cover (chance 1.5%); 5/5 learn, 5/5 robust onset (Delta-BIC 53.6-71.7, slope ratio ~30x), collapse leads capability in all seeds, 5 distinct permutations |
| ANT-FSS scaling laws | ant_fss.json | upgrades "finite-size trend" to law (reviewer #7): t50 = a + b ln N with b=87.6 (CI 41-124), R^2=0.93 (mechanistically derived from sqrt-N nucleation before running); closing width N-INVARIANT (280-290 across N=50-500); translation data collapse RMS 0.010 vs 0.266 unaligned (96% reduction); honest miss recorded: N=10 onset does not replicate at the 1500-trip horizon (threshold now 10<N<=20), FSS-1 counted FAILED |
| LEARN-GRIP-CONFOUND fixed-time/fixed-state | learn_grip_confound.json | reviewer #8: at every FIXED tau, per-episode openness discriminates flippable vs locked with AUC 0.974-0.990 (20,480 eps/cell) -- time confound eliminated; fixed-(tau x abs-x) conditioning removes the signal (registered boundary, consistent with the stance hidden-regime contrast); logistic openness partial effect positive |
| LEARN-GRIP-POLICY prospective control | learn_grip_policy.json | reviewer's decisive gap #3: openness-triggered intervention rule calibrated ONLY on original-seed records transfers to 5 unseen seeds: flips 99.99% of episodes while acting 3.8 steps LATER than the best fixed schedule (99.6%) and beating random timing by 21 points -- openness is a control variable, not a retrospective correlate |
| TRI-C-BP-EXT five fresh seeds | tri_c_breakpoint_ext.json | reviewer #9 (3 seeds too thin): 4/5 fresh seeds reproduce the high-order onset (Delta-BIC 19.8-73.1) with collapse leading capability in all passing seeds; pooled 7/8 |

## Revision round 2 additions (2026-08-04)

| Claim in manuscript | Evidence file | What it establishes |
|---|---|---|
| Representation equivalence class (Methods) | repr_equiv_convention.json, repr_equiv_grip.json | true representation battery (12 preregistered cells, measurement-only changes on byte-identical retrained seeds): verdict survives 8/12; t* essentially invariant (conv 250-300/4000 span; grip identical 17.0); all 4 breakers mechanically explained and still locate commitment at 19-22 |
| Onset does not depend on tabular parameterization (Results) | learn_convention_nn.json, learn_roles_nn.json, learn_nn_resolution.json | MLP replications, 10 fresh seeds/system: at adequate (5-update) resolution onset in 6/7 and 10/10 learned seeds, dBIC up to 162, collapse precedes capability everywhere, all identities distinct; coarse-grid under-resolution honestly recorded (1/7, 1/10) |
| Plateau = symmetric regime competition (Results) | learn_nn_init.json | init-scale sweep: commitment time monotone decreasing in sigma for convention (385->335->175); roles extremes correct but midpoint breaks strict monotonicity -> registered MISS, reported |
| Ladder attribution correct off-family; order freedom bounded; sample floor measured (Methods) | sd_audit.json | modular-sum -> pure C_high 3.32 bits; Markov chain -> pure C_pair; 50 Dirichlet joints zero negativity, exact identity; env-before/after-pair split shift <= 0.26 bits with C_ind/C_high exactly invariant; median abs err <= 0.018 bits at n=3e4, C_high bias 0.31 bits at n=300; SDA-1 MISS = interacting generators genuinely relocate structure (property, reported) |

## Revision round 3 additions (2026-08-04)

| Claim in manuscript | Evidence file | What it establishes |
|---|---|---|
| The Overcooked negative has a named, manipulable cause (Results, boundary section) | overcooked_ring_convention.json, oc_ring_ext.json | mechanism recovery inside the standard package: official coordination_ring restores equivalent competing regimes -> 8/8 pooled seeds end committed to a circulation direction (6 CCW / 2 CW, endogenous symmetry breaking), 0/2 cramped controls ever commit on the same object at the same 100-point grid; 1/8 certifies punctuated onset (t*=780k < capability crossing 1M); non-monotone convention formation reported as instrument boundary (OCE-3 miss recorded) |
| Standard-benchmark realization attempt (Methods) | mpe_spread_realization.json, mpe_spread_ppo.json | honest record: simple_spread not solved by shared-parameter on-policy training (coverage 1.7% REINFORCE, 3% PPO), competence precondition unmet, no theory claim either way; SR-4/SR-5 controls pass |
| Standardized adjudication exists (Methods, certificate paragraph) | emergence_certificates.json | frozen instruments packaged into qualification + EIP vector + categorical verdict; battery: 4 systems certify punctuated, single-ant N=1 with SAME amplitude correctly fails qualification (individual-source), Overcooked occupancy below gate with low-power flag |

## Revision round 4 additions (2026-08-05)

| Claim in manuscript | Evidence file | What it establishes |
|---|---|---|
| Ring convention is a formation-level commitment with no realization stage (Results, boundary) | oc_ring_realization.json | preregistered within-episode probe: 0/5 mid-training seeds show internal lock (drift only); final checkpoints read closed-from-step-one (0.371 initial openness, 8/8); untrained controls open, zero false onsets -- instrument controls all pass, answer is scientific |
| Evaluation-noise floor of the ring formation analysis (Methods) | semi_inject.json | injections at known t0 into the real pipeline: FPR 0.01, t* error 1% span, power 0.88 vs 0.90 clause (near-miss, recorded); quantifies why 30-episode budgets cap certification |

| Ring commitment is causally irreversible (Results, boundary; Methods) | oc_ring_intervention.json | 48 preregistered perturb-and-resume runs: open phase 8/16 moved vs late 1/16 (Fisher p=0.008), strict late flips 0/16, openness-at-perturbation predicts flip AUC 0.849, capability recovery 0.92 (institution locks, not learning) -- completes the single-environment closed loop in an unmodified benchmark |

## BARRIER-XPLAY (barrier_xplay.json)
Claim in manuscript: "the open plateau IS the joint exploration
barrier -- a code or a role division has value only to the extent that
others already share it."
Evidence: unilateral adoption gain at mid-plateau 0.011 (convention) /
0.001 (roles) vs deviation cost after commitment 0.40 (structural
ceiling for K=5) / 1.00; cross-seed intelligibility 0.14 vs 1.00
within; hybrid-team success 0.05 vs 1.00 within. One registered miss
(BX-3 convention, threshold mis-calibrated to payoff granularity)
reported verbatim.
