# Sharp reviewer attacks and the revision plan

Drafted 2026-07-23. Purpose: anticipate the harshest technically
competent reviews of the possibility-collapse paper, and specify the
revision that answers each one WITHOUT abandoning the core story:

> Emergence is a spontaneous, selective and persistent regime-level
> collapse of the effective joint state-action-trajectory possibility
> space; collapse decomposes by source (env / individual / pair /
> high-order); its formation can show a commitment window that
> precedes visible capability and is causally load-bearing.

Each item: the attack (in the reviewer's voice), why it bites, the
answer we already have, and the revision required.

---

## R1. "Possibility collapse is just policy convergence renamed."

Attack: every RL run reduces policy entropy; every learner commits.
You have re-described ordinary training with new vocabulary.

Why it bites: our own JC data show C_individual dominates the ladder
(0.28-0.52 bits) -- most raw collapse IS individual contraction.

Answer in hand: the ordinary-learner control (N ~= -0.003, old
battery) passes burst but fails lower-order novelty; the ladder
separates marginal contraction from relational/environmental
channels; the claim never rests on total collapse alone.

Revision:
- State the minimum boundary (counterfactual reachability, structural
  breakpoint, non-hardcoded, persistence) as the DEFINITION, with
  total collapse explicitly declared insufficient.
- Report C_individual as the "ordinary learning" channel by default;
  headline results must be carried by C_env / C_rel / C_high and by
  the formation-history/intervention evidence.
- Add a matched "solo-agent" control: same PPO, same layouts, partner
  replaced by a wall/noop -- its ladder should be C_individual-only.

## R2. "Your source labels are contract-relative, not objective."

Attack: SD-4 shows that hiding E misattributes 0.265 bits of
environment collapse to the pair channel. So the decomposition depends
on what the analyst declares as environment. Different analysts,
different emergence types.

Answer in hand: SD-4 was preregistered BY US as a boundary statement;
the project's observer-contract stance already covers this.

Revision:
- Promote contract-relativity from a limitation to a stated principle
  (like frame-dependence in physics): every source claim carries its
  declared contract C = (phi, H, nu, E, H0).
- Add a sensitivity table: recompute the Overcooked ladder under
  plausible alternative E declarations (layout; layout x time-bin;
  none) and show which claims are stable across contracts and which
  flip -- claims that flip are reported as contract-local.

## R3. "The estimator does not scale and the testbed is a toy."

Attack: 36-cell joint tables with two agents prove nothing about
possibility spaces of size 10^n; C_high is identically zero in every
real-system result you show; entropy estimates from 60 episodes are
biased.

Answer in hand: none sufficient. This is a real gap.

Revision (experiment E4, mandatory before submission):
- A >= 3-agent learned system (LBF 3-agent, infrastructure exists in
  the workspace) with the full ladder including C_high via per-context
  pairwise-maxent IPF.
- Bias control: Miller-Madow or NSB correction + subsampling curves
  (report the ladder at 25/50/100% of rollouts; claims must be stable).
- A scaling statement: the ladder needs per-context joint tables over
  DECLARED coarse action/state alphabets, not the raw exponential
  space; state this as the scope of the instrument.

## R4. "Your collapse curves have the same grid-dependence you
exposed in Pythia."

Attack: you demoted burstiness partly because checkpoint thinning
flips verdicts (your own 9/9 result). Your formation curves use 8
checkpoints. The 'largest collapse interval' and the 'commitment
window' inherit exactly that fragility.

Answer in hand: partial -- we registered no burstiness claim on the
8-point grid, and the window was validated causally (INT-1/2), which
is grid-independent evidence.

Revision:
- Rerun the formation curve on one seed with a 2x denser grid
  (16 checkpoints) and show the window location is stable.
- Report J only with grid-sensitivity bands; the commitment window is
  defined by the INTERVENTION effect, not by the curve shape alone.
  Make this definitional move explicit in the text.

## R5. "The intervention margins are within noise, and your early cut
IMPROVED the system -- so the 'cut' is a curriculum knob, not a
surgical lesion."

Attack: one seed per condition; M differs by 1.4-2.4 points between
none/commit/late; early-cut score BEAT the control (36.8 vs 30.8).
Your intervention changes learning dynamics wholesale; attributing the
commit-window effect to "emergence formation" is post hoc.

Answer in hand: INT-1/2 directions were preregistered and the window
location was transferred from a different seed (93001 -> 93101),
which is nontrivial. The early-cut improvement is disclosed.

Revision (confirmatory E3, mandatory):
- >= 5 seeds per condition, plus a RANDOM-window condition and a
  dose-response arm (180k / 360k / 720k cuts at the commit position).
- Primary endpoint: seed-mean M and C_rel orderings with exact
  permutation tests; effect sizes with CIs, not point comparisons.
- Add formation curves DURING intervention runs (checkpoints saved)
  so the "delayed formation" reading of the early-cut effect is
  testable rather than narrated.
- Frame the early-cut result as a discovered phenomenon (curriculum
  effect of decoupling) with its own follow-up, not as noise.

## R6. "Your analytic ground truth is circular: the knobs are defined
in the same vocabulary as the ladder."

Attack: you build a generator whose four knobs are literally
individual/env/pair/high mechanisms, then show the ladder recovers
them. Diagonal dominance is guaranteed by construction.

Answer in hand: partially fair; the knobs are mechanistic (copying,
parity, tilt) rather than information-theoretic, but the kinship is
real.

Revision:
- Add OFF-DESIGN generators whose mechanisms are not expressed in
  ladder vocabulary: Kuramoto-style phase coupling, ant-recruitment
  dynamics, a small transformer-in-context task. Show the ladder
  assigns labels that match mechanism-level do-interventions.
- Add the cross-dissociation grid (large collapse without burst,
  burst without persistence, same curve different mechanism) as
  blind classification, scored against mechanism truth.

## R7. "The maxent ladder is Schneidman/Amari connected information;
total correlation splits are classic. Where is the novelty?"

Attack: C_total = C_ind + C_env + C_pair + C_high is a nested
maximum-entropy hierarchy, published decades ago. PID/O-information
already type multivariate structure.

Answer in hand: true about the algebra; the novelty is what the
ladder is APPLIED to and what is done with it.

Revision -- position explicitly in Related Work:
- The ladder algebra is inherited (cite connected information /
  maxent hierarchies / PID).
- The contributions are: (i) the ladder applied to counterfactual
  FUTURES from identical mid-episode states (real-vs-cut), not to
  static equal-time statistics; (ii) the formation-history curve over
  training (genesis, not product); (iii) the causally validated
  commitment window (no correlational measure provides this); (iv)
  the source-typed verdict replacing accept/reject gates.
- Add the baseline race (E5): MI/TC/O-info/CE computed on the same
  data cannot (a) separate copied from learned genesis, (b) locate
  the intervention window. Show it, don't assert it.

## R8. "All real-system evidence is one small benchmark."

Attack: two Overcooked layouts, tiny PPO nets, author-chosen macro
basins. External validity near zero.

Revision:
- E4 (3-agent LBF) as the second learned domain with the full ladder.
- Reuse the existing learned-basin discovery (k-means, LB-1..3
  passed elsewhere in the project) to remove the hand-basin
  objection for the Overcooked phi.
- Keep the analytic battery + classic exemplars (ant, Kuramoto,
  grokking bridges already in the workspace) as convergent validity,
  clearly labeled as such.

## R9. "M is confounded: any cooperative task loses score when you
ghost the partner, emergent or not."

Attack: M > 0 for the scripted pair too (+5.8 in your own comparison
pilot). So M measures partner-dependence, not emergent organization.

Answer in hand: correct, and already visible in our own data.

Revision:
- Never report M alone; define the null band from mechanism-matched
  scripted/marginal controls and report Delta-M above that band.
- The certificate is the CONJUNCTION (G with CI excluding the
  scripted null) AND (Delta-M above the scripted band) AND matched
  partner-action marginals (TV diagnostic already computed).

## R10. "The definition moved three times; the frozen battery
contradicts the final ontology."

Attack: the manuscript battery rejects common-cause systems; the new
theory calls them environment-mediated emergence. Reviewers will find
the contradiction in your own supplementary code.

Revision (writing, mandatory):
- Freeze definition v2.0 (multi-source typology) in one place; the
  narrative presents v1 (punctuated collapse) as the preregistered
  starting hypothesis and its partial falsification as a RESULT, in
  one dedicated section -- never as shifting definitions between
  sections.
- Re-run the four-quadrant battery with source-typed outputs
  (script -> externally-specified; common cause -> env-mediated;
  coincidence -> below reachability boundary; local feedback ->
  relational) and present BOTH labelings, explaining the upgrade.
- The old accept/reject outputs stay in the record as the v1
  instrument (the project's own falsification evidence).

## R11. "Micro collapse with macro expansion: you predicted macro
collapse (JC-5), it failed, and you declared victory. Unfalsifiable."

Attack: when entropy falls you call it collapse; when it rises you
call it 'macro capability creation'. Heads you win.

Answer in hand: JC-5 was preregistered in a specific direction and
reported as a miss; the micro/macro level distinction exists in the
theory text (ant bridge example) but was not encoded in the
prediction.

Revision:
- Declare the level structure ex ante: the collapse claim lives at
  the declared joint micro-action level; macro-branch entropy is a
  separate registered quantity with its own predicted direction PER
  SYSTEM CLASS (failure-dominated early regimes: macro entropy may
  rise with competence; success-saturated regimes: it should fall).
- State the falsification condition explicitly: if micro joint
  collapse does NOT occur while a stable regime forms (relative to
  the declared reference), the theory is wrong for that system.
  Keep JC-5 as a retained miss that motivated the level split.

## R12. "The intensity score is ad hoc."

Attack: E_k = (M_k B_k R_k)^(1/3) is numerology.

Revision: report the emergence profile as a VECTOR
(per-channel magnitude, temporal concentration with grid bands,
persistence, value). Provide the composite only in supplementary
material, axiomatized like the existing record axioms (nullity,
monotonicity, boundedness), or drop it.

## R13 (NMI-specific). "Interesting measurement theory. What does
machine intelligence gain?"

Attack: NMI wants a capability, a risk insight, or a control lever,
not a taxonomy.

Answer in hand: the commitment window IS a control lever (INT-1/2:
equal-budget cuts at the window damage the coupled regime most; the
early-cut curriculum effect shows constructive use too).

Revision:
- Make the applied claim explicit: monitoring C_rel/G during training
  gives (a) early warning of regime formation before reward moves
  (t_seed < t_visible, 3/3 seeds), (b) a window where targeted
  intervention prevents or shapes the regime (INT-1/2), and (c) a
  diagnostic separating environment-following from genuinely coupled
  teams at matched performance (comparison pilot).
- E6 transfer experiment: if endogenously formed regimes transfer to
  new partners/layouts better than copied ones at matched product,
  genesis becomes a deployable model-selection signal.

---

## Revision priority order

1. E3 confirmatory (multi-seed + random window + dose-response) --
   answers R5, the single most likely rejection reason.
2. E4 three-agent ladder with C_high > 0 in a learned system --
   answers R3/R8.
3. Definition v2.0 freeze + re-labeled four-quadrant battery --
   answers R10 (pure writing + one rerun, cheap).
4. E5 baseline race on stored data -- answers R7.
5. Contract sensitivity table + solo-agent control + Delta-M null
   band -- answers R1/R2/R9 (cheap, mostly stored data).
6. Dense-grid formation curve -- answers R4 (one training run).
7. Off-design generator battery -- answers R6.
8. Writing: level structure declaration (R11), vector profile (R12),
   applied framing (R13).

## Execution status (2026-07-23, second pass)

Definition v2.0 frozen in EMERGENCE_DEFINITION_V2.md; all items below
preregistered in V2_ALIGNMENT_PREREGISTRATION.md before running.

- R1: answered by decision (channel-reporting discipline + existing
  ghost-cut nulls; trivial solo control rejected, decision recorded).
- R2: DONE -- contract sensitivity table, CS-1/CS-2 PASS; relational
  label stable under contract refinement, hidden-E inflation ~7x.
- R4: dense-grid pipeline RUNNING (14-point grid, same seed 93001).
- R5: DONE, resolved NEGATIVELY -- E3C confirmatory (5x5 incl.
  random window, exact permutation tests) did not replicate the
  pilot: p(M commit < random) = 0.325, condition effects below seed
  noise; C_rel lowest in the UNCUT condition. The reviewer was
  right: the pilot margins were noise. The causal-window claim is
  dropped per the frozen falsification clause; the early-cut
  curriculum effect was also seed noise (E3C-4 no-replication).
  Retained honest finding: the final organization is robust to any
  single 360k-step feedback lesion -- formation is re-entrant, a
  persistence property. (overcooked_e3c_analysis.json)
- R6: DONE -- Kuramoto off-design battery, KUR-1/2/3 PASS, KUR-4
  preregistered may-miss confirmed as the mathematically correct
  attribution (3-way sync is pairwise-implied).
- R9: DONE -- Delta-M null band; NB-1 registered MISS (seed 93002 at
  2M inside the band): strengthens the "M never alone" guardrail,
  M-claims deferred to E3C seed means.
- R10: DONE -- v2 typology relabel of the frozen battery, RL-1 PASS
  (bit-for-bit preservation, sha256 verified).
- R11/R12/R13: writing tasks, now anchored in the v2 definition doc
  (level structure section 7, guardrails section 8, falsification
  section 9).
- R3 (3-agent C_high in a real system), R7/R8 (estimator lineage,
  breadth): next wave after E3C/DG complete.

---

## Execution status (2026-07-23, third pass -- wave 4)

- R3/R8 (scale, toy testbed): PARTIAL. TRI-B gives a real 3-agent
  LEARNED system whose regime is measured by the same ladder
  (C_pair ~= 1.01 bits, 3/3 seeds), plus the registered finding
  that learning compiles the parity constraint down to low order
  (C_high ~= 0.001, TRIB-3 may-miss confirmed). A learned C_high > 0
  system remains open and is said so in the paper. The simultaneous
  variant is a registered testbed failure (coordination trap), kept.
- R4 (grid dependence): CLOSED for the descriptive window (DG-1/2
  PASS on the 14-point grid); t_seed robustification still owed.
- R5 (intervention noise): CLOSED BY WITHDRAWAL. E3C fired its
  falsification clause; the causal window claim is out of the paper.
  INT-1/2 phrasing in earlier revisions must be read as superseded.
- R6 (circular ground truth): CLOSED. KUR off-design battery plus
  BENCH-72: blind recovery of source/M/J/t*/rho/V across 72 factorial
  cells with M-vs-J dissociation (M shape-invariant, J strictly
  ordered) and zero-collapse verdicts on revelation/metric-artifact
  pseudo-controls -- the Schaeffer-style metric artifact is now a
  measured negative control, answering the strongest form of R6/R4.
- R7 (maxent ladder novelty): CLOSED (E5 baseline race) and
  STRENGTHENED by E1-C: identical products, observational baselines
  matched, yet the profile separates the systems on fresh seeds.
- R9 (M confounded): CLOSED (null band NB + "M never alone"
  guardrail; one-seed-in-band miss retained).
- R13 (what does it buy): the deployable claim is now (a) early
  detection (t_seed < t_visible), (b) matched-product diagnosis of
  collapse composition (E1-C), explicitly NOT (c) window-targeted
  intervention (withdrawn). E6 transfer remains future work.
- New liability disclosed proactively: EP-1/EP-2 registered misses
  (episode-time monotone commitment failed; cyclic re-opening). The
  two-timescale claim is not made in the paper until a cycle-aligned
  variable passes a fresh preregistration.

---

## Execution status (2026-07-24, fourth pass -- V3 wave)

- R3 (learned high-order regime): CLOSED. TRI-C blocks low-order
  compilation with private iid cues; learning then builds the
  textbook irreducible XOR carrier (C_high 0.94-0.96 bits, pairwise
  ~0.0004 bits, 3/3 seeds; TRIC-1..4 all pass). Combined statement
  for the paper: C_high is not unlearnable, it is UNFAVORED (TRI-B
  + TRI-C). Contract relativity reproduced on the learned system
  (declaring E reattributes the collapse to C_env).
- R-abruptness (the v2 over-correction, GPT audit): CLOSED by V3 +
  the RE battery + TRI-C-BP. B5 (onset-type breakpoint by model
  comparison with persistence checks) holds on the ant joint space
  (RE-2, all four pass) and on the learned high-order channel
  (TRI-C-BP, Delta-BIC 38-73, 3/3 seeds), and correctly EXCLUDES
  the ordinary learner (RE-1 deceleration knee) while refusing a
  verdict on stored LM grids (RE-3, onset unresolvable). The
  onset-vs-deceleration dissociation is preregistered and passed,
  not post hoc.
- R5' (causal meaning of the breakpoint): PARTIAL, honestly scoped.
  ANT-INT: timing leverage maximal AT the hinge (AI-2), robustness
  after it (AI-3), but outcome leverage peaks BEFORE it (AI-1
  dropped as frozen). ANT-INT-B quantitative law failed as frozen
  (rho 0.62). The per-episode conditional form (ANT-INT-C, fresh
  seeds, declared three-strikes drop clause) then PASSED all three
  predictions: flip rate 0.000 -> 0.205 across per-episode openness
  bins, 0/2600 flips once the episode is closed, permutation
  p < 5e-5. The paper claims three differentiated causal quantities
  PLUS conditional controllability (the episode's remaining
  openness as the control variable); still no window-targeted
  control at training time.
- BP-FRESH: BPF-1 pass / BPF-2 fail recorded; learning-time B5 for
  Overcooked remains unclaimed -- the honest sentence stands.

---

## Execution status (2026-07-26, fifth pass -- breadth & robustness)

- R8 (breadth / "one toy family"): SUBSTANTIALLY CLOSED. B5 now
  confirmed in three mechanistically unrelated classes: stigmergic
  ants (RE-2), gradient-learned high-order coordination (TRI-C-BP
  + TRI-C-BP-N, 13/13 seeds, Delta-BIC 38-169), and the Kuramoto
  synchronization transition (KUR-BP-R2, 3/3 onset above
  criticality, 3/3 gated null below, C_pair carries 99.9% of the
  collapse). The two-sided Kuramoto result (breakpoint exactly
  above criticality) is the strongest external-validity card: B5
  tracks a textbook phase transition it was never tuned on.
- R-detector ("your hinge test is a dowsing rod"): CLOSED BY
  MATURATION IN PUBLIC. Three registered misses (KURBP-2 bar
  inconsistency, KURBP-3 flat-series false positive, KURR-1
  saturation-knee capture) each produced a definitional fix (V3.1
  effect-size gate; saturation-truncated window; unified thinning
  bar), and every fix was re-tested on FRESH seeds. The final
  contract is stated in EMERGENCE_DEFINITION_V3.md and the paper
  can print the full audit trail.
- R-seeds ("n=3 everywhere"): ADDRESSED where load-bearing: the
  learned high-order breakpoint now rests on 13/13 seeds across two
  preregistrations; the vulnerability program on 19 trained systems
  (3 VUL-MAT + 16 VUL-MAT-B).
- R-vulnerability: scoped to what held. Frozen results: dominant-
  channel destruction, dose monotonicity, env-vs-high-order attack
  dissociation (VM-1/2/3), exact immunity of low-relational-share
  seeds (VMB-2, 13/13 at 0.000 loss). The rank law is declared
  unresolvable (no high-share seed in 16 trainings) and NOT
  claimed; the sharp threshold pattern is descriptive only. New
  substantive finding stated descriptively: implementation
  degeneracy -- 13/16 fresh TRI-B trainings compile the parity
  constraint to INDIVIDUAL order.
- R-two-timescales: CLOSED for Overcooked in cycle-aligned form
  (EP-CYCLE all three pass; thin margins disclosed), on top of the
  ant episode timescale (RE-2, ANT-INT-C) and the learning
  timescale (t_seed, TRI-C-BP).
- Remaining declared open items: Overcooked learning-time B5
  (BP-FRESH mixed, unclaimed), LM onset resolution (RE-3
  unresolvable on stored grids), E3C re-entrance quantities
  (registered follow-up, compute-heavy), window-targeted
  training-time control (withdrawn, stays withdrawn).

---

## Execution status (2026-07-26 afternoon, sixth pass -- the
## abruptness laws)

- R-abruptness-completeness ("your breakpoint is existence-only;
  where are the laws?"): CLOSED. Two independent control parameters
  measured under frozen preregistrations: (i) FINITE SIZE --
  onset absent for a solitary chooser at any gain/grid/object
  (five registered misses, AG/AF/AFB), present at N = 10, 12x
  sharper at N = 100 (ANT-COLONY-BP, all three predictions pass);
  (ii) FEEDBACK STRENGTH -- breakpoint time obeys critical slowing
  down (t* 6.7 -> 1.8 monotone in K, 10/10) and closing slope
  rises monotonically with K (KUR-SCALE, all pass).
- R-ant-flagship ("your fine-grid ant curve contradicts RE-2's
  hinge"): PRE-EMPTED, disclosed by us first. RE-2's existence
  verdicts stand under its frozen contract; its onset-type reading
  is withdrawn (deceleration into zero at 1-trip resolution) and
  the onset lives at colony scale (V3.2). The re-scoping is in the
  preregistration ledger, dated, with the diagnostic chain.
- R-object ("basin entropy vs state entropy -- you switch objects
  when convenient"): CLOSED BY CLASSIFICATION, not by choice.
  V3.2 fixes the object classes ex ante for all future tests:
  current-state objects carry B5; endpoint projections are
  early-warning instruments, structurally deceleration-shaped
  under autocatalysis (proved by the AF/AFB misses). The division
  is falsifiable: an endpoint curve with a genuine onset, or a
  current-state collective curve without one, breaks it.
- The theory now answers WHY abruptness: it is the generic
  signature of collective autocatalytic closure -- fluctuations
  ~1/sqrt(N) must compound before closure accelerates -- and it
  vanishes smoothly as N -> 1. Emergence is abrupt BECAUSE it is
  collective; that sentence is now backed by preregistered data at
  every load-bearing word.

---

## Execution status (2026-07-26 evening, seventh pass -- NMI ML gap)

- Fatal NMI attack: "Your headline onset-type B5 never appears in
  a real trained ML system." RESULT: the attack is VALID and must
  be conceded, not spun. OC-STATE-BP (policy action entropy at a
  fixed reference set) and OC-OCC-BP (trajectory/role occupancy)
  both fail onset 0/3 on the existing Overcooked BP-FRESH
  checkpoints. They show gradual collapse and role selectivity, not
  a structural onset. This is now in the preregistration ledger and
  in the story, not hidden in supplement.
- Attempted learned finite-size rescue: LEARN-N-BP failed because
  sampled REINFORCE plurality never learned; LEARN-N-EXACT and
  LEARN-ETA learned strongly but always as deceleration/convergence
  (0/25 onset in the eta sweep). Conclusion: smooth optimization of
  a consensus potential is not abrupt emergence. The finite-size
  law is established for collective stochastic dynamics (ants) and
  physics (Kuramoto), but DOES NOT automatically transfer to smooth
  learned population optimization.
- Revised NMI-safe claim: onset-type possibility collapse is
  confirmed in (i) physical synchronization, (ii) collective ant
  colonies at scale, and (iii) learned high-order toy coordination
  with an information bottleneck (TRI-C, 13/13). Real deep MARL
  currently supports gradual trajectory-space collapse, source
  decomposition, early warning, and role selectivity -- NOT onset
  B5. This narrower claim is less flashy but far more defensible.
- Manuscript implication: do not submit as "we discovered the
  universal law of emergence in machine intelligence". Submit, if
  at all, as "a measurement theory of possibility-collapse
  emergence, with a sharp audit trail of where abrupt onsets do and
  do not occur". The negative ML results are a feature only if they
  are central to the paper's honesty and boundary conditions.

## Addendum to seventh pass (2026-07-27): nonlinear learned quorum

LEARN-QUORUM-BP was the last reasonable rescue for the NMI ML-onset
gap in this wave: a learned population with a nonlinear collective
threshold, exact gradients, N up to 50. It failed onset 0/20 while
learning/collapsing strongly. This strengthens the negative lesson:
learned optimization commonly produces deceleration/convergence,
not onset-type emergence, unless the task imposes a special
information bottleneck (TRI-C). A reviewer will accept the honesty;
they will not accept a universal ML-emergence claim.

Editorial implication: the manuscript can still be interesting for
NMI only if framed as a measurement theory with explicit positive
and negative boundary conditions. If framed as "we prove abrupt
possibility collapse is the signature of machine emergence", the
reviewer should reject it.

## Addendum to seventh pass (2026-07-27): lottery objection and V3.3

New fatal attack: "A lottery win is low probability and surprising.
If emergence is unexpected possibility collapse, why is a lottery not
emergence?"

Answer in hand: DEF-CAL directly calibrates this boundary. The
minimum gate is now endogenous regime formation, operationalized as
`D and G and R` before any intensity score:
- LOTTERY has high S (6.64 bits) but D=0, R=0 and G=0; it is excluded.
- RANDOM_MASK has high S, D and R but G=0; externally imposed
  collapse is excluded.
- NUCLEATION has high S/X and passes D/G/R; it qualifies as
  endogenous regime formation even though its event-aligned curve
  does not satisfy the old onset-type B5 clause.

Revision forced: stop presenting B5 as a universal existence gate.
The defensible ontology is two-stage:
1. qualification = endogenous persistent future-distribution
   reorganization;
2. intensity profile = M, J/B5, S, X, G, R, A, V plus source profile.

Reviewer implication: this resolves the lottery counterexample and
improves conceptual clarity, but it also weakens the strongest
equivalence thesis. The paper should claim a report-card / conditional
mechanism theory, not "all emergence is punctuated collapse".

## Addendum to seventh pass (2026-07-27): Potts crosswalk

New reviewer demand: "Show that your B5/J profile connects to a
known theoretical distinction, not just author-chosen examples."

Answer in hand: CEB-POTTS is the cleanest classic crosswalk so far.
The 2D Potts model has a known transition-order contrast: q=2 is
continuous, q=10 is first-order. Under the frozen control-axis
contract both systems order, but q=10 has a stronger collapse profile
and a much larger hysteresis loop:
- hinge Delta-BIC: q=10 32.6 vs q=2 16.6;
- post-hinge slope: q=10 -0.172 vs q=2 -0.129;
- hysteresis: q=10 0.771 vs q=2 0.059, about 13x.

Revision implication: present B5/J as a graded punctuatedness
dimension, not a binary emergence label. Finite q=2 can still show a
hinge on a finite grid; the decisive evidence is that the profile
orders the known continuous-vs-first-order distinction correctly.

## Addendum to seventh pass (2026-07-27): Vicsek finite-size rescue

Earlier attack: "Vicsek is a canonical flocking emergence model, but
your N=200 run was smooth; classic emergence contradicts B5."

Answer in hand: CEB-VICSEK-FS re-ran the control-axis contract across
N={100,400,1600} with matched cooling/heating scans. The earlier null
is now scoped as small/medium finite-size smoothing, not a decisive
contradiction:
- N=100/400: no control-axis B5 (Delta-BIC 1.34 and -3.17);
- N=1600: B5 becomes resolvable (Delta-BIC 12.11, onset_type true);
- hysteresis increases with N: 0.080 -> 0.093 -> 0.106.

Revision implication: Vicsek supports the conditional finite-size
story, but modestly. The effect is weaker than Potts q=10 and should
not be oversold; it is evidence for scale-dependent sharpening in a
canonical flocking system, not a universal claim that every Vicsek
run is punctuated.

## Addendum to seventh pass (2026-07-27): EEC ladder toy failure

New reviewer attack: "Your EEC mechanism ladder is a hand-written toy
that can be tuned until it gives the desired hinge."

Answer in hand: the attack is valid, and our own preregistered toy
ladder confirms the danger. EEC-LADDER and EEC-LADDER-B both failed
their frozen control clauses. The first produced non-monotone profile
strength; the second still gave a spurious B5 in the smooth scheduled
control and failed the threshold / anti-shortcut conditions.

Revision implication: do not use the hand-written EEC ladder as
positive evidence. Its role is methodological: it shows why the
machine-intelligence flagship must be a real spatial collective task
with local observation, geometry, sparse reward and no scripted
probability schedule. A reviewer will accept the negative toy result
as discipline; they will reject it if presented as proof.

## Addendum to seventh pass (2026-07-27): Life boundary test

New reviewer attack: "Game of Life is a canonical weak-emergence
case. If it does not show your B5, your definition excludes classic
emergence."

Answer in hand: CEB-LIFE makes this a boundary, not a hidden
contradiction. Because Life is deterministic, exact future entropy is
zero unless a perturbation ensemble is declared. Under the frozen
single-cell perturbation contract, seeded BLOCK/BLINKER/GLIDER are
low-G exemplars. R-PENTOMINO does show future-outcome collapse
(openness about 0.40 -> 0 by t=30), but not onset-type B5
(Delta-BIC 7.83, onset_type false).

Revision implication: this strongly supports V3.3 and weakens the old
universal equivalence thesis. Some weak-emergence cases are best
described as deterministic computational regime reorganization, not
punctuated onset. The paper must explicitly accept this instead of
reclassifying Life after the fact.

## Addendum to seventh pass (2026-07-27): symmetry-breaking G evidence

New reviewer attack: "You say emergence is spontaneous, but where do
you prove the concrete regime was not externally specified?"

Answer in hand: SYM-BRIDGE directly tests the external-underdetermined
/ internally-selected criterion. In the symmetric condition, A and B
are externally equivalent. Across 120 episodes final A fraction is
0.508, but each episode locks strongly (mean lock 0.946) and the
median openness curve passes onset-type B5 (Delta-BIC 153, t*=280,
thinning stable). The early sign at t* predicts final side perfectly
in this batch, giving the "unexpected before, intelligible after
precursor" evidence. In the biased control, final A fraction is 1.0
and the curve is deceleration-shaped, not onset-type B5.

Revision implication: this is a strong support cell for G and
spontaneous symmetry breaking. It does not solve the learned-ML
flagship gap, but it makes the definition much harder to attack with
"you just measured externally specified convergence."

## Addendum to seventh pass (2026-07-27): learned bridge pilot fails

New reviewer attack: "You have non-learned ant/bridge dynamics, but
no learned machine-intelligence bridge."

Answer in hand: LEARN-SYMBRIDGE tried the simplest learned bridge
pilot and failed completely. A shared policy over {A bridge, B bridge,
idle} trained with sampled REINFORCE under sparse quorum reward did
not learn in 20/20 seeds: final exact success stays around 0.2%,
policy entropy remains near 1.0, and no B5 is applicable.

Revision implication: do not present this as a flagship. It is a
negative design result: a one-step sparse quorum wrapper has no usable
learning signal. A credible ML flagship must be a real spatial task
with trajectory-level affordances, local observations and likely
curriculum/autocurriculum. This keeps the ML gap open but sharply
specifies what the next experiment must contain.

## Addendum (2026-07-28): Potts q-scan strengthens crosswalk

Reviewer demand: "q=2 vs q=10 is cherry-picked. Show the profile
tracks the known Potts transition-order boundary."

Answer in hand: CEB-POTTS-QSCAN extends to q={2,3,4,5,8,10} and
L={32,48}. The aggregate profile separates q<=4 from q>4, matching
the known continuous-vs-first-order boundary:
- L=32: mean hysteresis 0.136 (q<=4) vs 0.650 (q>4); mean Delta-BIC
  6.02 vs 14.00.
- L=48: mean hysteresis 0.187 vs 0.621; mean Delta-BIC 9.57 vs 22.07.

Miss retained: the registered size-sharpening clause fails in this
small two-size run, so no Potts finite-size law is claimed. The result
supports external transition-order validity, not a complete scaling
analysis.

## Addendum (2026-07-28): utility via controllability prediction

New reviewer attack: "Even if the profile classifies emergence, why
is it useful for machine intelligence or control?"

Answer in hand: SYM-BRIDGE-INT turns the profile into an intervention
forecast. A matched counter-regime impulse is applied at different
times. Pre-intervention openness predicts whether the final bridge
side can still be switched:
- openness: 0.987, 0.947, 0.889, 0.819, 0.622, 0.362;
- switch rate: 1.000, 1.000, 0.945, 0.815, 0.435, 0.160;
- tau-level rank correlation: 0.943;
- pooled episode-level rank correlation: 0.664.

Revision implication: this is a core NMI-usefulness result. The
profile is not just a descriptive taxonomy; it predicts a concrete
control question: "Can a bounded intervention still change the
macro-regime, or has the system already committed?" This must be
migrated to the learned spatial flagship, but it already establishes
the actionability target.

## Addendum (2026-07-28): learned utility audit is weak

Reviewer attack: "Your utility result is in a hand-coded bridge. Does
the same profile predict intervention effect in learned MARL?"

Answer in hand: not yet. A retrospective audit of the existing MPE
simple_spread deep MARL probe shows the registered counterfactual
effect exists (D3), but commit_collapse_bits does not predict
episode-level intervention effect size. Pooled rank correlations:
commit collapse vs assignment-JS = 0.011; commit collapse vs absolute
win-gap = -0.101. Early potential has only a weak positive relation
to assignment-JS (0.141).

Revision implication: do not claim learned-system profile utility from
the old MPE audit. The claim remains strong in SYM-BRIDGE and must be
tested prospectively in the learned spatial flagship with frozen
profile-to-intervention predictions.

## Addendum (2026-07-28): Vicsek dense finite-size scan

Reviewer attack: "The Vicsek N=1600 B5 could be a one-off statistical
power artifact."

Answer in hand: CEB-VICSEK-DENSE scans N={100,200,400,800,1600,3200}
under the same control-axis contract. B5 appears only at larger N:
{800,1600,3200}. Rank trends with N are positive for Delta-BIC
(0.714), max adjacent drop (0.886), and hysteresis (0.486).

Caveat retained: raw hysteresis is noisy and not perfectly monotone
at small N, so this is not a clean scaling law. It is enough to defend
the finite-size interpretation of the earlier N=200 null and to align
Vicsek with the ant finite-size story.

## Addendum (2026-07-29): learned transport feasibility

Reviewer attack: "The learned bridge failed because it was a bad
one-step sparse task. Does adding multi-step transport help?"

Answer in hand: yes, but not enough. LEARN-TRANSPORT-VEC adds
multi-step threshold object dynamics and side-neutral transport
reward. It improves learnability substantially over LEARN-SYMBRIDGE:
success rises to roughly 0.60-0.65 instead of ~0.002. But policies
remain high-entropy (~0.97) and no seed crosses the registered 0.8
success / low-entropy regime bar.

Revision implication: this is a direction-finding pilot, not flagship
evidence. It supports the diagnosis that multi-step affordances help,
but the final learned spatial experiment needs state-dependent neural
policies, richer geometry and likely curriculum.

## Addendum (2026-07-29): state-dependent learned transport learns

Reviewer attack: "You still do not have a learned spatial system that
actually learns the collective task."

Answer in hand: LEARN-TRANSPORT-STATE is the first clear positive on
learnability. A state-dependent neural policy conditioned on object
position/velocity solves the symmetric threshold transport task in
10/10 seeds, with final success 1.0. Final side selection is balanced
across seeds (learned_frac_right = 0.5), so the side is not externally
specified.

Important limitation: this is not yet punctuated learned emergence.
Outer B5 appears in 0/10 seeds, and the registered within-episode
realization-collapse proxy fails. Policies tend to encode a side
preference already at the symmetric state. Thus the result supports
learned self-selected coordination, but not onset-type learned
regime formation.

Revision implication: this materially improves the ML story but does
not close the flagship gap. The next version must save checkpoints
and test prospective utility: can the profile predict when the learned
transport convention can be switched or disrupted?

## Addendum (2026-07-30): learned transport utility and realization

Reviewer attack: "The learned transport task learns, but does the
profile predict anything useful in a learned system?"

Answer in hand: not yet. LEARN-TRANSPORT-UTILITY is a prospective
utility pilot on the learned transport task. All 5/5 seeds learn, but a
counter-regime impulse never switches final side at any tested tau. This
means the learned policy is a rigid side convention from the beginning
of the episode. The non-learned SYM-BRIDGE utility law remains strong,
but learned-system controllability utility is still an open gap.

Reviewer attack: "Can a learned system show the within-episode
realization version of possibility collapse at all?"

Answer in hand: yes, in a constrained but informative quadrant test.
LEARN-TRANSPORT-EQUIVARIANT uses a left-right equivariant policy that
ties left/right logits at the symmetric state without specifying which
side should win. It solves the task in 5/5 seeds, preserves high initial
openness (mean H0 = 0.874), keeps final sides balanced within evaluation
episodes, and then collapses action entropy within the episode (mean
drop = 0.866). Thus the learned policy has the capability, while each
episode realizes one regime through self-amplification of early motion.

Important limitation: this is not a strict punctuated-B5 learned
flagship. The collapse saturates too early; 0/5 seeds pass the robust
episode-level B5 detector because most hinge windows are too short. The
claim should be: learned realization collapse is now demonstrated, but
learned punctuated realization requires a richer/longer geometry that
slows commitment enough for robust breakpoint testing.

Revision implication: the ML section should now separate three facts:
(1) ordinary/state-dependent learned transport can self-select a
convention without punctuated emergence; (2) equivariant learned
transport can preserve episode openness and realize a regime within an
episode; (3) a full NMI flagship still needs a temporally resolvable
B5 and preferably a learned-system intervention utility law.

## Addendum (2026-07-31): external-review triage and the two decisive gaps

An external GPT review (working from a slightly older snapshot) was
triaged. Points adopted:

1. Narrative repositioning. The paper's central claim is the V3.3-style
   framework claim -- "a possibility-space measurement framework that
   identifies, types, and predicts macro-regime formation" -- not "all
   emergence is punctuated collapse". This matches the frozen two-level
   structure (qualification D-G-R, then intensity profile) and is what
   the evidence supports.
2. The two decisive gaps are (a) a prospective learned-system utility
   law with a baseline race against generic predictors (reward,
   entropy, order parameters, change-point detectors), and (b) a
   learned system whose regime formation is temporally resolvable so
   the intensity dimensions are testable, with two timescales analyzed
   separately (formation across training vs realization within
   episode).
3. Do not force training-level B5. Episode-level commitment in a
   learned system is a valid and valuable result on its own.
4. Keep all failures in the record; freeze all future flagship
   experiments prospectively (already policy since V3.3).

Points discarded or downscaled:

1. The full multi-algorithm program (MAPPO+MASAC, 15 seeds, N=64,
   VMAS) is beyond current compute; the mechanism-first lightweight
   versions test the same causal claims and can be scaled later.
2. The review's scoring predates LEARN-TRANSPORT-EQUIVARIANT, which
   already closed the "can a learned system show realization collapse
   at all" quadrant (5/5 learn, initial openness 0.874-0.911, episode
   drop 0.72-0.87, seeds/episodes side-balanced).

Experiments launched in response (both preregistered):

- LEARN-TRANSPORT-EQ-UTILITY: interventions on the equivariant learned
  policy at multiple within-episode times, with openness racing |x|,
  |v| and tau at predicting side-switchability. This is the learned
  analogue of SYM-BRIDGE-INT, viable now because the equivariant
  policy, unlike the pre-committed state policy of
  LEARN-TRANSPORT-UTILITY, keeps early openness.
- LEARN-GRIP-TRANSPORT: a two-phase task (collective grip accumulates
  before pushing can move the object) that structurally delays side
  commitment, targeting the first temporally resolvable learned
  episode-level B5. LTES showed the miss is not physics speed but the
  absence of a pre-commitment phase.

## Addendum (2026-07-31, later): both gaps materially closed

Reviewer attack: "There is still no learned system with punctuated
possibility collapse, and no learned-system utility law."

Answer in hand, both preregistered:

1. Learned punctuated realization exists. LEARN-GRIP-TRANSPORT +
   LGT-B: a REINFORCE-trained shared policy (no side or grip shaping)
   solves a grip-then-push transport task in 5/5 seeds; the
   within-episode side-openness stays at 1.000 for 18-19 steps and then
   collapses abruptly to 0.095, passing the full frozen B5 adjudicator
   in 5/5 seeds (Delta-BIC 45.8-52.7, t* = 16-18). The immediate
   contrast with LEARN-TRANSPORT-EQUIVARIANT-SLOW (identical reward
   family, no preparation phase, 0/5 B5) is a preregistered mechanism
   experiment: mechanically delayed commitment, not learning itself,
   generates resolvable punctuated collapse. This also vindicates the
   EEC-style conditional claim with a learned system on both sides of
   the condition.

2. Learned controllability law exists. LEARN-TRANSPORT-EQ-UTILITY:
   switch probability under a bounded counter-regime impulse is 1.0 at
   tau <= 8, 0.898 at tau = 12, 0.280 at tau = 20; openness AUC 0.9955.
   Two honest caveats are recorded: the frozen rank-correlation clause
   failed (0.266) due to switch saturation, and |x|/|v| baselines are
   marginally higher in AUC because a 1D system makes openness and the
   order parameter nearly redundant.

3. One object-class lesson was re-learned prospectively: the grip
   task's total action entropy is the wrong object (grip-phase
   determinism masks side uncertainty); the side-openness object was
   frozen (LGT-B) after seeing only seed 0's console line and before
   any curve inspection. This is the V3.2 object-class doctrine
   applied, not post hoc metric shopping.

Remaining weakness for the strongest referee: openness vs order
parameter separability requires a system where the two decouple
(higher-dimensional geometry or hidden coordination), and the
formation timescale (across training) of the grip system has not yet
been profiled. LEARN-GRIP-UTILITY (frozen) tests whether the LGT-B
breakpoint t* marks the closing of the intervention window, unifying
detection and prediction in one learned system.

## Addendum (2026-07-31, evening): breakpoint leads, not marks, the window

LEARN-GRIP-UTILITY outcome: the controllability window exists in the
flagship (switch 1.0 through tau=16, 0.27 at tau=30) and side-openness
predicts switchability at AUC 0.996, but the strict identity
"breakpoint = window closing" is falsified (0/5 aligned; the window
closes 5-10 steps after t*). Mechanism: the equivariant policy's
commitment is state-conditional, so shortly after t* a bounded impulse
can still carry the object across the symmetry point and the policy
amplifies the new side.

This failure improves the paper. The defensible claim is now sharper
and more useful: in a learned system, the possibility-space breakpoint
is a LEADING indicator of controllability loss, with a mechanistically
explained lag. Detection (B5 at t*), prediction (openness AUC), and
the causal window (switch curve) are now three distinct, preregistered
measurements in one flagship system.

## Addendum (2026-08-01): the mean-field objection is answered

Reviewer attack: "Your learned flagship is one shared multinomial --
a mean-field system. The joint possibility space is a product; the
multi-source decomposition is vacuous there."

Answer in hand: LEARN-STANCE-TRANSPORT gives each of 8 agents an
individual internal stance and local (3-neighbor, resampled, noisy)
observations, trained by per-agent REINFORCE with no conformity or
side shaping. All 5 seeds learn (success 1.0), final sides are
balanced across episodes (0.49-0.51), and the exchangeable source
ladder delivers the headline result: per-agent marginal entropies
remain at 0.9997-1.0 bits while total correlation reaches 6.83-6.97
bits (max 7). The learned collective's possibility collapse is
attributed entirely to relational structure -- individual agents
remain individually unpredictable while the group becomes almost
perfectly coordinated. No prior experiment in this project (and to
our knowledge no learned-MARL emergence claim elsewhere) has shown
this decomposition prospectively.

Honest failures recorded: the declared stance-openness object
collapses within ~3 steps (free lean actions + strong field
feedback), so the separability clause (openness vs |x|) failed and no
episode-level B5 is claimed on this object. The commitment memory
migrates to the physical state once consensus is instantaneous. The
sticky-consensus follow-up (LEARN-STANCE-STICKY, frozen) extends the
consolidation phase mechanically; if it also fails, the separability
claim is dropped for 1D transport geometries.

## Addendum (2026-08-02): separability established; integrity audit

Reviewer attack: "Openness never beats the order parameter; your
profile is a redundant re-description of obvious state variables."

Answer in hand (after fixing a disclosed float32 NaN defect and
rerunning both stance experiments with identical seeds): in
LEARN-STANCE-STICKY, where attitude inertia extends the hidden
consensus phase, pooled openness AUC = 0.886 exceeds |x| (0.849),
|v| (0.853) and time (0.824), and openness rank correlation is also
highest. In the non-sticky control the ranking reverses (0.843 vs
0.904). The pair is mechanistic evidence: possibility openness has
predictive value beyond the order parameter precisely when a genuine
hidden-coordination phase exists -- which is where a practitioner
would need it, since the order parameter is silent there.

Integrity note: one output file (learn_transport_utility.json) had
its verdict corrected by hand when the undefined rank correlation was
discovered; to guarantee every number is code-generated, the fixed
script was rerun and the file regenerated. The NaN-corrupted stance
outputs are archived as *_nanbug.json with the defect, fix, and rerun
preregistered in the alignment document.

## Addendum (2026-08-02, evening): statistical strength and scope of the
## LLM discussion

Reviewer attack: "Five seeds is anecdote, not evidence."

Answer in hand: LEARN-GRIP-EXT extends the flagship to 10 seeds with
the identical code path and zero exclusions: 10/10 learn, 10/10 pass
the frozen side-openness B5 adjudicator, breakpoints concentrated at
t* = 16-18 (two seeds at 22/24).

Algorithm check completed: under advantage actor-critic on the
byte-identical environment, the phenomenon reproduces in shape and
location in all 5 seeds (plateau 16-18, primary hinge Delta-BIC
37.7-45.5, t* = 14-16), but the frozen 4/5 robust-B5 clause narrowly
fails at 3/5 -- two seeds miss only the parity-thinned Delta-BIC >= 10
threshold (6.6/8.1), a detector-power effect of halving a 19-point
window. Per the frozen falsification clause the paper claims strict
robust-B5 for REINFORCE only and reports A2C as a strong partial
replication. No post hoc threshold adjustment was made; the honest
wording costs little because the primary hinge evidence is large in
every seed.

Editorial decision on LLM material: informal manifold/analogy
narratives about LLM emergence (scaling as manifold connection,
hallucination as topological holes) are NOT included in the paper.
They are unfalsifiable in our framework's terms and duplicative of
the scoped, testable treatment already in
LLM_AGENT_EMERGENCE_CROSSWALK.md (capability formation as
formation-axis collapse in mechanism space; metric-artifact
control per Schaeffer). At most one Discussion sentence links the
formation axis to reported LLM capability jumps, marked as untested
in this work.

## Addendum (2026-08-02): matched-parameter causal contrast

Reviewer attack: "Your sticky-vs-nonsticky comparison confounds
stickiness with horizon, tau grid and flip count."

Answer in hand: LEARN-STANCE-CONTROL reruns the exact sticky code
path (all constants imported from the sticky module, same seeds) with
the single change STICK_P 0.25 -> 1.0. The separability ranking
reverses exactly as predicted (openness AUC 0.886 -> 0.811 while |x|
goes 0.849 -> 0.884), and the relational-collapse attribution is
unchanged. The claim "openness carries controllability information
beyond the order parameter iff a hidden consolidation phase exists"
now rests on a single-parameter causal contrast, not a cross-task
analogy. Formation-axis honesty is also complete: the fine-grid run
confirms formation is fast but smooth (0/5 B5 at 5-update
resolution), so no punctuated-formation claim is made anywhere.

## Addendum (2026-07-31, evening): the flagship quadrant is complete

LEARN-GRIP-FORMATION closes the two-timescale question in the same
system. All three frozen clauses pass: 5/5 learn; the formation axis
shows zero punctuated collapse -- instead the outcome-openness object
EXPANDS from 0 to ~0.65 as the capability forms (~100 updates); and
the realization axis reproduces 5/5 episode-level B5 (Delta-BIC
47-62). The flagship therefore demonstrates, prospectively and in one
learned system: smooth/expansive capability formation, punctuated
capability realization, an openness-based early warning of
controllability loss, and a preregistered mechanism contrast (no
preparation phase -> no resolvable B5). This is the
constraint-affordance duality made operational: training opens the
macro capability space while each episode closes the joint
possibility space.

## Revision round vs the external 4/10 review (2026-08-03)

Point-by-point status against the nine core criticisms:

1. Representation/contract dependence (deepest): ANSWERED by
   REPR-ROBUSTNESS -- onset verdicts unchanged in 100% of 243 analysis
   contracts per system (grip / ant N=100 / TRI-C), t* within 1-11% of
   span, frozen cell reproduces every published verdict; object choice
   documented as theory-specified (grip raw action entropy RISES and is
   reported as such). Manuscript claim reworded from "survives the
   change-the-metric attack" to the measured invariance statement.
2. Constructed learned positives: ANSWERED by LEARN-CONVENTION (4/5
   onset, no gate, 5 distinct codes) and LEARN-ROLES (5/5 onset,
   Delta-BIC 53-72, 5 distinct permutations). Grip/XOR rescoped in the
   text as mechanism-isolating controls that are "not load-bearing".
   Both new systems are from the reviewer's own suggested list.
3. Deep-MARL negative / NMI relevance: REFRAMED with new evidence --
   the boundary (joint exploration barrier present vs absent) is now
   demonstrated inside learning, not asserted; Overcooked remains the
   honest gradual case and is now diagnostic rather than damning.
4. Title/claim tension: FIXED -- title is now "Emergence as collapse of
   the effective possibility space"; abstract carries the boundary
   condition; "laws" wording replaced except where a law is actually
   fit (see 7).
5. Decomposition foundations: ADDRESSED in Methods -- non-negativity by
   construction, information-projection (not parametric-model) status,
   canonical stage order, contract-relativity as the only movable
   choice. (Deeper SI derivations remain a to-do for submission.)
6. Detector researcher-degrees-of-freedom: ANSWERED by
   DETECTOR-VALIDATION -- frozen detector, held-out labelled library:
   FPR 0.000 on all control families at every density/noise tested,
   power 1.000 at the operating point, monotone power in grid density,
   graceful noise degradation, location agreement with ruptures Binseg;
   honest 0-power floor at 12 points retained.
7. "Laws" overstated: ANSWERED by ANT-FSS -- t50 = a + b ln N (R^2
   0.93, CI [41,124]) derived mechanistically BEFORE the run, width
   N-invariance, translation data collapse (RMS 0.010 vs 0.266); FSS-1
   miss (N=10 at longer horizon) recorded as failed sub-prediction.
8. Controllability = time proxy: ANSWERED by LEARN-GRIP-CONFOUND --
   AUC 0.974-0.990 at every FIXED tau (20,480 episodes/cell); the
   fixed-|x| null is recorded and folded into the existing
   hidden-regime boundary claim (stance contrast).
9. Statistical uniformity: PARTIALLY addressed (large-n conditional
   analyses, bootstrap CIs on the scaling slope); TRI-C remains at 3
   seeds -- candidate for a final robustness batch before submission.

Third decisive gap (prospective control value): LEARN-GRIP-POLICY
preregistered and running -- openness-triggered intervention timing,
calibrated only on original-seed records, tested on 5 unseen seeds
against fixed-time and random baselines.

## Round-3 external review (score 6/10) - response ledger (2026-08-04)

The reviewer's three pre-submission priorities, all now executed:

1. TRUE representation battery (their sharpest point: our 243-cell
   battery was adjudication-contract robustness, not representation
   robustness). DONE: REPR-EQUIV, 12 measurement-only representation
   cells on byte-identical retrained seeds. Result: verdict survives
   8/12 (< the registered 90%, recorded as RE-1 MISS), location
   essentially invariant everywhere (1.25% span / identical), all four
   breakers mechanically characterized. Manuscript now claims a
   MEASURED EQUIVALENCE CLASS, not blanket invariance; the 243-cell
   battery is renamed "adjudication-contract robustness" in Methods.

2. Neural convention/roles at 10 seeds. DONE: LEARN-*-NN + NN-RES.
   Onset in 6/7 and 10/10 learned seeds at adequate resolution;
   coarse-grid under-resolution recorded; bonus mechanistic finding
   (random init pre-breaks symmetry, compressing the plateau; init
   sweep supports in convention, mixed in roles - registered MISS on
   strict monotonicity).

3. Source-decomposition audit (mixed sources, off-family, order,
   samples). DONE: SD-AUDIT. Off-family attribution exact; filtration
   argument settles order-swap; env/pair declaration freedom bounded
   at 0.26 bits; sample floor quantified; SDA-1 mixed-source MISS
   reported with the correct interpretation (realized structure vs
   generator labels).

Additional wording fixes from the review: "law" -> "consistent with
logarithmic nucleation"; controllability scoped (openness is an
information-sufficient transform of physical state in grip; value =
computable without privileged state + transfers as a rule); Fig. 6
caption "Laws and scope" -> "Scaling and scope".

Honest-miss count for this round: RE-1, NN-INIT roles monotonicity,
SDA-1, plus coarse-grid NN-1/NN-2 - all recorded in the
preregistration ledger with no post-hoc threshold changes.

## Round-4 additions (2026-08-04, standard-environment recovery)

Addressing the one remaining major criticism ("all positives are
self-built small systems; deep-MARL evidence is negative"):

1. The negative now has a NAMED CAUSE and a RECOVERY: cramped_room
   lacks equivalent competing joint regimes; the official
   coordination_ring layout restores them, and with identical PPO
   mechanics every one of 8 preregistered seeds ends committed to a
   circulation direction (both directions realized across seeds),
   while cramped controls never commit. Collapse precedes capability
   in the certified seed. Two clauses missed and recorded: onset
   certification is 1/8 (convention formation is non-monotone in
   training time -- instrument boundary, stated in the paper), and
   one ring seed's policy-entropy object showed a hinge (OCR-4).
2. MPE simple_spread attempt: competence precondition unmet at two
   trainer strengths; recorded as training failure, reported in
   Methods, no theory claim.
3. "What counts as emergence for an arbitrary system" is now
   answered by a packaged certificate (frozen thresholds only,
   vector intensity + categorical verdict, deliberate refusal of a
   scalar score); its battery demonstrates the discriminative point
   that bare entropy cannot make (N=1 vs N=100 at equal amplitude).

Honest-miss ledger this round: OCR-1, OCR-4, OCE-3, MPE competence
(twice). No thresholds changed post hoc anywhere.

## NONMONO-CERT (2026-08-04): honest negative, development closed

Attempted the obvious response to "ring onset certification is only
1/8": extend certification to non-monotone commitment via a
settled-openness envelope (future-max = irrevocable closure; direct
quantification of the persistence clause). Preregistered validation
before touching ring data. Three rounds on synthetic non-monotone
libraries: power 0.30 / 0.71 / 0.59 against required 0.90 (FPR always
<= 0.05). Final-round stop clause honored: instrument NOT applied to
ring seeds, OCE-3 stands, failure reported in Methods. This is the
system working as designed -- the alternative (iterating until the
ring seeds certify) is exactly the detector-engineering criticism
reviewers raised in round 1.

## Round-5 additions (2026-08-05, GPT external review response)

Adopted the external reviewer's program: single-environment closed
loop instead of environment shopping. Status:
1. OC-RING-REAL (realization probe): preregistered, run once, main
   clauses MISSED (0/5), all six control clauses PASSED. Registered
   interpretation: ring commitment is formation-level only; episodes
   drift, they do not internally lock. Reported in the paper as the
   mechanistic complement of grip.
2. SEMI-INJ (real-substrate injection bridge): FPR 0.01, t* 1% span,
   power 0.88 vs 0.90 (near-miss recorded); quantifies the
   evaluation-noise floor.
3. OC-RING-INT (causal commitment test): preregistered with four
   falsifiable clauses (flip asymmetry, zero late flips, openness AUC,
   capability-recovery control); 48 byte-identical resumed-training
   runs; in progress. Declared the FINAL experiment of the program.

## OC-RING-INT final (2026-08-05): ALL FOUR CLAUSES PASS

The single-environment closed loop demanded by the external review is
complete: standard benchmark (unmodified coordination_ring), true
learning (stored PPO checkpoints), matched controls (cramped formation;
untrained + final-checkpoint realization), causal intervention
(perturb-and-resume, 48 runs), and openness as the predictor of
steerability (AUC 0.849). Commitment is causally locked (0/16 late
flips) while capability relearns (0.92 recovery): the institution, not
the learning, is what closes.

## Round-6 addition: BARRIER-XPLAY
Criticism addressed: "the 'joint exploration barrier' is a narrative
label, never measured." Response: deterministic replay of both learned
positive systems with snapshots; frozen protocol measured (a) mean
unilateral adoption gain at mid-plateau = 0.011 (convention, best of
120 codes) / 0.001 (roles), i.e. no unilateral gradient toward any
regime before commitment; (b) unilateral deviation cost after
commitment = 0.40 (the structural ceiling for K=5) / 1.00; (c) mutual
exclusivity of converged regimes: cross-seed intelligibility 0.14 vs
1.00 within, hybrid-team success 0.05 vs 1.00 within. One registered
miss (BX-3 convention threshold mis-calibrated to payoff granularity)
reported verbatim. The barrier is now a measured object.
