# Emergence, definition V3 (frozen 2026-07-23T16:55+08:00)

Supersedes EMERGENCE_DEFINITION_V2.md and its v2.1 amendment, which
remain in the repository as the audit trail of how the definition
moved. V3 exists because v2 OVER-CORRECTED: it demoted abruptness
entirely to a phenotype on the strength of v1 falsifications that
had tested a DIFFERENT hypothesis (burstiness of performance/outcome
metrics under fragile detectors), not the current one (a regime
breakpoint in the joint possibility space). GPT's audit of
2026-07-23 identified this conflation; the v2.1 amendment already
restored breakpoint necessity (B5); V3 freezes the full corrected
statement under its own name.

## 1. Definition

EN: Emergence is a spontaneous and persistent regime-level collapse
in the effective joint state-action-trajectory possibility space,
characterized by a structural breakpoint in its collapse dynamics
and decomposable into environmental, individual, pairwise and
higher-order sources.

中文：涌现是联合状态—动作—轨迹有效可能性空间的一次自发且可保持的制度
性坍缩，其形成表现为坍缩动力学的结构断点，并可分解为环境、个体、两两
及高阶来源。

## 2. Boundary conditions (all required)

- B1 selectivity: the collapse closes trajectory families
  selectively (regime-compatible futures survive), not uniformly.
- B2 persistence: the regime persists, or recovers after bounded
  perturbation (rho).
- B3 provenance (spontaneity): the final regime is not written in by
  action masking, policy overwrite, central command or rule change.
  Provenance is a DECLARED boundary; no distributional statistic
  identifies it (E1 impossibility result; BENCH-72 mask control).
- B4 non-transience: transient pseudo-regimes fail (BENCH-72
  transient-sync control).
- B5 breakpoint (the V3 restoration): the collapse dynamics of the
  effective joint possibility space (total or a declared source
  channel) contain a repeatable, contract-robust structural
  breakpoint t*: the system passes from an open / slowly organizing
  phase into a distinctly stronger closure-and-commitment phase.
  Detection is by model comparison (two-regime vs one-regime fits,
  Delta-BIC or likelihood ratio) with grid/representation
  persistence checks -- never by single-step deltas, raw second
  differences, or performance-curve jumps. J (temporal
  concentration) remains an intensity phenotype: sigmoid-like
  formations can satisfy B5 with small J.

## 3. What the v1 falsifications actually established

They falsified DETECTORS and OBJECTS, not B5:
(i) J-as-existence-criterion on performance/outcome curves
(ordinary learner passes burst gates); (ii) breakpoints read off
1-D order parameters (ANT-3's route-commitment span); (iii)
single-step detectors without grid-persistence (Pythia thinning
flips). None of these measured the joint possibility space with a
model-comparison detector. The three cases are therefore OPEN under
V3 and are re-adjudicated by the RE battery (preregistered).

## 4. Sources, intensity, spectrum

- Source typology unchanged from v2: environment-mediated,
  individual-parallel, pairwise, higher-order -- all are TYPES of
  emergence; the interaction cut and the maxent ladder DECOMPOSE,
  they do not gatekeep.
- Intensity is the profile (M, B/J, rho, V; C_env, C_ind, C_pair,
  C_high) -- calibrated end-to-end on BENCH-72, where M is provably
  shape-invariant while J orders shapes.
- Collapse event spectrum (from the EP registered miss): in cyclic
  or multi-goal systems the possibility space re-opens and
  re-collapses per cycle/subtask. Emergence events are therefore a
  SEQUENCE {(t_k, M_k, B_k, rho_k, Gamma_k)}; whole-episode
  monotonicity is not required and was the wrong frozen shape.
  Within-episode claims must use cycle/subtask-aligned macro
  variables, frozen in advance.

## 5. Causal program (E3C scoped precisely)

E3C falsified ONE strong hypothesis: that a single transient
(360k-step) feedback lesion inside the predicted window permanently
prevents the regime, more than a random-window lesion. It did NOT
show that the commitment moment lacks causal meaning. The measured
phenomenon is RE-ENTRANCE: regime interrupted -> possibilities
re-open -> same attractor re-forms. The registered follow-up
program replaces "one pulse destroys" with four quantities:
collapse delay (Delta t_collapse), re-entry time (T_reentry),
dose-response D -> P(regime re-forms), and path switching (same
attractor vs different basin). Episode-time commitment-window
interventions (the ant-story analog) remain untested and are a
separate, future preregistration.

## 6. Vulnerability program (design constraints, frozen wording)

The E1-C profile separation licenses a NEUTRAL prediction only:
systems with larger C_env should show larger regime-profile change
under environment-cue interventions -- "change" may be performance
loss, adaptation, or regime switching; "more fragile" is NOT
presupposed. Any vulnerability experiment must use the full
four-type intervention matrix (cue / individual / pairwise /
higher-order perturbations at matched budgets), held-out prediction
of the most-affected channel, standardized effect sizes and shares
of total collapse, not raw channel ratios.

## 7. Falsification conditions for V3

- RE battery: if, measured on the joint possibility space with the
  frozen hinge detector and persistence checks, (a) recognized
  emergence exemplars (ant TRAIL commitment) show NO robust
  breakpoint, or (b) breakpoints flip arbitrarily under reasonable
  thinning in all real systems tested, B5 falls and abruptness
  returns to phenotype status (that was v2; the move would then be
  evidence-driven, twice).
- Ground-truth recovery failing (BENCH-72 already passed).
- Collapse signals never leading visible capability (currently
  passing: t_seed < t_visible 3/3).
- The framework adding nothing over MI/PID/O-info/CE baselines
  (E5 already passed against this).

## Amendment V3.1 (2026-07-24): effect-size gate in the B5 detector

The KUR-BP subcritical control exposed a detector defect: on a long
FLAT openness series (no collapse at all, drop 0.000), the hinge
model comparison still returned Delta-BIC 23 for a slope change of
-1.4e-5 -> -8.9e-5 -- statistical significance without physical
content. B5 is a claim about the collapse dynamics; a series that
does not collapse has no collapse dynamics to have a breakpoint in.

Frozen for all future B5 tests (past verdicts unaffected -- every
prior tested series had drops far above the gate): the hinge test
is APPLICABLE only if total openness drop across the declared
analysis window is >= 0.1; otherwise the verdict is "no collapse,
B5 not applicable". Thinning persistence uses the RE-2 bar: thinned
Delta-BIC >= 2 with onset type preserved and t* shift <= 2 coarse
grid steps; the full-grid existence bar stays Delta-BIC >= 10 for
new preregistrations.

This amendment was adopted through a registered miss (KURBP-3), not
by re-adjudicating it away: KUR-BP's outcomes stand as frozen, and
the amended contract was re-tested on fresh seeds (KUR-BP-R).

## Amendment V3.2 (2026-07-26): object classes and the finite-size
## law of the breakpoint

Adopted through six registered outcomes (ANT-GAIN miss, ANT-FINE
AF-1 miss / AF-2 pass, ANT-FINE-B misses, ANT-COLONY-BP passes).

1. OBJECT CLASSES. B5 is a claim about the CURRENT-STATE joint
   possibility space (the joint state-action table now: Kuramoto
   raw-phase joint table, TRI-C joint action table, a colony's
   behavioral openness H2(p_t)). ENDPOINT-PROJECTION objects
   (basin entropy of long-horizon cloned continuations, as in RE-2
   / EP / ANT-FINE) measure outcome PREDICTABILITY, which under
   autocatalytic amplification saturates at maximal rate from the
   start -- such curves are structurally incapable of onset-type
   B5 and serve as EARLY-WARNING instruments (the t_seed family).
   The two object classes are complementary, not competing: early
   warning saturates first, the current-state breakpoint follows,
   completion follows that (measured ordering, ANT-COLONY-BP:
   prediction early, t* = 350, completion 651).

2. FINITE-SIZE LAW. The onset-type breakpoint is a COLLECTIVE
   phenomenon. It requires the system's spontaneous fluctuations
   to be small relative to saturation (scale separation ~
   1/sqrt(N)). A solitary chooser (N = 1) shows deceleration only,
   at any feedback gain and grid resolution (five registered
   misses); N = 10 shows onset; N = 100 shows a 12x slope kink
   sharpening with N (ANT-COLONY-BP). Kuramoto (N = 200) and
   learned policies (uniform initialization, small gradient steps)
   satisfy the scale-separation condition, which is why they
   passed B5 directly. Corollary: "abruptness" in the definition
   is not an added axiom; it is the generic signature of
   collective autocatalytic closure, and it VANISHES smoothly as
   N -> 1 -- exactly the intuition that emergence is about many
   parts committing together.

RE-2's verdicts stand under their frozen contract; RE-2 is
re-scoped as the early-warning + commitment-before-completion +
controllability leg of the ant story, while onset-type B5 for ants
lives at colony scale (ANT-COLONY-BP).

## Amendment V3.3 (2026-07-27): emergence qualification and intensity

Adopted after the DEF-CAL definition calibration
(`outputs/definition_calibration.json`). V3.0's wording made B5 sound
like a universal existence gate. The accumulated classic-case and
definition-boundary results now require a two-level structure:

1. EMERGENCE QUALIFICATION. A candidate event must first pass
   `D and G and R`:
   - D: a regime-level reorganization of effective future
     possibilities, not merely a rare sample from an unchanged
     generator.
   - G: the concrete macro-regime is generated by the system's
     internal dynamics under the declared contract, not directly
     specified by masking, policy overwrite, central command or a rule
     that uniquely fixes the regime.
   - R: the regime persists or recovers under bounded perturbation.

   This gate solves the lottery objection. A lottery win has high
   prior surprisal but no future-distribution reorganization, no
   internal generative regime, and no persistence. A rare critical
   nucleus can be a trigger, but the emergence event is the
   self-maintaining regime it creates.

2. EMERGENCE INTENSITY PROFILE. After qualification, report the
   continuous profile
   `(M, J/B5, S, X, G, R, A, V; C_env, C_ind, C_pair, C_high)`:
   - M: collapse magnitude.
   - J/B5: temporal concentration / structural breakpoint strength.
   - S: prior surprisal of the final macro-regime.
   - X: explanation gain, i.e. how much more predictable the final
     regime becomes after early internal precursors but before full
     visibility.
   - G: spontaneous generation degree.
   - R: persistence / recovery.
   - A: macro affordance or capability gain.
   - V: value sign.

3. STATUS OF B5. B5 is no longer a universal qualification gate. It is
   the defining marker of PUNCTUATED emergence and remains central to
   the strong-onset story, but gradual emergence and rare-seed
   nucleation can pass `D and G and R` while having weak or
   representation-dependent B5. This moves the paper's strongest
   defensible claim from universal equivalence (all emergence is
   punctuated collapse) to a conditional mechanism theory:
   possibility-space dynamics distinguishes rare samples, hard
   specification, gradual organization and punctuated regime
   formation.

DEF-CAL outcomes: LOTTERY is excluded despite high S; scheduled MASK
is excluded because it is externally determined; RANDOM_MASK is
excluded despite high S/D/R because G=0; NUCLEATION passes D/G/R and
has high S and X, but its event-aligned curve does not satisfy the
old onset-type B5 clause. This is recorded as a definition refinement,
not as a hidden threshold change.
