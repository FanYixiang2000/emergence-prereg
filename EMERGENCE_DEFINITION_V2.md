# Emergence, definition v2.0 (frozen)

Frozen: 2026-07-23. Supersedes the v1 "endogenous reorganization"
gate for all NEW analyses. The v1 battery and every stored output
remain untouched as the historical record of how v1 was falsified
into v2 (that history is part of the paper, not an embarrassment).

---

## 1. The definition

> **Emergence is a spontaneous, selective and persistent regime-level
> collapse of a system's effective joint state-action-trajectory
> possibility space, measured under a declared observer contract, and
> decomposable by source into environment-mediated, individual,
> pairwise and higher-order channels -- each channel a distinct TYPE
> of emergence, not a gate on its existence.**

中文冻结版：

> 涌现是系统联合状态—动作—轨迹有效可能性空间的一次自发的、选择性
> 的、可保持的制度性坍缩；它在声明的观察者合同下测量，并可按来源分
> 解为环境介导、个体并行、两两协作与高阶不可约四个通道——每个通道是
> 涌现的一种类型，而不是涌现存在与否的闸门。

## 2. Observer contract (nothing is claimed outside it)

Every claim is relative to a declared contract

    Contract = (phi, H, nu, E, H0, I)

- phi: the macro variable(s) and the level structure (which level is
  micro, which is macro; collapse direction is declared PER LEVEL);
- H: time horizon and checkpoint grid (with grid-sensitivity report);
- nu: the perturbation/counterfactual set (what "cut" means);
- E: the declared environment variables (source attribution is
  relative to E; SD-4 in the analytic battery proves declared-vs-
  hidden E moves collapse between C_env and C_rel by construction);
- H0: the admissible lower-order reference class (independent,
  additive, pairwise, common-environment models);
- I: the interventions the system admits.

## 3. Minimum boundary (what is NOT emergence, under any channel)

All four must hold; each is operationalized, none is aesthetic:

- B1 counterfactual reachability: the alternatives that collapsed
  must have been genuinely reachable (P_before(tau) > eps under nu),
  not items on a nominal action list.
- B2 structural breakpoint: the collapse must be a regime change
  relative to the system's own baseline or a matched null -- ordinary
  per-step uncertainty reduction does not count.
- B3 no external hard-coding: the final macro-regime must not be
  directly specified by an external controller, reward clause or
  edit. Environmental selection pressure is allowed; environmental
  dictation of the specific regime is not. (This is the user-approved
  "spontaneity" clause: it excludes central scripts and mid-training
  policy edits, and it is why the commitment-window CUT is an
  instrument, not an emergence generator.)
- B4 persistence: the collapsed regime must persist or recover under
  the declared perturbations (R). Transient coincidence is excluded.

## 4. Source typology (the core of v2)

Total collapse decomposes along the nested maximum-entropy ladder

    C_total = C_env + C_individual + C_pair + C_high

with H(Q0) >= H(QE) >= H(QI) >= H(Qpair) >= H(P) (monotonicity is a
theorem of the nesting; verified exactly in the analytic battery,
SD-5, max violation 0.0).

| Channel | Reads as | Example | Status under v2 |
|---|---|---|---|
| C_env | common information closes joint futures | focus-fire after all see the low-HP enemy | environment-mediated emergence |
| C_individual | each policy contracts on its own | independent learners all converge to attack | parallel/individual emergence; requires the novelty certificate N to be distinguished from ordinary fast learning |
| C_pair | pairwise constraint closes combinations | leader-follower | pairwise cooperative emergence |
| C_high | beyond every pairwise model | role-lock, parity | higher-order collective emergence (the strongest class) |

v1's error, corrected: v1 used the interaction cut as an accept/
reject gate ("only local feedback is emergence"). v2 uses the same
instrument as a SOURCE DECOMPOSER. Common-cause coordination is not
"pseudo-emergence"; it is environment-mediated emergence when B1-B4
hold. This matches the user's four arguments (independent contraction
in MARL, focus-fire, synchronization-as-relation) and the 2026
environment-driven-emergence literature.

## 5. Certificates (existence vs. genesis vs. product)

- Existence: B1-B4 + C_total breakpoint under the contract.
- Source profile: (C_env, C_individual, C_pair, C_high) at the
  regime, plus their formation curves over training/organization
  time s.
- Genesis certificate (our original contribution vs. causal
  emergence): G_s = JS(P_s^real, P_s^cut) > 0 over a formation
  history M_0 -> ... -> M_s, with t_seed < t_visible (measured:
  t_seed ~ 320k < t_visible ~ 740-800k on Overcooked seeds
  93001-93003) and a causally verifiable commitment window
  (intervention preregistered as E3C).
- Product certificate: N (lower-order irreducibility), R
  (persistence), A (macro efficacy). This is where causal emergence
  lives inside our framework -- as a product check, not the theory.

## 6. Temporal phenotype (burst is a shape, never a gate)

J (temporal concentration of [Delta G]_+ or [Delta C]_+) classifies
transitions as punctuated vs. gradual. It is reported WITH its grid
band (multi-resolution J values), because burstiness is provably
grid- and representation-dependent (our Pythia thinning result;
Schaeffer-style metric artifacts). "Punctuated endogenous
possibility-collapse emergence" is the strongest named subclass, not
the definition.

## 7. Level structure (micro collapse, macro creation)

Collapse direction is declared per level. The measured Overcooked
fact -- micro joint-action entropy falls while macro basin entropy
rises during formation (JC-5 registered miss, all three seeds) -- is
the predicted signature "micro possibilities get organized -> macro
capability gets created", not a contradiction. Any claim of the form
"emergence = entropy drop" without a level declaration is rejected by
this document.

## 8. Guardrails adopted from the sharp-review audit

- M (macro gain) is never reported alone; it is compared against the
  mechanism-null band (NB preregistration).
- C_individual-only claims are labeled ordinary-learning-channel
  unless N > 0.
- Source labels always carry the contract (CS sensitivity table).
- Registered misses are retained and interpreted (JC-5, OTC-C4,
  the early-cut curriculum observation).

## 9. Falsification conditions for v2 as a whole

- If the commitment-window intervention is indistinguishable from a
  random equal-budget window (E3C-2 fails), the genesis story loses
  its causal leg and reduces to a descriptive estimator paper.
- If the formation window is a checkpoint-grid artifact (DG fails),
  the temporal claims are withdrawn.
- If the ladder mislabels off-design generators (KUR fails), the
  source typology is instrument-bound and must be narrowed.

## 10. Evidence map (stored files, none modified)

- Analytic ground truth for the ladder: collapse_source_decomposition
  .json (SD-1..SD-5, including the SD-4 contract-relativity proof).
- Real-system genesis curves: overcooked_genesis_curve_curve_s93001/
  2/3.json (t_seed < t_visible, 3/3 seeds).
- Real-system joint collapse + decomposition: overcooked_joint_
  collapse_s93001/2/3.json (C_individual-dominant with nonzero
  C_relational; JC-5 miss retained).
- Same-product different-genesis pilot: overcooked_genesis_
  comparison_pilot.json (clone/marginal separate from learned on G;
  product matching failed and is disclosed -- fix scheduled, E1).
- Intervention pilot: overcooked_intervention_{none,early,commit,
  late}_s93101.json (single-seed; confirmatory = E3C).
- v1 historical battery: collective_constraint.json (untouched);
  v2 reinterpretation: collective_constraint_v2_typology.json (new
  file, pure relabel, RL preregistration).

## 11. Post-freeze evidence updates (dated addenda; body above is
## unchanged)

2026-07-23: the E3C confirmatory intervention (5 conditions x 5
seeds, random-window control, frozen permutation plan) FAILED the
causal predictions: p(M commit < random) = 0.325; C_rel lowest in
the uncut condition; condition differences (~1-2 in M) are well
below seed noise (sd ~3.8-4.9). The falsification clause in section
9 fires for the causal leg: THE COMMITMENT-WINDOW INTERVENTION CLAIM
IS DROPPED. What survives: the descriptive formation-history claims
(t_seed < t_visible, 3/3 seeds; collapse-burst interval), and a new
honest finding -- the final organization is robust to any single
360k-step feedback lesion, i.e. formation is re-entrant, which is a
persistence (R) property, not a fragility. Section 5's phrase "a
causally verifiable commitment window (intervention preregistered as
E3C)" must be read together with this addendum: the verification was
attempted and did not confirm. The genesis certificate remains
descriptive-predictive (early detection), not interventional, until
a redesigned lesion passes a NEW preregistration.

2026-07-23, later: the E1 product-matched comparison fired its
falsification clause (E1-2 FAIL): at matched product, a
noise-handicapped scripted pair has G >= the learned system's G,
because the scripted partner genuinely reacts to noise-perturbed
shared state. Consequence for section 5: single-time-point G is a
COUPLING measure and belongs to the product side; it must not be
called a genesis certificate. Genesis is carried ONLY by (i) the
formation history over M_0 -> M_s (systems with no formation
process have no genesis object at all) and (ii) the provenance
boundary B3. Two independent registered results (OTC-C2 clone, E1
noisy script) now support this; the E1 impossibility reading -- at
matched product and stochasticity, no single-checkpoint cut
statistic identifies endogeneity -- is adopted as a theorem-level
claim of the paper, replacing the earlier hope that G alone
separates script from learned.

Also 2026-07-23: DG passed both grid-robustness predictions
(descriptive window survives a 14-point grid), with the retained
caveat that the t_seed statistic is sensitive to end-point G noise
and needs a robustified preregistered definition before flagship
use.

2026-07-23, wave 4: four further registered results, adopted into
the definition's evidentiary base.

(1) BENCH-72 (bench72_factorial.json, B72-1..6 all PASS): the
full-factorial calibration promised by the roadmap's Claim 2. Blind
source recovery 72/72; M invariant across temporal shapes while J
strictly orders punctuated > sigmoid > gradual in all 24 groups --
the formal proof that amplitude (M) and abruptness (B/J) are
distinct quantities, i.e. section 6's demotion of burst to a
phenotype is now a measured property, not a stipulation.
Revelation-only and metric-artifact pseudo-controls yield exactly
zero collapse (metric artifacts are excluded by measurement);
external mask/overwrite yield large real collapse and are excluded
only by the declared B3 flag, which is the same lesson E1 taught for
G: distributional instruments measure collapse, provenance requires
the declared boundary.

(2) E1-C (overcooked_profile_confirmatory.json, both predictions
PASS on fresh seeds after the E1-B estimation): at matched product,
the source profile separates learned from scripted+noise through
C_env (0.0137 vs 0.0005) and total collapse (0.081 vs 0.030),
non-overlapping CIs. Standing claim: same product, different
collapse COMPOSITION is detectable even where single-point G fails.
C_relational is unclaimed.

(3) TRI-B (triad_relational_collapse_sequential.json): a real
3-agent learner forms the parity regime (3/3 seeds) and carries it
relationally (C_pair ~= 1.01 bits), but compiles it down to
individual+pairwise order (C_high ~= 0.001; TRIB-3 registered
may-miss confirmed). Finding: learned C_high > 0 remains
undemonstrated in this workspace; section 4's higher-order channel
is calibrated (SD-3, BENCH-72) but its learned realization is open.

(4) EP (overcooked_episode_collapse.json, EP-1/EP-2 registered
misses): whole-episode monotone commitment is the wrong shape for
cyclic regimes; within-episode openness collapses fast in the first
cycle (median 0.60 -> 0.00 by t=40) and re-opens at each cycle
boundary. The two-timescale claim of the roadmap is therefore NOT
certified yet; any retry requires a cycle-aligned macro variable
frozen in advance.

---

# v2.1 amendment (2026-07-23T16:20+08:00, user-directed realignment)

The v2 freeze OVER-CORRECTED on abruptness. Driven by the user's
clarification ("突变肯定是必要的。只不过突变的不是任务进度，而是可能
性空间"), the definition is amended:

1. NECESSARY CONDITION (new boundary B5): emergence requires a
   repeatable, contract-robust REGIME BREAKPOINT t* in the collapse
   dynamics of the effective joint possibility space (total or a
   declared source channel): the system passes from an open/slow-
   organizing phase into a distinctly stronger possibility-closure
   and macro-commitment phase. "Breakpoint" means a dynamical phase
   change detectable by model comparison (two-regime vs one-regime),
   robust within the declared grid/representation band -- NOT a
   literal discontinuity, NOT fast collapse, NOT a large single-step
   delta, and NOT abruptness of task progress, performance or any
   visible capability metric.
2. J (temporal concentration) remains an INTENSITY phenotype: it
   quantifies how concentrated the collapse is around t*, and may be
   small (sigmoid-like formation) while B5 still holds.
3. What the v1 falsifications actually killed, re-stated precisely:
   (i) J-as-existence-criterion; (ii) breakpoints measured on task/
   performance curves; (iii) single-step-delta detectors without
   grid-persistence checks. They did NOT test B5: the v1 ant battery
   (ant_contrast.py, ANT-3) measured a ONE-DIMENSIONAL colony order
   parameter (route-commitment deviation) with a 10-90% SPAN gate --
   the wrong object and a J-style gate. Its "gradual" verdict is
   therefore evidence about J on an order parameter, not about the
   existence of a joint-possibility-space breakpoint. The ant
   breakpoint question is OPEN and is exactly the E7 prediction
   (t_collapse < t_completion).
4. Consequence for E3C: the failed intervention lesioned TRAINING-
   time feedback. Under B5's two-timescale reading (learning-time
   and episode-time each host their own emergence events), the
   episode-time commitment-window intervention -- the direct analog
   of the ant story -- has never been tested. E3C's withdrawal is
   scoped to training-time single transient lesions only.
