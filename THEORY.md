# Formal statements behind the possibility-collapse framework

This document collects the framework's formal content in one place: the
root trajectory-space definition (Proposition 0, three identities), eight
further propositions with proofs, one process-metric lemma, and pointers to the scripts that
verify each statement numerically on measured (not hypothetical) systems.
The deductive chain is: 0 grounds the definition and licenses the basin
estimator; 1 relates representation jumps to distribution movement;
2 shows tested single signatures are insufficient on the audited battery; 3 gives the
usefulness identity that powers selectivity crediting and the phase
boundary; 4 controls estimation error; 5 gives scoped derivability and
threshold-insufficiency results for audited signatures, with rival
equations computed to machine precision within declared candidate
families as witnesses.
Numerical verification lives in `verify_theory_bounds.py`
(`outputs/theory_bounds_verification.json`) and, for Proposition 5, in
`exact_prior_formalisms.py` (`outputs/exact_prior_formalisms.json`) plus the
second-domain single-signal audits in `contextual_lbf_single_signal_audit.py`.

## Setup

Fix a finite set of macro-basins `B` and an observer filtration. At time (or
checkpoint) `t` the observer's possibility distribution is `P_t`, a
probability distribution over `B`, estimated by Monte Carlo rollouts (policy
systems) or by the model's own predictive distribution on held-out inputs
(learning-process systems).

Definitions used throughout:

- collapse from reference: `C_t = KL(P_t || P_0)` (bits)
- collapse burst: `B_t = max(C_t - C_{t-1}, 0)`
- stepwise collapse: `K_t = KL(P_t || P_{t-1})`
- macro-representation: `R_t = E_{B ~ P_t}[phi(B)]` for a fixed feature map
  `phi: B -> R^d`
- representation jump: `J_t = ||R_t - R_{t-1}||_2`
- `diam(phi) = max_{b, b'} ||phi(b) - phi(b')||_2`

## Proposition 0 (the trajectory-space definition and why basins may stand in for it)

This is the paper's root definition, stated in the space where "possibility"
lives: full trajectories. Everything else in this document is about
estimating it. Let `tau` be a system trajectory with prior law `P(tau)`
(the open possibility space), and let `M` be a macro-structure variable
(which basin/ability/outcome materialized). Collapse strength of a
structure `m` is

    C(m) = KL( P(tau | M = m) || P(tau) )        [bits]

Three identities connect this definition to everything we measure.

**(0a) Average collapse is mutual information.**

    E_{m ~ P(M)} [ C(m) ] = I(tau; M) = H(tau) - H(tau | M).

Proof. Expand: `sum_m P(m) sum_tau P(tau|m) log( P(tau|m) / P(tau) )
= sum_{m,tau} P(m,tau) log( P(m,tau) / (P(tau) P(m)) ) = I(tau; M)`. QED.

Interpretation: averaging the specific information associated with each
declared macro-readout recovers mutual information. Because `M` is an
observer-declared readout of `tau`, this conditioning identity is descriptive:
it does not make `M` a cause or make the readout observer-independent.
Causality enters through the later do-contrasts. Verified to machine precision on empirical
rollout distributions, trained and untrained (`verify_theory_bounds.py`,
prop0, mi_identity_gap < 1e-9).

**(0b) The rarity law: a deterministic readout carries event surprisal.**
If `M` is a deterministic function of the trajectory (each `m` corresponds
to a trajectory set `A_m`), then `P(tau | A_m) = P(tau) / P(A_m)` on
`A_m`, hence

    C(m) = -log2 P(A_m).

Proof: substitute into the KL sum; the ratio is constant `1/P(A_m)` on
`A_m` and the conditional mass sums to 1. QED.

This formalizes a narrower statement: broad, high-probability readouts carry
little trajectory information. Measured (gridworld, exact to < 1e-9):
under the UNTRAINED policy the sacrifice-rescue structure has prior mass
`P_0(A_m) = 0.007`, i.e. initial surprisal 7.16 bits; the trained system realizes
it with `P = 0.333` (1.58 bits) -- the ecological rate of the contexts where
it is useful (1/3 of episodes), a 5.58-bit log-likelihood gain. By contrast the failed_noise structure --
"something happened" -- has untrained rarity 0.06 bits: appearance alone
carries no collapse, which is why "something appeared, hence emergence"
(the colloquial use) measures nothing.

Rarity does NOT certify endogeneity: a designer can hardwire a behavior that
triggers only in a rare context. Endogeneity and acquisition therefore remain
separate empirically audited components; they are not corollaries of the
rarity identity.

**(0c) Data processing: basin measurements are conservative.**
All experiments estimate distributions over a finite basin partition
`B = g(tau)`, not over raw trajectories. For any two conditions (e.g.
`do(a)` vs `do(a')`),

    JS( P(B | do a), P(B | do a') ) <= JS( P(tau | do a), P(tau | do a') )

and likewise for KL, by the data-processing inequality applied to the
coarse-graining channel `g`. Basin-level JS/KL contrasts are therefore
LOWER bounds on their trajectory-space counterparts. Entropy and entropy
differences do not inherit this divergence bound; Potential and entropy-drop
summaries remain observer- and partition-scale dependent. Verified on
do-contrast rollouts (rescue: JS_basin 0.90 <= JS_traj 1.00; bridge:
0.55 <= 1.00).

Together: (0a) grounds the descriptive substrate in standard information
theory, (0b) quantifies readout surprisal, and (0c) licenses basin-level
JS/KL intervention contrasts as conservative estimates. The six-component
criterion is then the answer to what (0a)-(0b) deliberately
leave open: WHICH collapses count (conditional, specific, useful,
endogenous, acquired) -- Propositions 1-3 show no single projection of
`P_t(B)` can answer that, and the batteries show the conjunction can.

Note on the composite form. The project's original sketch wrote strong
emergence as a product, `E(m) = C(m) * S(m) * N(m) * (1 - G(m))`
(collapse x structure x non-decomposability x non-prespecification). The
registered criterion keeps the same components but replaces the product
with a conjunction of frozen per-component thresholds, for a measured
reason: a product lets a large factor buy back a deficient one (e.g.
`useful_habit`'s huge usefulness gap, +4.4, would compensate its zero
selectivity under any calibration of the product). The original battery
demonstrates named exclusions for selectivity, usefulness and endogeneity;
potential, specificity and acquisition are pinned by dedicated controls and
the later registered refinement rather than unique accuracy drops in that
matrix. The product is a useful mnemonic; the conjunction is the falsifiable
instrument.

## Formation definition and the revelation-null theorem

Proposition 0 is a substrate identity, not by itself a definition of an
emergence EVENT. Because `M=g(tau)`, its `C(m)=-log P(M=m)` can be
positive for a coin toss, a completed football match or a scripted
program once the result is conditioned on. Calling that specific
information "formation" would merely rename outcome revelation.

To distinguish formation from revelation, fix an observer contract

    O = (psi, nu, H, J, M0)

where `psi` maps future trajectories to possibility classes, `nu` is a
declared distribution of admissible counterfactual perturbations, `H` is
the horizon, `J` names the internal mechanism/channel whose contribution
is tested, and `M0` is a matched null mechanism. At the SAME history
`h_t`, time and external environment define

    P_real,t(b) = Pr_nu[ psi(tau_{t:t+H}) = b | h_t, M ]
    P_null,t(b) = Pr_nu[ psi(tau_{t:t+H}) = b | h_t, M0 ].

The mechanism-attributable possibility transformation is

    C_O(t) = H(P_null,t) - H(P_real,t)          (constraint; signed)
    G_O(t) = JS(P_real,t, P_null,t)             (reorganization)

and its formation over a window is the growth of this contrast,

    Delta G_O[t0,t1] = G_O(t1) - G_O(t0)

(with the analogous signed `Delta C`). A candidate emergence event must
also be selective and persist/recover; `A`, `L` and signed `V` remain
type annotations as above.

**Revelation-null theorem.** If the tested mechanism makes no difference
under the matched contract,

    P_real,t = P_null,t  for every t,

then `C_O(t)=G_O(t)=Delta G_O=0`, regardless of how concentrated the
ordinary posterior over success/failure becomes as time passes.

Proof. Equality gives equal entropies and `JS(P,P)=0` at every time;
subtracting zeros gives zero formation. QED.

Consequences:

1. A coin toss or ordinary successful execution can have arbitrarily
   large Proposition-0 surprisal but zero mechanism-attributable
   formation.
2. A central script gives zero when the script is retained in the null
   and only agent-agent coupling is cut.
3. A common environment gives zero after conditioning on/retaining the
   environment while agent channels are cut.
4. A genuine internal interaction/circuit may give positive `G_O` even
   when entropy does not fall (`C_O~0`): reorganization is more general
   than literal contraction.
5. For a deterministic system, `nu` must be a declared counterfactual
   perturbation ensemble. With a delta `nu` on one exact microstate both
   laws are point-valued and no stochastic formation claim is available.

The four-mechanism role-lock experiment is a machine witness of this
theorem: central script/common cause give `C=G=0`, independent
coincidence gives `C=G=0` and low `R`, local feedback gives
`C=1.585`, `G=0.459`, `M=0.667`, `R=1`. This theorem is the formal reason
that "all successful events collapse" does not trivialize the framework.

## Structured possibility collapse: the retained core story

The framework keeps "possibility collapse" as its organizing intuition, but
not in the naive result-revelation sense. The measured object is not
`{success, failure}` becoming known. It is the system's **counterfactual
joint future** under a declared contract being selectively reorganized by an
internal mechanism. In short:

    emergence transition =
        endogenous structured possibility collapse into an effective
        macro-regime.

Equivalently, an empirical emergence transition occurs when internal
interactions selectively collapse or reorganize counterfactual joint futures
toward a novel, lower-order-irreducible and persistent macro-regime. The
word "collapse" is therefore directional and structural: individual futures
that were formerly relatively independently composable become constrained to
macro-regime-compatible joint trajectories. This retains the original
intuition -- "many incompatible futures become committed to one organized
future" -- while excluding ordinary outcome revelation.

The two quantities above separate the general phenomenon from its direction:

    G_O(t) = JS(P_real,t, P_null,t)          # reorganization magnitude
    C_O(t) = H(P_null,t) - H(P_real,t)      # contraction/expansion direction

Thus `C_O > 0` means the tested interaction compresses joint futures,
`C_O < 0` means it expands reachable futures, and `C_O ~= 0` with
`G_O > 0` means the amount of possibility is similar but the structure is
rearranged. Literal entropy decrease is one important case, not the
definition.

This is also the clean distinction from causal emergence. Causal emergence
asks whether an already specified macro-regime has distinctive causal
efficacy relative to micro or alternative scales. Structured possibility
collapse asks how that macro-regime was generated: whether it was imposed by
a script, driven by a common environment, copied from an external policy, or
formed endogenously through internal feedback. The product certificate
(`N,R,A`) can overlap with causal emergence, PID and robustness analyses; the
genesis certificate (`P_real` versus `P_null/cut` through time) is the
additional object.

## Proposition 1 (collapse bounds representation jump)

For every `t`:

    J_t  <=  diam(phi) * TV(P_t, P_{t-1})  <=  diam(phi) * sqrt( ln2 * K_t / 2 )

where `TV` is total variation and `K_t` is in bits (the `ln 2` converts bits
to nats for Pinsker's inequality).

Proof. Write `Delta(b) = P_t(b) - P_{t-1}(b)`. For any unit vector `u`,
the scalar projection `f_u(b) = u^T phi(b)` has range at most
`diam(phi)`. Total-variation duality therefore gives

    |sum_b Delta(b) f_u(b)| <= diam(phi) * TV(P_t, P_{t-1}).

Taking the supremum over unit `u` yields the stated vector-norm bound.
The second inequality is Pinsker:
`TV <= sqrt(KL_nats / 2)` with `KL_nats = ln2 * K_t`. QED.

Consequences that the paper uses:

1. Burst-like collapse CAN produce representation jumps (upper bound is
   loose only through `diam(phi)`), which is why jump-based emergence
   observables fire when possibility collapse happens. This is the formal
   content of the "representation jump is an observable consequence"
   positioning (fig13, fig18).
2. The converse fails: `J_t` large implies only that the distribution
   moved (TV large), not that the movement was a collapse toward a useful
   basin. Movement between equally-open mixtures, or convergence toward a
   prespecified basin, produce the same jumps. This is the formal reason
   the false-positive audits (fig14) had to exist, and why the criterion
   needs usefulness and endogeneity components that no norm of `R_t` can
   supply.

Verified numerically on all four bridge regimes and on both grokking
conditions in `verify_theory_bounds.py` (every step of every trajectory).

## Proposition 2 (single-observable insufficiency on the audited battery)

For each single observable
`O in {potential H(P_0), stepwise/burst collapse, specificity JS, usefulness gap, representation jump, causal-emergence EI}`
no one-sided threshold reproduces all predeclared battery labels. The best
hindsight-selected threshold accuracies are 0.8-0.9.

Evidence: exhaustive threshold sweeps plus illustrative witnesses from
(`criterion_battery_measurements.csv`, `refined_confirmation_external.csv`,
`grokking_collapse_summary.json`):

| observable | S+ (accepted) | S- (rejected) | shared/reversed signal |
|---|---|---|---|
| potential | latent_conditional (H0 = 1.15) | useful_habit (H0 = 1.11) | open futures |
| collapse burst | grokking (burstiness 3974) | prewired (burstiness 28376) | sudden collapse |
| specificity JS | noise_policy (0.74) | harmful_decoy (0.65) | trigger-contingent futures |
| usefulness gap | latent_conditional (+3.4) | useful_habit (+4.4) | positive gap |
| representation jump | uncertain_preference bridge run | pure_team probe (fig14) | large J_t |
| causal-emergence EI (macro - micro) | latent_conditional (+0.71) | shaped_process (+0.93) | macro do-variable beats micro |

The pairs illustrate shared or reversed signals; the exhaustive sweeps, not
each pair alone, establish the maximum threshold accuracy.

(The EI row uses the charitable episodic proxy; Proposition 5c replaces
it with Hoel's exact EI on the enumerated chains and the failure
worsens: exact CE separates nothing -- its best threshold is the
trivial classifier. Likewise 5d for the exact Rosas Psi.)

This motivates a conjunction rather than a single scalar on the audited
battery. The original ablations uniquely pin selectivity, usefulness and
endogeneity; dedicated controls and the later registered refinement support
potential, specificity and acquisition.

## Proposition 3 (counterfactual necessity credits selectivity)

Let a system trigger with probability `p_c` in context `c` (contexts drawn
with probability `w_c`), let `V_c(a)` be the expected return in context `c`
under `a in {trigger, non-trigger}`, and define usefulness as

    U = E[return | behavior] - E[return | do(non-trigger)].

Then

    U = sum_c w_c * p_c * ( V_c(trigger) - V_c(non-trigger) ).

Proof. Conditional on context `c`, behavior return equals
`p_c V_c(trigger) + (1 - p_c) V_c(non-trigger)`; subtracting
`V_c(non-trigger)` leaves `p_c (V_c(trigger) - V_c(non-trigger))`; take the
`w_c`-weighted sum. QED.

Consequences:

1. A selective system (`p_c ~ 1` where the trigger helps, `p_c ~ 0` where it
   harms) receives the full positive terms and none of the negative ones --
   `U` measures exactly the value of the selection structure.
2. A blind system (`p_c = p` for all `c`) receives `p * sum_c w_c
   (V_c(trigger) - V_c(non-trigger))`, the sign of the *marginal* effect:
   this is why `blind_trigger` and `marl_untrained` can show positive `U`
   in environments where triggering helps on average -- and why usefulness
   alone cannot replace selectivity (Proposition 2, row 4).
3. The phase-boundary prediction (fig22) is this identity applied with the
   closed-form `V_c` values of the parametrized environment.

## Proposition 4 (plug-in estimation is consistent, with O(1/n) bias)

Let `P_hat_n` be the empirical distribution of `n` i.i.d. rollout outcomes.
Then `P_hat_n -> P` almost surely, and the plug-in entropy and JS estimates
converge with bias `O((|B| - 1) / n)` (Miller-Madow). With `|B| = 4` and
`n >= 36` per estimate (the battery setting), the bias bound is < 0.09 bits,
below every margin the verdicts rely on (Proposition 2 margins and the
threshold plateaus in `threshold_sensitivity_summary.json`).

Empirical counterpart: the estimator-robustness grid (fig20) shows every
qualitative conclusion is unchanged across `n in {12..96}` and probe
temperatures, and the threshold-sensitivity analysis shows the registered
cutoffs sit on wide accuracy plateaus (one honest exception documented
there: `noise_policy` sits at H0 = 0.508, near the 0.5 potential cutoff).

## Proposition 5 (coverage of the audited emergence signatures)

This is the formal content of the scoped claim that the audited
signatures are projections of the root object.
The claim has two halves: (i) DERIVABILITY -- each
published emergence quantity is a deterministic functional of the root
object plus the observer's declared value/feature/decomposition maps;
(ii) AUDITED VERDICT INSUFFICIENCY -- none of these scalar quantities,
under its published sign rule or a hindsight-optimal one-sided threshold,
reproduces all predeclared battery labels. The second statement is scoped to
the measured testbed and declared candidate families.

**Setup.** The root object is the intervened family of trajectory laws

    P = { P(tau | c, do(w)) : c contexts, w interventions (incl. none) }

over training time, together with the declared value, representation,
macro-feature and micro-decomposition maps. Every
quantity below is a functional Q = F_Q(P) obtained by (projection step)
coarse-graining tau and/or restricting the intervention set, then
(functional step) applying a fixed information-theoretic or metric map.

**(5a) Performance jump (LLM emergent abilities, Wei et al.).** Let
`u: B -> R` be a task score and `M_t = E_{P_t}[u]`. Then `M_t` is the
one-dimensional projection `phi = u` of `P_t(B)`, and by the Prop. 1
argument

    |M_t - M_{t-1}| <= diam(u) * TV(P_t, P_{t-1})
                    <= diam(u) * sqrt( ln2 * K_t / 2 ).

Every ability jump is a bounded symptom of stepwise collapse. Lost in
the projection: potential (was the future open?), provenance (who
collapsed it?), and metric nonlinearity (Schaeffer et al.'s critique is
the statement that `diam(u)` depends on the grader, confirmed from the
mechanism side by our tail-facts result). Witness: hindsight-optimal
accuracy 0.800 on the battery; accepts `shaped_process`.

**(5b) Representation jump.** `J_t = ||E_{P_t}[phi] - E_{P_{t-1}}[phi]||`
is the `phi`-projection of the same displacement; Prop. 1 is the
derivation and Prop. 2 row 5 the witnesses (direction-blind,
provenance-blind, strength-scale measures noise).

**(5c) Causal emergence, exact EI (Hoel et al. 2013).** The one-step
TPM `T(s, .) = P(s_{t+1} | do(s_t = s))` is a restriction of `P` (one
step ahead, state interventions only). Hoel's effective information is
the functional

    EI(T) = I(X_{t+1}; X_t),  X_t ~ uniform  =  (1/N) sum_s KL(T(s,.) || mean-row),

and CE = max over coarse-grainings of `EI(macro) - EI(micro)`. Two
things are discarded by construction: the BEHAVIORAL measure (the
uniform intervention distribution never observes which states/actions
the system itself selects -- endogeneity and selectivity are invisible)
and the VALUE function (EI is sign-blind). Exact computation on the
enumerated policy-closed chains (10 systems, state = (mode, context,
positions, switch, t), `exact_prior_formalisms.py`): micro EI is
9.6-11.3 bits and no candidate macro beats it (CE < 0 everywhere,
range -8.83..-7.01); the hindsight-optimal threshold in EITHER
direction reaches 0.8 = the trivial all-negative classifier, missing
both true positives; the ordering also fails (`blind_trigger` -8.83 is
on the far side of `latent_conditional` -8.46). The earlier flavored
proxy (episodic macro-vs-micro do-EI, `prior_metrics_comparison.py`)
gave CE its most charitable reading and it still misfired on
`shaped_process`; the exact form is strictly worse here.

**(5d) Information-decomposition emergence, exact Psi (Rosas et al.
2020).** The pooled one-step joint of the behavioral occupancy measure
is a two-time marginal of `P(tau)`; Rosas' practical criterion is the
functional

    Psi(V) = I(V_t; V_{t+1}) - sum_j I(X^j_t; V_{t+1}),   verdict: Psi > 0.

Discarded by construction: the value sign and the correctness of
selection (Psi reads "macro predictive structure beyond the parts",
which forced or wrongly-selective coordination possesses in abundance).
Exact computation (same chains, V maximized over four supervenient
features x two micro decompositions, i.e. the rival's best case): the
natural Psi > 0 verdict scores 0.3 -- it flags six non-emergent systems
and its TOP scorer is `wrong_selector` (+0.59, the system that triggers
in exactly the wrong mode), while missing `noise_policy`. The
hindsight-optimal threshold reaches 0.9 only with the direction
INVERTED relative to the theory's own sign convention (low Psi =
emergent), and still misses `latent_conditional`.

**(5e) PID synergy (Williams-Beer line).** Same two-time behavioral
marginals, projected through a joint-vs-marginal MI difference; the
sampled version is in `prior_metrics_comparison.py` (best acc. 0.9,
direction-inverted operating point). Lossiness cause identical to (5d).

**Coverage statement.** Each audited quantity in (5a)-(5e) is `F_Q(P)` --
derivable from the augmented root object, hence "covered" in this scoped
sense. On the audited battery, no scalar projection reproduces all
predeclared labels under its published sign rule or a hindsight-optimal
one-sided threshold; (5c)/(5d) evaluate the two formal rivals from their
published equations within declared candidate families. The six-component
criterion retains value sign, conditional selectivity, endogeneity and
acquisition jointly; this is a testbed result, not a universal ordering
theorem over all emergence definitions.

Scope note (kept deliberately narrow): "covered" means derivable-and-
audited on systems with an explicit trajectory law and value structure.
It does NOT mean the prior definitions' every application domain has
been re-measured, and it does NOT claim Psi/EI are wrong in their home
use cases (quantifying macro predictive structure / intervention-level
compression); the claim is that as EMERGENCE VERDICTS they are
projections that provably discard the verdict-carrying information.

Second-domain witness. The Contextual LBF confirmation repeats the
single-projection test in the strongest full six-component domain rather than
only in the gridworld battery. Giving each behavior-only signal a
hindsight-optimal one-sided threshold against the full CLBF verdict yields a
maximum accuracy of 0.86 on the registered ten-seed confirmation and 0.88 on
the five-seed post-confirmation extension. Potential and acquisition can score
higher as single components in this particular domain, but they are
definition-internal observables: potential is the openness component, and
acquisition uses the same-seed initialization twin to measure learned
provenance. They are not standalone prior definitions. This matters for the
paper's wording: the experiment supports "prior-like behavior signals are
lossy projections" more directly than the stronger and false claim that no
individual component can ever coincide with the conjunction on a finite
domain.

## Proposition 6 -- What survives a refinement of the observer's basins

Let `B'` be a refinement of `B`, so `B = h(B')`. For any distribution
`P` and any two intervention laws `P_a, P_a'`:

    H(B') >= H(B)
    JS(P_a(B'), P_a'(B')) >= JS(P_a(B), P_a'(B)).

If the declared value is coarse-measurable, `u'(b') = u(h(b'))`, then

    E_{P_a}[u'(B')] - E_{P_a'}[u'(B')]
      = E_{P_a}[u(B)] - E_{P_a'}[u(B)].

Consequently, a Potential or JS-Specificity pass measured on a coarse
partition remains a pass under any refinement at the same absolute threshold,
and the usefulness gap is exactly invariant when value is coarse-measurable.
A failure need not remain a failure: refinement can reveal distinctions hidden
by the coarse observer.

Proof. `H(B') = H(B) + H(B'|B)`. The JS inequality is data processing applied
to `h`. Value invariance follows by grouping the fine-state expectation by
`h(b')`. QED.

Important non-result: entropy contraction is not refinement-monotone:

    Delta H(B') - Delta H(B)
      = H_pre(B'|B) - H_post(B'|B),

whose sign is unrestricted. This is why the paper treats entropy-drop
``collapse'' as a partition-scale summary while reserving conservative
trajectory claims for KL/JS.

## Proposition 7 -- Error bound for model-based do-contrasts

Let `P_a, P_b` be the target future-basin laws under two actions and
`Q_a, Q_b` the rollout-model laws. Suppose

    TV(P_a, Q_a) <= epsilon_a,
    TV(P_b, Q_b) <= epsilon_b,

and the declared basin value has range
`R = max_b u(b) - min_b u(b)`. Then

    | [E_{P_a}u - E_{P_b}u] - [E_{Q_a}u - E_{Q_b}u] |
      <= R (epsilon_a + epsilon_b).

Thus a model-estimated usefulness gap that exceeds threshold `t` by more than
`R(epsilon_a + epsilon_b)` certifies a true gap above `t`, conditional on the
state-sufficiency, consistency and continuation-policy assumptions stated in
the manuscript.

Proof. For any function of range `R`,
`|E_P u - E_Q u| <= R TV(P,Q)`. Apply this once to each intervention and use
the triangle inequality. QED.

For Potential, if `K` basins and `epsilon = TV(P,Q) <= 1 - 1/K`, the
Fannes--Audenaert continuity bound gives

    |H(P) - H(Q)| <= h_2(epsilon) + epsilon log2(K - 1).

These bounds turn the identification caveat into a measurable margin
requirement; they do not estimate `epsilon` automatically.

## Proposition 8 -- Stability of a conjunctive verdict

Write every registered component as a signed margin `z_i`, with pass iff
`z_i >= 0`, and let `hat z_i` satisfy
`|hat z_i - z_i| <= e_i`. If every measured pass has
`hat z_i > e_i` and every measured failure has `hat z_i < -e_i`, then the
entire conjunctive verdict is unchanged by all errors inside those bounds.
If the component bounds hold with probabilities at least `1-delta_i`, verdict
stability holds with probability at least `1-sum_i delta_i`.

Proof. Each component sign is fixed by its margin; conjunction preserves all
fixed signs. The probability statement is the union bound. QED.

This proposition clarifies threshold transfer: a threshold learned in one
observer/task is not automatically universal. Transfer is warranted only for
components whose target-domain margins exceed estimator, observer and model
errors. The robustness grids estimate some of those margins empirically.

## Lemma -- A bounded process-burst statistic with the same verdict

For window burst `b >= 0` and median background burst `m > 0`, define the
registered ratio `r=b/m` and

    q = b/(b+m) = r/(1+r).

The map is strictly increasing, and

    r >= 5  iff  q >= 5/6.

For `m=0,b>0`, set `q=1` (the limiting infinite-ratio case); for `m=b=0`,
set `q=0`. The bounded statistic removes numerical explosions caused by the
`1e-6` denominator without changing the frozen decision rule. Exploratory
re-analysis of 27 stored runs finds zero verdict mismatches
(`process_proxy_robustness.py`).

`verify_observer_bounds.py` stress-tests Propositions 6--7 and the lemma on
10,000 random distribution pairs each and the 27 measured process runs:
zero violations, zero bounded-ratio verdict mismatches
(`outputs/observer_bounds_verification.json`).

## Measured coupling between the trajectory-space object and basin observers

`trajectory_basin_coupling.py` computes, exactly (chain-rule path KL on the
enumerated battery chains, zero Monte-Carlo error), how much of each
intervention's trajectory-space contrast the declared basin observer
retains (`outputs/trajectory_basin_coupling.json`):

- DPI: zero violations in every system/intervention (basin KL <= path KL).
- Rarity identity: exact to machine precision in all ten systems.
- The projection is where the verdict-relevant information lives, not a
  loss to be minimized: interventions produce LARGE path-space contrasts in
  控制系统 as well (converged_team 32.2 bits, shaped_process 31.9 bits of
  trajectory KL) but the basins retain almost none of it (3%, 0.01%),
  whereas the emergent latent_conditional retains 32% -- an order of
  magnitude more. Raw trajectory displacement cannot rank emergence;
  value-bearing coarse-graining is what makes useful collapse visible.
- Degenerate/undefined cases align with provenance: forced constructions
  (blind_trigger, harmful_decoy, useful_habit) have zero path KL against
  their own natural law (their behaviour IS the intervention), and
  anti_selector's restricted support breaks absolute continuity -- exactly
  the systems whose provenance the endogeneity component tests.

## Three-layer structure: definition, assumptions, protocol

The framework has three distinct layers that must not be conflated:

1. **Root definition (ontology).** Emergence-as-useful-possibility-collapse
   is defined on trajectory laws (Proposition 0): an endogenously triggered,
   selective, useful contraction of an open conditional future distribution
   that stabilizes a macro-structure.
2. **Identifiability assumptions (observer contract).** A declared observer
   map: basin partition, value function, intervention set, time resolution,
   rollout model. Propositions 6--8 and the resolution note below bound what
   survives changes of this contract.
3. **Operational protocol (the six-component test).** A conservative,
   high-specificity identification procedure under a declared contract. The
   six components are NOT claimed to be individually necessary conditions of
   the root definition. Measured evidence for their roles is asymmetric:
   dropping selectivity, usefulness or endogeneity admits a named gridworld
   counterexample; on the Contextual LBF domain (75 evaluated systems,
   `component_ablation_witnesses.json`) only conditional selectivity is
   non-redundant (dropping it admits the two borderline learned seeds and no
   controls), while the other five components are empirically redundant there
   because every control fails at least two components simultaneously. That
   is the intended design: a conjunction whose components back each other up
   against generic imitations and are individually load-bearing only against
   targeted ones.

## The non-triviality bridge: why ordinary decisions are not emergence

Objection: every decision, Bayesian update or converged policy contracts the
future distribution; why is that not all "emergence"? The criterion answers
with measured exclusions rather than rhetoric:

- **Mere choice / converged action selection** fails Potential at
  measurement time: a deterministic or near-deterministic policy holds no
  open multimodal future (measured: greedy controllers at 0.0 bits in both
  deep-MARL domains; the chess quiet-position lesson).
- **Ordinary gradual learning** fails the process proxy's burst component
  and, at the episode level, acquisition-without-selectivity: tail
  facts/words accrue slowly and are rejected at every Pythia scale.
- **Exogenous injection** (scripted structure, forced triggers, teacher
  curricula) fails endogeneity/acquisition by provenance: scripted_coop,
  forced_commit, team_nearest, prewired -- each passes multiple behavioural
  components and is rejected on provenance.
- **Useless collapse** (memorization inside grokking runs, harmful decoys)
  fails usefulness under do-contrast despite genuine distributional
  contraction.

What the collapse must additionally deliver -- and what the accepted cases
share -- is a *new, stable, causally load-bearing macro-structure*: a basin
(cooperative role assignment, consumption order, strategic continuation,
reusable linguistic ability) whose forced removal degrades value
(do-contrasts) and which was acquired rather than prewired (initialization
twins). Persistence is now directly measured on the saved Contextual LBF
policies under a prospectively frozen perturbation battery
(PERSISTENCE_PREREGISTRATION.md): the acquired selective structure is fully
stable across horizon changes and observation noise up to sigma 0.2 (40/40
retention cells; noise degradation unmeasurably small -- a benign registered
failure of the strict-monotonicity prediction), never appears in
initialization twins (70/70 cells), and has a measured, narrow spatial
generalization boundary: transfer to novel layouts is partial for
selectivity (5/10 at the 50% retention bar) and negative for value on all
ten seeds -- a registered failure kept in place. Stabilized collapse is
therefore an empirical property of the accepted cases, while its
generalization scope is a measured boundary rather than an unstated
assumption; promoting persistence to a formal seventh component remains
declared future work.

## Emergence magnitude versus emergence velocity

The ordinary-learner probe and the 2.8B grid results are reconciled by one
formal distinction the framework now states explicitly. The full protocol
measures emergence MAGNITUDE under a declared observer contract O:

    E(M; O) ~ [future-law reorganization] x [causal utility, do-contrast]
              x [endogenous acquisition] x [stability under declared D]

(as always, the conjunction of frozen per-component thresholds is the
falsifiable instrument; the product is the mnemonic). Emergence VELOCITY is
the temporal concentration of that reorganization along a training
trajectory -- what the four-component process proxy's burstiness measures at
a given checkpoint grid. The two are different quantities:

- gradual emergence is possible (high E accumulated at low velocity);
- fast ordinary learning is possible (high velocity, but E's selectivity/
  provenance components never tested by the proxy -- the measured
  ordinary-learner acceptance);
- the 2.8B S1/S7 failures are velocity measurements dropping below a
  grid-relative detection threshold while the magnitude components
  (usefulness, controls) still behave correctly.

Consequently the process proxy is an acquisition-shape (velocity)
instrument. It can flag when reorganization is temporally concentrated and
useful; it cannot certify magnitude, which requires the episode-level
selectivity/specificity/provenance components. No lone proxy pass is treated
as an emergence verdict anywhere in the project.

The status of the original burst-collapse hypothesis is now explicit. The
first version of the framework used burst-like collapse as the operational
separation between ordinary learning and emergence; the stored
`collapse_burst_experiment.py` remains the historical hypothesis. The
boundary audit (`burst_boundary_audit.json`) shows why it cannot be the
definition: a fast ordinary learner has burstiness 6805--22822 and passes
the old proxy 6/6, yet is rejected by the lower-order novelty gate
(`N_cap = -0.003`); the Deneubourg ant trail is accepted as collective
emergence (`D = 0.942`, `R = 0.966`) while its 10--90 commitment spans
24.8% of the horizon; and the Pythia-2.8B agreement verdict flips under
9/9 checkpoint-thinning cells. Burst is therefore retained as a
predictive evidence channel, especially when aligned with jumps and
controls, but rejected as either a sufficient or necessary definition of
emergence.

### Capability-specific novelty gate: why not all learning is emergence

For learned capabilities, the ordinary-learner result makes one further
condition necessary. A capability candidate must exhibit a **structural
novelty gap** relative to a frozen lower-order hypothesis class with the
same inputs:

    N_cap(H0) = Perf(full learned system) - sup_{h in H0} Perf(h).

Here `H0` is part of the observer contract and must be justified before the
comparison (for example, additive input effects or an architecture lacking
the compositional layer at issue). `N_cap` is not a universal scalar:
acceptance means that process-level collapse is accompanied by a measured
failure of the declared lower-order mechanism, not merely that optimization
made a predictor more confident. It is the capability analogue of asking
whether a collective constraint disappears when agent coupling is cut.

The direct boundary experiment
(`capability_novelty_boundary.json`) exposes and repairs the old proxy false
positive. On the ordinary five-class coarse-sum task, the full learner
scores 0.924 while a frozen additive classifier scores 0.927
(`N_cap = -0.003`): ordinary learning is rejected despite the old process
proxy accepting 6/6 runs. On modular addition the corresponding scores are
1.000 versus 0.000 (`N_cap = 1.000`); for induction they are 0.987 for the
two-layer model versus 0.116 for the one-layer control
(`N_cap = 0.871`). Thus abruptness is neither necessary nor sufficient.
Gradual formation can qualify, but only when the resulting useful, stable
organization is counterfactually unavailable to the declared lower-order
mechanism.

This result supports a necessary boundary, not a complete universal
definition of representational novelty. A maliciously weak `H0` can inflate
the gap, so lower-order controls, input information, optimization budget and
threshold must be frozen and defended. The old 6/6 ordinary-proxy acceptance
remains in the record rather than being relabelled after the fact.

## Temporal resolution and burst detection (measured, not assumed)

Burstiness verdicts are relative to the checkpoint grid. The held-out
scaling runs measured this dependence rather than assuming it away: at 2.8B
the agreement collapse is spread over several early intervals and fails the
burst threshold on the full published grid, yet every 2--4x thinning of the
same series re-aggregates the mass and flips the verdict to accept
(`held_out_scaling_robustness.json`; registered failures S1/S7). The
direction is systematic: coarsening merges adjacent increments, so a
spread-out collapse can only gain window share, while refinement can split
a single apparent burst. Process-level verdicts therefore carry their grid
as part of the observer contract; the framework predicts and audits this
dependence (radius/thinning grids) instead of claiming grid invariance.

## Axioms of the continuous record (A1-A8, machine-verified)

The continuous record `G_chi = (Y, M, V, A, Q, R, U)` under contract
`chi` is constrained by eight axioms. Each is verified numerically on
fresh random ensembles or pinned to the registered experiment that
established it (`verify_record_axioms.py`,
`record_axioms_verification.json`; all pass):

- **A1 Nullity.** If the action is independent of the future basin, the
  causal magnitude is exactly zero: identical do-laws give JS = 0
  (20,000 random systems, max deviation 0.0).
- **A2 Boundedness.** Every normalized dimension lies in its declared
  range (P, S, M, Q, A in [0,1]; V, E_adapt in [-1,1]; 20,000 random
  profiles).
- **A3 Monotonicity.** Along any mixture path from the do-block law
  toward a distinct do-trigger law, measured magnitude is
  non-decreasing (2,000 paths x 11 points, no violation).
- **A4 Data processing.** No basin coarsening can increase the do-law
  divergence (20,000 random coarsenings, no violation); measured
  magnitude can only be lost, never manufactured, by a poorer observer.
- **A5 Context sensitivity.** Merging contexts cannot raise measured
  selectivity above the true per-context separation.
- **A6 Value separability.** Flipping the value function flips V and
  leaves M unchanged within estimator noise (generator calibration
  GC-4).
- **A7 Provenance separability.** A fully prewired system has Q = 0 at
  unchanged M and S (generator calibration GC-5); structure and its
  provenance are measured by different dimensions.
- **A8 Abstention under non-identifiability.** When world-model error
  exceeds the certificate margin or probe coverage fails, no hard
  verdict is emitted (world-model closure follow-up: 20/20 mismatches
  caught, zero silent wrong verdicts).

## Admissible observer contracts and the identification interval

A contract `chi = (g, V, H, pi_roll, I, boundary)` is *admissible*
(`chi` in `C_adm`) iff:

1. the basin map `g` is measurable from future trajectories alone (no
   access to the intervention label or to training provenance);
2. no intervention in `I` writes directly into the value's argument
   (do-operators act on actions/decoding, never on the recorded
   outcome);
3. the value function `V` is declared before any confirmatory
   measurement (frozen protocols);
4. the rollout model's error is bounded by the certificate margin or
   the verdict abstains (Proposition 7 + A8);
5. refining `g` may only add resolution: verdict-relevant components
   are stable under refinement in the sense of Proposition 6 (a
   refinement can reveal, never delete, do-law separation, by A4 read
   in reverse);
6. the system boundary is declared (Terminology).

Observer dependence then becomes a computable *identification
interval*: for any dimension X of the record,

    [X_lo, X_hi] = [inf_{chi in C_adm_measured} X_chi,
                    sup_{chi in C_adm_measured} X_chi]

reported over the stored admissible contracts rather than asserted to
be a point. Measured instance (five stored contracts, 15 CLBF learned
seeds, `contract_ranking_stability.json`): the per-seed E_struct
identification intervals have median width 0.14; the seed RANKING is
stable across contracts (mean pairwise Spearman 0.76); and no interval
of any control overlaps the adaptive layer (E_adapt > 0 for 15/15
learned, = 0 for all controls under every contract). The point is not
that the number is observer-free -- it is that the observer's freedom
is bounded, published and audited.

## Proposition S -- The spatial-collapse bridge (total correlation)

For N agents with future-basin variables `B_1..B_N`, the joint
contraction relative to the independence null,

    C_spatial = sum_i H(B_i) - H(B_1,...,B_N),

is exactly the total correlation / multi-information of the agents'
futures (Watanabe 1960; McGill 1954) and equals
`KL(joint || product of marginals)`. Machine-verified
(`verify_spatial_bridge.py`, all checks pass):

- S-A identity and nonnegativity on 20,000 random joint laws
  (max gap < 1e-12);
- S-B under independence the joint entropy is exactly `N * H_1`
  (checked N = 1..8): the open possibility space -- and therefore the
  collapse available to coordination -- grows linearly with the
  population. This formalizes "more agents feel more emergent."
- S-C exact monotonicity of C_spatial in the coupling strength of a
  latent-copy family: coordination IS contraction of the joint future
  space; each individual becomes predictable from the others exactly
  to the extent C_spatial > 0. This formalizes "coordination is
  (candidate) emergence."
- S-D the blind spot, stated as mathematics: a deterministic script
  attains the MAXIMUM C_spatial = (N-1) log2 k, and two
  mechanistically different generators with the same joint law are
  indistinguishable. Structural collapse is provenance-blind by
  construction; the adaptive layer (value, endogeneity, acquisition)
  is the only place scripts can be excluded -- which is what the
  battery's scripted counterexamples and the exact-Psi audit measure.

Connections: integration-style measures (Tononi, Sporns & Edelman
1994) build on this quantity; PID synergy (Williams & Beer 2010;
Rosas 2020) decomposes related multivariate information. The bridge
places the swarm intuition, integration measures and the synergy
audit on the same trajectory-law substrate, with the provenance
blindness derived rather than asserted.

## The emergence-type lattice (coordinates, not a total score)

What is unified is not "every emergence equals one number" but "every
emergence claim can be located, compared and falsified in one
testable coordinate system." Universal core (measurable in any system
with trajectories, no learning required):

    N  collective non-additivity   co-information I(X;Z) - sum I(Xi;Z)
                                    (analytic on logic gates; Prop. S
                                    connects it to total correlation)
    D  interaction dependence      loss of macro structure under
                                    marginal-preserving surrogate
                                    decoupling (do-operator on the
                                    coupling, not the components)
    A  causal autonomy             EI(macro) - EI(micro), exact on
                                    enumerated chains (Hoel's quantity
                                    as a DIMENSION, not a rival)
    R  robustness                  recovery RATIO after irrelevant
                                    micro perturbation (attractor = 1,
                                    transient = 0; persistence absent
                                    perturbation is trivial)

Adaptive extension (the existing six-component machinery):
L/Q acquisition, S selectivity, V signed value, P persistence.

Lattice (min-combinations; no dimension can compensate another):

    weak emergence        N, D, R pass          (Bedau/Chalmers)
    causal emergence      weak AND A > 0        (Hoel)
    adaptive emergence    weak AND L            (this paper's
                                                 certificate)
    functional emergence  adaptive, V reported with sign

Philosophical strong emergence (macro facts underivable in principle
from complete micro facts) is declared OUTSIDE the empirical
framework: it is a different kind of claim, not a higher score, and
no measurement in this paper adjudicates it.

Prior definitions as projections, now with per-dimension residence:
PID/synergy lives on N; causal emergence on A; self-organization on
D+R; phase-transition order parameters and LLM "sudden abilities" are
formation DYNAMICS (abruptness T), neither necessary nor sufficient
(the metric-artifact row of the adversarial matrix, agreeing with
Schaeffer); learned conventions add L and S; utility is only V and is
NOT necessary (harmful congestion is emergence with V < 0).

Validation (emergence_coordinates.py): analytic truths recovered on
calibration families; frozen thresholds transferred blind to held-out
families (Kuramoto, Life, learned conventions) reproducing the
literature's own labels 4/4 -- including Chalmers' glider as weak-not-
adaptive; adversarial matrix 8/8 rejected on the predicted dimension.
Three rounds of specification errors were caught by the battery's own
exact computations and are retained in the ledger.

### Two swarm questions the coordinates answer (ant double-bridge)

A reader raised two objections that the coordinate system resolves
cleanly (ant_contrast.py, a Deneubourg double-bridge: two equal routes
around a central obstacle, identical individual rules except the
pheromone channel; thresholds copied from the coordinates battery).

*"An ant bridge is built bit by bit -- where is the possibility
collapse, and must it be abrupt?"* The collapse is not in any single
ant's motion; it is the contraction of the DISTRIBUTION over which
route the colony commits to. Both routes start equiprobable (route-
choice entropy ~ 1 bit); positive feedback progressively concentrates
that distribution until the colony is committed (commitment dev 0 ->
0.99). Crucially this contraction is measured to be GRADUAL: the
10%-90% collapse is spread over a quarter of the foraging horizon and
no single trip carries more than ~15% of it (an abrupt step would put
~100% in one trip). Abruptness (the formation-dynamics dimension T) is
therefore not required for possibility collapse -- it is a separate,
non-necessary property, consistent with the metric-artifact row of the
adversarial matrix.

*"Is an ant finding food around an obstacle not emergence?"* Under the
collective coordinates it is not weak emergence: a lone ant navigating
around the obstacle produces no colony-level consolidation (rate 0.00)
and there is no coupling to break (D 0.00). It is genuine individual
adaptation, but the coordinates deliberately separate individual
adaptation from COLLECTIVE emergence. Turn on the pheromone channel and
the same individuals become a system with D 0.94 and R 0.97 -- weak
emergence by the same rule used for the held-out families (ANT-1..3,
all passing; N -0.15, redundancy-dominated in many-body swarms and
reported descriptively, not gated on).

## Irreducibility: surviving the five frontier objections

Five 2025-2026 papers raise the bar for any collapse- or higher-order-
style account of emergence. They do NOT compete with possibility
collapse one-for-one, but together they rule out the naive version
("the group became more correlated / more concentrated, therefore
emergence"). We take each objection as a design constraint and show
(emergence_irreducibility.py, exact on small discrete systems with a
Gaussian analytic anchor) that the instrument already meets it or is
upgraded to meet it. The upgrade is to stop scoring the MAGNITUDE of the
contraction and score only its ENDOGENOUS, IRREDUCIBLE, macro-causal
part.

**Environment-Driven Emergence (the no-go).** A common, time-varying
environment can produce synergy-dominated / negative O-information with
NO interaction; higher-order statistics do not imply higher-order
interaction. This is the sharpest threat because "possibility collapse"
and "co-information" are both statistical. We reproduce it in its
strongest form: a deterministic common cause X1 = e1, X2 = e2,
X3 = e1 XOR e2 is DISTRIBUTIONALLY IDENTICAL to a genuine three-way
role-lock -- both have total correlation 1 bit, O-information -1
(synergy), and irreducible higher-order connected information
C_irr = 1 bit. No statistic of the joint distribution can tell them
apart. The framework separates them by causal structure, two agreeing
ways: (i) conditioning on the environment, C_irr|E = 0 for the common
cause vs 1 bit for the role-lock; (ii) the do-operator on the coupling
(cut agent-agent channels, keep the environment), D_higher = 0 vs 1.
Same distribution, opposite verdict. This also demotes the framework's
own N (co-information): N is necessary but NOT sufficient, exactly the
ENVDRV point, and the load-bearing quantity is the interventional
C_irr|E / D pair, not any statistic.

**PITHON (higher-order grows from pairwise).** Observing temporal
higher-order structure is not novel if it is reducible to pairwise edge
dynamics. We make reducibility the test: C_irr = KL(P || P^(2)), the
connected information beyond the pairwise maximum-entropy model
(Schneidman 2003; Amari 2001). A Markov-chain (pairwise) system has
C_irr = 0 by construction and is classified weak/reducible; only
structure that survives the best pairwise explanation counts.

**Cognitive Agent Networks (collapse can be pathological).** Consensus
can be premature convergence on a bad basin, so magnitude cannot
measure quality. Two results: (a) magnitude != strength -- across five
systems the Spearman correlation between total contraction and
C_irr|E is <= 0; the maximum-contraction system (redundant consensus,
C_total = 2 bits) has C_irr|E = 0, while the strong case has SMALLER
contraction (1 bit). (b) A functional/pathological double dissociation
at MATCHED magnitude: a consensus tuned to the role-lock's contraction
(0.998 vs 1.000 bits) has value gain 0 and C_irr|E = 0, while the
role-lock has value gain +0.5 and C_irr|E = 1 bit. Collapse is scored
by irreducibility and signed value, never by size.

**Causal Emergence 2.0 (multi-scale causal contribution).** Causal
power is distributed across scales; a single scale is a slice, not a
rival. A (causal autonomy, EI-macro minus EI-micro) is already a
DIMENSION here, not a competitor, and the lattice reports it per system
rather than seeking one "best" scale -- consistent with CE 2.0's
reframing.

**Krakauer, Krakauer & Mitchell (LLM emergence).** A sudden benchmark
jump is not emergence (already the metric-artifact row of the
adversarial matrix, agreeing with Schaeffer); genuine emergence needs a
coarse-grained variable that predicts/controls and screens off micro
detail, and the knowledge-in provenance must be accounted for. Our
positive certificate requires exactly a macro variable with
irreducible joint structure plus interventional load-bearing (C_irr|E
and D), and the acquisition/endogeneity components (L, endogeneity)
address knowledge-in vs knowledge-out. Notably co-information of a
chosen readout (macro_gain) is reported but NOT gated on: it is 0 for
the genuinely strong role-lock (a degenerate readout) and can be
positive for independent bits, so it is neither necessary nor
sufficient -- again only joint irreducibility carries the verdict.

### Strong vs weak by reducibility, not magnitude

The upgraded definition, tested above:

    emergence = endogenous, selective contraction of the counterfactual
    reachable joint-future set, that is IRREDUCIBLE to environment,
    independent adaptation, and all lower-order interactions, and that
    carries macro causal load.

Weak / reducible emergence: the contraction is accounted for by
independent learning, a common environment, pairwise structure, an
ordinary attractor, or a known task constraint --

    C_joint  ~  C_environment + sum_i C_i + sum_{i<j} C_ij .

Strong emergence: after conditioning on the environment and removing
all lower-order structure, an irreducible joint constraint remains and
is interventionally load-bearing --

    C_irr|E  =  C_joint - C_(<= pairwise) | environment  > 0 ,
    D_higher >= threshold .

Strength is therefore graded by REDUCIBILITY, not by how much the
possibility space shrank. An environment that leaves a single road open
produces enormous contraction but zero C_irr|E (weak); three agents
that exclude only a few trajectories yet form a role-interlock no
pairwise or common-cause model reproduces have small contraction but
positive C_irr|E (strong). Philosophical strong emergence (macro facts
underivable in principle) remains outside this empirical scale -- a
different kind of claim, not a larger C_irr.

Validation (emergence_irreducibility.py): IR-1 the no-go is reproduced
in both directions (Gaussian common cause redundancy Omega +0.30 bits;
deterministic common cause synergy Omega -1, C_irr 1 bit); IR-2 the
framework rejects it (C_irr|E 0, D 0) despite a signature identical to
the role-lock; IR-3 pairwise systems are reducible (C_irr 0); IR-4 the
role-lock is strong (C_irr|E 1, D 1); IR-5 magnitude does not rank
strength (Spearman <= 0; max-contraction system has C_irr|E 0);
IR-6 functional/pathological dissociate at matched magnitude. All six
frozen predictions pass; misses would be retained in the ledger.

## Possibility collapse, made precise: endogenous collective constraint

An earlier framing invited a fatal objection: if the possibility space
is {task succeeds / fails} or {bridge forms / does not}, then EVERY
successful process is a "collapse" -- a coin toss, a single agent
avoiding an obstacle, a scripted plan. Outcome-collapse is result
revelation, not emergence; and a single agent choosing one of its own
actions is ordinary decision-making, not emergence. The possibility
space must be the JOINT-ACTION branch set of the agents, and the
question is whether AGENT-AGENT INTERACTION endogenously reorganizes it.

Let $A^i_{t:t+\tau}$ be agent $i$'s action branch. The interaction-free
baseline is not the raw Cartesian product but the reachable joint
distribution with agents decoupled while their individual abilities,
the environment, common signals, any controller, and each single-agent
marginal are held fixed:

    P_broken( A^{1:n} )   vs   P_real( A^{1:n} ).

This yields a two-part certificate. The GENERATION certificate asks how
the structure forms:

    C  collective constraint   = H(P_broken) - H(P_real)   (joint
                                 branches pruned by interaction)
    G  reorganization          = JSD(P_real, P_broken)      (structure
                                 changed even when entropy is unchanged)
    M  endogenous macro gain   = P(Z|real) - P(Z|broken)    (capability
                                 that survives ONLY with interaction)

The PRODUCT certificate asks what the structure is: the same
N (irreducibility, C_irr|E), R (persistence) and A (macro causal
autonomy) as above. Possibility collapse is exactly the C > 0 face of
this object -- a sub-mechanism, not the definition.

**Micro-freedom down, macro-capability up.** Emergence is not a
decrease in total entropy. In the ant bridge the support ants LOSE
individual freedom (their joint branches are pruned, C > 0) while the
colony GAINS a macro capability it never had (crossing the gap, M > 0).
Emergence reorganizes micro freedom into macro capability; some
collective behaviours even keep future options maximally open, so
"entropy went down" is neither necessary nor definitional.

**The decisive test: same outcome, different mechanism**
(collective_constraint.py). Three-agent role-lock, structure
$Z: a_1+a_2+a_3 \equiv 0 \pmod 3$ (order-3 irreducible: every pair is
unconstrained). Four mechanisms are matched on the FINAL OUTCOME and on
single-agent marginals: a central controller, a common environment,
independent coincidence, and endogenous local feedback. The first three
plus local feedback are DISTRIBUTIONALLY IDENTICAL (uniform over the 9
valid configs, uniform marginals, $P(Z)=1$) -- no statistic of the
joint distribution separates them. The interaction-broken counterfactual
does: only local feedback shows C = 1.585 bits, G = 0.459, M = +0.667,
C_irr|E = 1.585, R = 1.0 and is accepted; the central controller and
common cause give C = G = M = C_irr|E = 0 (structure survives cutting
agent channels because it was external), and independent coincidence is
rejected on persistence (R = 0.30). Each imposter fails on the PREDICTED
certificate component. This is the four-quadrant table a referee
demands, passed (CC-1..5).

**Real-system bridge (Overcooked, read-only).** The externally
timestamped Overcooked round-1 artifacts do not include the learned
policy checkpoints or step-by-step trajectories in this checkout, so the
full interaction-broken `C,G,M;N|E,R` replay cannot be claimed without
retraining or restoring those files. Rather than conceal this, we record
the limitation and measure the strongest bridge available from the
frozen metrics (overcooked_collective_constraint.py). The joint branch is
`(context, first-potter role)`. The broken baseline cuts context-role
dependence while keeping the role marginal, yielding
`C_ctx = I(context; role)` and `G_ctx = JSD(P_real,P_broken)`. Among the
8 preregistered accepted learned seeds, 8/8 show positive context-role
constraint (C_ctx >= 0.05, G_ctx >= 0.01); all 12 learned seeds retain
positive macro gain against do-block; accepted seeds remain persistent
under contract B (8/8). In contrast, the high-scoring scripted and
BC-clone external controllers have C_ctx = 0 in 12/12 seeds: they always
force the same role, so they succeed without endogenous context-specific
collective constraint. This supports the new certificate in a public
benchmark but also leaves the strongest reviewer demand explicit: a
complete Overcooked replay with saved policies and a true agent-channel
cut remains future work unless the checkpoints are restored.

This reframes the intuitions the reader raised. Why do MORE agents look
more emergent? A larger joint-action possibility space to be
endogenously constrained. Why is coordination emergence? Because
interaction makes agents mutually predictable -- each is derivable from
the others -- which is C > 0 on the joint branches. Why is
reward-shaped "focus fire" NOT emergence? A process reward compresses
the outcome into a single peak the designer specified; the constraint is
exogenous (like the central controller above), so C_endogenous ~ 0. Why
does OpenAI hide-and-seek read as genuine emergence? Under sparse
outcome reward the joint search space is vast, and a tiny jointly-
reachable branch is progressively locked in by the agents' own
interaction -- high C, G, M with endogenous provenance. The definition:

    emergence = endogenous formation of collective constraints that
    reorganize independently-composable joint-action possibilities into
    a stable, causally efficacious macro-organization;

possibility collapse (C) and reorganization (G) certify the GENESIS;
irreducibility (N|E), persistence (R) and macro autonomy (A) certify the
PRODUCT; usefulness V is a signed annotation, never a gate; abruptness
is a separate, non-necessary dynamical property; philosophical strong
emergence stays outside the empirical scale.

## Canonical validation: possibility spaces, not stories

To make the definition reusable rather than private, the canonical
possibility-collapse matrix (canonical_possibility_collapse.py) exports
row-wise validation evidence across public/classic examples and negative
controls. It explicitly separates analytic truth, constructed mechanism
truth, canonical convergent validity and external empirical evidence.
Each row names the possibility space under test:
relative phase futures for Kuramoto; heading futures for Boids; spatial
configuration futures for Schelling; local orbit futures for Life
gliders; route futures for ant double-bridges; context-conditioned role
futures for Overcooked; output/computation futures for grokking and
induction heads. This is the key shift: the framework does not ask
whether "a macro structure exists"; it asks which previously reachable
future branches were selectively closed or reorganized, by what
mechanism, and whether the resulting regularity stabilizes.

The battery contains 19 rows and all match their expected public or
analytic status. Canonical positives pass: Kuramoto supercritical
synchronization, Boids flocking, Schelling segregation, Game-of-Life
gliders (weak-not-adaptive), ant trails, Overcooked learned role
conventions, grokking and induction heads. Negatives fail on the
predicted route: Kuramoto subcritical lacks stable collapse; ant solo is
individual adaptation; common-driver, central-controller and metric-jump
rows are pseudo-emergence; Overcooked scripted/BC controls are high
performing but externally specified; memorizer/no-structure/one-layer
capability controls fail usefulness, burstiness or architectural
possibility. This is an intended reusable validation matrix for later
work, not a mysterious scalar written into nature. Only analytic and
constructed rows are ground truth; agreement with canonical examples is
convergent validity and public-system rows are external evidence.

This also restores the scope of "possibility collapse" beyond
multi-agent coordination. Collective emergence is the joint-action
special case; capability emergence is the computation/output-space
special case; causal emergence is a macro-causal validation layer after
the collapse has stabilized. Macro structure and synergy are therefore
not the definition itself; they are observables of a stabilized collapse.
The remaining limitation is explicit: public Overcooked has a read-only
bridge row because the stored artifacts lack checkpoints/trajectories
for a full interaction-broken replay. The matrix records this limitation
instead of hiding it.

Life exposes a necessary observer-contract boundary. Conditional on a
fully specified microstate and deterministic update rule, the future is
point-valued and stochastic collapse is exactly zero. Under a declared
counterfactual ensemble of admissible local perturbations, however, the
glider orbit can be tested for robustness and dependence on the Life
interaction rule; that contract gives D/R weak-emergence evidence. These
are different questions, not interchangeable estimates. Accordingly the
current Life row supports the literature's weak-emergence classification
under the perturbation contract but does not prove that a fixed glider
undergoes probabilistic temporal collapse. Any universal claim must be
relational to `(system, possibility map, horizon, intervention ensemble,
baseline)`.

## What is NOT claimed

- No claim that collapse alone is emergence (Proposition 1, consequence 2;
  the non-triviality bridge above lists the measured exclusions).
- No claim that the six components are jointly *sufficient* for every
  notion of emergence outside the measured families; the claim is necessity
  evidence for selectivity, usefulness and endogeneity from ablations,
  control support for the remaining components, and out-of-sample transfer of
  the conjunction (fig21, fig25).
- Proposition 4 covers sampling noise; Propositions 6--8 delimit partition,
  rollout-model and margin robustness. None makes an authored observer map
  uniquely correct.
