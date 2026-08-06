# Emergence as punctuated collapse of the effective possibility space

**Authors.** [Author list withheld for double-anonymized review]

---

## Abstract

Emergence is invoked across physics, biology and machine learning, yet
each field certifies it with a different instrument, so observationally
identical macroscopic behaviours can be labelled emergent or not
depending only on the measure chosen. We propose a single operational
definition: emergence is a spontaneous, persistent, regime-level
collapse of a system's effective joint state–action–trajectory
possibility space, and we decompose that collapse by source into
environment-mediated, individual, pairwise and higher-order channels,
each a distinct type. Using a preregistered protocol, we calibrate the
instrument on analytic ground truth, validate it on off-design physics,
and apply it to learned multi-agent systems. A structural breakpoint in
the collapse marks emergence where theory predicts it and is absent
where it predicts convergence, its timing and sharpness obey
finite-size and criticality laws, and remaining openness predicts
whether an intervention can still change the outcome.

---

## Introduction

"More is different" is a claim about organization: many interacting
parts settle into a collective regime that none of the parts specified
in advance^1. The word *emergence* names that phenomenon in
statistical physics, in the self-organization of animal collectives, in
the appearance of new capabilities in large neural networks, and in
multi-agent reinforcement learning. Yet the fields do not share an
instrument. Integrated-information and causal-emergence measures ask
whether a coarse-grained description has more causal power than its
microstate^2,3; partial-information decomposition asks how much of a
target is carried synergistically^4; phase-transition theory tracks an
order parameter^5; the machine-learning literature reports "emergent
abilities" as sharp upturns of a benchmark against model scale^6.

These instruments disagree in a way that is not cosmetic. A recent and
influential critique showed that some reported emergent abilities are
artefacts of discontinuous metrics: change the score function and the
sharp jump disappears^7. This is not an isolated embarrassment; it is
the generic risk of defining emergence on the *observable of
convenience* (a benchmark, an order parameter, a success rate) rather
than on the object that actually reorganizes. Two systems can produce
byte-identical macroscopic outcomes — identical joint distributions,
identical marginals, identical task success — through entirely
different underlying couplings (Fig. 1a). If "emergence" is read off the
outcome, these systems are indistinguishable; if it is a property of how
the possibility space closes, they are not.

We take the second view and make it operational. Consider the joint
space of what every part *could* still do together — the effective
joint state–action–trajectory space. Before organization this space is
wide; a colony of ants approaching a gap can still form a bridge, cross
elsewhere, or disperse. Emergence, on our account, is the moment this
space *collapses* onto a narrow committed regime: the colony decides,
together, to build. Crucially the collapse is not the completion. The
bridge does not yet exist when the collective possibility space has
already contracted onto building it. This is the intuition our
instrument is built to measure, and to measure in a way that survives
the change-the-metric attack that sank naïve benchmark-based claims.

Our contribution is threefold. First, a definition with a two-level
structure: a qualification (spontaneous, regime-level, persistent) and
a quantitative *emergence intensity profile* whose axes — amplitude,
abruptness, and the source decomposition — are separately measured, so
that abruptness is a phenotype rather than a smuggled existence
criterion. Second, a calibrated measurement contract, matured through
preregistered self-falsification, that recovers the source of a
collapse on analytic ground truth (72/72 cells) and on physics the
instrument was never designed around. Third, and most consequential for
this journal's readers, the finding that in *learned* systems the same
collapse admits a clean dissociation of timescales, obeys quantitative
laws, predicts controllability, and — importantly — is **not** generic:
ordinary deep multi-agent optimization collapses smoothly, and we
report exactly where the punctuated form does and does not appear.

## Results

### An operational definition on the right object

We measure openness as the normalized entropy of the effective joint
possibility distribution, \(O_t = H(P_t)/H(P_\mathrm{ref})\), and
collapse as \(C_t = 1 - O_t\). Because literal support never shrinks
for a stochastic policy, we work throughout with the *effective*
possibility count \(N_\mathrm{eff}=2^{H}\); this is the formal version
of the \(10^n \to 2^n\) intuition. A system *qualifies* as emergent
when its collapse is regime-level (a structural change in the joint
distribution, not a marginal one), endogenous (generated internally,
declared through a provenance boundary rather than assumed), and
persistent (it recovers after perturbation). Given qualification, we
report an intensity profile rather than a binary label.

The decisive move is the source decomposition. Total collapse splits
along a nested maximum-entropy ladder,
\[
C_\mathrm{total} = C_\mathrm{env} + C_\mathrm{individual} + C_\mathrm{pair} + C_\mathrm{high},
\]
so that the interaction cut is a *decomposer*, not a gate. On analytic
ground truth with four independent knobs, each knob moves only its own
component, the ladder is monotone, and hiding an environmental common
cause deterministically re-attributes collapse from \(C_\mathrm{env}\)
to the relational channels (all five preregistered checks pass;
Fig. 1b). This contract-relativity is not a bug: it is the precise,
measurable statement that "higher-order" is defined only relative to
what an observer declares exogenous. On a full factorial of 72 analytic
cells (4 sources × 3 temporal shapes × 2 stabilities × 3 magnitudes)
the instrument recovers the source in 72/72 cells, and — the point that
defends us against the metric-artefact critique — the amplitude \(M\) is
invariant across temporal shape (relative range 0.000 across all
groups) while the abruptness \(J\) strictly orders punctuated > sigmoid
> gradual. Amplitude and abruptness are measured as different
quantities; a revelation-only or discontinuous-metric control produces
exactly zero collapse.

### A structural breakpoint marks emergence, and its absence marks convergence

Qualification distinguishes emergent from non-emergent organization;
the *breakpoint* distinguishes onset from convergence. We fit the
collapse curve with a two-regime model and compare it to a one-regime
fit by ΔBIC, inside a saturation-truncated window and behind an
effect-size gate — two amendments we were forced to adopt when a flat,
subcritical control produced a spurious hinge and when an S-curve's
saturation knee stole the fit (both preregistered misses, both fixed
definitionally on fresh seeds rather than by tuning).

With the matured detector, an *onset*-type breakpoint (slow → fast)
appears exactly where the theory places emergence and is absent where
it places convergence (Fig. 6c). It appears in a committing ant colony,
in a learned high-order coordination task, and at the Kuramoto
synchronization transition. It is absent — correctly — in an ordinary
supervised learner, whose collapse begins at maximum rate and
decelerates (a *knee*, not an onset), and it is unresolvable, and so
not claimed, on stored language-model checkpoints whose entropy has
already collapsed before the second saved step.

### Emergence obeys laws: finite size and criticality

The breakpoint is not a detector artefact but a lawful feature of
collective self-amplification. In an ant-commitment model a single
chooser *never* shows onset at any gain or grid; ten concurrent ants do
(ΔBIC 18.4); one hundred sharpen it twelvefold (ΔBIC 217.2), with the
open phase flattening and the collapse steepening as \(N\) grows
(Fig. 6a). Abrupt possibility collapse is thus a genuinely collective
phenomenon: it requires fluctuations (\(\sim\!1/\sqrt{N}\)) small
against saturation. In the Kuramoto system the same instrument obeys
criticality: across coupling \(K=0.9\to2.5\) (10/10 runs onset) the
breakpoint time falls monotonically (6.7 → 1.8 — critical slowing down
approached from above) and the closing slope rises monotonically
(0.032 → 0.199), while every subcritical run is correctly gated null
and the collapse is carried by the pairwise channel in 3/3 seeds
(Fig. 6b). Abruptness therefore has two independent, manipulable
control parameters — system size and feedback strength — both
preregistered and both confirmed.

### In learned systems, formation and realization dissociate

The most consequential results are in learned agents, because here the
possibility space is shaped by optimization rather than fixed physics.
We designed a two-phase collective-transport task in which sixteen
agents must jointly grip an object before they can push it, structurally
delaying the choice of side. Trained with policy gradients, 10/10 seeds
learn the task (success ≥ 0.995) and 10/10 exhibit the predicted
signature: the side-openness of the joint policy holds at a plateau for
~17–19 steps and then collapses, with an onset breakpoint in every seed
(ΔBIC 45.8–52.7, \(t^\*\) concentrated at 16–18; Fig. 2). A
preregistered mechanism control — the same task without the gripping
phase — shows no such breakpoint (0/5), establishing that it is the
structural delay, not learning per se, that produces punctuated
collapse. The phenomenon reproduces under a different algorithm
(advantage actor–critic on the byte-identical environment): 5/5 seeds
reproduce the plateau-then-collapse shape and the primary hinge
(ΔBIC 37.7–45.5, \(t^\*\) 14–16). We report this as a strong partial
replication rather than full algorithm independence, because the strict
robustness clause (surviving subsample thinning in ≥4/5 seeds) passes
in only 3/5 — two seeds miss solely on thinned-subsample detector power,
and we do not adjust the threshold post hoc.

The same learned system cleanly separates the two senses of emergence
that are usually conflated (Fig. 3). Along the *realization* axis —
within a single episode — collapse is punctuated (5/5 breakpoints).
Along the *formation* axis — across training — the capability appears
*smoothly*: outcome-openness expands (0 → ~0.65 within ~100 updates,
0/5 breakpoints), and at fine 5-update resolution the success rise is
fast but has no structural onset (0/5). Capability formation is smooth
and expansive; capability realization is punctuated. This is the
constraint–affordance duality made measurable: micro-level possibility
contraction is what *creates* macro-level capability.

### The source typology transfers to learned coordination

Each channel of the ladder is realizable by learning (Fig. 4). A
hidden-coordination task with eight individually-observing agents
produces a purely *relational* collapse: per-agent marginals stay open
(individual entropy ≈ 1.0 bit) while total correlation rises to
6.8/7 bits. When the low-order route is blocked with private
information, learning builds a genuinely irreducible *higher-order*
carrier: the textbook XOR structure emerges with \(C_\mathrm{high}\)
0.94–0.96 bits and pairwise ≈ 0.0004 bits throughout the whole
formation history (3/3 seeds), and declaring the private cues exogenous
re-attributes the identical collapse to \(C_\mathrm{env}\) — the learned
version of contract-relativity. Together these give a sharp statement:
higher-order structure is not unlearnable but *unfavoured* — gradient
learning selects the lowest-order implementation available and builds
irreducible structure only when information constraints force it. Off
design, the same ladder correctly reads out Kuramoto oscillators
(uncoupled → null; common driver → \(C_\mathrm{env}\); single edge →
\(C_\mathrm{pair}\)), a mechanism vocabulary it never saw. And at matched
task performance, the source *profile* separates a learned Overcooked
pair from a noise-matched scripted pair through its
environment-conditioned component (\(C_\mathrm{env}\) 0.0137
[0.0123, 0.0156] vs 0.0005 [0.0004, 0.0007], non-overlapping), even
though a single-point coupling measure cannot — establishing that
provenance lives in the collapse composition, not in any scalar.

### Remaining openness predicts controllability

If emergence is the closing of a possibility space, then how open that
space still is should predict whether the outcome can still be steered.
It does (Fig. 5). In the ant system, per-episode intervention flip-rate
rises strictly with the episode's own remaining openness (0.000 →
0.0006 → 0.054 → 0.205 across openness bins); a closed episode
(openness < 0.1) flips 0/2600 times, and flipped episodes are 0.58
openness units more open than unflipped ones (permutation
\(p < 10^{-4}\), 8,372 paired counterfactuals). In the learned grip
system, a counter-regime impulse switches the outcome with probability
1.0 up to \(t=16\) and only 0.27 by \(t=30\), and side-openness
predicts switchability with AUC 0.996. A matched-parameter contrast
isolates the mechanism: when stances are inertial (a hidden
consolidation phase exists), joint openness predicts switchability
*better* than the physical order parameter (AUC 0.886 vs 0.849); when
the single stickiness parameter is set so no hidden phase exists — all
other constants imported unchanged — the advantage reverses (0.811 vs
0.884). Openness carries controllability information beyond the order
parameter exactly when there is a hidden regime to consolidate.

### Where the punctuated form does not appear

The claim is falsifiable and we report its boundary honestly (Fig. 6c).
Onset-type collapse is **not** generic to learned optimization. In real
deep multi-agent RL (Overcooked, two agents) the learned policies show
gradual trajectory-space collapse and increasing role selectivity that
track capability but produce no structural onset (0/3 on both a
policy-entropy and a trajectory-occupancy object). A registered rescue
sequence — smooth population consensus, quorum-threshold populations,
and learning-rate scans — also produced no onset (0/20 in the quorum
population). Among classic cases the picture is deliberately mixed:
Vicsek flocking, Swift–Hohenberg pattern selection and Schelling
segregation organize strongly but collapse *gradually* under our object,
while the Potts model reproduces the known continuous-versus-first-order
distinction through hinge strength and hysteresis. The honest
conclusion is not "everything is punctuated collapse" but a
classification: possibility-collapse profiles sort systems into
punctuated, gradual, parameter-axis and early-warning forms, and
onset appears specifically when the system is forced across a new joint
regime.

## Discussion

We have argued, and measured, that emergence can be defined on the
object that actually reorganizes — the effective joint possibility
space — rather than on the observable of convenience, and that this
choice dissolves the metric-artefact problem that has dogged
emergent-ability claims: amplitude and abruptness are provably separate
axes, and a discontinuous scoring function produces zero collapse under
our instrument. The definition is generative rather than nominal. It
supplies a source typology that transfers from analytic ground truth to
physics to learned agents; a breakpoint whose presence and absence
track the theory's own predictions across ant colonies, a
synchronization transition and learned high-order coordination; two
quantitative laws for abruptness; and a control-theoretic payoff, since
remaining openness predicts whether an intervention can still flip the
outcome.

For machine intelligence specifically, the formation/realization
dissociation clarifies a persistent confusion. "Emergent abilities" of
scaled models and the within-run appearance of a coordinated strategy
are different events on different timescales: our learned flagship shows
the first as smooth and expansive and the second as punctuated, in the
*same* system. This reframes the debate about whether capabilities
"suddenly" appear: the capability may form smoothly while each
deployment of it commits abruptly.

The limitations are real and stated as scope, not hidden. Onset-type
collapse is not generic to deep MARL at current scale; our learned
positive controls are a designed grip task and an information-bottleneck
coordination toy, chosen because they *force* a joint-regime crossing.
We do not claim a causally verified commitment window — a
one-pulse-destroys-the-regime hypothesis failed its own falsification
clause, and formation proved re-entrant, which we report as a
persistence property. We make no breakpoint claim for stored
language-model checkpoints, whose grids are too sparse to resolve onset.
And the universal equivalence "emergence = onset breakpoint" remains
unproven: several canonical cases collapse gradually. What we offer is
narrower and, we believe, more useful — a calibrated, preregistered,
falsifiable instrument that measures *how*, *when* and *through which
channel* a possibility space closes, and that earns its claims by
reporting, in the open, every place it says no.

## Methods

*Possibility space and openness.* For a system of \(n\) parts we define
the effective joint possibility distribution \(P_t\) over the relevant
state–action–trajectory support at analysis time \(t\), estimated from
rollouts or exact enumeration where tractable. Openness is
\(O_t=H(P_t)/H(P_\mathrm{ref})\) with \(H\) the Shannon entropy in bits
and \(P_\mathrm{ref}\) the maximum-entropy reference on the same
support; collapse is \(C_t=1-O_t\) and effective possibility count is
\(N_\mathrm{eff}=2^{H(P_t)}\). Entropies are estimated with a
Miller–Madow-style bias correction and, for the joint objects, with the
nested ladder below rather than by direct high-dimensional estimation.

*Source decomposition.* The nested maximum-entropy ladder fits, in
order, an environment-conditioned model, an independent-individual
model, a pairwise model and the full joint, attributing the successive
entropy reductions to \(C_\mathrm{env}\), \(C_\mathrm{individual}\),
\(C_\mathrm{pair}\) and \(C_\mathrm{high}\). The provenance boundary
(which variables are declared exogenous) is fixed before analysis; all
contract-relativity results (analytic SD-4, learned TRI-C) are
consequences of moving this declared boundary, not estimation error.

*Breakpoint detector (B5).* Collapse curves are fit with one- and
two-regime piecewise-linear models; the breakpoint is accepted when the
two-regime ΔBIC exceeds 10 and survives parity-thinning of the
checkpoint grid, the curve passes an effect-size gate (max drop ≥ 0.1),
and the analysis window is truncated at saturation (first point within
5% of the final value). An *onset* type requires
\(|\text{slope}_\text{after}| > |\text{slope}_\text{before}|\); the
reverse ordering is a *deceleration* knee. The gate and
saturation-truncation were added in response to two preregistered
control failures (a flat subcritical false positive; a saturation-knee
capture) and re-tested on fresh seeds.

*Learned systems.* The grip-transport flagship uses 16 agents, grip
threshold 6, a two-phase grip-then-push dynamics (grip gain 0.06, decay
0.01), trained by REINFORCE (1,200 updates, batch 512, lr 2×10⁻³) and,
for the robustness check, by advantage actor–critic on identical
environment parameters. The hidden-coordination flagship uses 8
individually-observing agents with per-agent stances; the sticky variant
sets stance-change probability 0.25 and the matched control sets it to
1.0 with all other constants imported unchanged. Overcooked runs use the
standard cramped-room layout. The ant, Kuramoto, Potts, Vicsek,
Swift–Hohenberg and Schelling models follow their standard definitions
(see Supplementary Methods).

*Preregistration and integrity.* Every predicted outcome was frozen in
a preregistration file with an explicit falsification clause before the
corresponding run; passes and misses are reported identically. All
numbers in the main text and figures are produced by a single extraction
script run directly against the raw output files; no value is
transcribed by hand. One float32 numerical defect (an entropy clamp that
rounded to a degenerate value) was found, fixed in code, and both
affected experiments were rerun with identical seeds, with the corrupted
outputs archived. No thresholds were changed after seeing results.

*Reporting summary.* Further information is available in the Reporting
Summary. Large language models were used only for manuscript editing and
did not contribute to experimental design, analysis or claims.

*Code and data availability.* All experiment scripts, preregistration
files, raw outputs and the number-extraction script are provided in the
accompanying repository.

---

## Display items

**Figure 1 | The problem and the instrument.**
(a) Four generators — central script, common cause, coincidence and
local feedback — constructed to have identical joint distributions,
identical marginals and identical macroscopic success, so that any
outcome-based emergence measure labels them identically. (b) The
source-decomposition ladder on analytic ground truth: four independent
knobs each move only their own component, and hiding an environmental
common cause re-attributes collapse from \(C_\mathrm{env}\) to the
relational channels (SD-1…SD-5 all pass). (c) The full-factorial
calibration (BENCH-72): amplitude \(M\) is invariant across temporal
shape while abruptness \(J\) strictly orders punctuated > sigmoid >
gradual; revelation-only and discontinuous-metric controls give zero
collapse.

**Figure 2 | Punctuated realization in a learned collective.**
Side-openness of the joint policy in the 16-agent grip-transport task
holds on a plateau (~17–19 steps) and then collapses; onset breakpoints
in 10/10 seeds (ΔBIC 45.8–52.7, \(t^\*\) 16–18). Inset: the
no-preparation mechanism control shows no breakpoint (0/5). Overlay: the
advantage actor–critic replication reproduces the shape and primary
hinge in 5/5 seeds.

**Figure 3 | Two timescales dissociate.**
Formation axis (across training): outcome-openness expands 0 → ~0.65,
no breakpoint (0/5), smooth even at 5-update resolution. Realization
axis (within episode): punctuated collapse, breakpoint in 5/5. The same
learned system is smooth in one sense of emergence and punctuated in the
other.

**Figure 4 | The source typology transfers to learning.**
Relational collapse in the hidden-coordination task (per-agent entropy
≈ 1.0 bit; total correlation → 6.8/7 bits). Learned higher-order XOR
carrier when the low-order route is blocked (\(C_\mathrm{high}\)
0.94–0.96 bits, pairwise ≈ 0.0004). Off-design Kuramoto ladder read-out.
Matched-performance profile separation in Overcooked (\(C_\mathrm{env}\)
non-overlapping CIs).

**Figure 5 | Remaining openness predicts controllability.**
Ant per-episode flip-rate rises with the episode's own openness (0.000 →
0.205; closed episodes 0/2600). Grip switch-probability window
(1.0 at \(t\le16\), 0.27 at \(t=30\); AUC 0.996). Sticky-vs-matched
control: openness beats the order parameter for switchability only when
a hidden consolidation phase exists (AUC 0.886 vs 0.849; reversed to
0.811 vs 0.884 in the control).

**Figure 6 | Laws and scope.**
(a) Finite-size law: no onset at \(N=1\), onset at \(N=10\) (ΔBIC 18.4),
12× sharpening at \(N=100\) (ΔBIC 217.2). (b) Kuramoto criticality:
breakpoint time 6.7 → 1.8 and closing slope 0.032 → 0.199 across
\(K=0.9\to2.5\); subcritical runs gated null. (c) The classification:
systems that cross a new joint regime show onset (ant colony, learned
high-order, Kuramoto); ordinary learners, smooth/quorum populations,
real two-agent deep MARL, and several classic gradual cases do not.

---

## References

1. Anderson, P. W. More is different. *Science* **177**, 393–396 (1972).
2. Hoel, E. P., Albantakis, L. & Tononi, G. Quantifying causal
   emergence shows that macro can beat micro. *Proc. Natl Acad. Sci.
   USA* **110**, 19790–19795 (2013).
3. Rosas, F. E. et al. Reconciling emergences: an information-theoretic
   approach to identify causal emergence in multivariate data. *PLoS
   Comput. Biol.* **16**, e1008289 (2020).
4. Williams, P. L. & Beer, R. D. Nonnegative decomposition of
   multivariate information. Preprint at arXiv:1004.2515 (2010).
5. Strogatz, S. H. From Kuramoto to Crawford: exploring the onset of
   synchronization in populations of coupled oscillators. *Physica D*
   **143**, 1–20 (2000).
6. Wei, J. et al. Emergent abilities of large language models. *Trans.
   Mach. Learn. Res.* (2022).
7. Schaeffer, R., Miranda, B. & Koyejo, S. Are emergent abilities of
   large language models a mirage? *Adv. Neural Inf. Process. Syst.*
   **36** (2023).
8. Kuramoto, Y. *Chemical Oscillations, Waves, and Turbulence*
   (Springer, 1984).
9. Vicsek, T. et al. Novel type of phase transition in a system of
   self-driven particles. *Phys. Rev. Lett.* **75**, 1226–1229 (1995).
10. Couzin, I. D. et al. Effective leadership and decision-making in
    animal groups on the move. *Nature* **433**, 513–516 (2005).
11. Bonabeau, E., Dorigo, M. & Theraulaz, G. *Swarm Intelligence: From
    Natural to Artificial Systems* (Oxford Univ. Press, 1999).
12. Reid, C. R. et al. Army ants dynamically adjust living bridges in
    response to a cost–benefit trade-off. *Proc. Natl Acad. Sci. USA*
    **112**, 15113–15118 (2015).
13. Baker, B. et al. Emergent tool use from multi-agent autocurricula.
    *Int. Conf. Learn. Represent.* (2020).
14. Carroll, M. et al. On the utility of learning about humans for
    human–AI coordination. *Adv. Neural Inf. Process. Syst.* **32**
    (2019).
15. Schelling, T. C. Dynamic models of segregation. *J. Math. Sociol.*
    **1**, 143–186 (1971).
16. Cross, M. C. & Hohenberg, P. C. Pattern formation outside of
    equilibrium. *Rev. Mod. Phys.* **65**, 851–1112 (1993).
17. Wu, F. Y. The Potts model. *Rev. Mod. Phys.* **54**, 235–268
    (1982).
18. Power, A. et al. Grokking: generalization beyond overfitting on
    small algorithmic datasets. Preprint at arXiv:2201.02177 (2022).
19. Krakovna, V. et al. Specification gaming: the flip side of AI
    ingenuity. *DeepMind Blog* (2020).
20. Goldenfeld, N. & Kadanoff, L. P. Simple lessons from complexity.
    *Science* **284**, 87–89 (1999).
