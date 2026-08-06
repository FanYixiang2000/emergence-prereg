# Flagship design: proving the possibility-collapse story

Drafted 2026-07-22T21:10+08:00. This document designs the decisive
experiment chain for the multi-source possibility-collapse theory of
emergence. Each experiment states its registered prediction AND its
falsification condition; the story is only claimed if the chain holds.

The claim to be proven:

> Emergence is a spontaneous, selective and persistent regime-level
> collapse of the effective joint state-action-trajectory possibility
> space; the collapse decomposes by source (env / individual / pair /
> high-order); its rate can show a commitment window that precedes
> visible capability and is causally load-bearing.

## Evidence already in hand (2026-07-22 pilots)

- SD-1..5: analytic instrument validation of the source ladder
  (exact ground truth, all passed; hidden-E boundary quantified).
- JC-1..5: on a real learning system (Overcooked, seed 93001), the
  largest joint-action collapse-rate increase falls exactly in the
  performance take-off interval (640k->1M); relational and
  environment channels grow late; the registered JC-5 miss showed
  micro collapse + macro expansion, matching the theory's own
  "micro possibilities organized -> macro capability created".
- OTC-G1..3: real-vs-cut G is CI-positive long before visible
  performance; the signed direction C crosses from expansion to
  compression at take-off; G peaks in the take-off window then falls
  after lock-in.
- OTC-C1..4: scripted G=0 vs learned G>0; copied (BC clone) shows
  instantaneous G>0 (single-time snapshots cannot separate copied
  from learned genesis); product matching not yet achieved (C4 miss).

## E3 (RUNNING FIRST, decisive): commitment-window intervention

Preregistration: COMMITMENT_INTERVENTION_PREREGISTRATION.md.
Question: is the observed collapse-burst window (640k-1M) causally
load-bearing for the formation of the coupled regime?
Method: equal-budget (360k steps) cuts of the partner co-adaptation
loop during training (frozen ghost partner + one-sided updates), at
early / commit / late positions, against a matched no-cut control,
all with identical seed and mechanics.
Falsification: if the commit-window cut damages final coupling no
more than random-position cuts, the commitment window is descriptive,
not causal, and the strongest claim of the paper must be dropped.

## E2 (RUNNING): multi-seed formation curves

Two further seeds (93002, 93003) of the full genesis curve; then the
joint-collapse ladder on their checkpoints.
Registered: MS-1 in >= 2/3 seeds the largest collapse increase falls
in or adjacent to that seed's take-off interval; MS-2 in >= 2/3 seeds
the signed C crosses from negative to positive no earlier than the
score reaches 25% of final.
Falsification: if the collapse-burst/take-off alignment does not
replicate, JC-3 was a single-seed artifact.

## E1 (designed, queued): product-matched Same Product, Different Genesis

Fix the OTC-C4 miss: handicap the scripted pair (action noise or
slower cook cadence) and/or use the best learned seed so that final
scores of scripted / copied / learned fall within a factor-2 band.
Then compare G over formation history (training checkpoints for
learned; nothing forms for scripted/copied by construction) and the
final product certificates.
Registered target: matched product, separated genesis
(G_formation_learned > 0; scripted/copied have no formation curve).
Falsification: if at matched product the learned system's formation
curve is indistinguishable from a copied policy evaluated at the same
product level, the genesis certificate adds nothing beyond causal
emergence.

## E4 (designed): three-agent system for C_high in a real learner

n=2 makes C_high identically zero (declared degeneracy). Use a
3-agent Level-Based Foraging (infrastructure already in workspace) or
3-agent gridworld team task; train with checkpoints; compute the full
ladder C_env + C_ind + C_pair + C_high (pairwise maxent via IPF on
3-agent action tables, per declared context).
Registered target: a learned cooperative regime with C_high > 0 that
no pairwise model explains; controls (independent learners with
shared context) show C_env/C_ind growth but C_high ~= 0.
Falsification: if learned cooperation never exceeds the pairwise
maxent reference, the high-order channel is empirically empty in
these systems and the typology loses its strongest tier.

## E5 (designed): baseline race

On the SAME stored rollouts/checkpoints, compute reward curves, action
entropy, MI/total correlation, O-information proxy, and a
causal-emergence-style macro efficacy A. Registered target: at least
one of (a) earlier detection of formation than reward, (b) source
identification impossible for correlational baselines (already shown
in the analytic battery via same-distribution different-mechanism),
(c) localization of an intervention window that baselines cannot
provide (E3). Falsification: if every readout is reproduced by a
baseline at equal budget, the framework is redundant.

## E6 (designed): transfer value of genesis

At matched product (E1), test new-partner and new-layout transfer.
Registered target: endogenously formed regimes adapt better than
copied regimes at matched product; if not, genesis has no
machine-intelligence value beyond bookkeeping (retained either way).

## Order of execution

E3 + E2 tonight (compute-bound only, ~1h); E1 next (needs handicap
calibration pilots, disclosed as pilots); E4 next session (new
environment plumbing); E5 mostly post-hoc on stored data; E6 after E1.
