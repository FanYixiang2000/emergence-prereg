# Overcooked Transition Certificate V1

Status: protocol scaffold, not yet a confirmatory flagship result.

This document freezes the V1 target for the next Overcooked experiment. The
goal is not to add another read-only Overcooked score. The goal is to make the
framework's central claim testable on a real machine-intelligence benchmark:
from the same simulator state, compare the true coupled continuation with an
interaction-broken continuation.

## Contract

For each system we declare:

- `S`: unmodified `overcooked_ai_py` two-agent Overcooked environment.
- `phi`: future trajectory to macro branch
  `(first potter after snapshot) x (delivery within horizon)`.
- `H`: continuation horizon, default 120 environment steps.
- `nu`: sampled intermediate states from natural policy rollouts, stratified
  by layout and snapshot time window.
- `I`: interventions that cut the partner feedback channel while preserving
  the environment dynamics.
- `H0`: lower-order / non-emergent references: scripted fixed roles, common
  context-role marginals, and ghost-partner replay cuts.

## Primary real-vs-cut certificate

At snapshot history `h_t`, define:

```text
P_real,t(Z) = Pr[phi(tau_{t:t+H}) = Z | h_t, both agents coupled]
P_cut,t(Z)  = Pr[phi(tau_{t:t+H}) = Z | h_t, partner ghost replay]
```

The ghost-partner cut replays a partner action trace sampled from another
episode with the same layout/time bin. Agent 0 still acts from its policy and
sees the resulting environment state; the ghost partner no longer adapts to
agent 0. This cuts the bidirectional feedback loop while keeping actions,
environment dynamics and task interface in the same benchmark.

Reported metrics:

- `G = JS(P_real, P_cut)`: endogenous future reorganization.
- `C = H(P_cut) - H(P_real)`: signed contraction/expansion.
- `M = score_real - score_cut`: macro effect over the continuation horizon.
- `J`: temporal concentration of `G_t` over snapshot bins.
- Matching diagnostics: partner action marginal total variation and branch
  support size.

## Registered smoke predictions

These are engineering sanity checks for the scaffold, not NMI-level claims.

- OTC-S1: The script exports a complete contract and all distributions sum to
  one for a small scripted smoke run.
- OTC-S2: Ghost replay changes the future distribution by a finite,
  non-negative `G` and reports partner-action marginal mismatch.
- OTC-S3: The output explicitly marks whether a learned checkpoint was used;
  no learned flagship claim is made unless a checkpoint is supplied.

## Flagship criteria, not yet claimed

The future confirmatory run must use newly saved checkpoints and trajectories.
It should compare central script, common-driver/marginal controls,
independent matched policies and learned local feedback at matched final
performance. The main predictions will be frozen separately before launch:
local feedback should have higher `G,N,R,A` than matched external mechanisms,
and early `G_t/N_t` should predict a commitment window whose disruption
selectively damages the macro organization.

## V2 addendum: genesis-curve pilot (frozen 2026-07-22T20:05+08:00, before launch)

Purpose: first executable test of the "formation curve" claim -- that the
real-vs-cut future reorganization `G_t` appears along training and can be
compared in time against the visible macro performance. This is a PILOT
(one fresh seed), not the confirmatory flagship.

Protocol (fixed before launch):

- Training: `train_mixed` mechanics unchanged (self-play PPO, shared net,
  two layouts, shaped-reward anneal over 60% of steps), fresh seed 93001,
  2,000,000 steps.
- Checkpoints saved when step count first crosses:
  40k, 80k, 160k, 320k, 640k, 1.0M, 1.5M, 2.0M.
- Certificate per checkpoint: identical to the pilot2m evaluation
  (layouts cramped_room + asymmetric_advantages, snapshot steps
  20/40/80/120/160, 10 episodes per layout, horizon 120, ghost-partner
  replay cut). Certificate seed = 93001 + checkpoint index.
- Curves: `G(s)`, `C(s)`, `M(s)`, `real_score(s)`, with a 1000-resample
  row bootstrap CI on `G(s)`.
- Definitions: `t_seed` = first checkpoint with `G >= 0.5 * G(final)`;
  `t_visible` = first checkpoint with `real_score >= 0.5 *
  real_score(final)`. Indices on the declared checkpoint grid.

Registered pilot predictions (outcomes retained either way):

- OTC-G1: all checkpoint certificates export complete normalized
  distributions with finite non-negative `G`.
- OTC-G2 (directional, may miss): `t_seed <= t_visible` on the declared
  grid -- endogenous future reorganization is measurable no later than
  half-maximal visible performance.
- OTC-G3 (replication): the final (2M) checkpoint reproduces the earlier
  pilot positive, `M > 0`, on a fresh seed.

Known limitations declared up front: one seed; checkpoint grid coarse
(the project's own Pythia thinning result shows grid dependence of
temporal shape claims, so no burstiness claim is registered here);
`G` at 6-bin resolution is a coarse macro branch variable.

## V3 addendum: genesis-comparison pilot (frozen 2026-07-22T20:10+08:00, before launch)

Purpose: first executable four-mechanism comparison at the certificate
level. Rows:

1. `scripted_roles` -- prewired role regime (already measured; fresh-seed
   replication here).
2. `learned` -- the 2M-step self-play checkpoint
   `overcooked_transition_pilot2m_s92003.pt` (fixed here so the row does
   not depend on the concurrently running genesis-curve seed).
3. `bc_clone_of_learned` -- per-agent supervised distillation of the
   learned policy from its own self-play trajectories (copied
   organization; no interactive formation history of its own).
4. `context_marginal` -- both agents sample independently from the
   learned policy's (layout, time-bin) marginal action table; common
   context preserved, state coupling removed by construction.

Certificate identical to the pilot2m evaluation (same layouts, snapshot
steps, episodes, horizon, ghost cut). Fresh certificate seeds 94001+.

Registered pilot predictions (outcomes retained either way):

- OTC-C1 (replication): scripted `G ~= 0` and learned `G > 0` with the
  row bootstrap CI excluding the scripted value.
- OTC-C2 (the honest negative that motivates the genesis curve): the BC
  clone, whose organization is COPIED rather than endogenously formed,
  still shows finite instantaneous `G > 0` -- i.e. a single-time
  real-vs-cut snapshot cannot separate copied from learned genesis;
  separation requires the formation-history curve and/or training-time
  interventions.
- OTC-C3: the context-marginal system has `G` at or below the scripted
  level (no state coupling to cut) and a degraded product score,
  reported as a product-matching failure diagnostic, not hidden.
- OTC-C4 (product matching): scripted, learned and clone final scores
  are within a declared factor-of-two band of each other; the marginal
  system is expected OUTSIDE the band and is reported as such.

## Recorded outcomes (2026-07-22T21:05+08:00; pilots, retained as registered)

V2 genesis-curve pilot (seed 93001, `overcooked_genesis_curve_curve_s93001.json`):

- OTC-G1 PASS; OTC-G2 PASS; OTC-G3 PASS (final M = +9.8).
- Substantive shape, stated honestly: G is bootstrap-CI-positive from
  40k steps (G~0.026-0.038) while score is < 7% of final; G peaks at
  1.0M (0.050) exactly at the performance take-off (score 2.0 -> 16.4);
  G then DECLINES (0.006-0.007) after the regime locks in while M keeps
  rising to +9.8. The signed direction C crosses from negative
  (interaction EXPANDS futures, -0.31 early) to positive (interaction
  COMPRESSES futures, +0.09 at 2M) with the crossing at the take-off.
  OTC-G2 passes partly because the 0.5*final threshold is weak when
  final G is small; the load-bearing observation is the peak-then-lock
  shape plus the C sign crossing, both of which need multi-seed
  confirmation before any flagship claim.

V3 genesis-comparison pilot (`overcooked_genesis_comparison_pilot.json`):

- OTC-C1 PASS (scripted G = 0 exactly; learned G CI [0.058, 0.183]).
- OTC-C2 PASS -- the registered honest negative: the BC clone shows
  instantaneous G = 0.150 > 0, so a single-time real-vs-cut snapshot
  cannot separate copied from learned genesis.
- OTC-C3 PASS (context-marginal G = 0.004, score 0).
- OTC-C4 FAIL, retained: scripted score 102 vs learned 41; the product
  is NOT matched. The flagship needs product matching (weaker script /
  stronger learned policy) before any "Same Causal Product" claim.

Related instrument battery: `collapse_source_decomposition.py`
(COLLAPSE_SOURCE_PREREGISTRATION.md) -- SD-1..SD-5 all passed on exact
analytic ground truth, including the registered SD-4 limitation that
the source decomposition is relative to a declared environment
variable (hidden-E misattributes 0.265 bits of env collapse to pair).

## Flagship experiment name: Same Causal Product, Different Genesis

The decisive comparison is not "does the macro-regime have causal efficacy?"
alone. Causal emergence and causal-abstraction methods already address that
product question. The Overcooked flagship must instead hold the product as
matched as possible and vary the genesis:

- prewired/scripted role regime;
- common-driver/context-role regime;
- copied or marginally matched policy regime;
- learned local-feedback regime.

The target pattern is:

```text
A_script ~= A_common ~= A_copied ~= A_learned
G_genesis_learned > G_genesis_script,common,copied
```

where `A` is macro product efficacy and `G_genesis` is the real-vs-cut
future reorganization that appears over training or within an episode. This
is the concrete way the framework differs from a static causal-emergence
score: it asks whether an effective macro-regime was endogenously generated,
not merely whether the macro-regime is effective after it exists.
