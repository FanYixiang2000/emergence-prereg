# Collapse-source decomposition battery (preregistration)

Frozen: 2026-07-22T20:20+08:00, before the script was executed.

## Why this experiment exists

The current theory upgrade (recorded in the project dialogue of
2026-07-22) redefines the object of study as: emergence is a
spontaneous, selective, persistent regime-level collapse of the
effective joint state-action-trajectory possibility space, with the
total collapse decomposable by SOURCE:

```text
C_total = C_env + C_individual + C_pair + C_high
```

Under this ontology, common-environment coordination and parallel
individual contraction are emergence CHANNELS to be labelled, not
pseudo-emergence to be rejected. The interaction cut becomes a source
decomposer, not a binary gate. No existing output in this workspace
implements that decomposition; the existing batteries (e.g.
`collective_constraint.py`) implement the older accept/reject
ontology and are retained unchanged as historical record.

This battery is the first instrument for the new ontology, built where
it can be checked exactly: an analytic three-agent system with exact
joint distributions (10^3 states enumerated, no sampling noise).

## System

Three agents, actions a_i in {0..9}, common binary environment signal
E with P(E=0)=P(E=1)=1/2. Binary reduction b_i = a_i mod 2.
Generative knobs, each in [0,1]:

- lambda_ind: each agent's base action distribution mixes
  uniform(10) -> concentrated on {0,1} (individual contraction).
- rho_env:  actions with parity matching E get weight multiplied by
  (1 + 4*rho_env) (environment-mediated tilt; conditionally
  independent given E).
- kappa_pair: agent 2 copies agent 1's action with probability
  kappa_pair (pairwise coupling; preserves marginals).
- gamma_high: with probability gamma_high, agent 3's action is
  resampled restricted to parities satisfying b1 XOR b2 XOR b3 = 0
  (three-way parity, invisible to any pairwise model by construction
  when rho_env = 0).

## Nested reference ladder (all computed exactly)

- Q0:    uniform over 10^3 joint actions.
- QI:    product of the three observed single-agent marginals.
- QE:    mixture over E of the product of per-agent conditionals
         P(a_i | E)  (conditional independence given declared E).
- Qpair: per-E maximum-entropy distribution matching all three
         pairwise conditional marginals P(a_i, a_j | E), computed by
         iterative proportional fitting, then mixed over E.
- P:     the true joint.

Components:

```text
C_individual = H(Q0) - H(QI)
C_env        = H(QI) - H(QE)
C_pair       = H(QE) - H(Qpair)
C_high       = H(Qpair) - H(P)
```

These telescope to C_total = H(Q0) - H(P) by construction.

## Registered predictions (outcomes retained either way)

- SD-1 (nesting): H(Q0) >= H(QI) >= H(QE) >= H(Qpair) >= H(P) at every
  grid point of the knob sweep, within 1e-6 tolerance.
- SD-2 (diagonal dominance): for each knob, raising it from 0.2 to 0.8
  with the other knobs at a fixed base of 0.2 increases its OWN
  component more than it increases any other component.
- SD-3 (pure-source dissociations):
  a. pure env (rho=0.8, others 0): C_pair and C_high each < 0.02 bits;
  b. pure pair (kappa=0.8, others 0): C_high < 0.02 bits, C_pair > 0.2;
  c. pure high (gamma=0.8, others 0): C_pair < 0.02 bits, C_high > 0.2.
- SD-4 (declared-observer boundary, expected LIMITATION): recompute
  the ladder with E HIDDEN (drop QE from the ladder, i.e. attribute
  env correlation downstream). Prediction: for the pure-env system the
  hidden-E ladder misattributes the env collapse to C_pair and/or
  C_high by > 0.1 bits. This registers, as a measured fact, that the
  source decomposition is relative to a declared environment variable
  -- the observer-contract boundary the theory already acknowledges.
- SD-5 (time phenotype separation): applying three time profiles to
  gamma_high (linear ramp, sigmoid k=12, single step), the recovered
  10-90% commitment widths of the C_high(t) curve order
  step < sigmoid < linear.

## Interpretation discipline

This battery validates the INSTRUMENT on analytic ground truth. It
does not by itself certify any real system as emergent, and it does
not overturn the frozen manuscript battery; it is the first evidence
layer for the multi-source possibility-collapse ontology.

# Part 2: joint-collapse curve on the real learning system

Frozen: 2026-07-22T20:40+08:00, before the script was executed.

## Why

The theory's PRIMARY object is the collapse of the joint
state-action possibility space itself (10^n -> 2^n), not the
real-vs-cut divergence G. The existing genesis-curve pilot measured G;
this part measures the collapse curve directly on the same eight
saved checkpoints (seed 93001, overcooked_genesis_curve_*).

## Protocol

- Checkpoints: the eight saved nets (40k..2M), unchanged.
- Rollouts: 30 self-play episodes per layout per checkpoint,
  200 steps each, fresh seeds 95001+.
- Declared environment variable E = layout (two values).
- Joint variable: per-step joint action (a0, a1), 36 cells.
- Ladder (computed from empirical tables, exact on the tables):
  Q0 uniform(36); QI product of unconditional marginals;
  QE mixture over E of per-E marginal products; P the empirical
  mixture joint. With n = 2 agents, the pairwise maxent equals the
  full joint, so C_high = 0 BY CONSTRUCTION and is reported as a
  declared degeneracy (a >= 3 agent system is needed for C_high).
- Components: C_individual = H(Q0) - H(QI); C_env = H(QI) - H(QE);
  C_relational = H(QE) - H(P). Normalized openness O = H(P)/H(Q0),
  collapse Cbar = 1 - O. Episode bootstrap (1000) for CIs.
- Macro-branch openness: H of the stored P_real basin distribution
  from the genesis-curve JSON, per checkpoint (no new data).

## Registered predictions (outcomes retained either way)

- JC-1 (collapse direction): Cbar(2M) > Cbar(40k) -- training
  contracts the effective joint action space.
- JC-2 (ladder sanity): all components >= -0.02 bits at every
  checkpoint (estimator tolerance).
- JC-3 (directional, may miss): the largest single-interval increase
  of Cbar falls in an interval that contains or is adjacent to the
  performance take-off interval (640k -> 1.5M on this seed's curve).
- JC-4 (relational constraint): C_relational at 2M exceeds its 40k
  value -- the trained regime holds inter-agent constraint beyond
  environment + marginals.
- JC-5 (macro-level collapse): the basin-distribution entropy at 2M
  is lower than at 40k (macro possibility collapse accompanies
  micro joint-action collapse).
