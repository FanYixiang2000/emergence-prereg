# Pre-registration: criterion transfer to an external swarm system

Registered: 2026-07-03, before running any criterion measurement on the
external battery below. This document fixes the target system, all metric
definitions, all thresholds, the label-assignment (audit) rules, and the
per-system predictions. The measurement script is
`external_swarm_criterion_transfer.py`; results are written to
`outputs/external_transfer_*.{json,csv}` and compared against this document.

## 1. Why this experiment exists

Everything measured so far lives in one benchmark family (discrete
sacrifice/contextual gridworlds designed by us). The strongest remaining
reviewer objection is: "your criterion only works on environments built to
satisfy it." This experiment answers that objection by transferring the full
five-component criterion, with thresholds *unchanged* from the internal
battery, to a pre-existing external system family:

- Source: `examples_6.29_MARL_SWARM/swarm_decoy_abstraction.py` and
  `decoy_target_rl.py` (continuous 2-D N-vs-N swarm combat with decoy /
  threat / fragile enemy roles, strictly local observations, rule-based
  controllers, and a REINFORCE target-selection MARL learner trained with
  team reward only and no role labels).
- Nothing about the state space, action space, dynamics, or observation
  model is shared with the gridworld family. Continuous positions, damage
  exchange, N simultaneous agents, target-selection actions.

## 2. Environment variant (fixed before measurement)

`ContextualDecoyEnv`: the external decoy benchmark (`front_decoys=True`)
with one added latent episode context, sampled uniformly per episode and
never given to any controller as a label:

- `passive` context: front high-HP enemies are classic decoys
  (HP 4.0, damage 0.005). Engaging them wastes the horizon (trap).
- `aggressive` context: the same front enemies are fragile-but-deadly
  ambushers (HP 0.9, damage 0.16). Bypassing them means sustained transit
  damage; clearing them first is the better strategy.

Agents can sense per-enemy HP and damage locally (these features already
exist in the external project's design); the context is *latent to the
observer* at t=0 and must be marginalized, exactly as in the gridworld
`uncertain_preference` regime.

Fixed parameters (finalized after a rule-based feasibility pilot, before any
criterion measurement): `n_agents = n_enemies = 6`, `horizon = 25`,
`agent_damage = 0.08`, `threat_damage = 0.16`, `decoy_period = 3`,
`front_decoys = True`, `P(aggressive) = 0.5`. These are exactly the external
project's own "tough front" calibration (`decoy_tough_front_h25_*` in its
outputs), under which the decoy trap genuinely binds (nearest-only win rate
0.0 in the passive context in the external project's own results). The
feasibility pilot showed one refinement to the registered predictions:
because blind engagement is harmful in `passive` but helpful in
`aggressive`, the *context-averaged* usefulness gap for `nearest_only` can
partially cancel and its sign is not confidently predictable; the registered
rejection route for `nearest_only` is therefore selectivity (saturated
engagement, p ~= 1) plus endogeneity, with usefulness left unpredicted.
Thresholds remain copied from the internal battery, untouched.

## 3. Definitions (fixed)

- Trigger event: total damage dealt to decoy-role enemies during the
  episode exceeds 0.3 HP ("the swarm engaged the front enemies").
- Macro-basins B (4): {win, loss} x {engaged, bypassed} using the trigger
  event flag for engaged/bypassed.
- Episode score (reuses the external project's own reward shape;
  the decoy-engagement term has coefficient 0, so the trigger itself is
  never directly rewarded):
  `score = 10*win + 2*mission_hp_dealt + 3*mission_kills - deaths - 0.02*steps`
- do_trigger: while any decoy-role enemy is alive and visible, restrict
  each agent's target candidates to decoy-role enemies; after the decoys
  are cleared the controller acts naturally (minimal forced trigger).
- do_non_trigger: remove decoy-role enemies from every agent's target
  candidates for the whole episode; everything else is unconstrained
  (minimal forbidden trigger).
- P(B) estimation: `n_eval = 120` full-episode Monte Carlo rollouts per
  condition per system, marginalizing over the latent context and the
  environment seed. H0 = entropy of the natural-behavior basin
  distribution.

## 4. Thresholds (copied unchanged from the internal battery)

Identical to `criterion_ablation_battery.THRESHOLDS` (registered there
before that battery ran):

- potential:    H0 >= 0.5 bits
- selectivity:  4 p (1-p) >= 0.5 with p = natural trigger rate
- specificity:  JS(P(B) | do_trigger, P(B) | do_non_trigger) >= 0.2 bits
- usefulness:   counterfactual necessity > 0
                (mean natural score - mean do_non_trigger score)
- endogeneity:  no hand-coded target-priority semantics and no process
                reward on the trigger (design flag, fixed per system below)

Full criterion = all five pass. No threshold may be adjusted after seeing
external measurements.

## 5. Systems and label-assignment (audit) rules

Ground-truth labels are assigned by *behavioral audit*, independent of the
criterion internals, exactly as in the internal battery (the `noise_policy`
precedent). Audit rule for the learned system:

- If the trained MARL policy's per-context trigger rates separate
  (rate in `aggressive` >= 0.7 AND rate in `passive` <= 0.3), it has
  acquired the latent-conditional engagement structure: label = emergent (1).
- If the rates do not separate, training failed to acquire the structure:
  label = non-emergent (0), and the criterion is predicted to reject it.

| system | description | endogeneity flag | predicted label | predicted full-criterion verdict |
|---|---|---|---|---|
| marl_learned | REINFORCE scorer trained on contextual env, team reward only, no role labels | endogenous | 1 by audit rule above (0 if audit fails) | matches audited label |
| marl_untrained | randomly initialized scorer, greedy | endogenous | 0 | reject (predicted to fail usefulness and/or specificity) |
| nearest_only | external hand rule: attack nearest | prespecified | 0 | reject (saturated/blind engagement: fails selectivity and usefulness; also endogeneity) |
| role_oracle | external hand rule: threat > fragile > non_decoy > nearest | prespecified | 0 | reject (never engages: fails selectivity; also endogeneity) |
| damage_aware | hand rule: highest sensed damage first, then nearest | prespecified | 0 | reject via endogeneity ONLY (predicted to pass potential, selectivity, specificity, usefulness) |

Key falsifiable predictions:

1. The full criterion agrees with the audited label on all five external
   systems, with thresholds transferred unchanged.
2. `damage_aware` passes all four measured components and is excluded only
   by endogeneity — the external counterexample that pins the endogeneity
   component, mirroring `shaped_process` internally.
3. The usefulness gap is *sign-flipping across contexts* for forced
   engagement (harmful in `passive`, helpful in `aggressive`), mirroring
   the rescue/bridge sign flip in the gridworld family.

## 6. What counts as failure

- If the full criterion misclassifies any external system relative to its
  audited label, that is a registered failure and will be reported as such.
- If `marl_learned` fails the audit (no conditional structure learned), we
  report the training failure and the criterion's verdict on the actual
  (non-emergent) behavior; we do not retrain with different rewards to
  force a pass. Only optimizer hyperparameters (iterations, learning rate,
  batch) may be adjusted, since they do not touch the reward or the
  environment.
