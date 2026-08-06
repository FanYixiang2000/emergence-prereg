# Contextual LBF six-component confirmation protocol

Status: author-maintained protocol frozen before training or evaluating any
fresh confirmation seed listed below.

Frozen: 2026-07-11. This is not a third-party timestamped registered report.

Frozen implementation:
`contextual_lbf_transfer.py`, SHA-256
`6864ea312ee5017917b84a0d270a479096fd982f67f0cec1bf264f39521d2089`.

## Pilot disclosure

Two design pilots were run before this freeze and are excluded from
confirmatory counts.

1. Seed 101 used a 12-step horizon, border-food layouts and binary-clearance
   value. The layout violated the benchmark's normal interior-food convention
   and the policy could not clear the task. It was rejected.
2. Seed 102 used the final interior layouts, 15-step horizon and the
   training-aligned discounted sparse reward. It passed all six components;
   all four controls were rejected. This one-seed result established
   feasibility but is not confirmation.

No threshold was tuned on either pilot. Thresholds are copied from the
previously frozen refined six-component criterion.

## Task

Base benchmark: `Foraging-5x5-2p-2f-coop-v3` from lbforaging 2.0.0.
Dynamics, standard observations, six discrete actions and sparse collection
reward are unchanged.

The wrapper balances two unlabeled geometric contexts:

- context 0: lexicographic food identity 0 is nearer to the team;
- context 1: food identity 1 is nearer.

The context label is never passed to a controller. Standard local observations
contain player/food geometry. Two interior layouts are used per context,
selected deterministically from the episode seed. Both foods require the two
level-1 players to cooperate. Episode horizon is 15.

Training: the existing MAPPO-style PPO implementation, 8,000 episodes per
policy seed, no trigger/process reward, no context label, all hyperparameters
frozen in code.

## Trigger, basins, value and interventions

- Trigger: food 0 is collected before food 1.
- Basins: `{win, loss} x {food0-first, food1-first}`.
- Value: discounted mean sparse team reward with the training discount
  `gamma=0.98`.
- `do_trigger`: until food 0 is collected, force the team's minimal
  cooperative path/load actions toward food 0, then release.
- `do_non_trigger`: the analogous intervention toward food 1.

The intervention planner is shared across systems and does not enter natural
learned-policy episodes. Invalid LOAD actions are mapped to NONE, matching the
environment's declared valid-action set and avoiding an lbforaging border/load
implementation error.

## Systems and expected labels

1. `learned`: PPO policy, expected emergent.
2. `initial_twin`: identical architecture reconstructed at the learned
   policy's initialization seed, expected non-emergent.
3. `team_nearest`: competent hand-coded geometry-sensitive coordinator,
   expected non-emergent through endogeneity and acquisition.
4. `fixed_food0`: hand-coded fixed target order, expected non-emergent.
5. `fixed_food1`: hand-coded fixed target order, expected non-emergent.

## Frozen six-component rule

- Potential: natural basin entropy >= 0.5 bits.
- Conditional selectivity:
  `|P(trigger|context0)-P(trigger|context1)| >= 0.5`.
- Specificity:
  `JS(P(B|do_trigger), P(B|do_non_trigger)) >= 0.2` bits.
- Usefulness:
  natural discounted score minus `do_non_trigger` score > 0.
- Endogeneity: learned/initial neural systems have no process reward or
  hand-coded trigger; scripted systems fail this component.
- Acquisition:
  learned selectivity minus its same-seed initialization twin >= 0.3.
  Static controls have acquisition 0.

Evaluation uses 80 episodes per context and condition. Episodes quantify
measurement uncertainty conditional on a policy. The training seed is the
population-level unit.

## Fresh confirmation seeds

`1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110`

No seed may be retrained or replaced because of an outcome. Crashes may be
rerun only with the same seed and unchanged code.

## Prospective predictions

CLBF-C1. The learned policy passes the full six-component rule on at least
9/10 fresh seeds. This gives an exact one-sided positive-seed sign-test
`p <= 11/1024 = 0.0107`; 8/10 does not meet the registered population-level
target.

CLBF-C2. The initialization twin and all three scripted controls are rejected
on every seed (40/40 non-learned verdicts).

CLBF-C3. `team_nearest` passes the four behavioral components and fails exactly
`{endogeneity, acquisition}` on at least 9/10 seeds.

CLBF-C4. Every learned seed has positive acquisition, and every initialization
twin fails acquisition.

CLBF-C5. The learned natural trigger rate is higher in context 0 than context 1
on every seed. Pooled usefulness is positive on at least 9/10 seeds.

CLBF-C6. A nonparametric bootstrap over the ten seed-level learned metrics has
a positive lower 95% bound for acquisition and usefulness. Episode-level
resampling is secondary and conditional on the trained policies.

## Failure handling

- All failed predictions remain failures.
- No change to layouts, horizon, controller, score, thresholds, evaluation
  count or seeds is allowed after this freeze.
- Any mechanics-only fix must be documented, must not inspect confirmation
  labels to choose behavior, and invalidates the code hash above.
- The existing non-contextual LBF probe remains a four-prediction mechanism
  instrument. It is not retroactively relabeled as a six-component test.

