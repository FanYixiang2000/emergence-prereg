# Pre-registration: within-episode collapse in a deep MARL system

Written and frozen BEFORE the probe below was run on any trained policy.
A feasibility pilot may tune TRAINING hyperparameters (PPO learning
rate, entropy bonus, epochs, steps) and ESTIMATOR parameters (rollouts
per step, probe temperature) only -- never the basin definition, the
do-operators, the success criteria, or the predictions.

## Why this experiment

The reviewer gap: the framework's mechanism-level evidence (within-
episode P_t(B) with do-operators) so far comes from tabular/DQN
gridworlds we designed. This experiment repeats the same measurement in
a standard external deep MARL benchmark nobody in this project designed:
PettingZoo/MPE simple_spread (3 agents, 3 landmarks, cooperative
coverage), with parameter-shared PPO neural policies -- a modern
(if small) deep MARL stack: neural policies, policy-gradient training,
continuous state, decentralized execution.

## The coordination possibility space

simple_spread has a built-in symmetry: agents are interchangeable, so
which agent ends up covering which landmark is undetermined at episode
start -- a genuinely multimodal future. The macro outcome (basin) of an
episode is the ASSIGNMENT: for each agent, the index of its nearest
landmark at the final step (27 outcomes for 3x3). "Coverage" (the
useful macro structure) = the assignment is a bijection. Landmark
positions are randomized every episode, so every episode presents a
fresh possibility space.

## Measurement (identical logic to within_episode_collapse_probe.py)

At every step t of an evaluation episode, snapshot the world state and
estimate P_t(B) by M Monte-Carlo rollouts of the learned stochastic
policy to episode end (the policy's own categorical distribution; the
environment dynamics are deterministic, so all openness is policy
openness = the agents' own undecided coordination).

    potential_t   = H(P_t(B)) bits          (27-way, max 4.75)
    collapse_t    = KL(P_{t+1} || P_t)
    commit step   = argmax_t collapse_t     (the symmetry-breaking step)
    P_t(win)      = sum of bijection-basin mass

Do-operators at the commit step, for agent 0 with realized final
landmark L (minimal interventions, mirroring do_trigger/do_non_trigger):

    do_commit : agent 0 moves greedily toward L for the remaining steps;
                other agents follow the policy
    do_block  : agent 0's policy is renormalized over actions that do
                not decrease distance to L; other agents follow policy

## Conditions

- `trained` (3 PPO seeds): full training on the standard reward.
- `untrained`: same architecture, random initialization.
- `greedy_nearest`: hand-scripted controller, every agent moves toward
  its CURRENT nearest landmark every step (locally optimal, no
  commitment structure beyond the initial geometry).
- `noise`: uniform random actions.

## Registered predictions

- D1 POTENTIAL: for trained policies, mean early-episode potential
  (steps 0-2) >= 1.0 bits -- the assignment really is open at start
  (untrained/noise are expected to have even higher potential; that
  alone must not qualify them, matching the quiet-chess-position
  lesson).
- D2 USEFUL COLLAPSE: trained policies collapse P_t(B) toward
  bijection basins: mean P_final(win) - P_0(win) > 0 and final
  bijection frequency >= 0.5 across episodes; noise and untrained
  stay below 0.35 final bijection frequency (3! / 27 = 0.22 is the
  uniform-assignment baseline).
- D3 LOCALIZATION + COUNTERFACTUAL: at the commit step of trained
  policies, do_commit vs do_block changes the win-basin probability:
  median [P(win | do_commit) - P(win | do_block)] > 0 across episodes,
  a majority of episodes positive (sign test p < 0.05, pooled over
  seeds).
- D4 GREEDY CONTRAST: the greedy_nearest controller reaches LOWER
  final bijection frequency than trained policies (duplicate targeting
  from symmetric geometry is not resolved by local greed), and its
  early potential is lower than the trained policy's (its future is
  fixed by the initial geometry: no genuine openness, convergence
  without possibility).

## Failure handling

As always: failed predictions are registered failures, reported with
their route; no threshold or definition changes in response.

## Outcome (recorded after the three-seed run; nothing above edited)

Training pilots and one estimator bug are documented in the script
docstring (allowed scope only: critic upgraded to MAPPO-style
centralized critic -- training only, execution and probes remain
decentralized -- plus LR/entropy annealing; the estimator bug made all
futures degenerate and was fixed before any registered run).
Aggregate over seeds 11/22/33 (`deep_marl_collapse_aggregate.json`):

- D1 PASS. Trained early potential 1.41-1.60 bits >= 1.0 on every
  seed; untrained (2.75) and noise (2.72) are higher, and as
  registered that alone does not qualify them (the chess quiet-position
  lesson, reproduced in a learned system).
- D2 **REGISTERED FAILURE**, two informative routes. (a) Final
  bijection rates 0.50 / 0.45 / 0.525: one seed sits 0.05 below the
  frozen 0.5 threshold (controls: untrained 0.25, noise 0.20 -- the
  ordering is clean, the margin is not met on 1/3 seeds). (b) The
  within-episode win-mass shift P_end(win) - P_0(win) was positive on
  only 1/3 seeds. Post-mortem: for a CONVERGED policy, P_t(win)
  estimated by rolling out that same policy is approximately a
  martingale -- P_0(win) already prices in the coordination the policy
  will perform, so no systematic rise should have been predicted. The
  quantity that does fall within episodes is the assignment ENTROPY
  (potential), and the collapse localizes at the commit step. The
  prediction was mis-designed for converged policies; reported as
  failed, with this route, and the martingale point is itself a
  finding worth keeping (possibility collapse under a converged policy
  shows up as entropy contraction at constant win-mass, not as
  win-mass growth).
- D3 PASS (the load-bearing counterfactual). At the commit step,
  do_commit vs do_block on agent 0 shifts the win-basin probability by
  a pooled median of +0.083 (per-seed medians +0.094 / +0.104 /
  +0.063), sign test 74 wins / 44 losses, pooled p = 0.0037 < 0.05.
  The symmetry-breaking commitment is counterfactually load-bearing in
  a neural MARL system. Inference note added during manuscript audit:
  the pooled episode p value is conditional on the three trained
  policies; the three positive seed medians are the appropriate
  training-randomness summary, and three seeds do not support a
  population-level significance claim.
- D4 PASS. The scripted greedy-nearest controller reaches bijection
  0.425 < every trained seed's rate, with early potential 0.0
  (deterministic given the initial geometry): convergence without
  possibility, exactly the registered contrast.

Net: 3/4 registered predictions pass; the failed one (D2) fails partly
by a 0.05 margin on one seed and partly by a prediction-design error
whose diagnosis (the martingale property of P_t(win) under the
behaving policy) is consistent with the C5 chess lesson: in systems
where the observer's rollout policy already contains the trigger,
useful collapse must be read from do-contrasts (D3, which passed), not
from shifts against the endogenous baseline.
