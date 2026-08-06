# Pre-registration: LBF cross-task deep MARL collapse probe

Status: FROZEN before any trained policy is probed. Written 2026-07-07.
Amendments after this point are labeled and dated; outcomes are recorded
in place after the main run.

## Why this experiment

The deep-MARL evidence (fig35) currently rests on one task
(PettingZoo/MPE simple_spread). The predictable referee objection is
"single-task demo". This probe repeats the SAME measurement pipeline --
P_t(B | s_t) by Monte-Carlo rollouts of the behaving policy from
snapshotted world states, do-operator contrasts at the maximal-collapse
step -- on a structurally different cooperative benchmark we did not
design: Level-Based Foraging (lbforaging 2.0.0, Christianos et al.,
the EPyMARL benchmark task family), gym id
`Foraging-5x5-2p-2f-coop-v3`.

Structural differences that make this a genuine cross-task test:

- discrete grid positions, not continuous positions/velocities;
- an irreversible resource-consumption event (a food, once loaded,
  is gone) instead of continuous coverage;
- FORCED cooperation: in the coop variant every food's level equals
  the sum of the players' levels, so no single agent can ever load a
  food alone -- the useful structure is a synchronized joint commitment,
  not a spatial assignment;
- sparse reward only at collection (no local_ratio shaping term).

## Observer possibility space

- Basin of an episode = the ORDER in which foods are consumed, as a
  tuple of food indices (foods indexed by their spawn position, sorted
  lexicographically at episode start):
  (), (0,), (1,), (0,1), (1,0) -- 5 outcomes.
- Win basins = full clearance: (0,1) or (1,0).
- P_t(B | s_t): from a world snapshot at step t (field matrix, player
  positions/levels, step counter), roll the behaving controller to the
  episode horizon N=48 times; count basins reached (foods already eaten
  before t keep their recorded order prefix).
- Potential_t = H(P_t(B)) in bits (max log2 5 ~ 2.32).
- Collapse_t = KL(P_t || P_{t-1}) in bits; commit step = argmax over t
  of Collapse_t.
- Do-operators at the pre-commit snapshot, applied to agent 0 with the
  target food = the next food actually consumed in the behaving episode
  (if none was consumed after the commit step, the nearest remaining
  food to agent 0):
  - do_commit: agent 0 moves greedily toward the target food and LOADs
    whenever adjacent; agent 1 stays on-policy.
  - do_block: agent 0's policy is renormalized over actions that do not
    decrease its distance to the target food, with LOAD forbidden while
    adjacent to it; agent 1 stays on-policy. Minimal restriction --
    everything else remains on-policy.

## Conditions

- trained_seed{11,22,33}: parameter-shared PPO actor on local
  observations with a MAPPO-style centralized critic (training only;
  execution and all probe rollouts decentralized), trained on the coop
  task with the environment's own sparse reward. No shaping, no process
  reward: endogeneity holds by construction.
- untrained: same architecture, freshly initialized.
- greedy_nearest: scripted -- each agent moves toward its nearest
  remaining food and LOADs when adjacent (no coordination protocol).
- noise: uniform random actions.

Training and estimator hyperparameters (episode budget, LR schedule,
entropy schedule, probe temperature implicit in the stochastic policy,
rollout count) are pilot-tunable BEFORE the main measurement, exactly
as in the simple_spread and chess protocols; pilot logs are kept and
cited. Basins, win definition, do-operators, and the four predictions
below are frozen now and not tunable.

## Registered predictions

- L1 (potential): trained policies keep genuine order-openness early in
  the episode: mean early potential (steps 0-2) >= 0.8 bits in every
  seed. (The two win basins alone give 1.0 bit if both orders stay
  reachable.)
- L2 (useful structure): trained full-clearance rate >= 0.5 in every
  seed; untrained and noise < 0.2. NOTE the martingale lesson from
  chess C5 / simple_spread D2 is priced in: we do NOT predict a
  within-episode rise of P_t(win) under the behaving policy.
- L3 (counterfactual necessity): pooled over trained seeds, at the
  pre-commit snapshot, median[ P(win | do_commit) - P(win | do_block) ]
  > 0 with two-sided sign-test p < 0.05. This is the registered
  usefulness reading (do-contrast, not observed improvement).
- L4 (double dissociation vs scripted greed): greedy_nearest shows
  LOWER early potential than every trained seed (deterministic policy,
  collapsed openness) AND a lower full-clearance rate than every
  trained seed (in the coop task uncoordinated greed cannot load any
  food unless both agents happen to choose the same one and LOAD in
  the same step).

## Failure handling

Any failed prediction is reported as a REGISTERED FAILURE with
diagnosis, exactly as R3/R5/T2/T3/T6/C5/D2 were. If PPO cannot reach a
competent policy on the coop task within the pilot budget (sparse
cooperative reward is known-hard), the fallback -- registered here,
before any run -- is the non-coop variant `Foraging-5x5-2p-2f-v3` with
the same basins, predictions, and thresholds; the switch, if taken,
will be reported with the failed coop training log.

## Outcomes

Main run 2026-07-07 (`lbf_collapse_probe.py --seeds 11 22 33
--train_episodes 8000 --tag main`; outputs/lbf_collapse_main.json;
training + estimator pilots documented in the script docstring: probe
temperature frozen at 6.0 after a sweep with a saved pilot net, and a
do-operator release-on-consumption fix, both before the main run).

- L1 PASS. Early potential 1.456 / 1.413 / 1.419 bits (threshold 0.8)
  -- both consumption orders genuinely open at episode start in every
  seed.
- L2 PASS. Trained full-clearance rate 0.90 / 0.90 / 0.93 (threshold
  0.5); untrained 0.033, noise 0.0 (threshold < 0.2). The coop
  fallback registered in "Failure handling" was NOT needed.
- L3 PASS. Pooled do-contrast median +0.042; 69 positive, 21 tied and
  0 negative episode contrasts; all three policy-seed medians are
  positive. The original pooled episode sign test is p = 3.4e-21.
  Inference note added during manuscript audit: that p value is
  conditional on the three trained policies and is not a
  population-level test over training seeds.
- L4 PASS. greedy_nearest: win rate 0.10 with 0.0 bits early potential
  vs trained minimum 0.90 / 1.413 bits -- the double dissociation
  (openness without usefulness in noise; usefulness never reached by
  blind greed; both only in the trained system) reproduces.

4/4 -- the first domain in the project where every registered
prediction passed. Consistent with the martingale lesson (chess C5,
simple_spread D2) having been priced into L2's design: no
within-episode P_t(win) rise was predicted, and usefulness was read
exclusively from the do-contrast.
