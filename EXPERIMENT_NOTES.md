# Experiment Notes

## 2026-06-30: Learned PTC Smoke Run

Command:

```bash
python3 run_learned_ptc_experiment.py \
  --train_episodes 8000 \
  --eval_episodes 3000 \
  --seed 11 \
  --eval_temperature 0.35
```

Compact result:

```text
regime,potential_modes,natural_trigger,choice_tension,endogenous_score
pure_individual,1.0000,0.0000,0.0000,0.0000
pure_team,1.0030,0.9997,0.0013,0.0013
linear_mixed,1.1073,0.0000,0.0000,0.0000
dense_shaping,1.0000,0.0000,0.0000,0.0000
uncertain_preference,2.9997,0.3297,0.8839,2.6516
```

## Interpretation

This is the first learned-policy evidence that the story can be made
reviewer-facing:

- `pure_individual` suppresses sacrifice, as expected.
- `pure_team` almost always sacrifices, but that makes the behavior
  unsurprising. It is single-mode team optimization, not latent emergence.
- `linear_mixed` and `dense_shaping` can succeed under a forced trigger, but the
  learned natural policy does not choose the trigger. This distinguishes
  externally imposed counterfactual success from endogenous emergence.
- `uncertain_preference` preserves multiple natural futures and selects the
  trigger at a non-saturated rate, producing the only high endogenous emergence
  score in this toy setup.

The useful lesson is that `forced trigger success` is not enough. The stronger
evidence pattern is:

```text
multimodal prior
+ nonzero but non-mandatory natural trigger
+ trigger-specific future reorganization
```

## Current Weakness

The environment is still macro-action based. A skeptical reviewer could say the
basins are too directly encoded by the action names.

The next version should be a spatial gridworld where the same basins are inferred
from event sequences:

- selfish escape: agent 0 reaches a safe exit while agent 1 fails;
- direct teamwork: both agents take the visible bridge;
- sacrifice rescue: agent 0 blocks/opens/holds a costly switch and agent 1
  reaches a delayed high-value goal;
- failed noise: neither coherent macro path completes.

The metric should then cluster or classify basins from trajectories, not from a
single macro action.

## 2026-06-30: Spatial PTC Smoke Run

Command:

```bash
python3 spatial_sacrifice_gridworld.py \
  --train_episodes 80000 \
  --eval_episodes 3000 \
  --seed 23 \
  --eval_temperature 0.25
```

Compact result:

```text
regime                 potential  natural_trigger  tension  necessity  score
pure_individual        1.0000     0.0000           0.0000   3.9997     0.0000
pure_team              1.0056     0.9993           0.0027   9.0000     0.0027
linear_mixed           1.2397     0.0000           0.0000   4.0033     0.0000
dense_shaping          1.2232     0.0500           0.1900   9.0000     0.2324
uncertain_preference   2.9993     0.3370           0.8937   5.6580     2.6526
random_noise           1.0741     0.9883           0.0461   7.5040     0.0493
```

## Interpretation

This is the second benchmark and the first spatial one. It strengthens the
evidence because the basin is not selected by a macro action. The agents must
move through a grid, and the sacrifice basin is inferred from the event chain:

```text
a0_step_on_sacrifice_switch -> hidden_gate_opens -> a1_reaches_high_value_goal
```

Main evidence:

- `pure_team` learns sacrifice almost deterministically, but this is not
  surprising; trigger choice tension is almost zero.
- `random_noise` also triggers almost deterministically after long training, but
  its potential remains near one mode. Noise is not structured possibility.
- `dense_shaping` occasionally triggers, but the potential and score remain far
  below `uncertain_preference`.
- `uncertain_preference` is the only condition that combines high potential,
  non-saturated natural trigger choice, high trigger specificity, and high
  counterfactual necessity.

This gives a stronger reviewer-facing distinction:

```text
team reward can make sacrifice optimal;
noise can make behavior irregular;
dense shaping can make coordination easy;
but only uncertain preference currently preserves structured multimodal choice.
```

## Next Weakness

The spatial benchmark is still tabular and centralized. The next evidence level
should add:

- multi-seed confidence intervals;
- decentralized learner variants;
- event-sequence clustering instead of hand-coded basin labels;
- direct connection to existing RUSP/OES/Event Order experiments.

## 2026-06-30: Spatial 2-Seed Sweep Smoke

Command:

```bash
python3 run_spatial_sweep.py \
  --seeds 23,29 \
  --train_episodes 25000 \
  --eval_episodes 1200 \
  --eval_temperature 0.25
```

Compact result:

```text
regime                 n  potential  trigger_rate  tension  necessity  score
dense_shaping          2  1.0609     0.0108        0.0428   9.0000     0.0457
linear_mixed           2  1.1899     0.0000        0.0000   6.2846     0.0000
pure_individual        2  1.0000     0.0000        0.0000   3.9475     0.0000
pure_team              2  1.4047     0.1400        0.4032   6.4967     0.7260
random_noise           2  1.0160     0.5000        0.0000   3.6650     0.0000
uncertain_preference   2  2.9979     0.3408        0.8987   2.1983     0.6607
```

This is only a smoke test. It confirms the multi-seed pipeline works, but it
also shows that short training can leave `pure_team` unstable across seeds.
Formal evidence should use longer training, more seeds, and confidence
intervals. The useful engineering lesson is that single-seed evidence is not
acceptable for the target venue.

## 2026-06-30: Spatial 5-Seed Stratified Sweep

After the 2-seed smoke test, the spatial benchmark was updated to use
stratified training contexts for `uncertain_preference`. Evaluation still uses
stochastic context sampling, but training no longer depends on whether a seed
happened to under-sample the latent-sacrifice context.

Command:

```bash
python3 run_spatial_sweep.py \
  --seeds 23,29,31,37,41 \
  --train_episodes 80000 \
  --eval_episodes 2500 \
  --eval_temperature 0.25
```

Compact result:

```text
regime                 n  potential  trigger_rate  tension  necessity  score ± 95%CI
dense_shaping          5  1.1388     0.0287        0.1110   9.0000     0.1287 ± 0.0842
linear_mixed           5  1.2202     0.0000        0.0000   4.0052     0.0000 ± 0.0000
pure_individual        5  1.0007     0.0000        0.0000   3.9969     0.0000 ± 0.0000
pure_team              5  1.0040     0.9995        0.0019   9.0000     0.0019 ± 0.0017
random_noise           5  1.3745     0.7193        0.2443   8.6879     0.4733 ± 1.0820
uncertain_preference   5  2.9953     0.3427        0.9006   6.8512     2.6635 ± 0.0663
```

This is the first result that begins to look like a stable controlled benchmark:

- `uncertain_preference` has the highest endogenous emergence score with a tight
  confidence interval.
- `pure_team` has high counterfactual necessity, but no trigger choice tension:
  sacrifice is mandatory, not surprising.
- `dense_shaping` has perfect counterfactual necessity under forced trigger, but
  the natural trigger rate remains low and the score is far below
  `uncertain_preference`.
- `random_noise` can sometimes create trigger behavior, but the confidence
  interval is huge and the potential is far lower. This supports the claim that
  noise is not structured possibility.

Current strongest reviewer-facing sentence:

```text
Across five seeds in a spatial sacrifice benchmark, uncertain preference is the
only condition that simultaneously maintains approximately three effective
future modes, a non-saturated natural trigger rate, high trigger specificity,
and stable endogenous emergence scores.
```

Remaining weakness:

- five seeds is still small for a final paper;
- the learner is centralized tabular Q-learning;
- basin classification is hand-coded from events;
- we still need a second non-toy MARL environment and a Go/search-tree external
  validation.

## 2026-06-30: Performance-Aware Spatial Sweep

The previous metrics were too focused on emergence signatures. A reviewer can
reasonably ask: does the surprising trigger matter for actual outcomes?

The spatial benchmark now reports:

- `natural_team_return_mean`: factual performance of the learned natural policy;
- `sacrifice_basin_rate`, `team_direct_basin_rate`, `selfish_basin_rate`: what
  actually happened;
- `counterfactual_necessity`: forced-trigger return minus forced-non-trigger
  return;
- `retrospective_importance`: within natural rollouts, team return of
  sacrifice-rescue trajectories minus non-sacrifice trajectories;
- `retrospective_gain_per_local_cost`: retrospective importance normalized by
  the immediate local switch cost.

Command:

```bash
python3 run_spatial_sweep.py \
  --seeds 23,29,31,37,41 \
  --train_episodes 80000 \
  --eval_episodes 2500 \
  --eval_temperature 0.25
```

Compact result:

```text
regime                 return  sacrifice  necessity  retro_importance  emergence_score
dense_shaping          6.0858  0.0287     9.0000     3.0003            0.1287
linear_mixed           4.1008  0.0000     4.0052     0.0000            0.0000
pure_individual        4.0002  0.0000     3.9969     0.0000            0.0000
pure_team              8.9986  0.9995     9.0000     4.2000            0.0019
random_noise           8.0359  0.7193     8.6879     4.9918            0.4733
uncertain_preference   6.3406  0.3427     6.8512     4.0465            2.6635
```

Important interpretation:

- If the claim were "uncertain preference maximizes team return", the current
  evidence would fail: `pure_team` has the highest natural return.
- The current claim is different: `pure_team` is high-performing but
  unsurprising because sacrifice is almost mandatory; `uncertain_preference` is
  where sacrifice remains a real choice and still has strong retrospective
  importance.
- This means the current benchmark supports "performance-relevant emergence
  signature", not "best-performing algorithm".

This is a useful but incomplete result. For a Nature-level story, we need a
contextual benchmark where always sacrificing is not best. In that benchmark, a
good emergent trigger should be:

```text
rare enough to be non-obvious,
selective enough to avoid over-sacrifice,
and important enough to improve final performance when it appears.
```

## 2026-06-30: Contextual Selective-Trigger Benchmark

A new benchmark was added in `contextual_sacrifice_gridworld.py`.

This benchmark directly addresses the concern that "learning sacrifice" is not
meaningful by itself. The episode has a visible mode:

- `rescue`: the switch is locally costly but opens the high-value goal.
- `bridge`: the switch is a decoy; direct teamwork is better.

Therefore, always sacrificing is no longer the right story. The desired behavior
is selective triggering.

Smoke command:

```bash
python3 run_contextual_sweep.py \
  --seeds 53,59,61 \
  --train_episodes 60000 \
  --eval_episodes 2000 \
  --eval_temperature 0.25
```

Compact result:

```text
regime                 return  rescue  bridge  over_sacrifice  selective  emergence
dense_shaping          9.4942  0.4992  0.5008  0.0000          1.0000     1.5489
linear_mixed           5.7207  0.0000  0.4303  0.0000          0.0000     0.0000
pure_individual        4.0027  0.0000  0.0007  0.0000          0.0000     0.0000
pure_team              9.5000  0.5000  0.5000  0.0000          1.0000     1.5443
random_noise           8.9452  0.4978  0.4375  0.0138          0.8707     1.0354
uncertain_preference   6.1083  0.1110  0.5757  0.0000          0.2421     0.3305
```

Interpretation:

- This benchmark shows that PTC metrics can be tied to factual performance:
  high-performing policies must trigger in rescue mode and avoid over-triggering
  in bridge mode.
- `pure_team` and `dense_shaping` become strong here. That is not a problem; it
  is evidence that the framework is not a method advertisement for uncertain
  preference.
- `uncertain_preference` is not universally better. It preserved latent
  possibility in the previous spatial benchmark, but in this contextual task it
  underperforms because it does not reliably learn the rescue trigger.
- `random_noise` obtains reasonable return but shows over-sacrifice and lower
  selectivity, which helps separate noisy success from structured triggering.

Reviewer-facing lesson:

```text
The phenomenon is not sacrifice itself. The phenomenon is selective
actualization: a locally costly trigger becomes meaningful only in the contexts
where it reorganizes the future into a higher-performing trajectory.
```

This is closer to the Nature-level story because it connects the emergence
signature to factual task performance.

## 2026-06-30: Possibility Preservation Tree

The user reframed the theory as a mathematical problem:

```text
The current optimum can cause final suboptimality.
Preserving possibility can improve the final outcome.
```

`possibility_preservation_tree.py` isolates this idea.

Command:

```bash
python3 possibility_preservation_tree.py --episodes 5000 --seed 71
```

Compact result:

```text
policy                   immediate  expected  option_value  success  emergence
myopic_greedy             5.0000     5.0000    0.0000        0.0000   0.0000
always_trigger           -1.0000     5.4428    0.4428        0.4956   0.9911
always_direct            -1.0000     3.5450   -1.4550        0.5050   0.0000
random_preserve          -1.0000     4.4124   -0.5876        0.4900   1.0336
possibility_preserving   -1.0000    10.0088    5.0088        1.0000   1.9999
```

Interpretation:

- `myopic_greedy` is locally optimal because it has the best immediate reward.
  But it closes future options and ends at return 5.
- `possibility_preserving` accepts an immediate local loss and obtains expected
  return around 10 after context is revealed.
- `always_trigger` and `random_preserve` show that preserving possibility is not
  enough by itself. The policy must preserve the option and then select the
  correct future basin.

This gives the cleanest mathematical statement so far:

```text
Emergence-relevant actions can be locally suboptimal because their value is not
in immediate reward, but in the future option set they keep alive.
```

This benchmark should become the conceptual bridge between the theory and the
larger MARL/Go experiments.

## 2026-06-30: Closed-Form Possibility Ablation

Concern:

```text
We cannot compare RL against a heuristic and claim that the theory wins.
The variables must be controlled.
```

`possibility_ablation.py` addresses this by removing learning entirely. All
policies are evaluated by closed-form expected return in the same finite-horizon
tree.

Formula:

```text
V_cash = R_cash

V_preserve = -c + p * R_trigger + (1 - p) * R_direct

preserve wins iff
  -c + p * R_trigger + (1 - p) * R_direct > R_cash
```

The local optimality trap region is:

```text
R_cash > -c        # cash-out is better immediately
V_preserve > V_cash  # preserving options is better finally
```

Default command:

```bash
python3 possibility_ablation.py
```

Default result:

```text
n_conditions                         4275
possibility_preserving_win_rate      0.7364
local_optimum_trap_rate              0.7319
mean_positive_option_value           4.8023
max_option_value                    12.8000
```

Boundary commands:

```bash
python3 possibility_ablation.py \
  --cash_start 6 --cash_stop 18 --cash_step 0.5 \
  --output_dir outputs/ablation_high_cash

python3 possibility_ablation.py \
  --cost_start 3 --cost_stop 8 --cost_step 0.5 \
  --output_dir outputs/ablation_high_cost
```

Boundary result:

```text
condition           win_rate  trap_rate  mean_positive_option_value
default             0.7364    0.7319     4.8023
high_cash_out       0.2573    0.2573     2.0841
high_preserve_cost  0.4568    0.4568     3.2171
```

Interpretation:

- The result is not caused by a training algorithm.
- The result is not caused by a heuristic having privileged information.
- The win region contracts when the immediate cash-out reward increases.
- The win region contracts when preservation cost increases.
- This matches the analytic inequality exactly.

Reviewer-facing claim:

```text
Possibility preservation is valuable in a precisely characterizable region:
when the expected value of future context-conditioned options exceeds the
immediate local optimum. PTC metrics are then used to detect this structure in
learned and spatial systems.
```

## 2026-06-30: Same-Solver Planning-Horizon Ablation

Concern:

```text
Even closed-form policies may be seen as different heuristics.
Can the same solver change its decision only because it sees farther?
```

`planning_horizon_ablation.py` uses the same Bellman solver, same environment,
and same reward function. The only variable is planning horizon.

Default command:

```bash
python3 planning_horizon_ablation.py
```

Default result:

```text
n_conditions                 4275
horizon_reversal_rate        0.5158
h2_preserve_rate             0.5158
mean_positive_option_value   3.6693
max_option_value            12.3500
```

Boundary result:

```text
condition           reversal_rate  mean_positive_option_value
default             0.5158         3.6693
high_cash_out       0.0966         1.7013
high_preserve_cost  0.2457         2.4512
```

Interpretation:

- Horizon 1 sees only immediate reward and chooses `cash_out` in the reversal
  region.
- Horizon 2 sees the context-conditioned future option and chooses
  `preserve_option`.
- Increasing cash-out value or preservation cost shrinks the reversal region.

Reviewer-facing claim:

```text
The result does not depend on comparing different algorithms. Under the same
Bellman optimality principle, expanding the visible future transforms a locally
optimal cash-out action into a globally suboptimal trap.
```
