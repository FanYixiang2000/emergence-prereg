# Preregistration: full six-component criterion on unmodified Overcooked-AI

Version: 1.0 (FINAL, to be frozen by external timestamp)
Prepared: 2026-07-18

FREEZE PROCEDURE. This document is frozen when it is pushed to the
public repository and tagged. No confirmatory training seed is launched
before that timestamp exists; the confirmation script refuses to run
without an explicit `--confirm-frozen` flag asserting this. After the
timestamp, nothing in this document changes; failures are retained.

## 1. Why this experiment

Every existing full six-component domain in the project (external
swarm, Contextual LBF, latent-context sequence model) uses
author-built wrappers. Overcooked-AI is a public coordination
benchmark with community-defined dynamics and sparse reward. This
experiment runs the frozen criterion on the UNMODIFIED benchmark:
original layouts, original dynamics, original sparse delivery reward,
no wrapper environments.

## 2. Disclosed design pilots (all excluded from confirmation)

- Single-layout trainability (seed 8801): self-play PPO with annealed
  community shaping reaches ~130 (cramped_room) and ~150
  (asymmetric_advantages) sparse reward per 400-step episode.
- Scripted role pairs (this repo's BFS cook/server subpolicies)
  deliver on both layouts; role asymmetry measured: in
  asymmetric_advantages the agent-0-cooks allocation scores 440 vs 240
  for the reverse; cramped_room is role-indifferent (220/220).
- Mixed-context pilots (seeds 8901-8904, 3M steps): the shared policy
  learns a context-dependent role allocation -- final first-potter
  agent-0 rates 0.68-1.0 in asymmetric_advantages vs 0.0-0.47 in
  cramped_room; cross-context separation 0.53-0.85 (>= 0.5 in 4/4
  seeds). Two seeds underlearned deliveries in one of the two layouts
  at 3M steps (sparse ~0 in that layout), motivating the 5M-step
  budget below; training failures in confirmation seeds are NOT
  excluded -- they count against OC-1 as specified. A
  forced_coordination pair never learned deliveries on
  forced_coordination and is dropped.
- One end-to-end criterion evaluation on a pilot policy (seed 8901
  checkpoint): potential 0.98 bits, selectivity 0.85, specificity
  1.00 bits, usefulness +90.0.

## 3. Frozen protocol

Environment: `overcooked_ai_py` (PyPI overcooked-ai), horizon 400,
layouts `cramped_room` (context 0) and `asymmetric_advantages`
(context 1), drawn uniformly per episode during training. The layout
is visible only through the standard 96-dim featurized observation;
no context label exists anywhere.

Training per seed: self-play PPO, shared parameters (the repo's
`overcooked_confirmation.py::train_mixed`, hash-anchored), 5,000,000
environment steps, community shaped rewards annealed linearly to zero
over the first 60% of steps, sparse delivery reward untouched. No
reward term references agent identity: the role allocation is never
prespecified.

Confirmation seeds: 77001..77012 (twelve). Pilot seeds excluded.

Candidate macro-structure and trigger: role allocation; trigger =
agent 0 is the first to place an ingredient in a pot.

Observer contract A: basins = (first potter: agent0/agent1/none) x
(>= 1 delivery or not), 6 cells; value = sparse team reward per
episode; interventions do-commit / do-block as implemented in
`overcooked_criterion.py` (minimal, releasing); rollout model = the
policy as trained (T=1); 40 evaluation episodes per context per
condition.

Observer contract B (re-evaluation only): value = delivery-success
indicator; horizon 300; all else as contract A.

Thresholds, copied unchanged from the frozen criterion: potential
>= 0.5 bits; conditional selectivity >= 0.5; specificity JS >= 0.2
bits; usefulness > 0; endogeneity (learned, not scripted/forced);
acquisition >= 0.3 selectivity gain over the initialization twin.

Systems per seed (5): learned; initialization twin (same architecture,
same torch seed, untrained); scripted role pair (cook = agent 0);
behavioural clone of the scripted pair (supervised, fresh net);
untrained-other (different random initialization).

## 4. Registered predictions

    OC-1  learned accepted (all six components) on >= 8/12 seeds.
    OC-2  all 48 control verdicts are rejections (4 controls x 12
          seeds); control failure routes are reported descriptively.
    OC-3  trigger direction matches the pilot: first-potter-agent-0
          rate is higher in asymmetric_advantages than in cramped_room
          on >= 10/12 seeds.
    OC-4  under contract B, all 12 initialization twins remain
          rejected (the declared-value change never rescues a twin).
    OC-5  seed-level primary inference: the learned usefulness
          do-contrast is positive on >= 10/12 seeds (exact one-sided
          sign test p < 0.05 at n = 12).

Any miss is retained as a registered failure. The population unit is
the training seed; episode-level statistics are conditional
measurement uncertainty only.

## 5. Analysis and reporting

Per-seed component values, verdicts and failure routes; seed-level
bootstrap CIs for usefulness and acquisition; both contracts reported;
all outputs hash-anchored in the manifest. The result enters the
manuscript as the public-environment confirmation figure regardless of
outcome.

## 6. OUTCOMES (recorded 2026-07-19, after the frozen run)

External timestamp: `github.com/FanYixiang2000/emergence-prereg`, tag
`v1.0-overcooked-prereg`, commit `8415e45`, pushed 2026-07-18 before
seed 77001 was launched. Seeds 77001-77012 trained 2026-07-18
(83-89 minutes each); pooled in
`outputs/overcooked_confirmation_pooled.json`.

    OC-1  PASS -- learned accepted on exactly 8/12 seeds. Every
          rejection (77001, 77005, 77006, 77008) routes through
          conditional selectivity (three also fail acquisition):
          those policies pot indiscriminately in both layouts, the
          correct verdict for context-blind competence.
    OC-2  PASS -- 48/48 control rejections. Routes: scripted roles
          and BC clones fail potential + selectivity + endogeneity +
          acquisition (12/12 each); initialization twins fail
          selectivity + acquisition (12/12) and usefulness (11/12);
          untrained-other fail selectivity + acquisition (12/12) and
          usefulness (7/12).
    OC-3  PASS -- first-potter-agent-0 rate higher in
          asymmetric_advantages on 12/12 seeds.
    OC-4  PASS -- contract B rejects all 12 twins.
    OC-5  PASS -- usefulness do-contrast positive on 12/12 seeds
          (exact one-sided sign test p = 2.4e-4; per-seed gaps
          +4.75 to +107.25 sparse-reward units).

No prediction failed; nothing was excluded or re-thresholded.
