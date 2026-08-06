# Commitment-window intervention (preregistration)

Frozen: 2026-07-22T21:15+08:00, before any intervention run started.

## Question

The seed-93001 formation curves located a candidate commitment window:
the largest joint-action collapse-rate increase AND the real-vs-cut G
peak both fall in 640k-1M training steps, exactly at the performance
take-off. Is that window causally load-bearing for the formation of
the coupled macro-regime, or merely descriptive?

## Design

Four conditions, identical mechanics and identical fresh seed 93101,
2,000,000 training steps each (train_mixed PPO mechanics unchanged
outside the window):

- `none`   no cut (matched control);
- `early`  co-adaptation cut during steps 80k-440k;
- `commit` co-adaptation cut during steps 640k-1,000k;
- `late`   co-adaptation cut during steps 1,500k-1,860k.

Equal budget: every cut lasts 360k steps.

The cut: on entering the window, agent 1 is frozen (a ghost copy of
the current network); during the window agent 1 acts from the frozen
copy and ONLY agent 0's stream enters the PPO update. This severs
bidirectional co-adaptation while keeping the environment, the task
and both agents' presence unchanged. On leaving the window, normal
self-play resumes.

## Evaluation (identical protocol for all four conditions)

At the final 2M checkpoint: the state-level real-vs-ghost transition
certificate (G, C, M, score; snapshot steps 20/40/80/120/160, 10
episodes/layout, horizon 120) and the joint-collapse ladder
(C_individual, C_env, C_relational; 30 episodes/layout, horizon 200).
Evaluation seeds fixed at 96001 + condition index.

## Registered predictions (outcomes retained either way)

- INT-1 (primary): the commit-window cut damages final coupling more
  than the equal-budget early and late cuts:
  `M_commit < min(M_early, M_late)` and `M_commit < M_none`.
- INT-2: same ordering for the relational collapse component:
  `C_rel_commit < min(C_rel_early, C_rel_late)`.
- INT-3 (selectivity, may miss): the commit cut damages organization
  more than raw capability: `score_commit >= 0.5 * score_none` while
  `M_commit < 0.5 * M_none`.

## Falsification statement

If INT-1 and INT-2 both fail -- i.e. the commit-window cut is no more
damaging than position-matched cuts -- then the commitment window
identified by the collapse curves is descriptive rather than causal,
and the paper's strongest claim (temporally targeted intervention on
emergence formation) must be dropped or weakened accordingly.

## Recorded outcomes (2026-07-23T10:25+08:00; pilot, retained)

- INT-1 PASS: M none/early/commit/late = +9.4 / +14.0 / +8.0 / +10.4;
  the commit-window cut gives the lowest final macro coupling effect.
- INT-2 PASS: C_rel = 0.0142 / 0.0165 / 0.0100 / 0.0187; commit cut
  gives the lowest relational collapse.
- INT-3 FAIL, retained: score_commit = 26.8 >= 0.5*30.8, but
  M_commit = 8.0 is not < 0.5*9.4; organization was damaged, not
  halved. The selectivity claim needs a stronger or better-normalized
  formulation before any confirmatory run.
- Unregistered observation, disclosed: the EARLY cut increased final
  score (36.8) and final G (0.155). Interpretation offered, not
  claimed: training against a frozen partner early acts as a
  curriculum and delays regime formation, so at 2M the early system
  is still in its high-G formation phase. This is consistent with the
  G-peaks-during-formation reading but requires dedicated follow-up
  (formation curves for the intervention conditions).
- Margins are small and this is one seed per condition; a
  confirmatory run needs >= 5 seeds per condition, a random-window
  condition, and dose-response over window length.

## Declared limitations

One seed per condition (this is a pilot; a confirmatory run needs
>= 3 seeds per condition); window positions chosen from the seed-93001
curve, applied to a different seed (93101), which makes INT-1 a
transfer test of the window location as well; the early cut may
additionally damage basic skill acquisition (disclosed confound,
mitigated by the equal-budget late-cut comparison).
