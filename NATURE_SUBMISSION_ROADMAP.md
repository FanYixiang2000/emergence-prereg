# Nature-Scale Evidence Roadmap

## Core Claim

Single-mode coordination is not emergence. A stronger emergence signature is:

```text
latent multimodal potential
+ endogenous non-saturated trigger choice
+ trigger-specific future collapse
+ counterfactual necessity
```

The target story is not "we propose a competing definition". The target story is:

> We provide a measurable framework that distinguishes reward-induced
> coordination, metric-induced pseudo-emergence, random unpredictability, and
> latent possibility emergence across multiple systems.

Equally important, the theory should be framed as an underlying mechanism that
can explain several existing observables:

```text
possibility-space collapse
        -> representation-space jump
        -> PID-style joint information
        -> macro causal / behavioral effectiveness
```

This framing is stronger than claiming that representation jumps are wrong.
Representation jumps are useful observables; possibility collapse is the
candidate lower-level process that can make them occur.

## What A Nature-Subjournal Version Needs

### 1. Cross-Benchmark Evidence

One toy benchmark is not enough. We need at least three tiers:

1. **Controlled ground-truth worlds**
   - Possibility-preservation finite-horizon tree.
   - Macro-action sacrifice MDP.
   - Spatial sacrifice gridworld.
   - Optional: stochastic trap/bridge variant.

2. **MARL-scale environments**
   - RUSP sacrifice/rescue task.
   - Existing OES/Event Order trajectories.
   - Existing swarm/SMAC-derived coordination traces if available.

3. **External validation system**
   - Go/KataGo search-tree positions for "key move" / "tesuji" / "sacrifice".
   - Optional LLM tool-use tasks as a later extension.

### 2. Unified Metrics

The same metric family must apply across systems:

- `potential_effective_modes`: structured future diversity before the trigger.
- `trigger_choice_tension`: natural trigger rate is neither 0 nor 1.
- `trigger_specificity_js_bits`: trigger-conditioned futures differ from
  non-trigger futures.
- `collapse_bits`: entropy decreases after trigger.
- `macro_predictability_gain`: event-order or macro-basin concentration rises.
- `counterfactual_necessity`: success drops when the trigger is removed.
- `representation_jump`: macro embedding changes abruptly after future
  distributions contract.

### 3. Strong Counterfactuals

Every benchmark needs these ablations:

- No trigger.
- Wrong trigger / greedy trigger.
- Dense reward shaping.
- Pure team reward.
- Pure individual reward.
- Linear scalarization.
- Uncertain preference / RUSP.
- Random-noise control that increases entropy without structure.

### 3.5 Analytic Controls

The theory also needs a part that is not dependent on a learning algorithm.
The finite-horizon possibility tree gives a closed-form condition:

```text
-c + p * R_trigger + (1 - p) * R_direct > R_cash
```

This proves that local immediate optimality can be globally suboptimal when it
closes valuable future options. The larger experiments should be presented as
empirical tests of this analytic structure, not as isolated RL demos.

The same-solver planning-horizon ablation further controls for heuristic
differences:

```text
same Bellman solver + same reward + same environment
short horizon -> cash-out
long horizon -> preserve possibility
```

### 4. Reviewer-Facing Surprise

The evidence must show actions that are:

- locally costly;
- not explicitly rewarded as process steps;
- not mandatory under the learned policy;
- retrospectively necessary for high-value futures;
- replaceable by neither greedy actions nor dense reward shaping.

### 5. Statistical Standards

For a serious submission we need:

- multi-seed sweeps, not one smoke run;
- confidence intervals or bootstrap intervals;
- effect sizes for all PTC metrics;
- pre-registered basin classifiers for controlled worlds;
- negative controls showing random entropy is not emergence;
- open code that reproduces all figures from raw rollouts.

### 6. Paper Figure Plan

1. Concept figure: spatial emergence vs temporal emergence.
2. Metric schematic: potential -> trigger -> collapse.
3. Controlled benchmarks: reward regimes vs PTC metrics.
4. Counterfactual panel: remove/replace trigger.
5. MARL extension: RUSP/OES/Event Order mapped into the same metrics.
6. Go extension: expert key moves have higher collapse signatures than ordinary
   high-value moves.
7. Representation bridge: collapse burst predicts latent/macro representation
   jump.

## Immediate To-Do List

1. Add a second controlled benchmark where basin labels come from spatial event
   sequences, not macro-action names.
2. Add multi-seed sweep utilities.
3. Add confidence intervals.
4. Add counterfactual-necessity metrics.
5. Add a random-noise control regime.
6. Add factual performance and retrospective-importance metrics.
7. Add a contextual sacrifice benchmark where always sacrificing is not optimal.
8. Connect learned benchmark output to the same JSON/CSV schema.
9. Turn current smoke results into reproducible paper-style plots.
10. Start a Go-analysis adapter that can later consume KataGo search outputs.

## Current Status

- Macro-action scripted benchmark: implemented.
- Macro-action learned benchmark: implemented.
- Spatial learned benchmark: implemented with 5-seed stratified statistics and
  performance-aware metrics.
- Multi-seed statistics: implemented for the spatial benchmark.
- Contextual selective-trigger benchmark: implemented as a 3-seed smoke sweep.
- Possibility-preservation tree: implemented as the clean mathematical core.
- Closed-form possibility ablation: implemented with cash-out and cost boundary
  sweeps.
- Same-solver planning-horizon ablation: implemented.
- Analytic ground-truth validation: implemented; structure-only evidence fails,
  combined utility evidence succeeds.
- Performance-closure capability ablation: implemented; full option-preserve +
  context-use structure improves return and success in the predicted region.
- Performance robustness sweep: implemented across mismatch payoff and payoff
  asymmetry settings.
- Paper figure generation: implemented as first-pass PNG figures.
- External sacrifice MARL adapter: implemented at summary level using existing
  conditional-sacrifice and mechanism-probe outputs.
- External decoy swarm adapter: implemented at summary/grid level using
  role-aware robustness outputs.
- External decoy trajectory adapter: implemented using existing per-agent,
  per-time target-role records. It separates collapse into decoy traps from
  useful non-decoy collapse.
- Collapse-burst information-theoretic experiment: implemented.
- PID-inspired synergy proxy experiment: implemented.
- Representation-jump bridge experiment: implemented as a controlled formula
  bridge from future-basin collapse to macro-representation jumps.
- Learned Q-representation stress tests: implemented for spatial and contextual
  benchmarks. These reveal an important false-positive risk: ordinary team
  reward can create large representation jumps.
- Within-episode future-distribution probe: implemented. `P_t(B | s_t)` is
  estimated by Monte Carlo rollouts of the learned policy inside real episodes,
  with a minimal do-operator showing useful collapse in rescue mode and harmful
  collapse in bridge mode.
- Unsupervised basin discovery: implemented. Raw event-sequence clustering
  recovers hand basins with 1.0 purity, answering the circularity objection.
- Multi-seed bootstrap CIs for the within-episode probe: implemented (5 seeds).
  Harmful collapse is sign-stable; rescue gap is honest-noisy (3/5 positive).
- Neural DQN replication: implemented. The rescue/bridge sign flip holds with a
  PyTorch MLP policy, and penultimate-layer embedding jumps track collapse
  bursts (correlation ~0.9) at training checkpoints.
- Criterion-ablation battery: implemented. Nine measured systems with
  structure-based labels; full criterion 9/9, each of selectivity/usefulness/
  endogeneity uniquely necessary on a named counterexample, single observables
  score 0.44-0.67.
- Estimator robustness: implemented. All sign conclusions and the potential
  ordering survive a 12-cell rollout-samples x probe-temperature grid.
- Pre-registered external transfer: implemented. Protocol, thresholds, audit
  rules, and predictions frozen in `EXTERNAL_TRANSFER_PREREGISTRATION.md`
  before measuring the external continuous swarm decoy family. All three
  registered predictions passed: 5/5 verdicts correct with thresholds
  unchanged, `damage_aware` excluded only by endogeneity (external
  counterexample pinning that component), and forced-engagement usefulness
  sign-flipping across latent contexts exactly as in the internal family.
- Multi-seed external replication: implemented. Three of four registered
  predictions replicate 5/5 seeds; the all-verdicts prediction failed on
  2/5 seeds (random untrained network accepted by marginal tension +
  accidental positive usefulness). Registered failure, reported as such.
- Conditional-selectivity refinement + out-of-sample confirmation:
  implemented. The failure motivated upgrading selectivity from marginal
  tension to per-context separation and adding a measured acquisition
  component (separation gain over own initialization); the refined criterion
  was frozen and then confirmed on five fresh external seeds and a fresh
  internal battery seed (`refined_criterion_confirmation.py`).
- Prospective phase-boundary prediction: implemented. Closed-form payoff
  accounting predicted the behavioral onset, the usefulness sign flip, and
  the acceptance boundary before training. The primary run matched 4/4
  scored non-tie points; three independent-seed replications matched 11/12,
  with stable outer phases and one false acceptance at G=7.
- Grokking bridge to large-model emergence: implemented. Grokking measured
  as delayed, sudden, useful, endogenous possibility collapse; memorizer /
  shuffled-label / prewired controls fail the registered components exactly
  as predicted. Generality sweep (second task, three seeds) and model-scale
  decomposition of the Wei-vs-Schaeffer debate implemented alongside.
- Transformer grokking replication: implemented. A 412k-parameter causal
  transformer reproduces the grokking/memorizer contrast under the frozen
  criterion; the process-level result now spans MLP, causal-transformer,
  and attention-only families.
- Induction-head validation: implemented. An externally discovered,
  externally explained phase change (Olsson et al. 2022) with an external
  architectural impossibility theorem (Elhage et al. 2021). The frozen
  training-process proxy accepts exactly the two-layer condition and rejects the
  one-layer / no-structure / memorizer controls, 16/16 checks across four
  seeds. The one-layer rejection route (usefulness) coincides with the
  external theorem.
- Public checkpoint series (zero authorial control): implemented.
  MultiBERTs seed_0 (110M-parameter BERT-base, 29 checkpoints published by
  Google Research), protocol frozen before download. Subject-verb
  agreement acquisition passes the frozen process proxy (potential 14.7 bits,
  burst coincides with the 0.50 -> 0.98 accuracy jump at ~2% of
  pretraining); random-target and shuffled-vocab controls fail on
  usefulness exactly as registered, 4/4 predictions. Replicated on ALL
  four remaining published seeds: 15/15 verdicts, identical anchor window
  (step 20k) on every seed.
- Phenomena battery on the public series: implemented. Five abilities
  total; reflexive and determiner agreement pass as registered; the two
  auxiliary gradualism predictions (facts, NPI) are registered failures
  reported as such, with NPI showing collapse (20k) preceding usefulness
  (40k) inside the public model.
- Burst-jump alignment statistics: implemented. Empirical window ranks are
  0.023-0.051 per accepted run across 11 runs; controls are 0.54-0.99.
  No omnibus p value is claimed because runs are dependent and some additions
  were exploratory.
- Evidence audit: implemented.
- MARL/swarm integration: summary-level first pass complete across sacrifice and
  decoy settings, plus trajectory-level decoy target-collapse evidence.
- External strategic system (the Go/KataGo slot): DONE via chess.
  240 externally annotated sacrificial key moves (lichess puzzle
  database, real human games) + 120 balanced quiet controls, measured
  with Stockfish future-basin playouts under a frozen protocol
  (`CHESS_PREREGISTRATION.md`). The annotated key move is locally costly
  (median -3 pawns) yet reaches the win basin at 0.78 vs 0.24 for the
  best deep alternative (240/240 sign test, p = 5.7e-73); quiet
  positions have higher potential but no useful collapse available.
  4/5 registered predictions pass; the C5 effect-size margin is a
  registered failure with the route recorded before the main run.
  Go/KataGo becomes an optional replication.
- Chess robustness: DONE. 12/12 perturbation cells (temperature x depth,
  basin thresholds, classical vs NNUE engine) preserve the core
  conclusions; C4 potential is estimator-scale-dependent in absolute
  value, as recorded in the pilot note.
- Prior detectors on the external system: DONE. The fig28 blind-spot
  the exploratory chess move-ranking comparison was removed from the
  main claim because subtracting a common within-position baseline does
  not test the martingale/do-contrast lesson.
- Over-acceptance closure (tail-gradualism rejection): DONE with three
  registered failures honestly reported. The criterion rejects rare-word
  accrual on the public MultiBERTs series (route burstiness+usefulness)
  while accepting the frequency-matched abrupt families; abruptness is
  measurably metric- and grid-dependent.
- Deep MARL within-episode probe: DONE. PettingZoo/MPE simple_spread
  with MAPPO-style PPO (3 seeds), protocol frozen in
  `DEEP_MARL_PREREGISTRATION.md`. D1 (potential), D3 (commit-step
  do-contrast, positive median in 3/3 policy seeds; episode-level
  p = 0.0037 conditional on those policies), D4 (greedy has zero potential and
  lower usefulness) pass; D2 is a registered failure whose diagnosis
  (P_t(win) is a martingale under the behaving policy) independently
  confirms the do-contrast methodology, fig35.

## Most Important Next Experiment

The strongest remaining addition is explicit future-distribution collapse from
raw rollouts or search trees:

```text
state/time -> distribution over future basins
trigger/action -> contraction of that distribution
counterfactual action -> different or worse future basin
```

The decoy trajectory result already shows useful versus trap-directed target
collapse. For a Nature-scale submission, the next step should repeat this at the
future-outcome level in MARL rollouts and then in Go/KataGo search trees.

## Can This Be Submitted Now?

Close, for the first time. The evidence chain now covers: mechanism
(within-episode `P_t(B)` with do-operators), definition necessity (two
battery generations with named counterexamples), external validity
(pre-registered transfer + multi-seed replication + out-of-sample
confirmation of a refined criterion), predictive power (phase-boundary
prediction from closed form), and the bridge to large-model emergence
(grokking + scale decomposition). The package also contains two registered
failures handled the right way (marginal-selectivity acceptance of a random
network; the R2 normalization check), which is what pre-registration is for.

What still separates this from a submission:

1. **A public checkpoint series with zero authorial involvement**: DONE
   TWICE. MultiBERTs (110M parameters, trained and published by Google
   Research) passes the frozen process proxy on an externally documented
   ability with pre-registered controls, replicated across all five
   published seeds (15/15 verdicts, fig31). The decoder-side replication
   is now DONE as well: Pythia-160m (EleutherAI), 21 published
   checkpoints via a public mirror, 7/8 registered predictions passed
   with one verdict-correct route miss -- agreement and head facts
   accepted, BOTH frequency-tail families rejected (fig38,
   PYTHIA_PREREGISTRATION.md). The architecture-family objection
   ("your public evidence is only an encoder") is closed.
2. **Scale of learners**: DONE at the 100M level. The evidence now spans
   MLPs, a 412k causal transformer, attention-only transformers, a
   110M-parameter public BERT, and a 160M-parameter public decoder
   (Pythia). A billion-parameter scaling-family sweep (410m/1b/1.4b on
   the same Pythia grid) would be nice-to-have, no longer load-bearing.
3. **Theory tightening**: done second pass -- Proposition 0 now grounds
   the framework in the original trajectory-space definition
   (C(m) = KL(P(tau|m) || P(tau))) with three verified identities:
   E_m[C(m)] = I(tau;M); the rarity law C(m) = -log2 P(A_m) (7.16 bits
   of untrained rarity for the rescue structure, concentrated to the
   1/3 ecological rate by learning); and the data-processing bound
   showing basin-level KL/JS intervention contrasts are conservative
   (entropy differences remain partition-scale dependent).
   Rarity is now explicitly separated from provenance: endogeneity and
   acquisition remain independently audited components. Together with the four earlier propositions
   (collapse bounds representation jumps via Pinsker; no single
   observable suffices, with measured witnesses; the usefulness identity
   that powers the phase boundary; plug-in estimator consistency) the
   deductive chain is complete and machine-verified
   (`verify_theory_bounds.py`, all checks pass).
4. **Writing**: the story must lead with the refined six-component
   criterion and treat the earlier marginal-tension version as the honest
   development path (registered failure included), not as parallel truth.
   This is now THE remaining task.

## Writing phase (started)

Deliverables in place:

- `figures/figure1_concept.png` (`generate_figure1_concept.py`): the
  editor-facing concept figure -- four regimes of P_t(B), only one is
  emergence, with the do-block counterfactual drawn in.
- `MANUSCRIPT.md`: full skeleton -- title candidates, abstract, the
  four-question main-text structure, Box 1 (formulas + frozen
  thresholds), the unification table (prior signatures as measured
  projections with named witnesses), six-main-figure plan mapping to
  existing figures, the unified registered-failure narrative (the
  martingale/do-contrast principle from chess C5 + deep-MARL D2), scoped
  claims (say / don't-say), methods pointers, prepared answers to the
  three predictable reviewer pressure points.
- `REPRODUCIBILITY.md`: figure -> script -> output -> preregistration ->
  runtime map, registered-failure index, pilot-log index.

Remaining, in order:

1. Full prose main text from the skeleton (intro + results + discussion).
2. Composite main figures 2-6 assembled from existing panels.

Completed strengthening (2026-07-07, evidence-hardening pass):

- Cross-task deep MARL replication DONE: Level-Based Foraging
  (forced-coop), registered protocol L1-L4 in LBF_PREREGISTRATION.md,
  3 seeds, 4/4 predictions pass; do-contrast is positive in 69/90
  evaluation episodes, tied in 21 and negative in none, with positive
  medians in 3/3 policy seeds
  (fig36). The "single-task deep-MARL demo" objection is closed.
- Statistical hardening DONE: bootstrap_intervals.py gives every
  headline effect size a 20k-resample 95% CI; the prior-detector
  comparison on chess has disjoint CIs (composite vs best single
  signal), so the detector ranking is not sampling noise.
- Theory root DONE: Proposition 0 (trajectory-space definition, MI
  identity, rarity law, data-processing bound), machine-verified.
- Published-form rival audits DONE (strawman risk reduced, not eliminated):
  Hoel's EI and Rosas' practical criterion Psi computed from their equations
  with zero
  Monte-Carlo error on enumerated policy-closed chains of all 10
  battery systems (exact_prior_formalisms.py, fig37). Exact CE < 0
  everywhere, best threshold = trivial classifier; exact Psi > 0
  scores 0.3 with wrong_selector as top scorer. Both at or below the
  charitable proxies -- blind spots are structural on this testbed.
  Proposition 5 states scoped derivability and audited threshold
  insufficiency; the comparison addresses the original equations within
  declared candidate families, not only flavored proxies.
