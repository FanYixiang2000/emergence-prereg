# Evidence Audit

This file translates the research direction into four reviewer-facing criteria.

## Criterion 1: The Discovered Structure Must Be Useful

Current status: partially satisfied.

Evidence:

- `possibility_preservation_tree.py` shows that preserving options can improve
  expected return from `5.0000` to `10.0088`.
- `possibility_ablation.py` shows positive option value across controlled
  parameter regions.
- `contextual_sacrifice_gridworld.py` connects selective triggering to factual
  task return.
- `performance_closure_benchmark.py` shows that the full capability chain
  improves both expected return and success rate relative to myopic and
  no-context ablations.
- `performance_robustness_sweep.py` shows the closure result persists across
  payoff mismatch and asymmetry settings.
- `external_sacrifice_ptc_adapter.py` maps existing sacrifice MARL summaries
  into PTC evidence and shows conditional sacrifice is distinct from blind
  sacrifice.
- `external_decoy_ptc_adapter.py` maps existing decoy/role-aware swarm summaries
  into local-optimality-trap evidence.
- `external_decoy_trajectory_ptc.py` shows that real target-selection
  trajectories can collapse into either a decoy trap or a useful non-decoy basin.
- `collapse_burst_experiment.py` distinguishes burst-like useful collapse from
  gradual reward-guided convergence.
- `synergy_pid_proxy_experiment.py` connects spatial emergence to joint
  information: XOR-style joint structure predicts the future basin where
  individual parts cannot.
- `representation_jump_bridge.py` connects the existing representation-jump
  view of emergence to possibility collapse: abrupt representation jumps arise
  when future-basin collapse is burst-like rather than merely gradual.
- `learned_representation_jump_probe.py` and
  `contextual_learned_representation_probe.py` stress-test this bridge on
  learned Q-vector representations and reveal an important false-positive risk:
  ordinary team-reward convergence can also create large representation jumps.
- `within_episode_collapse_probe.py` estimates state-level future-basin
  distributions `P_t(B | s_t)` inside real learned episodes and shows with a
  minimal do-operator that the same trigger action opens a high-value future in
  rescue mode (+2.6 to +10.7 return) and destroys value in bridge mode (-2.6 to
  -7.3 return).
- `unsupervised_basin_discovery.py` recovers the basins from raw event
  sequences with 1.0 purity and matching effective-mode counts, removing the
  hand-label circularity objection inside these benchmarks.
- `run_within_episode_sweep.py` adds 5-seed bootstrap CIs: harmful bridge-mode
  collapse is sign-stable in 5/5 seeds; the uncertain-preference rescue gap is
  positive on average but seed-noisy (3/5), an honestly reported limitation.
- `neural_within_episode_probe.py` replicates the rescue/bridge sign flip with
  a PyTorch DQN and shows neural-embedding jumps tracking collapse bursts
  (correlation ~0.9), so the mechanism evidence is not a tabular artifact.
- `criterion_ablation_battery.py` gives the original definition-validation
  evidence: its nine-system, five-component harness classifies 9/9, while
  dropping selectivity, usefulness, or endogeneity each admits a named
  counterexample and single-observable definitions score 0.44-0.67.
- The later registered refinement adds conditional selectivity and acquisition;
  the resulting six-component rule passes the ten-system internal confirmation
  and fresh-seed external confirmation. Potential, specificity and acquisition
  are pinned by dedicated controls/refinement, not by unique accuracy drops in
  the original nine-system ablation matrix.
- `estimator_robustness_check.py` shows the sign conclusions and potential
  ordering survive all 12 estimator settings (rollout samples x probe
  temperature).

Important limitation:

- PTC structure alone is not utility. A multimodal future can be useless.

## Criterion 2: The Discovered Structure Must Be True

Current status: satisfied in analytic, learned, public-checkpoint and
externally annotated strategic systems, with domain-specific limitations.

Evidence:

- `possibility_ablation.py` defines analytic ground truth:

```text
preserve wins iff
  -c + p * R_trigger + (1 - p) * R_direct > R_cash
```

- `planning_horizon_ablation.py` shows the same Bellman solver changes action
  when horizon increases.
- `ptc_ground_truth_validation.py` validates evidence against analytic labels.

Default validation result:

```text
auc_structure_only       0.4978
auc_option_value         0.9838
auc_combined_evidence    1.0000
combined_accuracy        1.0000
```

Interpretation:

- Structured possibility alone is not enough.
- Utility alone is strong but incomplete.
- Combined evidence matches the analytic ground truth.

## Criterion 3: The Evidence Must Be Complete

Current status: improving, still incomplete.

Current positive controls:

- possibility-preserving policy beats myopic cash-out;
- horizon-2 Bellman solver preserves options where horizon-1 cashes out;
- full option-preserve + context-use capability improves task completion;
- spatial uncertain preference shows high PTC signature;
- contextual benchmark connects selective triggering to performance.
- external sacrifice MARL summaries show RUSP/fixed conditional sacrifice ahead
  of blind team sacrifice under the external PTC score.
- external decoy/role-aware swarm summaries show role-aware targeting avoids
  local decoy traps and improves win rate.
- external decoy target-selection trajectories show a crucial separation:
  nearest-only has high collapse into decoys, while role-aware targeting has high
  useful non-decoy collapse.
- collapse-burst synthetic evidence separates strong emergence from ordinary
  convergence and reward-shaped convergence.
- synergy proxy evidence separates joint-structure emergence from unique or
  redundant prediction.
- representation-jump bridge evidence connects our mechanism to a familiar
  observable definition of emergence.

Current negative controls:

- `myopic_greedy`: locally optimal but closes options.
- `always_trigger`: can be useful but not selective.
- `random_preserve`: preserves possibility but often chooses wrong.
- `random_noise`: can create irregular behavior but is unstable and not
  structured.
- `false_multimodality`: structured futures without utility gain.

- chess within-state probe: externally annotated sacrificial key moves in
  real human games are locally costly (median -3 pawns) yet selectively
  reach the win basin (0.78 vs 0.24 for the best alternative, 240/240);
  balanced quiet positions have higher potential but no useful collapse.
  4/5 registered predictions pass, one effect-size margin is a registered
  failure (`CHESS_PREREGISTRATION.md`).
- chess robustness: the core conclusions hold in 12/12 estimator / basin /
  engine perturbation cells, including a classical (pre-NNUE) engine.
- exploratory chess move ranking: retained in outputs but removed from
  the main evidence. Its preferred score subtracts a common
  within-position baseline, so the ranking does not test the registered
  martingale/do-contrast lesson. The key-vs-best-alternative do-gap and
  robustness grid remain load-bearing.
- tail-gradualism rejection on the public model: rare-word competence
  accrues slowly (top-1 0 -> ~0.2 over 2M steps) and is correctly NOT
  classified emergent under frozen thresholds -- the criterion rejects a
  natural (non-artificial) ability on a system trained by another lab.
  Three registered failures en route (pair-metric tail families;
  top-1 tail facts) establish, with data, that abrupt-vs-gradual verdicts
  depend on metric and grid resolution.

Since added (evidence-hardening pass, 2026-07-07):

- cross-task deep MARL replication (Level-Based Foraging, forced-coop,
  frozen protocol, 3 seeds, 4/4 registered predictions, fig36) --
  closes the single-task deep-MARL objection;
- bootstrap 95% CIs on every headline effect size
  (outputs/bootstrap_intervals.json); prior-detector ranking on chess
  now has disjoint CIs;
- EXACT rival formalisms (Hoel EI, Rosas Psi) computed with zero
  Monte-Carlo error on enumerated chains (exact_prior_formalisms.py,
  Prop. 5) -- closes the "flavored strawman" objection: both exact
  forms perform at or below their proxies, blind spots structural;
- LBF estimator-robustness grid (lbf_robustness_grid.py, criteria
  frozen in-script): pooled do-contrast positive with p <= 2.7e-20 in
  all four probe-temperature cells (T = 2..8), greedy double
  dissociation direction in all cells; the absolute 0.8-bit potential
  threshold fails at T <= 3 (the declared observer-scale dependence,
  same direction as chess C4) -- closes the "probe temperature was
  tuned to pass" objection;
- prior detectors on the deep-MARL domain (lbf_prior_detectors.py):
  round-1 registered prediction FAILED for the performance detector
  (set lacked a competent imitation; archived in
  lbf_prior_detectors_round1.json), fixed by adding scripted_coop
  (structure 100% prespecified) with a frozen round-2 prediction --
  which PASSED: performance accepts scripted_coop (win 0.85 = trained
  range, best acc 0.875); the structure detectors score scripted_coop
  ABOVE the trained range (specificity 0.94, Psi 0.95, EI 0.68 vs
  trained 0.55-0.72) -- hand-coded coordination is tighter than
  learned coordination, and none of the five signals can see who
  wrote it; no single detector reaches 1.0 on the 8-system set
  (best 0.875), while the composite criterion rejects scripted_coop
  and forced_commit on endogeneity/acquisition by construction.

Since added (decoder pass, 2026-07-08):

- Pythia-160m decoder-side replication (pythia_collapse_probe.py +
  pythia_tail_gradualism.py, PYTHIA_PREREGISTRATION.md, fig38): the
  frozen thresholds accept agreement (foreshadow burst preceding the
  0.49 -> 0.93 jump) and head facts, reject both controls and BOTH
  frequency-tail families on a public autoregressive decoder; 7/8
  registered predictions passed, one verdict-correct route miss
  (PY-T2) kept -- closes the "encoder-only public evidence" objection.

Since added (NMI hardening pass, 2026-07-13):

- Contextual LBF full six-component confirmation now carries the strongest
  external learned-system evidence: 9/10 fresh learned policies pass all six
  components, 40/40 non-learned controls reject, all registered predictions
  pass, and seed-bootstrap lower bounds are positive for usefulness and
  acquisition.
- A post-confirmation Contextual LBF extension adds five more fresh training
  seeds without changing thresholds: 4/5 learned policies pass, 20/20 controls
  reject, and all learned usefulness/acquisition effects are positive. This is
  supportive robustness evidence, not a new preregistered significance claim.
- A Contextual LBF single-signal audit gives every behavior-only signal a
  hindsight-optimal threshold. On the registered confirmation, no behavior-only
  signal exceeds 0.86 accuracy against the full six-component verdict; on the
  extension, the maximum is 0.88. Acquisition alone reaches 1.0 in these runs
  because it is a definition-internal provenance/learning component using the
  same-seed initialization twin, not a prior definition.
- Pythia scaling was reopened after a data-integrity failure in the old 2.8B
  mirror run. The mirror-invalid outputs are quarantined. The downloader now
  uses official Hugging Face revisions and safetensors shards; a 2.8B smoke
  check shows step0 and step1000 are no longer identical (agreement about
  0.50 -> 0.77). Official 1.4B/2.8B runs are in progress and should not be used
  in manuscript claims until `pythia_scaling_summary.json` is regenerated and
  checked.

Since added (seed-power and consistency pass, 2026-07-14):

- Deep-MARL seed extension: three further simple_spread and five further LBF
  policies trained under unchanged frozen probes
  (`deep_marl_collapse_seed_extension.json`, `lbf_collapse_seed_extension.json`).
  All eight new seeds have positive mean do-contrasts. Combined seed-level
  inference (`hierarchical_marl_analysis_combined.json`): exact one-sided sign
  tests p=0.016 (6 simple_spread seeds) and p=0.004 (8 LBF seeds), with
  positive cluster-bootstrap mean intervals [0.032, 0.156] and [0.159, 0.390].
  The simple_spread win-shift reading (D2) failed again on the extension while
  the do-contrast stayed positive -- the martingale diagnosis replicated on
  fresh data. Reported separately from the registered three-seed runs.
- Contextual LBF threshold-sensitivity rescoring
  (`contextual_lbf_threshold_sensitivity.json`): frozen cutoffs interior to
  wide stable plateaus for potential/specificity/acquisition; conditional
  selectivity identified and reported as the binding component; every control
  rejected at every grid point of every sweep.
- Manuscript-number consistency audit (`verify_manuscript_numbers.py`):
  26/26 headline numbers match the stored outputs, including CLBF
  confirmation/extension, Pythia 160m/1B, deep MARL and chess.
- Pythia 1.4B checkpoints are being pre-downloaded via the official
  huggingface_hub client (16/21 complete locally); an offline smoke run on
  local checkpoints (step0/1000/16000) produced distinct, sane values,
  confirming the local-cache path end to end.

Since added (held-out scaling completion, 2026-07-16):

- The registered Pythia scaling protocol (1B/1.4B/2.8B) is complete.
  Agreement transfers cleanly to 1B and 1.4B (same anchored window step 1000;
  burstiness 11.9/7.3; usefulness 0.47/0.43; all controls rejected). At 2.8B
  the registered S1 prediction FAILS and is kept: usefulness passes (0.42)
  with the window again at step 1000, but burstiness is 3.2 < 5 -- collapse
  spreads across several early intervals at the published grid resolution.
  S5's head-facts half also fails at the three larger scales (perfect final
  accuracy with sub-threshold burstiness), while both frequency-tail families
  are rejected at every scale. S2/S3/S4/S6 pass. Totals: 4/5 agreement
  acceptances, 10/10 control rejections, 8/8 tail rejections
  (`outputs/pythia_scaling_summary.json`).
- Checkpoint integrity: per-revision SHA-256 audits
  (`verify_pythia_checkpoints.py`) confirm all 21 cached 1.4B revisions are
  pairwise distinct, and detected two upstream repository defects at 2.8B:
  step64000 duplicates step143000 in both published weight formats (excluded
  from the run) and step32000's single-file safetensors is a stale copy of
  the final weights (rebuilt from the authentic per-revision
  pytorch_model.bin; tensor-identity verified; step96000/step128000
  bin-cross-checked with zero mismatching tensors). The stale file remains
  quarantined beside the rebuilt one.
- The manuscript-number consistency audit was extended to the new scales:
  36/36 checks pass (`verify_manuscript_numbers.py`).
- A leave-one-component-out audit across all 75 evaluated Contextual LBF
  systems (`component_ablation_witnesses.py`) makes the conjunction's
  internal logic explicit: dropping conditional selectivity admits the two
  borderline learned seeds (1104, 1204) and no controls; dropping any other
  single component changes no verdict, because every control fails at least
  two components simultaneously. Individually targeted counterexamples for
  selectivity/usefulness/endogeneity remain the gridworld battery's role.
  This is now stated in the manuscript (definition vs. observer contract
  vs. identification protocol) instead of implying six independent
  necessity proofs.
- THEORY.md gained three sections implementing the reviewer-facing
  clarifications: the three-layer structure (root definition /
  identifiability assumptions / operational protocol), the non-triviality
  bridge (why ordinary decisions, gradual learning, exogenous injection and
  useless contraction are excluded, each with measured witnesses), and the
  temporal-resolution section that turns the 2.8B S1/S7 registered failures
  into a measured grid-dependence statement.
Since added (reviewer-hardening pass, 2026-07-16 afternoon):

- Prospective discovery REPLICATION on a second year (2016-03; addendum
  frozen before download): 4/4 registered predictions pass again (AUROC
  0.725 vs 0.730 primary; precision 0.347 = 2.95x base rate). Notably the
  same-family shallow eval-gap baseline dropped 0.762 -> 0.652 across months
  while the mechanism-level do-gap was stable -- the collapse score is the
  distribution-stable predictor (`chess_discovery_replication_2016_03.json`).
- Referee-threshold sensitivity (labels only): AUROC monotone 0.61 -> 0.79
  as the label tightens 100 -> 250 cp; lift >= 2.27x at 125 cp and above;
  top-decile hit rate 4.5x base (`chess_discovery_referee_sensitivity.json`).
- Exact trajectory-basin coupling on the enumerated battery
  (`trajectory_basin_coupling.py`): zero DPI violations, rarity identity
  exact; controls have LARGE path-space contrasts (32.2/31.9 bits) with
  near-zero basin retention (3%, 0.01%) while the emergent system retains
  32% -- raw trajectory displacement cannot rank emergence, and the
  value-bearing basin projection is where useful collapse becomes visible.
  Degenerate cases align exactly with provenance (forced constructions have
  zero path KL against their own natural law). This closes the "the
  trajectory-space ontology is decoration" objection with exact numbers.
- Consistency audit now 42/42.

Since added (persistence and separability pass, 2026-07-16 evening):

- PERSISTENCE measured (the theory's "stable macrostructure" is no longer an
  unmeasured word): prospectively frozen perturbation battery on the saved
  CLBF confirmation policies (PERSISTENCE_PREREGISTRATION.md). The acquired
  structure is fully retained across horizon changes and observation noise
  up to sigma 0.2 (40/40 cells; noise degradation unmeasurably small -- a
  benign registered failure of strict monotonicity), never appears in
  initialization twins (70/70 cells), and has a measured spatial boundary:
  novel-layout transfer is partial for selectivity (5/10 at the 50% bar)
  and negative for value on all ten seeds (registered failure kept). A
  layout-specification bug in the first P1 attempt (lexicographic
  food-identity convention) was fixed before the reported run and the
  botched output quarantined (contextual_lbf_persistence_layoutbug.json).
- Trajectory-KL implementation independently cross-validated: chain rule vs
  full path enumeration agree to 7.6e-15 bits over 2000 random chains
  including singular and zero-KL cases; uniform declared rules for
  infinite-KL/degenerate support recorded in the coupling JSON. The
  "ordering preserved" phrasing was corrected: basin projection is a
  task-relevant filter (Spearman 0.37), not a numerical approximation.
- Component-directed witness matrix assembled from stored outputs
  (component_witness_matrix.json): every component has at least one
  measured system whose rejection runs through it (EXACT witnesses for
  selectivity and usefulness; EXACT-PAIR provenance witnesses for
  endogeneity/acquisition on 15/15 seeds across two domains; designed
  witnesses for potential and process-proxy burstiness). This separates
  in-domain empirical redundancy (leave-one-out) from cross-domain
  separability.
- Consistency audit now 45/45.

Since added (external-validity pass, 2026-07-16 late afternoon):

- THIRD FULL SIX-COMPONENT DOMAIN, sequence modality
  (`latent_context_lm.py`, LATENT_CONTEXT_PREREGISTRATION.md): a two-layer
  causal transformer trained only by next-token prediction on synthetic
  sequences with two unlabelled latent contexts. After one disclosed
  two-seed pilot, ten fresh seeds: 10/10 learned models pass all six
  components (thresholds copied unchanged from CLBF), 40/40 controls
  rejected, seed-bootstrap lower bounds positive (usefulness 0.481,
  acquisition 0.918). Registered failure kept: the oracle router was
  predicted to fail exactly {endogeneity, acquisition} but is
  intervention-inert in sequence space, so it fails four components
  (0/10 exact route) -- an informative modality contrast with embodied
  domains. The full criterion now spans swarm, Contextual LBF and a
  sequence model.
- Cross-engine-family referee for discovery
  (`chess_discovery_cross_engine.json`): re-labelling all 800 positions
  with Stockfish 11 CLASSICAL evaluation (pre-NNUE family) keeps the
  discovery signal (AUROC 0.743 main / 0.669 replication; lift 2.7x/2.6x;
  NNUE-vs-classical label agreement ~96%). Closes the same-engine
  circularity objection.
- Adversarial observer audit (`adversarial_observer_audit.json`):
  bijective relabelling exact to 2e-16; under 1000 random micro-cell
  partitions the do-contrast still passes specificity for 80% of random
  observers (no partition hides it) while the declared value-bearing
  partition is the most informative (100th percentile); controls can never
  be rescued because four components are partition-independent. Both
  reading directions were declared in advance.
- Ordinary-learner boundary probe (`ordinary_learner_control.json`):
  the four-component process proxy ACCEPTS a strong smooth supervised
  learner (6/6 runs at final accuracy 0.92-0.93) -- a deliberate,
  disclosed scope finding: the proxy measures acquisition shape, not
  emergence; the episode-level components and paired controls carry the
  discriminative weight. A first failed design attempt (97-class ordinal
  task too hard, accuracy 0.07) is quarantined as
  ordinary_learner_control_attempt1_failed_design.json.
- Capability novelty boundary (`capability_novelty_boundary.json`):
  a frozen additive same-input classifier matches the ordinary learner
  (0.927 versus 0.924; gap -0.003), but cannot solve modular addition
  (0.000 versus grokking 1.000); the independent induction comparison gives
  0.116 one-layer versus 0.987 two-layer (gap 0.871). All four registered
  predictions pass. This supplies a necessary lower-order novelty gate and
  rejects the ordinary learner despite retaining its 6/6 process-proxy
  acceptance. It does not establish a universal choice of lower-order
  hypothesis class.
- Burst-boundary audit (`burst_boundary_audit.json`):
  the original burst-collapse hypothesis is retained as a historical
  hypothesis but scoped by a double dissociation. Burst is not sufficient
  (ordinary learner min burstiness 6805, old proxy 6/6, novelty verdict
  false), not necessary (ant trail D 0.942/R 0.966 with a 0.248 gradual
  commitment span), and grid-relative (Pythia-2.8B agreement flips in 9/9
  thinning cells). Four bursty held-out controls still fail usefulness.
- Overcooked transition scaffold
  (`overcooked_transition_certificate_smoke_*.json`):
  first executable state-level real-vs-ghost replay contract. From identical
  simulator snapshots it compares coupled continuations with ghost-partner
  cuts and exports `G,C,M,J` plus partner-action marginal diagnostics.
  Scripted smoke gives the expected null (`G=0`, `C=0`, `M=0`, TV 0.004);
  initial-policy smoke gives finite diagnostic movement (`G=0.043`, TV
  0.079, `M=0`). This is infrastructure for the future learned flagship,
  not evidence that the round-1 learned Overcooked claim now has a full
  interaction-broken certificate.
- Learned Overcooked transition pilots
  (`overcooked_transition_pilot_audit.json`):
  first trained checkpoints are now audited with the same real-vs-ghost
  machinery. The early checkpoints are informative negatives: 40k gives
  `G=0.009`, `M=0`; 500k gives `G=0.012`, `M=-0.625` with partner TV 0.025.
  A longer 2M single-seed pilot gives the first learned positive candidate:
  `G=0.055`, signed `C=0.216`, `M=+14.6`, partner TV 0.025. This establishes
  end-to-end feasibility for learned checkpoints, but remains a single-seed
  pilot rather than the matched-mechanism / early-prediction / intervention
  flagship required for the main claim.
- Consistency audit now 109/109.

Since added (consolidation pass, 2026-07-16 evening):

- Persistence upgraded to DUAL retention curves
  (`persistence_retention_curves.json`): within the declared
  temporal-observational family both activation retention (~1.00) and
  causal retention (0.82-1.00) stay high with usefulness positive 10/10 in
  every cell; on novel layouts activation drops to 0.37 and causal
  retention is negative -- "stable causal macrostructure under the declared
  perturbation family D" is now the exact, measured claim.
- Sequence-domain generalization audit
  (`latent_context_generalization.json`, frozen predictions): no false
  triggers under wrong-token distractors (LG1 pass), causal value persists
  wherever activation persists (LG3 pass, zero violations), but marker
  positions outside the training range collapse selectivity (LG2 registered
  failure, 0/10 and 1/10). Convergent with the CLBF novel-layout boundary:
  in BOTH full-criterion domains the acquired structure is
  support-neighborhood-stable and positionally/geometrically bounded --
  a cross-domain regularity, reported as such in the manuscript.
- Observer audit completed with the control-side full-verdict null
  (`adversarial_observer_controls.json`): over 13 controls x 1000 random
  partitions, full-verdict acceptances = 0; the strongest a malicious
  partition achieves is inflating the two distributional components (<=83%).
  Manuscript wording updated to "observer choice cannot manufacture a full
  verdict" (not "observer arbitrariness closed") and "relabelling
  invariance" (not "code correctness proof").
- Emergence magnitude vs velocity distinction added to THEORY.md and the
  manuscript: the full criterion measures magnitude under a declared
  contract; burstiness measures velocity at a grid; gradual emergence and
  fast ordinary learning are both possible, which reconciles the
  ordinary-learner acceptance, the 2.8B S1/S7 failures and the tail
  rejections in one statement.
- Chess referee renamed to "classical-versus-NNUE cross-evaluation check"
  (same Stockfish search lineage disclosed; truly independent referees are
  declared future strengthening).
- Reproducibility pack: `Makefile` (audit / figures / paper /
  small-reproduction targets, all tested), `generate_manifest.py` ->
  `manifest.json` (21 claims with tiers and SHA-256s, 13 protocol hashes,
  193 output hashes), `INVALID_DATA_REGISTRY.md`, `requirements-lock.txt`.
- PREDICTION_LEDGER.md stratified into four tiers; the manuscript now
  states that main-text claims rest on Tiers 1-2 only and that the ledger
  count is a transparency record, not a sample size.
- Registered prediction S7 was scored on the held-out scales
  (`held_out_scaling_robustness.py`): FAILED at 130/162 = 80.2% < 90%
  condition-level thinning agreement, with a diagnostic pattern -- the 2.8B
  agreement verdict flips to ACCEPT in all nine thinning cells, so the S1
  burstiness failure is grid-relative (coarser grids re-aggregate the
  spread-out collapse), not an absent transition. Radius sensitivity shows
  the same borderline behavior (1.4B rejects at radius 0; 2.8B accepts at
  radius 2). Recorded in the preregistration outcomes and the ledger
  (SC-S1..S7); a new Extended Data figure (ed_fig9_pythia_scaling.png)
  documents the five-scale family and the registered failures.

Still needed (all optional):

- more seeds for non-contextual deep-MARL mechanism probes, or a clear
  demotion of those probes to mechanism illustrations rather than population
  evidence;
- learned agents under identical optimizer settings;
- Go/KataGo replication of the chess probe;
- SMACv2-class deep MARL scale-up;
- completion and audit of the official Pythia scaling-family sweep.

## Criterion 4: The Work Must Matter Academically

Current status: supported as a scoped methodological contribution; broad
generality beyond the tested learning systems and chess remains unproven.

Potential contribution:

```text
Emergence as possibility-preserving escape from local optimality traps.
```

Why this matters:

- It separates emergence from mere coordination.
- It separates useful latent possibility from random multimodality.
- It gives a mathematical condition for when local optimality closes valuable
  futures.
- It gives a controlled way to connect emergence, planning horizon, option
  value, and counterfactual necessity.

Reviewer-facing claim:

```text
The key evidence is not that an agent sacrifices. The key evidence is that a
locally suboptimal action preserves future options, becomes retrospectively
necessary under the realized context, and improves final outcomes in the
parameter region predicted by the analytic inequality.
```

## Current Strongest Result

The strongest current evidence is analytic and controlled:

1. Closed-form inequality defines the useful region.
2. Same-solver horizon ablation shows action reversal.
3. Ground-truth validation shows structure-only metrics are insufficient, while
   combined PTC + utility evidence matches analytic labels.
4. Performance-closure ablation shows the full capability improves final return
   and success rate in the predicted region.
5. Robustness sweep shows the closure rate remains stable across payoff settings.
6. Existing sacrifice MARL summaries provide first external support: RUSP ranks
   highest by summary-level external PTC score, while team is penalized for blind
   sacrifice.
7. Collapse-burst evidence directly tests the "sudden possibility collapse"
   component.
8. PID-inspired synergy evidence tests the "joint structure, not individual
   part" component.
9. Existing decoy swarm summaries provide a second external support: role-aware
   structure avoids immediate local traps and preserves winning futures.
10. Existing decoy target-selection trajectories provide the first external
    trajectory-level support: both controllers collapse, but only role-aware
    collapse is useful.

## Current Weakest Point

External validity has moved from "weakest point" to "partially closed": the
pre-registered transfer battery applied the criterion, thresholds frozen, to
the pre-existing continuous swarm decoy family and passed all three
registered predictions (5/5 verdicts correct; endogeneity pinned by an
external counterexample; sign-flipping usefulness replicated).

What remains before submission:

- multi-seed replication of the external transfer battery: DONE (plus the
  refined-criterion confirmation on fresh seeds);
- a second external domain with no authorial control at all: DONE twice
  over -- the MultiBERTs public checkpoint series (training-process level)
  and the chess key-move probe (within-state level, real human games,
  external annotations);
- a public third-party environment with an EXTERNAL timestamp: DONE --
  Overcooked-AI, preregistration pushed and tagged before any
  confirmatory seed, all five registered predictions passed (item 35);
- raw trajectory rollouts from larger deep MARL systems for explicit
  future-distribution metrics (remaining nice-to-have).

## Reviewer Risk Register

Current answer to "can this be submitted to a Nature sub-journal now?": yes as
a high-risk submission, but the evidence does not justify claiming a universal
definition of emergence. The strongest defensible contribution is a scoped,
intervention-aware measurement framework with unusually extensive
falsification and public-system transfer.

Likely reviewer objections:

1. **"This is a rebranding of representation jump / phase transition."**
   Current response: reposition possibility collapse as an underlying mechanism,
   not a replacement definition. The representation-jump bridge gives the first
   formula-level connection:

   ```text
   C_t = KL(P_t(B) || P_0(B))
   R_t = E_{B ~ P_t}[phi(B)]
   J_t = ||R_t - R_{t-1}||_2
   ```

   The neural checkpoint bridge now validates the relation on learned
   embeddings; public MultiBERTs/Pythia checkpoint traces extend the
   process-level test, while still not identifying a unique representation
   mechanism.

2. **"The core evidence is synthetic or toy."**
   Current response: analytic controls, external decoy trajectories,
   within-episode `P_t(B | s_t)` from learned rollouts, and a DQN replication
   showing the same rescue/bridge sign flip with a neural learner. The
   simple_spread and LBF studies add two deep-MARL tasks. Go/KataGo remains an
   optional extension, not a prerequisite claimed by the manuscript.

3. **"The basin labels are hand-designed."**
   Current response: `unsupervised_basin_discovery.py` recovers the basins from
   raw event sequences with 1.0 purity in all tested regimes, and the
   effective-mode counts agree in the key uncertain-preference regime.
   External/deep unsupervised basin recovery remains a limitation and is stated
   as such; it is not used to support the public-checkpoint conclusions.

4. **"Collapse can be bad, trivial, or reward-shaped."**
   Current response: negative controls distinguish random instability, blind
   sacrifice, dense shaping and decoy-trap collapse. Chess engine alternatives
   and deep-MARL do-blocks provide stronger counterfactual tests, although only
   the learned episode systems instantiate the full six-component criterion.

5. **"The theory predicts observables after the fact."**
   Current response: the phase-boundary study preregisters parameter regions
   and confirms the predicted emergence/non-emergence boundary in later
   learned runs.

6. **"The formula is not identifiable from real systems."**
   Current response: `P_t(B | s_t)` is estimated by Monte Carlo rollouts of the
   learned policy inside episodes, with the observer marginalizing over latent
   contexts; bootstrap intervals, chess search alternatives and two deep-MARL
   probes quantify uncertainty. Identifiability remains conditional on the
   declared basin map and intervention family.

7. **"Representation jump creates false positives."**
   Current response: yes, and the learned Q-vector probes demonstrate this
   directly. Pure team reward and dense shaping can show large representation
   jumps. The neural checkpoint bridge confirms it at the embedding level:
   burst-jump correlation is ~0.9 for both pure team and uncertain preference,
   so jump-collapse alignment alone does not separate regimes. Separation comes
   from open potential (H0 > 1 bit) plus the do-operator return-gap sign. This
   strengthens the argument that representation jump is an observable, not a
   sufficient definition.

8. **"Single-seed results."**
   Current response: the within-episode probe now has a 5-seed sweep with
   bootstrap CIs. Harmful bridge collapse is sign-stable (5/5); the
   uncertain-preference rescue gap is positive on average but noisy (3/5),
   reported honestly. Remaining need: more seeds and longer training for the
   paper version, plus the same sweep for the neural probe.

9. **"Why exactly these definition components? Any of them could be dropped."**
   Current response: the criterion-ablation battery. Nine measured systems with
   structure-based ground-truth labels; the original five-component harness
   classifies 9/9. Dropping selectivity admits `useful_habit` (a beneficial
   reflex with zero choice tension), dropping usefulness admits
   `wrong_selector` (selective but value-destroying), dropping endogeneity
   admits `shaped_process` (process-rewarded trigger). Single-observable
   definitions (potential-only, specificity-only, usefulness-only) score only
   0.44-0.67. Potential and specificity are not uniquely pinned on this
   battery; their necessity rests on fig4 ground-truth validation and the
   converged_team entropy contrast, stated explicitly.

10. **"The estimates depend on probe hyperparameters."**
    Current response: a 12-cell grid over rollout samples and probe temperature
    preserves the rescue-positive / bridge-negative signs and the
    open-vs-collapsed potential ordering in every cell.

11. **"The labels in your battery are circular."**
    Current response: labels are assigned from audited behavior (per-mode
    trigger rates and returns), not from regime names. The audit changed one
    label against our initial expectation (`noise_policy` learned genuine
    selective structure despite reward noise), and the change is documented in
    the code and README rather than hidden.

12. **"Your selectivity component is just marginal tension; randomness can sit
    mid-range."**
    Current response: correct, and the multi-seed external replication found
    exactly this failure. On 2/5 seeds the full registered criterion accepted
    `marl_untrained`: a random network whose marginal trigger rate happened
    to sit mid-range and whose marginal usefulness gap happened to be
    positive (engaging helps in about half the episodes regardless of who
    engages). We report this as a registered failure, not a nuisance. The
    refinement -- conditional selectivity, per-context trigger-rate
    separation >= 0.5, plus a measured acquisition component (separation
    gain over the same system at its own initialization). Acquisition is
    informative only in the external neural/swarm systems; the internal
    forced-behaviour battery retains five components. Because the refinement was
    chosen after seeing the failures, re-scoring old data (24/25 external,
    10/10 internal on the five informative components, with the new
    `anti_selector` counterexample) is only a
    consistency check; the load-bearing evidence is the out-of-sample
    confirmation on five FRESH external seeds and a fresh internal seed with
    the criterion frozen first (`refined_criterion_confirmation.py`).

13. **"The criterion only works on environments built to satisfy it."**
    Current response: the pre-registered external transfer
    (`external_swarm_criterion_transfer.py`, protocol frozen in
    `EXTERNAL_TRANSFER_PREREGISTRATION.md` before measurement). The full
    criterion, with thresholds copied unchanged from the internal battery,
    was applied to the pre-existing continuous swarm decoy benchmark:
    continuous 2-D positions, 6-vs-6 combat, target-selection actions,
    strictly local observations -- nothing shared with the gridworld family.
    All three registered predictions passed: (p1) the criterion agreed with
    audited labels on 5/5 systems, accepting only the REINFORCE learner that
    a behavioral audit showed had acquired latent-conditional engagement
    (engagement rate 1.00 in the aggressive context, 0.00 in the passive
    context, with no role labels and no process reward on the trigger);
    (p2) `damage_aware` -- a hand rule that reads the sensed-damage feature
    -- passed all four measured components and was excluded only by
    endogeneity, giving an external counterexample that pins that component;
    (p3) the forced-engagement gap sign-flipped across latent contexts
    (~ -20 passive, ~ +22 aggressive), replicating the rescue/bridge
    structure. The multi-seed replication is now done (see item 12 for the
    one weakness it exposed and how it was handled). Remaining need: a
    second external domain with zero authorial control (Go/KataGo).

14. **"A definition should predict, not just classify."**
    Current response: the prospective phase-boundary experiment
    (`phase_boundary_prediction.py`). Closed-form payoff accounting derived
    three boundaries in the control parameter G (behavioral onset at G = 5,
    usefulness sign at G = 9, second onset at G = 11) before any training.
    The primary learner sweep across G in {3,5,7,9,11,13,16} matched all
    four scored non-tie points, including a
    middle phase (G = 7) where the learner is selective but value-destroying
    -- the wrong_selector pattern arising naturally from payoffs rather
    than by construction. Registered ties (G = 5, 9, 11) were excluded from
    scoring in advance. Three independent-seed replications matched 11/12
    scored non-tie points, with one false acceptance at G=7; the outer phases
    were stable in 3/3 seeds. The second onset arrived later than the payoff tie
    (G = 16 rather than 13) due to learning friction at a +2 margin,
    reported as a deviation.

15. **"The thresholds were tuned to make the batteries perfect."**
    Current response: `threshold_sensitivity_analysis.py` rescans every
    numeric threshold over multipliers [0.4, 1.6] (usefulness over +-1.0
    return units) with everything else fixed. External accuracy stays 1.00
    everywhere except separation x1.6 (0.96). Internal accuracy is 1.00 for
    all reductions and within +-20% of everything except the potential
    threshold, where noise_policy (H0 = 0.508 bits) sits just above the
    0.5-bit line: raising that threshold 20% misclassifies it. This knife
    edge is reported, not hidden; it is also expected, because noise_policy
    is the system whose openness barely survives sigma = 4 reward noise.

16. **"Existing emergence measures already do this."**
    Current response: `prior_metrics_comparison.py` runs five
    literature-style single-signal detectors (representation jump, sharp
    metric jump, do-operator specificity alone, PID/Rosas-style synergy,
    and Hoel-style causal-emergence EI gain: EI of a max-entropy macro
    do-variable minus EI of a max-entropy micro do-variable) on the same
    10-system battery, each with its hindsight-OPTIMAL threshold. They cap
    at 0.80-0.90. Four misclassify the genuinely emergent
    latent_conditional (shared-policy blind spot); the EI detector's top
    scorer is the reward-SHAPED system (+0.93 vs +0.71 for the emergent
    one) -- macro-beats-micro is real in both, but EI only compares
    intervened models, so it cannot see natural selectivity, the sign of
    usefulness, or provenance. This is the precise, measured sense in
    which the framework subsumes prior definitions: each prior signal is a
    projection of the mechanism that fires with it, and each has a named
    measured counterexample that the conjunction excludes. The comparison
    is deliberately generous to the priors (oracle thresholds) and still
    favors the multi-component criterion (1.000 with registered
    thresholds).
    STRENGTHENED (published-form audit): `exact_prior_formalisms.py`
    replaces the two flavored formal rivals with quantities computed from
    their published equations, with zero Monte-Carlo error on the enumerated
    policy-closed chains (state = mode x context x positions x switch x
    t; softmax policy in closed form). Hoel's exact EI (max-entropy
    interventions, five candidate coarse-grainings, best taken): CE < 0
    for all 10 systems (micro EI 9.6-11.3 bits, no macro beats it);
    hindsight-best threshold = the trivial all-negative classifier
    (0.8), missing BOTH true positives. Rosas' exact practical
    criterion Psi (four supervenient features x two micro
    decompositions, best taken): the published verdict Psi > 0 scores
    0.3, its top scorer is `wrong_selector` (+0.59) and it misses
    `noise_policy`; hindsight-best (0.9) requires inverting the
    theory's own sign and still misses `latent_conditional`. The blind
    spots are structural on this testbed, not artifacts of proxy
    simplification. The scoped comparison addresses the original equations
    within declared candidate families (Prop. 5, THEORY.md).

17. **"External basins are still hand-defined."**
    Current response: `external_unsupervised_basins.py` clusters label-free
    episode observables (steps, survivors, HP, damage concentration,
    high-initial-HP damage share) with k-means. Purity against the hand
    basins is 0.84-0.99 across all five external systems, and the potential
    component's verdicts are unchanged when H0 is computed over clusters
    instead of hand basins. The registered effective-mode agreement check
    failed (clusters split loss episodes finer than the four hand basins);
    reported as a failed operationalization -- the discovered structure is
    finer, not contradictory.

18. **"The phase-boundary result is single-seed."**
    Current response: three independent seeds pooled give 11/12 non-tie
    prediction matches; the outer phases are 3/3 stable. The single miss is
    at G = 7, where the registered middle-phase usefulness margin is about
    -0.3 return units and sampling noise can flip the sign. Sharper
    statistics at the boundary (more probe episodes) is the known fix.

19. **"Your framework says nothing about large-model emergent abilities."**
    Current response: the grokking bridge (`grokking_collapse_bridge.py`).
    The model's predictive distribution on held-out inputs is a possibility
    distribution; grokking is a delayed, sudden, useful, endogenous collapse
    of that distribution, and the registered process-level criterion
    separates it from a memorizer (fails usefulness), a shuffled-label task
    (fails usefulness and burstiness), and a prewired model trained on the
    evaluation distribution (fails endogeneity only -- the process-level
    analogue of shaped_process/damage_aware). The same run also contains an
    early collapse WITHOUT usefulness (memorization), which the registered
    anchored window refuses to credit. Generality (second task, more seeds)
    and the scale-sweep decomposition of the Wei-vs-Schaeffer debate are
    covered by `grokking_generality_sweep.py` and
    `scale_emergence_decomposition.py`.

20. **"All target phenomena were chosen or built by you; the criterion has
    never faced an ability the community discovered independently."**
    Current response: the induction-head experiment
    (`induction_head_emergence.py`, fig29). Induction heads are an
    externally discovered, externally named, externally explained phase
    change (Olsson et al. 2022), with an external impossibility theorem
    (one-layer attention-only transformers cannot implement the circuit,
    Elhage et al. 2021). We trained attention-only transformers on
    variable-offset repeated sequences and applied the process-level
    criterion imported frozen from the grokking bridge. The two-layer model
    shows a sharp useful collapse (entropy 5.8 -> 0.1 bits coinciding with
    the copy-accuracy jump 0.02 -> 0.99) and is the only accepted condition;
    the one-layer control -- identical in everything but depth -- plateaus
    at 0.11 accuracy and is rejected via usefulness, so the external
    architectural theorem and the criterion's verdict coincide. 12/12
    fresh-seed sweep checks passed. Honest notes: a fixed-offset pilot was
    solved positionally by the one-layer model (task revised, pilot log
    preserved, criterion untouched).

21. **"The process-level results are an MLP artifact."**
    Current response: `transformer_grokking_replication.py` (fig30) repeats
    the grokking/memorizer contrast with a 412k-parameter causal
    transformer (the architecture family of the original grokking report)
    under the frozen criterion: grokking passes all four components,
    memorizer fails usefulness. With fig23/fig27 (MLP) and fig29
    (attention-only), the process-level instantiation now spans three
    architecture families. Honest note: the weight-decay-1.0 pilot showed
    slingshot instability; optimizer hyperparameters only were retuned
    once, with the unstable pilot log preserved.

22. **"Validate on a public checkpoint series with zero authorial
    involvement."**
   Current response: done (`multiberts_collapse_probe.py`, fig31,
    protocol frozen in `MULTIBERTS_PREREGISTRATION.md` before download).
    Google Research's MultiBERTs seed_0 series (Sellam et al., ICLR 2022)
    provides a 110M-parameter BERT-base
    with 29 published intermediate checkpoints we did not train or
    influence. The externally documented ability (long-range subject-verb
    agreement) passes the frozen process proxy: potential 14.7 bits, collapse
    burst coinciding with the accuracy jump 0.50 -> 0.98 in the first 40k
    steps, usefulness gain +0.48. Both registered controls fail exactly as
    predicted. The random-target control shares the agreement entropy/KL
    trace but fails usefulness; the shuffled-vocabulary control disrupts the
    ability readout and is also rejected on usefulness. Thus collapse
    symptoms alone cannot pass even on a public 110M-parameter system. 4/4 registered
    predictions passed, and the result replicates on ALL four remaining
    published seeds (15/15 verdicts, anchor window at step 20k on every
    seed). Honest notes: the late-pretraining accuracy drift (0.98 peak ->
    0.    93 at 2M steps) is reported, and burst timing is limited by the
    published checkpoint grid. UPDATE (decoder pass): the Pythia
    replication is DONE -- Pythia-160m, 21 published checkpoints
    fetched via a public mirror of the blocked hub, frozen protocol in
    PYTHIA_PREREGISTRATION.md. Agreement emerges (0.49 -> 0.93,
    burstiness 27.6, gain 0.47) with both controls failing usefulness
    despite sharing the relevant collapse/burst substrate; the foreshadow burst precedes the
    jump. 7/8 registered predictions passed (fig38). The
    zero-authorial-control requirement is met on BOTH architecture
    families the LM debate involves.

23. **"One ability on the public model is anecdote, not a battery; and
    every public-model rejection went through usefulness."**
    Current response: the registered phenomena battery
    (`multiberts_phenomena_battery.py`, fig32) adds reflexive agreement,
    determiner-noun agreement, country-capital facts, and NPI licensing.
    The two syntax families pass as registered. The two
    auxiliary-gradualism predictions (facts R3, NPI R5) FAILED -- both
    abilities came out abrupt and burst-coincident -- and are reported as
    registered failures with the honest reading: templated probes on
    high-frequency material hit ceiling early, so at this grid every
    probed ability is an abrupt useful collapse in the first 2-4% of
    pretraining, syntax/facts first, NPI one interval later
    (collapse-at-20k, usefulness-at-40k -- the collapse-precedes-jump
    signature visible inside a public model). A genuinely gradual ability
    that the criterion rejects on this system remains an open item; the
    shuffled-vocab and random-target controls still pin usefulness.
    UPDATE (decoder pass): on Pythia the open item is closed -- the
    frozen criterion REJECTS both frequency-tail families on the public
    decoder (tail_facts via burstiness 3.2 < 5, a ramp with no dominant
    burst; tail_words via usefulness, gain 0.056) while accepting head
    facts on the same checkpoints. Rejections now go through TWO
    different components, answering the "every rejection went through
    usefulness" half of this objection as well.

24. **"The burst-jump 'coincidence' was never quantified."**
    Current response: `burst_alignment_test.py` (fig32 right). Every accepted
    run's ability jump lands in a top collapse-burst window (empirical
    window rank 0.023-0.051 per run, 11 runs across
    MultiBERTs/grokking/induction); noise-jump controls sit at 0.54-0.99.
    No omnibus p value is reported because runs are dependent and three
    phenomena were exploratory additions. The registered R4
    bound (p <= 3/27) holds for every emergent MultiBERTs family. The
    script documents the caveat that this is a coincidence statistic and
    cannot replace the usefulness component (random_target shares the
    collapse series and shows small p with a noise jump).
    UPDATE (multiplicity): conservative Holm values are all >=0.252 and the
    smallest BH value is 0.058. The manuscript now labels alignment as
    descriptive local rank evidence, not family-wise significance.

25. **"Only one external family measures all six components."**
    Current response: Contextual LBF confirmation is complete. After two
    excluded design pilots, code, ten fresh seeds, thresholds and predictions
    were frozen (`CONTEXTUAL_LBF_PREREGISTRATION.md`). Nine of ten learned
    policies pass all six components; seed 1104 is retained at selectivity
    0.4875 < 0.5. All ten show the predicted context ordering, positive
    usefulness and positive acquisition. All 40 initialization/scripted
    controls reject. The strongest control, a competent team-nearest
    coordinator, passes all four behavioral components and fails exactly
    endogeneity/acquisition on 10/10 seeds. Seed-bootstrap lower 95% bounds:
    usefulness 0.053, acquisition 0.634. All six registered group predictions
    pass. Honest scope: the contextual layout wrapper is author-designed even
    though dynamics, observations, actions and sparse reward are from the
    recognized LBF benchmark.

26. **"Episode count, checkpoint grid and unbounded ratios inflate
    confidence."**
    Current response: `hierarchical_marl_analysis.py` gives seed--episode
    cluster intervals. LBF remains positive [0.072, 0.153], but
    simple_spread's mean interval crosses zero [-0.005, 0.187]; exact
    three-seed sign p=0.125 in both. `process_proxy_robustness.py` maps the
    ratio threshold 5 exactly to bounded q>=5/6 (27/27 unchanged), varies
    windows, and checks 243 thinning cells (93.8% agreement). The tail-control
    flips are retained as evidence of real checkpoint-grid sensitivity.

27. **"One scalar vs a six-threshold AND rule is an unfair comparison;
    give the priors multivariate freedom."**
    Addressed by `fair_baseline_comparison.py`: hindsight AND rules over
    prior-signal pairs/triples (each with its own threshold and direction),
    logistic regression and a depth-2 tree on all five transportable
    signals plus exact EI/Psi, and two-component AND rules over our own
    components. In-sample 0.9-1.0, LOOCV 0.7-0.8, and every baseline frozen
    and transferred to the fresh-seed battery reaches at most 0.9 while
    always misclassifying latent_conditional; the frozen six-component
    protocol scored 10/10 on the same fresh seed (stored earlier,
    untouched).

28. **"Random partitions do not test observer dependence; two plausible
    contracts might disagree."**
    Addressed by `dual_observer_contracts.py`: a second declared contract
    (horizon 12, undiscounted success value, speed-refined trigger-resolving
    basins) re-scores all 75 stored CLBF systems under frozen thresholds.
    Controls: 60/60 rejected under both contracts. Structural layer: agrees
    14/15 learned seeds. Value layer: 5 conservative flips (4 usefulness,
    1 borderline selectivity), recorded as registered misses DO-1/DO-3.
    Reading: structural verdicts travel, value verdicts are declaredly
    contract-relative -- now measured, and stated in the manuscript.

29. **"Both chess referees share the Stockfish lineage."**
    Addressed twice. `chess_realized_outcome.py`: an engine-free referee
    from realized game results -- directionally consistent (humans play the
    shallow-best move far more often at flagged positions: 0.53 vs 0.34 and
    0.43 vs 0.27; realized-score gain positive both months) but the
    registered interaction is null (p = 0.54), underpowered at n = 75
    flags/month, kept as registered miss RO-1.
    `chess_discovery_toga_referee.py`: relabelling with Toga II 3.0
    (Fruit 2.1 lineage -- independent search implementation and handcrafted
    evaluation, no shared code with Stockfish) under the frozen referee
    rule; registered prediction TG-1 (do-gap AUROC > 0.60 both months).

30. **"Positions may be correlated through players; position-level
    inference understates uncertainty."**
    Addressed by `chess_clustered_inference.py`: mover identity recovered
    from the PGNs (92 and 86 multi-position mover clusters per month);
    mover-cluster bootstrap AUROC intervals [0.615, 0.829] and
    [0.613, 0.834] -- wider than naive position-level intervals but
    excluding chance in both months. Reported in the manuscript.

31. **"Endogeneity is provenance metadata; the system boundary is
    undeclared."**
    Addressed in Methods ("Terminology and the system boundary"): the
    observer contract now declares the system boundary explicitly;
    endogeneity is paired with measured acquisition (initialization-twin
    contrast), and the known boundary case (a supervised clone of a
    script passes acquisition relative to its declared training signal)
    is stated rather than hidden.

33. **"The possibility space has to be hand-crafted by someone who already
    understands the system."**
    Addressed by `learned_basin_clbf.py`: k-means on raw trajectory
    features (no semantic labels) discovers the basin partition per seed;
    all 15 discovered partitions resolve the macro-structure, 60/60
    controls remain rejected, 14/15 learned systems remain accepted, and
    74/75 verdicts agree with the hand-basin protocol (LB-1..LB-3 all
    pass). The observer contract still declares the FEATURE space and k,
    which is stated -- but semantic basin authorship is not required.

34. **"Measured openness is just the policy's sampling stochasticity."**
    Addressed by `independent_rollout_audit.py`: under near-greedy
    decoding (T=0.2) every learned seed keeps >= 0.5 bits of potential
    (IR-1 15/15) and twins are rejected under all models (IR-3 15/15).
    The diffuse-model prediction IR-2 FAILED (8/15) and is retained:
    T=2.0 noise dilutes selectivity and erases the value contrast --
    the rollout policy is a behaviour-changing declared contract item,
    not a free estimator knob.

32. **"The substrate must produce a prediction no prior theory makes."**
    Addressed by `strength_gradient_battery.py` + `strength_gradient_fine.py`:
    the rarity identity predicts a GRADED emergence strength for the same
    macro-structure under different provenances (scripted / process-shaped /
    outcome-only). Registered outcome: seed-mean provenance rarity orders
    0 < 0.39 < 0.65 bits at matched final competence (~0.99), with the
    open-space rarity identical (6.28 bits) by construction; discovery is
    ~2x later without process shaping. The ST-3 suddenness-ordering miss is
    retained (both acquisitions are step-like at the measured grids):
    strength is carried by provenance rarity and discovery time, not burst
    shape. This operationalizes the folk gradient "prescribed < shaped <
    discovered" (non-emergent / weak / strong).

35. **"Every environment is author-designed; nothing was externally
    preregistered."**
    Addressed by the Overcooked-AI confirmation: public, unmodified
    benchmark; protocol, thresholds and predictions OC-1..5 frozen and
    pushed to a public repository (tag `v1.0-overcooked-prereg`,
    commit `8415e45`) with the timestamp preceding every confirmatory
    seed; the run script refuses to start without asserting the
    timestamp. All five registered predictions passed: learned
    accepted 8/12 (registered line exactly 8/12), controls rejected
    48/48 with the layered failure routes, trigger direction 12/12,
    contract-B twin rejections 12/12, usefulness do-contrast positive
    12/12 (p = 2.4e-4). This closes the single largest external
    criticism (author-designed-world circularity).

36. **"The continuous profile is descriptive wrapping around the
    binary verdict."**
    Partially addressed, honestly bounded: cross-contract ranking of
    E_struct is stable (mean Spearman 0.76, RS-1 pass); RS-2 failed as
    registered -- no structure-only score separates learned from
    scripted (the layering's own claim, re-derived); the
    predictive-validity battery PV-1..3 failed as registered (early
    causal magnitude does not predict boundary-cell acceptance; early
    performance is better; descriptively the early value axis of the
    same record reaches AUROC 0.81). The profile earns its place as a
    calibrated record (orthogonality + dose-response), not as an early
    predictor at phase boundaries.

37. **"Why exactly these dimensions, these normalizations, these
    divergences? Nothing shows they measure the constructs they
    name."**
    Addressed by `generator_calibration.py`: a six-knob ground-truth
    generator (selectivity, causal reorganization, signed value,
    acquired fraction, acquisition steepness, retention), each knob
    controlling exactly one construct BY CONSTRUCTION, measured through
    the full finite-sample estimator pipeline. GC-1 diagonal dominance
    passed with zero violations; GC-3 nullity, GC-4 value separability,
    GC-5 provenance separability all passed; GC-2's off-diagonal rule
    failed on two couplings and is retained (frozen exemption list
    omitted the matched acquisition pair; persistence responds to
    structure at 0.29 of diagonal because retention is retention OF
    structure). Plus `verify_record_axioms.py`: eight axioms (nullity,
    boundedness, monotonicity, data processing, context sensitivity,
    value/provenance separability, abstention) machine-verified, all
    pass. Plus the admissible-contract family and identification
    intervals in THEORY.md (median E_struct interval width 0.14 over
    five stored contracts).

38. **"The record should predict, component by component, its own
    endpoints -- not one global label."**
    Addressed by `convergent_validity.py` (fresh 9800-series seeds):
    CV-1 passed -- early causal magnitude predicts final structure
    where early performance is uninformative (Spearman 0.56 vs -0.09).
    CV-2/CV-3 failed as registered and are retained: near the phase
    boundary early performance predicts the value endpoint as well as
    early usefulness, and the record adds no LOO R^2 at n=20. The
    honest claim: predictive content is axis-specific.

39. **"Does the framework say anything about collective/crowd
    emergence (swarm-mind phenomena like Twitch Plays Pokemon)?"**
    Addressed by `crowd_vote_domain.py`: a collective-control family
    where the system's action is the vote-aggregation mode itself.
    50/50 controls rejected with exactly the declared routes (blanket
    democracy = forced convergence, fails potential; blanket anarchy
    fails usefulness through falls -- the historical counterfactual,
    reproduced by do-block 10/10; hand switcher and BC clone fail
    exactly endogeneity+acquisition). CR-1 retained miss (7/10; the
    three rejections learn blanket democracy -- context-blind
    competence, the same route as the rejected Overcooked seeds) and
    CR-5 retained miss (selectivity of the convention is graded). Two
    design pilots quarantined and disclosed. The historical event
    itself is classified observational-scale: no do-operators, no
    twin -- exactly what the instrument table declares.
