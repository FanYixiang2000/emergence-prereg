# Prediction ledger: every frozen prediction, its outcome, in one table

Purpose (reviewer-facing): the project's forking-paths defense is that
every domain ran under a preregistration whose predictions were frozen
before measurement and whose failures were kept. That defense is only
checkable if the predictions live in one place. This ledger indexes
every frozen prediction/check, its outcome, and the file recording it.
Nothing here is new analysis; every row points to the primary document.

Convention: PASS = the frozen prediction came out true. FAIL = it came
out false, kept as a registered failure, never re-thresholded. Rows
are grouped by domain in rough chronological order.

## Tier structure (the count is NOT a sample size)

The ledger mixes items of very different inferential weight. Tiers:

- **Tier 1 -- primary confirmatory hypotheses** (~20 rows): the
  preregistered predictions that carry the paper's conclusions
  (full-criterion confirmations in three domains: GW-1, SW/EXT
  confirmation rows, CLBF-C1..C6, LC-1..LC-6; discovery CD-1..CD-4;
  the curated chess do-contrast CH-C1..C4).
- **Tier 2 -- replication and transfer hypotheses**: fresh seeds, second
  months, held-out scales, cross-task and cross-modality transfers
  (CD-R, SC-S1..S7, MB/PY scale rows, phase-boundary replication,
  persistence PS-1..PS-4, generalization LG-1..LG-3).
- **Tier 3 -- diagnostic and mechanistic predictions**: detector blind
  spots, route predictions, boundary probes; failures here sharpen scope
  without changing the primary conclusions.
- **Tier 4 -- software and artifact invariants**: hash audits,
  relabelling invariance, cross-implementation checks, number-consistency
  audits. These support trust in the pipeline, not scientific inference.

Main-text claims rest on Tiers 1-2 only; Tier 3-4 items are reported as
audit infrastructure. No family-wise statistic is computed over the ledger
because the rows are heterogeneous and dependent.

## Internal gridworld family

| ID | Prediction (one line) | Outcome | Recorded in |
|---|---|---|---|
| GW-1 | 10-system internal battery: frozen conditional-selectivity rule matches all audited labels (five informative components; acquisition not scored internally) | PASS 10/10 | criterion_ablation_battery.py, refined_selectivity_check.py outputs |
| GW-2 | Registered named drop-ablation routes fail as predicted | PASS (selectivity, usefulness and endogeneity routes; potential/specificity pinned by dedicated controls rather than unique accuracy drops) | criterion_battery_measurements.csv, fig19 |
| GW-3 | Prior single-signal detectors: EI blind spot predicted in docstring before running (cannot see natural selectivity/provenance) | PASS (top scorer = shaped_process) | prior_metrics_comparison.py, prior_metrics_comparison.json |
| GW-4 | EXACT Hoel EI / Rosas Psi keep the same structural blind spots as the proxies | PASS (CE: trivial classifier; Psi>0 acc 0.3, top scorer wrong_selector) | exact_prior_formalisms.py, exact_prior_formalisms.json, Prop. 5 |
| GW-5 | Phase boundary: per-cell verdicts derived from Prop. 3 BEFORE training the G-grid | PASS (non-tie match rate 1.000, single seed) | phase_boundary_prediction.py outputs, fig22 |
| GW-6 | Phase boundary replicates on 3 seeds | MISS at G = 7 only (11/12; middle-phase usefulness margin ~0.3 return units, sign flips under sampling noise) | EVIDENCE_AUDIT.md item 18 |
| GW-7 | Unsupervised basins: cluster purity high AND effective-mode agreement with hand basins | PARTIAL: purity PASS (0.84-1.0); agreement check FAIL (clusters split loss episodes finer) | unsupervised_basin_discovery.py, EVIDENCE_AUDIT.md item 17 |

## External swarm transfer

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| E-1 | Original episode criterion matches audited labels on all 5 external systems, thresholds transferred unchanged | PASS 5/5 | EXTERNAL_TRANSFER_PREREGISTRATION.md |
| E-2 | damage_aware passes all four measured components, excluded ONLY by endogeneity | PASS | same |
| E-3 | Forced-engagement usefulness sign-flips across latent contexts | PASS | same |
| E-4 | Multi-seed replication: 4 registered predictions across 5 fresh seeds | 3 PASS 5/5 seeds; all-verdicts prediction FAIL on 2/5 seeds (untrained net accepted via marginal tension + accidental usefulness) | same (outcome section) |
| E-5 | Refined criterion out-of-sample: 5 fresh external seeds + fresh internal seed | PASS (full six components: 25/25 external; five informative internal components: 10/10; anti_selector fails exactly usefulness; seed 8431 training failure correctly rejected per protocol) | refined_criterion_confirmation.py outputs, README |

## Grokking / induction heads / scale

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| GK-1 | Grokking passes frozen training-process rule (now reported as the process proxy); memorizer / shuffled-label / prewired fail via named routes | PASS (prewired fails endogeneity ONLY, as predicted) | grokking_collapse_bridge.py outputs, fig23 |
| GK-2 | Generality: 2 tasks x 3 seeds | PASS (grokking emergent 6/6; controls fail as registered) | fig27 outputs |
| GK-3 | Transformer replication + seed sweep (3 seeds x 4 conditions) | PASS 12/12 | transformer_grokking_replication.py outputs, fig30 |
| IH-1 | Induction heads: frozen process proxy accepts two-layer, rejects one-layer / no-structure / memorizer; 4 seeds | PASS 16/16 (one-layer rejection route = usefulness, coinciding with the external impossibility theorem) | induction_head_emergence.py outputs, fig29 |
| SC-1 | Scale decomposition R2: accuracy <= normalized collapse + 0.15 everywhere | FAIL (normalization mis-designed; the substantive reading -- ability jump and collapse transition in the same scale region -- held and is reported with the failure) | README scale section, fig26 |

## Capability novelty boundary: ordinary learning is not automatically emergence

The ordinary-learner probe is a retained falsification: the old
four-component process proxy accepts all 6/6 fast smooth learners. Therefore
abruptness/gradualness and useful output collapse cannot decide capability
emergence. The added necessary product test compares the full learner with a
frozen lower-order hypothesis class seeing the same inputs:
`N_cap = accuracy(full) - accuracy(lower-order)`. The threshold 0.30 and all
predictions below were frozen before this run. This is a capability-specific
boundary test, not a claim that one additive baseline universally defines
representational novelty.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CN-1 | Ordinary coarse-sum: additive baseline accuracy >= 0.85 and novelty gap <= 0.10 | PASS (full 0.924, additive 0.927, gap -0.003) | capability_novelty_boundary.py, capability_novelty_boundary.json |
| CN-2 | Modular addition: additive accuracy <= 0.10, full grokking accuracy >= 0.90 and novelty gap >= 0.80 | PASS (full 1.000, additive 0.000, gap 1.000) | same |
| CN-3 | Induction architectural composition: two-layer minus one-layer accuracy gap >= 0.70 | PASS (0.987 vs 0.116, gap 0.871) | same |
| CN-4 | Retain old process-proxy false positive (ordinary 6/6 accepted), but novelty-qualified rule rejects ordinary and accepts modular grokking + induction | PASS (3/3 boundary classifications) | same |

## Burst-collapse boundary: original hypothesis retained but scoped

The original operational story treated burst-like possibility collapse as the
separating mark of emergence. This audit keeps that hypothesis visible and
tests its logical status using only stored outputs. The result is a
scope-fixing double dissociation: burst remains a valuable acquisition-shape
signal, but it is neither sufficient nor necessary for the emergence verdict.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| BB-1 | Burst is not sufficient: ordinary learner passes old process proxy 6/6 with all burstiness ratios >= 5, but lower-order novelty rejects it | PASS (min burstiness 6805; old proxy 6/6; N_cap -0.003, novelty verdict false) | burst_boundary_audit.py, burst_boundary_audit.json |
| BB-2 | Burst is not necessary: ant TRAIL accepted as weak collective emergence while 10-90 commitment is gradual | PASS (D 0.942, R 0.966; span 0.248 of horizon, max step 0.154) | same |
| BB-3 | Burst verdicts are grid-relative: Pythia-2.8B agreement has usefulness but fails bounded_burst on the full grid, flipping in all 9 thinning cells | PASS (full emergent 0; usefulness true; bounded_burst false; thinning agreement 0/9) | same |
| BB-4 | Burst alone cannot decide controls: at least one stored control has bounded_burst true but fails usefulness and is rejected | PASS (4 bursty failed controls across held-out Pythia scales) | same |

## MultiBERTs (public checkpoints, zero authorial control)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| MB-P1..P4 | Agreement emergent w/ burst-jump coincidence; random-target fails usefulness; shuffled-vocab fails usefulness with identical entropy/KL; jump in dense region | PASS 4/4 | MULTIBERTS_PREREGISTRATION.md |
| MB-G1..G5 | Reflexive + determiner emergent every seed; all random-target controls fail usefulness; windows <= 200k; bootstrap CIs clear thresholds | PASS 5/5 (then replicated: 15/15 verdicts on all 5 published seeds, same anchor window) | same |
| MB-R1,R2,R4 | Reflexive/determiner emergent, burst top-3 aligned | PASS (alignment p = 1/27 each) | same |
| MB-R3 | Country-capital facts NOT emergent (gradualism) | FAIL (emergent at 20k; post-mortem: high-frequency facts learn early) | same |
| MB-R5 | NPI NOT emergent (hard/late) | FAIL (emergent at 40k; two-phase: collapse at 20k at chance accuracy, usefulness at 40k) | same |
| MB-T1 | head_facts emergent (pair metric) | PASS | same |
| MB-T2 | tail_facts NOT emergent (pair metric) | FAIL (ceiling by 20k; probe-format post-mortem recorded) | same |
| MB-T3 | tail_words NOT emergent (pair metric) | FAIL (same route) | same |
| MB-T5 | head_facts_top1 emergent | PASS | same |
| MB-T6 | tail_facts_top1 NOT emergent | FAIL (sigmoid's steepest 20k segment exceeds the windowed-gain threshold; grid-resolution reading recorded) | same |
| MB-T7 | tail_words_top1 NOT emergent | PASS (rejected via burstiness AND usefulness; the process proxy CAN reject a real, slowly-learned ability on a public system) | same |

## Pythia (public DECODER checkpoints, zero authorial control)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| PY-Y1 | Agreement emergent (all four components), burst-jump coincidence under the anchored window | PASS (H_pre 8.86 bits, burstiness 27.6, usefulness gain 0.47; acc 0.49 -> 0.83 -> 0.93 across steps 512/1000/2000) | PYTHIA_PREREGISTRATION.md |
| PY-Y2 | random_target fails usefulness | PASS (gain 0.045) | same |
| PY-Y3 | shuffled_vocab fails usefulness | PASS (gain -0.024) | same |
| PY-Y4 | Anchored window at or before step 16k; H_pre >> 1 bit | PASS (window at step 1000, ~0.7% of training) | same |
| PY-T1 | head_facts emergent | PASS (burstiness 16.4, gain 0.35) | same |
| PY-T2 | tail_facts NOT emergent, route usefulness | VERDICT CORRECT, ROUTE MISS (rejected via burstiness 3.2 < 5; the windowed gain passed -- the slow ramp has no dominant burst, so the burst component catches the gradualism instead) | same |
| PY-T3 | tail_words NOT emergent, route usefulness | PASS (gain 0.056) | same |
| PY-S1..S3 | 410m scaling replication: agreement emergent, both controls fail usefulness, window at/near the 160m window | PASS 3/3 (window at step 1000, identical to 160m; attractor dip replicates and deepens) | same |
| SC-S1 | Held-out scales (1B/1.4B/2.8B): agreement passes all four proxy components at every scale | **FAIL at 2.8B** (1B/1.4B pass with window at step 1000; 2.8B: usefulness 0.42 passes, burstiness 3.2 < 5 -- collapse spread across early intervals at the published grid) | PYTHIA_SCALING_PREREGISTRATION.md |
| SC-S2 | Both controls fail usefulness at every held-out scale | PASS 6/6 | same |
| SC-S3 | Agreement accuracy > 0.8 by step 4000 at every held-out scale | PASS 3/3 (steps 1000/1000/2000) | same |
| SC-S4 | Largest collapse increment no later than one interval after the largest agreement jump, >= 2/3 scales | PASS 3/3 (burst precedes the jump by two intervals at every scale) | same |
| SC-S5 | Head facts accepted >= 2/3 scales; both tail families rejected >= 2/3 scales | **head half FAIL** (0/3 held-out scales accept; perfect final accuracy with sub-threshold burstiness); tail half PASS (6/6 rejections) | same |
| SC-S6 | Bounded transform leaves all primary verdicts unchanged | PASS | same |
| SC-S7 | Thinning agreement >= 90% of condition-level verdicts | **FAIL** (130/162 = 80.2%; diagnostic: 2.8B agreement flips to accept in 9/9 thinning cells -- the S1 failure is grid-relative) | same; held_out_scaling_robustness.json |

## Chess (external strategic system)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CH-C1 | Key move locally costly vs best reply; greedy is not | PASS (median -3.0 pawns; 80% strictly costly) | CHESS_PREREGISTRATION.md |
| CH-C2 | useful_shift(key) beats greedy and random, sign tests p < 1e-3 | PASS (238/240, 239/240) | same |
| CH-C3 | P(win/do key) - P(win/do deep_alt) >= 0.15, sign test | PASS (gap 0.539, 240/240) | same |
| CH-C4 | Median potential >= 1.0 bits under frozen observer | PASS (1.19 bits) | same |
| CH-C5 | Quiet control: shift margin >= 0.25 vs sacrifice shift | FAIL on the margin (endogenous baseline already prices in the key move; do-contrast is 0.539 vs ~0; martingale lesson recorded) | same |
| CH-G1 | Robustness: C1-C3 conclusions hold across 12 perturbation cells incl. classical-eval engine | PASS 12/12 (C4 absolute potential fails in the 3 strongest-observer cells -- the declared estimator-scale dependence) | chess_robustness_grid.json |
| CD-1 | Uncurated discovery: AUROC(do-gap) >= 0.70 vs independent deep-referee labels | PASS (0.730; 400 uncurated positions, base rate 0.095) | CHESS_DISCOVERY_PREREGISTRATION.md |
| CD-2 | Frozen flag precision >= 2x base rate | PASS (0.24 = 2.53x; 75 flagged, recall 0.47) | same |
| CD-3 | do-gap AUROC beats tactical-density and material baselines | PASS (0.730 > 0.635 > 0.589; same-family deep-eval-gap baseline 0.762 reported head-to-head without a registered direction) | same |
| CD-4 | Flagged median potential >= 1.0 bits | PASS (1.92 bits) | same |
| CD-R1..R4 | Frozen replication of CD1-CD4 on a different year (2016-03) | PASS 4/4 (AUROC 0.725; precision 0.347 = 2.95x base; potential 1.98 bits; do-gap stable across months while the shallow eval-gap baseline drops 0.762 -> 0.652) | same |

## Latent-context sequence model (third full six-component domain)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| LC-1 | Learned passes full six-component rule >= 9/10 fresh seeds | PASS 10/10 (selectivity 0.95-1.00; usefulness 0.475-0.500; acquisition 0.888-0.988) | LATENT_CONTEXT_PREREGISTRATION.md |
| LC-2 | All 40 non-learned verdicts rejected | PASS 40/40 | same |
| LC-3 | Oracle router fails exactly {endogeneity, acquisition} >= 9/10 | **FAIL, informative route** (rejected 10/10 but on FOUR components: deterministic router is intervention-inert in sequence space, so specificity=0 and usefulness=0 too) | same |
| LC-4 | All learned acquisitions positive; all twins fail acquisition | PASS 10/10 + 10/10 | same |
| LC-5 | Context ordering on every seed; usefulness positive >= 9/10 | PASS 10/10 + 10/10 | same |
| LC-6 | Seed-bootstrap lower bounds positive (usefulness, acquisition) | PASS ([0.481,0.494], [0.918,0.955]) | same |

## Persistence of the acquired structure (Contextual LBF, saved policies)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| PS-1 | >= 8/10 policies retain >= 50% of own baseline selectivity under each perturbation except strongest noise | **FAIL via one route** (horizon/noise: 40/40 cells pass; novel layouts: 5/10 -- spatial transfer is partial, mean selectivity 0.80 -> 0.29) | PERSISTENCE_PREREGISTRATION.md |
| PS-2 | >= 8/10 policies keep positive usefulness in >= 5 settings | PASS (10/10 at 6-of-7), with the shared failing setting reported: usefulness negative on novel layouts for all ten | same |
| PS-3 | Every initialization twin stays below 0.5 selectivity everywhere | PASS (70/70 cells) | same |
| PS-4 | Mean selectivity non-increasing in noise | **FAIL, benign route** (flat 0.800/0.800/0.803/0.798 up to sigma 0.2 -- more noise-robust than predicted; monotonicity broken by 0.003 jitter) | same |

## Deep MARL (simple_spread + LBF)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| DM-D1 | Trained early potential >= 1.0 bits; controls' higher openness must not qualify them | PASS | DEEP_MARL_PREREGISTRATION.md |
| DM-D2 | Within-episode win-mass rise + bijection >= 0.5 all seeds | FAIL (1 seed 0.45; win-mass is a martingale under the behaving policy -- prediction mis-designed, diagnosis kept) | same |
| DM-D3 | do_commit vs do_block shifts win probability (pooled sign test) | PASS (median +0.083, p = 0.0037) | same |
| DM-D4 | Greedy scripted: lower bijection and lower openness than trained | PASS | same |
| LBF-L1..L4 | Cross-task replication: openness, useful structure, do-contrast, greedy contrast | PASS 4/4 (first clean sweep; martingale lesson priced into L2's design) | LBF_PREREGISTRATION.md |
| LBF-G1 | do-contrast positive with sign-test p < 0.05 in EVERY probe-temperature cell (T = 2..8) | PASS (worst cell 20W/0L, p <= 2.7e-20) | lbf_robustness_grid.py docstring + json |
| LBF-G2 | Trained-vs-greedy double dissociation direction in every cell | PASS (absolute 0.8-bit threshold fails at T <= 3: declared observer-scale dependence, same direction as CH-G1's C4 cells) | same |
| LBF-DET-1 | Round 1: every single detector misclassifies forced_commit or pays elsewhere | FAIL for the performance detector (it separated the initial 7-system set -- the set lacked a competent imitation; archived) | lbf_prior_detectors.py docstring, lbf_prior_detectors_round1.json |
| LBF-DET-2 | Round 2 (frozen before rerun): with scripted_coop added, performance accepts it; structure detectors score it at/above trained; no single detector reaches 1.0 | PASS (performance best 0.875 missing scripted_coop; specificity/Psi/EI rank scripted_coop ABOVE trained: 0.94/0.95/0.68) | lbf_prior_detectors.json |

## Contextual LBF full criterion (fresh seeds)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CLBF-C1 | Learned full six-component pass on at least 9/10 fresh policy seeds | PASS 9/10 (seed 1104 retained: selectivity 0.4875 < 0.5) | CONTEXTUAL_LBF_PREREGISTRATION.md, contextual_lbf_confirmation_analysis.json |
| CLBF-C2 | Initialization twin + three scripted controls rejected on every seed | PASS 40/40 | same |
| CLBF-C3 | Competent team_nearest fails exactly endogeneity + acquisition on at least 9/10 seeds | PASS 10/10 | same |
| CLBF-C4 | Learned acquisition positive every seed; every initialization twin fails acquisition | PASS 10/10 + 10/10 | same |
| CLBF-C5 | Context ordering every seed; usefulness positive on at least 9/10 | PASS 10/10 + 10/10 | same |
| CLBF-C6 | Seed-bootstrap lower 95% bounds positive for acquisition and usefulness | PASS (0.634 and 0.053) | same |

## Reframing-revision audits (fair baselines, dual observer, realized outcome)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| FB-1 | Frozen multivariate prior-signal baselines (fitted with hindsight on the original battery) transfer to the fresh-seed battery below the six-component protocol's stored 10/10 | PASS (best transferred baseline 0.9; every baseline misclassifies latent_conditional; two-component AND rules 0.7-0.8) | fair_baseline_comparison.py, fair_baseline_comparison.json |
| DO-1 | All 15 learned CLBF systems accepted under the second plausible observer contract | **FAIL** (10/15; four flips through usefulness under the success-value/horizon-12 contract, one borderline selectivity 0.4625) | dual_observer_contracts.py, dual_observer_contracts.json |
| DO-2 | All 60 CLBF controls rejected under the second contract | PASS 60/60 | same |
| DO-3 | Contract A and contract B verdicts agree on all 75 systems | **FAIL** (70/75; all disagreements conservative, concentrated in the declaredly value-relative component) | same |
| RO-1 | Realized-outcome interaction: playing the shallow-best move helps realized score MORE at flagged than unflagged positions (pooled) | **FAIL** (interaction -0.009, permutation p = 0.54; underpowered at n = 75 flags/month) | chess_realized_outcome.py, chess_realized_outcome.json |
| RO-2 | Realized-score gain from playing the shallow-best move at flagged positions positive in each month | PASS (+0.002 and +0.024; wide CIs) | same |
| CL-1 | Mover-cluster bootstrap of the discovery AUROC excludes 0.5 in both months (Tier-4 robustness re-analysis, declared in docstring) | PASS ([0.615, 0.829] and [0.613, 0.834]; 92/86 multi-position mover clusters) | chess_clustered_inference.py, chess_clustered_inference.json |
| TG-1 | Different-lineage engine referee (Toga II / Fruit family): stored do-gap AUROC > 0.60 in both months under the frozen 150 cp rule | **FAIL** (0.663 / 0.534; Toga cp scale compressed, base rates 0.0075/0.0125 leave 3-5 positives). Disclosed quantile-matched follow-up also below the rule (0.615 / 0.562). Reading: a ~2005-era engine cannot reliably label value-critical decisions at this depth; the check cannot distinguish predictor non-transfer from weak-referee label noise. Retained; an equally strong independent-lineage referee (e.g. Leela) remains open | chess_discovery_toga_referee.py, chess_discovery_toga_referee.json |
| ST-1 | Strength gradient: scripted provenance has zero collapse (pattern probability 1 at every checkpoint) | PASS (by construction, stated) | strength_gradient_battery.py, strength_gradient_battery.json |
| ST-2 | Seed-mean provenance-rarity ordering: scripted 0 < shaped < outcome-only | PASS (0 < 0.39 < 0.65 bits; open-space rarity identical at 6.28 bits) | same |
| ST-3 | Suddenness ordering: outcome-only > shaped | **FAIL, grid-resolution route** (both provenances complete acquisition inside one 3k-episode checkpoint interval; suddenness ~19 for both); disclosed 40-checkpoint/10k-episode follow-up: discovery 2x later and pre-discovery rarity higher for outcome-only, suddenness still unordered | same + strength_gradient_fine.json |
| ST-4 | Final competence comparable (final pattern probability >= 0.8 in >= 4/5 seeds, both trained provenances) | PASS (means 0.99 and 0.99) | same |

## Possibility-space construction audits (learned basins, rollout models)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| LB-1 | Machine-discovered basins (k-means on raw trajectory features, no labels): all 60 CLBF control verdicts remain rejections | PASS 60/60 | learned_basin_clbf.py, learned_basin_clbf.json |
| LB-2 | Learned systems accepted with discovered basins on >= 13/15 seeds | PASS 14/15 | same |
| LB-3 | Verdict agreement with hand basins >= 70/75; all discovered partitions identifiable (MI >= 0.1 bits) | PASS 74/75; 15/15 identifiable | same |
| IR-1 | Learned potential >= 0.5 bits under near-greedy decoding (T=0.2) on >= 13/15 seeds | PASS 15/15 (openness is not sampling noise) | independent_rollout_audit.py, independent_rollout_audit.json |
| IR-2 | Learned verdicts accepted under BOTH rollout models (T=0.2 and T=2.0) on >= 13/15 seeds | **FAIL** (8/15; diffuse T=2.0 dilutes selectivity to 0.40-0.48 and erases the value contrast on most failing seeds -- injected noise changes the behaving system, identifying the rollout policy as a declared contract item, not an estimator knob) | same |
| IR-3 | Initialization twins rejected under both rollout models on 15/15 seeds | PASS 15/15 | same |

## Consolidation round (cross-fit basins, harmful emergence, provenance, 2-D surface)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| XF-1 | Cross-fitted low-level partitions identifiable on >= 13/15 seeds per method (4 methods) | PASS (15/15, 14/15, 15/15, 15/15) | crossfit_lowlevel_basins.py, crossfit_lowlevel_basins.json |
| XF-2 | Controls rejected >= 58/60 per method | PASS (60/60 all four methods) | same |
| XF-3 | Mean verdict agreement with hand basins >= 0.85 across methods | PASS (0.957) | same |
| HE-1..4 | Learned harmful emergence: harmful-direction selectivity, structural components, value-sign split (U_private > 0 > U_team), acquisition | PASS 5/5 on all four (U_private +7.5, U_team -2.0) | learned_harmful_emergence.py, learned_harmful_emergence.json |
| MP-1..5 | Matched-behaviour provenance: behaviour matched; structural magnitude matched for learners; acquisition boundary (clone counts as acquired); provenance-rarity ordering clone < shaped < outcome-only; clone counterfactually distinguishable from script | PASS on all five (clone spec 0.38-0.45 vs script 1.0; rarity 0.33 < 0.39 < 0.65) | matched_provenance.py, matched_provenance.json |
| P2D-1 | 2-D phase surface: all 15 non-fragile cells match the pre-derived verdict by seed majority | **FAIL by one cell** (14/15; G=13, w=0.25: 2/5 seeds acquire the structure -- low context frequency makes discovery stochastic; retained) | phase_2d_prediction.py, phase_2d_prediction.json |
| P2D-2 | Acceptance region non-rectangular exactly as derived (flips along w within G=10 and along G within w=0.10) | PASS | same |
| CE-1 | Contract ensemble (declared expectation): all controls contract-invariant negative; most learned seeds >= 6/7 contracts; relative band = known borderline seeds | PASS (420/420 control rejections; 10/15 learned >= 6/7; relative band {1103,1104,1105,1107,1205}) | contract_ensemble_analysis.py, contract_ensemble_analysis.json |

## Continuous-profile construct calibration

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CAL-1 | Orthogonal (alpha, beta) grid: M responds to total reorganization, A to steepness; original frozen rule scored cross-terms by Spearman | **Original rule FAIL, retained** (cross-axis series are practically constant, range < 1e-4, so rank correlation ranks numerical jitter); disclosed corrected rule (responsiveness = range > 0.1 AND Spearman >= 0.9; orthogonality = range < 0.02) PASSES on all rows/columns (M range 0.50 over beta, < 1e-4 over alpha; A range 0.50 over alpha, 0 over beta) | profile_calibration.py, profile_calibration.json |
| CAL-2 | Dose-response: partial do-block strength lambda in {0,.25,.5,.75,1}; M(lambda) and value degradation both monotone (Spearman >= 0.9, seed means) | PASS (both 1.0 on 5/5 seeds; e.g. M = 0.00/0.10/0.36/0.80/1.00) | same |
| PROF-1..5 | Taxonomy profiles separate along declared axes: learned high-struct positive-V; twin near-zero struct; scripted struct with Q=0; harmful same-struct opposite-V under two values; clone reduced M vs script | PASS 5/5 (assembled read-only from stored outputs) | profile_existing_systems.py, profile_existing_systems.json |

## Continuous-profile ranking stability and predictive validity

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| RS-1 | Cross-contract Spearman of the learned-seed E_struct ranking >= 0.5 (mean over 10 contract pairs) | PASS (mean rho 0.76; range 0.44-0.93) | contract_ranking_stability.py, contract_ranking_stability.json |
| RS-2 | Class separation: every control's E_struct below every learned seed's, in every contract | **FAIL, retained** (the competent scripted coordinator has high structure BY DESIGN; a structure-only score cannot separate learned from scripted -- the layered definition's own claim, re-derived by the continuous record). Disclosed follow-up RS-2b: on E_adapt (acquisition included) learned 15/15 positive, controls 0 by construction | same |
| PV-1 | Early (25%-training) causal magnitude predicts final acceptance, AUROC >= 0.75 | **FAIL** (0.626; structure is already formed by 25% at boundary cells) | predictive_validity.py, predictive_validity.json |
| PV-2 | Early M beats early performance as predictor | **FAIL** (0.626 vs 0.783) | same |
| PV-3 | Spearman(early M, final usefulness) >= 0.5 | **FAIL** (0.33) | same |
| PV-obs | Disclosed observation (not registered): the early VALUE component of the same profile is the strongest early predictor (AUROC 0.808) -- near the phase boundary, prediction is carried by the value axis, consistent with the acceptance flips running through usefulness | descriptive | same |

## World-model closure (simulator-free measurement)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| WM-1 | Calibration to simulator improves monotonically with training data (seed-mean TV over K = 200..20000) | PASS (e.g. 0.64 -> 0.30 -> 0.09 -> 0.05) | world_model_closure.py, world_model_closure.json |
| WM-2 | At K = 20000 no non-abstaining model verdict disagrees with the simulator verdict | PASS (15/15 match) | same |
| WM-3 | Ensemble-disagreement margin rule catches every mismatch (no silent wrong verdict at any K) | **FAIL** (20 mismatches at K <= 1000, none caught: data-starved models share the same bias, so ensemble spread underestimates error) | same |
| WM-F | Disclosed follow-up: coverage-augmented rule (margin AND incomplete-rollout fraction < 0.10) catches all mismatches without abstaining at K = 20000 | half PASS: catches 20/20 mismatches (no silent wrong verdict anywhere); F-2 FAIL -- abstains at K = 20000 because forced interventions still hit unseen state-action pairs in 39-50% of rollouts (conservative, not wrong) | world_model_closure_followup.py, world_model_closure_followup.json |

## Collapse-bridge identity (machine-verified)

| ID | Statement | Outcome | Recorded in |
|---|---|---|---|
| TH-B | Proposition B: action-attributable contraction = pi-averaged interventional divergence = expected retrospective update = I(A;B); deterministic policies have zero action-attributable collapse | Verified: 10,000 random systems max gap < 1e-15; deterministic boundary exactly 0; trained gridworld trigger step gap < 1e-16 | verify_bridge_identity.py, bridge_identity_verification.json |

## Six-parameter generator calibration and record axioms

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| GC-1 | Diagonal dominance: each knob's matched dimension responds most (declared couplings exempted) | PASS (no violation; e.g. J[S][s]=0.99, J[M][b]=0.55, J[V][v]=1.52, J[Q_rel][q]=0.99, J[A][a]=0.13, J[R][r]=0.98) | generator_calibration.py, generator_calibration.json |
| GC-2 | Off-diagonal responses < 0.25 of column diagonal | **FAIL, retained** (two violations: Q_raw responds to its own knob q -- a specification error in the frozen exemption list, disclosed follow-up exempts matched pairs; R responds to b at 0.29 of diagonal -- persistence is retention OF structure, a structural coupling of the constructs, stated) | same |
| GC-3 | Nullity: s=0 gives S<0.05, b=0 gives M<0.05, v=0 gives \|V\|<0.05 | PASS (0.011 / 0.000 / 0.004) | same |
| GC-4 | Value separability: flipping v flips V, leaves M within 0.05 | PASS | same |
| GC-5 | Provenance separability: q=0 gives Q<0.05 at unchanged M, S | PASS (Q_raw 0.000; M and S unchanged) | same |
| A1-A8 | Continuous-record axioms (nullity, boundedness, monotonicity, data processing, context sensitivity, value/provenance separability, abstention): machine verification | All 8 pass (20,000-sample ensembles; A6/A7 pinned to GC-4/5; A8 pinned to the world-model follow-up) | verify_record_axioms.py, record_axioms_verification.json |

## Convergent validity (component -> matching endpoint)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CV-1 | rho(M_early, M_final) > rho(perf_early, M_final) on 20 fresh boundary-cell seeds | PASS (0.56 vs -0.09: structure predicts structure where performance is uninformative) | convergent_validity.py, convergent_validity.json |
| CV-2 | rho(U_early, U_final) > rho(perf_early, U_final) | **FAIL, retained** (0.385 vs 0.415: near the boundary, early performance predicts the value endpoint as well as early usefulness -- consistent with the PV lesson) | same |
| CV-3 | Record adds LOO R^2 over early performance for U_final | **FAIL, retained** (0.057 -> -0.029 at n=20; no incremental value-endpoint content at this sample size) | same |

## Overcooked-AI public-environment confirmation (externally timestamped)

Protocol, thresholds and predictions frozen in
`OVERCOOKED_PREREGISTRATION.md` and pushed to the public repository
(`github.com/FanYixiang2000/emergence-prereg`, tag
`v1.0-overcooked-prereg`, commit `8415e45`, 2026-07-18) **before any
confirmatory seed was launched**. Twelve seeds (77001-77012), two
unmodified layouts, four controls per seed.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| OC-1 | Learned accepted (all six components) on >= 8/12 seeds | PASS (exactly 8/12; every rejection routes through conditional selectivity, the correct verdict for policies potting indiscriminately in both layouts) | overcooked_confirmation.py, overcooked_aggregate.py, overcooked_confirmation_pooled.json |
| OC-2 | All 48 control verdicts are rejections | PASS (48/48; scripted roles and BC clones fail endogeneity+acquisition despite high competence; twins and untrained fail selectivity+usefulness) | same |
| OC-3 | Trigger direction matches pilot (first-potter-agent-0 higher in asymmetric_advantages) on >= 10/12 seeds | PASS (12/12) | same |
| OC-4 | Under contract B all 12 initialization twins remain rejected | PASS (12/12) | same |
| OC-5 | Learned usefulness do-contrast positive on >= 10/12 seeds, exact sign test p < 0.05 | PASS (12/12, p = 2.4e-4) | same |

## Overcooked round-1 continuous profiles (read-only, declared expectations)

| ID | Expectation | Outcome | Recorded in |
|---|---|---|---|
| OP-1 | Every accepted learned seed has E_adapt > 0 | HOLDS (8/8) | overcooked_profiles.py, overcooked_profiles.json |
| OP-2 | Every control has E_adapt = 0 | HOLDS (48/48) | same |
| OP-3 | The four selectivity-rejected learned seeds rank below every accepted seed on E_struct, without the record seeing the verdict | HOLDS (max rejected 0.653 < min accepted 0.695, no overlap) | same |

## Overcooked collective-constraint bridge (read-only; limitation retained)

This is the first real-system bridge from the new collective-constraint
certificate back onto the externally timestamped Overcooked confirmation.
It is deliberately **not** claimed as the full interaction-broken
`C,G,M;N|E,R` flagship: the stored round-1 artifacts in this checkout
contain per-seed metrics but not the learned `.pt` checkpoints or
step-by-step trajectories needed to replay policies and cut the
agent-agent channel while holding the environment fixed. That limitation
is itself registered as OCC-5 and retained.

What can be measured read-only: the joint branch is
`(context, first-potter role)`, not `{success, failure}`. The
interaction-free baseline cuts context-role dependence while keeping the
role marginal, giving `C_ctx = I(context; role)` and
`G_ctx = JSD(P_real, P_broken)`. Macro gain is the frozen Overcooked
do-block reward gap; persistence uses the already declared contract-B
success-indicator re-evaluation.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| OCC-1 | Among the 8 preregistered accepted learned seeds, >= 7 have C_ctx >= 0.05 bits and G_ctx >= 0.01 bits | PASS (8/8) | overcooked_collective_constraint.py, overcooked_collective_constraint.json |
| OCC-2 | Scripted roles and BC clones have C_ctx <= 0.01 bits in 12/12 seeds while retaining high natural score -- external coordination is not endogenous collective constraint | PASS (12/12 scripted, 12/12 clone; C_ctx=0) | same |
| OCC-3 | Learned macro gain M = natural_score - do_block_score is positive in 12/12 seeds | PASS (12/12; same sign as OC-5, now attached to the generation certificate) | same |
| OCC-4 | Accepted learned seeds retain conditional_selectivity >= 0.5 and usefulness > 0 under contract B in >= 7/8 accepted seeds | PASS (8/8) | same |
| OCC-5 | Full C,G,M;N|E,R interaction-broken certificate is NOT available from stored round-1 artifacts because learned checkpoints / trajectories are absent | LIMITATION RETAINED (full_certificate_available=false) | same |

Reading: the stored Overcooked evidence now supports the new framework's
context-conditioned joint-branch constraint in a real public benchmark
and rejects high-performing scripted/clone external coordination on the
predicted route. It does **not** yet close the strongest reviewer gap:
the complete interaction-broken replay must be run from saved or newly
trained policies.

## Overcooked transition certificate scaffold (state-level real-vs-ghost)

This is the first executable step toward the NMI-critical flagship, not the
flagship itself. `OVERCOOKED_TRANSITION_CONTRACT.md` freezes the V1
measurement contract: from the same simulator state, compare the coupled
future distribution with a ghost-partner continuation that replays another
partner action trace from the same layout/time bin. It reports `G=JS`,
signed `C`, macro score gain `M`, temporal concentration `J`, and partner
action-marginal diagnostics. No learned checkpoint exists in this checkout,
so the smoke tests below deliberately make no learned-policy claim.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| OTC-S1 | Script exports a complete contract and normalized real/cut distributions on a small Overcooked smoke run | PASS (scripted and initial smoke distributions sum to 1) | overcooked_transition_certificate.py, overcooked_transition_certificate_smoke_*.json |
| OTC-S2 | Ghost replay gives finite non-negative G and reports partner-action marginal mismatch | PASS (scripted G=0.000, TV=0.004; initial G=0.043, TV=0.079) | same |
| OTC-S3 | Output explicitly marks whether a learned checkpoint was supplied; no learned flagship claim is made without one | PASS (`learned_checkpoint_supplied=false` in both smoke outputs) | same |
| OTC-P1 | Small learned pilots are audited honestly before any flagship claim | BOUNDARY + PILOT POSITIVE (40k: G 0.009, M 0; 500k: G 0.012, M -0.625; 2M: G 0.055, C 0.216, M +14.6, partner TV 0.025). Single-seed pilot only, not a flagship claim | overcooked_transition_learned_pilot.py, overcooked_transition_pilot_audit.json |

## Crowd-vote aggregation domain (collective control; Twitch-Plays-Pokemon-inspired)

Two design pilots disclosed and quarantined
(`crowd_vote_domain_pilot1_contextblind.json`: trigger conditioned on
the context label, selectivity manufactured by the observer;
`crowd_vote_domain_pilot2_ortrigger.json`: trigger/basin definitional
mismatch). Final spec aligns the trigger with the basin's own
majority-mode event; thresholds and CR levels unchanged throughout.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CR-1 | Learned meta-controller passes all six components on >= 8/10 seeds | **FAIL, retained** (7/10; the three rejections all learn blanket democracy -- context-blind competence correctly rejected via selectivity, the same route as the four rejected Overcooked seeds) | crowd_vote_domain.py, crowd_vote_domain.json |
| CR-2 | All 50 control verdicts rejections with declared routes | PASS (50/50; always_democracy fails potential+selectivity = forced convergence; always_anarchy fails usefulness via falls -- the historical counterfactual; scripted switcher and BC clone fail exactly endogeneity+acquisition; twin fails selectivity/usefulness/acquisition) | same |
| CR-3 | Learned usefulness do-contrast positive >= 9/10 | PASS (10/10) | same |
| CR-4 | do-block raises ledge fall rate by >= 0.3 on every seed | PASS (10/10) | same |
| CR-5 | Field-context anarchy fraction >= 0.7 on every seed | **FAIL, retained** (7/10; three seeds keep partial democracy in the field where its cost is small -- the convention's context-selectivity is graded, not binary) | same |

## Convention bifurcation (mechanistic follow-up to the CR-1 / Overcooked rejections)

Anomaly followed up: in two independent domains, ~1/3 of seeds learn a
context-BLIND convention and are correctly rejected via selectivity.
This battery manipulates the democracy time cost d (5 levels x 10
fresh seeds) and measures a reference value gap (scripted selective
switcher minus scripted always-democracy, no learner involved).

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| BF-1 | Selective-seed fraction non-decreasing in d | PASS (0.1, 0.2, 0.8, 1.0, 0.9) | convention_bifurcation.py, convention_bifurcation.json |
| BF-2 | Endpoints: <= 3/10 at d=1, >= 7/10 at d=3 | PASS (1/10 and 9/10) | same |
| BF-3 | Sign of the measured reference value gap predicts the majority basin at >= 4/5 grid points | PASS (4/5; the miss is d=1.5 where the gap is +0.12, i.e. the transition region itself) | same |
| BF-4 | 25%-training snapshot predicts the final convention basin on >= 70% of seeds | **FAIL, retained** (0.50 = chance: the convention basin is decided late in training, consistent with the PV lesson that value sorting happens after structure saturates) | same |

Reading: the certificate's selectivity rejections are lawful -- blind
conventions are a competing training attractor whose basin odds follow
the value gap between conventions (Proposition 3 applied to convention
selection). This explains, without re-scoring, the retained CR-1 miss
and the four rejected Overcooked seeds.

## Meta-collapse commitment (anomaly follow-up to BF-4; Tier 3)

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| MC-1 | Median hard-signature commitment time > 25% of training | nominal PASS (0.97) but reported as a MEASUREMENT ARTIFACT: the hard-argmax signature flips on near-tied Q jitter, so "commitment" lands at the last jitter | meta_collapse_commitment.py, meta_collapse_commitment.json |
| MC-2 | Commitment distribution broad (IQR >= 0.20) | **FAIL, retained** (IQR 0.03; clock-like because the measure tracks jitter, not decision) | same |
| MC-3 | Blind seeds accumulated fewer anarchy field successes by commitment | **FAIL, retained** | same |
| MC-4 | Population convention entropy falls mostly after 25% | **FAIL, retained -- and the informative one**: population entropy NEVER falls (~1 bit throughout); the 50 seeds settle into a stable 26/22/2 split. The convention space does not collapse at the population level: a stable bifurcation, not a universal attractor. Also corrects the d=2.0 selective fraction to ~0.52 at n=50 (the 8/10 at n=10 was sampling); BF-1..3 conclusions unchanged | same |

## Meta-collapse margins (disclosed follow-up MC-F; declared, not registered)

Soft signature = Q(democracy) - Q(anarchy) margins traced every 250
episodes, 24 fresh seeds, to separate decision from jitter.

| ID | Declared expectation | Outcome | Recorded in |
|---|---|---|---|
| F-1 | Post-last-flip median \|margin\| >= 3x pre-flip (decision grows a moat) | **FAIL, retained** (0/24 at the 3x level; margins grow steadily rather than step-like) | meta_collapse_margins.py, meta_collapse_margins.json |
| F-2 | De-jittered commitment mid-training and stochastic (median in [0.2,0.8], IQR >= 0.15) | half: IQR 0.16 passes, median 0.82 just outside -- commitment is genuinely late AND stochastic | same |
| F-3 | The basin difference lives in the FIELD state, not the cliff state | PASS 24/24 (1.00): both conventions end with strong positive cliff margins (+2.4 to +3.3); selective seeds end with negative far-field margins, blind seeds positive, no overlap. The bifurcation is decided by whether the learner discovers that anarchy suffices on open ground -- not by hazard learning | same |

## Canonical exemplars (universality coverage; frozen expectations)

| ID | Expectation | Outcome | Recorded in |
|---|---|---|---|
| CE-1 | Boids: within-episode heading contraction monotone in alignment coupling | PASS (0.00 -> 1.64 bits over the 5-point grid) | canonical_exemplars.py, canonical_exemplars.json |
| CE-2 | Boids: empirical 3-bird total correlation monotone, < 0.1 at zero coupling, > 1.0 at full | PASS (0.04 -> 2.70 bits; Proposition S made empirical) | same |
| CE-3 | Boids: do-decouple JS >= 0.2 at high coupling, <= 0.05 at zero | PASS (1.000 vs 0.000 bits) | same |
| CE-4 | Schelling: segregation monotone in tolerance threshold; do-freeze load-bearing at tau = 0.7 | PASS (0.753 < 0.921 < 1.000; JS 0.331 at 0.7 and 0.000 below -- load-bearing only near tipping, an unregistered but clean observation) | same |
| CE-5 | Life: deterministic soup gives one future (H(final\|init) = 0 exactly); across-soup class entropy > 0.3 bits | PASS (50/50 replays identical; entropy 1.36 bits) | same |

Classification delivered: flocking and segregation are structural
emergence (open space, coupling-monotone contraction, counterfactually
load-bearing) that fails the adaptive layer because nothing is
acquired; Life is substrate pattern formation with zero
action-attributable collapse by the bridge identity. The framework
speaks the field's own examples, with numbers.

## Universal-observer recipe (one semantics-free possibility space, three domains)

One frozen recipe (bag-of-opaque-tokens + raw numerics, standardized,
k-means seed 0, k by silhouette over 2..8), byte-identical code across
domains. Two disclosed corrections after run 1 (denominator slip in
the frozen U-1 text: the battery has NINE systems; and run 1 compared
retrained policies against the STORED battery's hand verdicts --
quarantined as universal_observer_run1_crosspolicy.json; the corrected
run scores hand and universal verdicts side by side on the same
policies).

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| U-1 | Battery: clustered-basin verdicts agree with same-policy hand verdicts at the ~90% bar (>= 8/9), both predeclared positives accepted | PASS (8/9; both positives accepted; the one disagreement is the converged_team control, where the semantics-free partition resolves micro-variation the declared contract deems irrelevant and so inflates potential -- residual observer-dependence living exactly where Proposition 6 (refinement) says it must) | universal_observer.py, universal_observer.json |
| U-2 | Crowd domain: verdict agreement >= 16/18; all 15 controls rejected | PASS (18/18; 15/15) | same |
| U-3 | No per-domain tuning (same function objects, same hyperparameters) | PASS (asserted in code) | same |
| U-C | Third domain, stored: CLBF cross-fitted low-level basins | 95.7% agreement, 60/60 controls under every method (crossfit_lowlevel_basins.json) | cited |
| U-4a | Public Overcooked (stored round-1 policies, 4 declared seeds, reduced 20-episode budget disclosed): universal-vs-hand agreement >= 18/20 | PASS (19/20) | overcooked_universal_observer.py, overcooked_universal_observer.json |
| U-4b | All 16 control verdicts rejected under the universal recipe | PASS (16/16) | same |
| U-4c | Accepted-seed learned policies remain accepted; 77005 remains rejected | **FAIL, retained** (77007 loses specificity under the semantics-free action-histogram basins -- coarser partitions trade sensitivity for objectivity and never manufacture acceptance; 77010's HAND verdict itself flips at the reduced budget, an instrument-budget sensitivity reported as measured) | same |

## Emergence-promoting selection (the record as an optimizable objective)

Arena: crowd domain at d = 1.5, the measured transition region (value
edge +0.12, natural discovery ~20-40%). Population selection on three
criteria, same total budget, 10 replicates per arm.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| PE-1 | Selecting on E_adapt raises the final selective-convention fraction over no-selection by >= 0.25 | PASS (0.78 vs 0.39) | emergence_promoting_selection.py, emergence_promoting_selection.json |
| PE-2 | Selecting on E_adapt beats selecting on value by >= 0.15 (the record sees structure that near-silent value cannot) | PASS (0.78 vs 0.53) | same |
| PE-3 | No value sacrifice (E-arm mean value within 0.3 of no-selection) | PASS (5.29 vs 5.54) | same |

Reading: the certificate is a hard conjunction, but the continuous
record is a scalar field any black-box optimizer can climb --
demonstrated: emergence can be PROMOTED by optimizing the measured
record, at no value cost, in the region where value alone is nearly
silent. This is the measured answer to "can the community optimize
against this standard."

## Emergence coordinates (analytic truths, blind families, type lattice)

The construct-validity battery: per-dimension ANALYTIC ground truths
(not generator parameters), thresholds frozen on calibration families
(Boolean gates, exact Markov chains, constructed processes), then
blind application to held-out families the calibration never saw.
Three runs of disclosed specification errors, every one caught by the
battery's own exact computations and retained: run 1 (majority-3
truth constant was an arithmetic error -- the ESTIMATOR was right and
the hand-written truth wrong; the designed A-positive chain was
exactly A = 0 by enumeration; the Kuramoto breaker was inert), run 2
(the unit-process breaker violated its own declared surrogate
definition), run 3 (static-pattern R truth misassigned; R conflated
recovery with the noise floor -- redefined as a recovery ratio).
Quarantined: emergence_coordinates_run{1,2,3,3b}*.json.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| EC-1 | Calibration recovery: N estimated within 0.1 bits of analytic values on five gates (XOR/parity exact 1.000, majority 0.430 vs 0.434); exact A signs on three chains (+0.189/0/-1.0); D and R within 0.15 of construction truths | PASS (final run; runs 1-3 misses retained above) | emergence_coordinates.py, emergence_coordinates.json |
| EC-2 | Blind lattice classification under frozen thresholds: Kuramoto supercritical = weak emergence, subcritical = reject, Life glider = weak-not-adaptive (Chalmers' canonical case), stored learned convention = adaptive | PASS 4/4 | same |
| EC-3 | Adversarial pseudo-emergence matrix: 8 rows each rejected (or accepted-with-negative-value for harmful congestion) on the PREDICTED dimension -- common driver fails D, central controller and redundant copying fail N, random/transient patterns fail R, static pattern fails D, thresholded smooth ability flagged as metric artifact | PASS 8/8 | same |

Type lattice reported (no weighted total): weak = min(N, D, R);
causal = weak AND A > 0; adaptive = weak AND acquired; functional =
adaptive with signed V. Philosophical strong emergence is declared
outside the empirical framework -- a different kind of claim, not a
higher score.

## Ant double-bridge: two reviewer questions, answered with numbers

Two questions a reader raised about swarms, addressed by a controlled
Deneubourg double-bridge contrast (two equal routes around a central
obstacle; identical individual rules except the pheromone channel).
Q1: "an ant bridge is built bit by bit -- where is the possibility
collapse, must it be abrupt?" Q2: "is an ant finding food around an
obstacle not emergence?" The N/D/R thresholds are COPIED from the
coordinates battery (0.3/0.5/0.6) and the weak-emergence verdict uses
the same weak(D,R) rule applied to the held-out families; N is
redundancy-dominated in many-body swarms and reported descriptively.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| ANT-1 | SOLO (individual navigation around the obstacle): colony consolidation rate < 0.3 and D < 0.5 -- individual obstacle-finding is NOT collective weak emergence under the frozen coordinates (it is individual adaptation; the coordinates separate the two). Answers Q2 | PASS (rate 0.00, D 0.00) | ant_contrast.py, ant_contrast.json |
| ANT-2 | TRAIL (stigmergy): D >= 0.5 and R >= 0.6 -- collective trail formation IS weak emergence, reproducing the double-bridge literature | PASS (D 0.94, R 0.97; N -0.15 redundancy-dominated, reported) | same |
| ANT-3 | Gradual, non-abrupt collapse: TRAIL colony commitment dev reaches >= 0.5 and the 10-90 collapse is spread over >= 10% of the foraging horizon (an abrupt step would be span ~ 0, one trip carrying ~all of it); SOLO never commits (dev < 0.3). Answers Q1: the collapse is in the DISTRIBUTION over which route the colony commits to, and it contracts progressively as pheromone accumulates -- abruptness is not required | PASS (TRAIL commitment 0.99, span 25% of trips, max single-trip step 0.15 of total vs 1.0 for a jump; SOLO commitment 0.00) | same |

## Irreducibility: surviving five frontier objections

Five 2025-2026 papers (Hoel Causal Emergence 2.0; PITHON temporal
higher-order from pairwise; Environment-Driven Emergence no-go;
Cognitive Agent Networks; Krakauer/Krakauer/Mitchell on LLM emergence)
rule out the naive collapse story. The upgrade: score only the
ENDOGENOUS, IRREDUCIBLE, macro-causal part of the contraction, graded
by REDUCIBILITY not magnitude. C_irr = KL(P || pairwise max-ent);
C_irr|E conditions on the environment; D_higher is the do-operator on
agent coupling. Thresholds C_irr strong >= 0.3 bits, reducible < 0.05,
D >= 0.5. Exact on discrete systems, Gaussian analytic anchor.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| IR-1 | The no-go is real, both directions: (a) Gaussian common cause -> Omega > 0 redundancy (closed form); (b) a deterministic common cause X1=e1, X2=e2, X3=e1 XOR e2 is DISTRIBUTIONALLY IDENTICAL to a genuine role-lock -- C_total 1 bit, Omega -1 synergy, C_irr 1 bit -- with no interaction. Higher-order statistics cannot separate interaction from common cause | PASS (Gaussian Omega +0.30 bits; common-cause synergy Omega -1.00, C_irr 1.00) | emergence_irreducibility.py, emergence_irreducibility.json |
| IR-2 | The framework defeats the no-go: for that common-cause system C_irr\|E < 0.05 bits AND D_higher < 0.5 -> rejected, despite a signature identical to the role-lock and large contraction. Same distribution, opposite verdict | PASS (C_irr\|E 0.00, D 0.00) | same |
| IR-3 | PITHON reducibility: a pairwise Markov-chain system has C_irr < 0.05 bits -> higher-order-looking structure fully reducible to pairwise; weak, not irreducible | PASS (C_irr 0.00) | same |
| IR-4 | Irreducible role-lock (positive): a genuine 3-way parity constraint has C_irr\|E >= 0.3 AND D_higher >= 0.5 -> strong emergence, irreducible beyond environment and all pairwise. macro_gain reported but NOT gated (it is 0 here -- a degenerate readout -- confirming co-information is neither necessary nor sufficient) | PASS (C_irr\|E 1.00, D 1.00; macro_gain 0.00 reported) | same |
| IR-5 | Magnitude != strength: across the five systems Spearman(C_total, C_irr\|E) <= 0; the maximum-contraction system (redundant consensus, C_total 2 bits) has C_irr\|E ~ 0 while the strong case has SMALLER contraction (1 bit) | PASS (max-C_total = consensus, C_irr\|E 0.00; role-lock C_total 1 < 2) | same |
| IR-6 | Functional vs pathological at MATCHED magnitude: a consensus tuned to the role-lock's contraction (within 0.1 bit) has value gain <= 0 and C_irr\|E ~ 0, while the role-lock has value gain > 0 and C_irr\|E >= 0.3 -- double dissociation at equal collapse | PASS (matched C_total 0.998 vs 1.000; consensus V 0.00 / C_irr\|E 0.00 vs role V +0.50 / C_irr\|E 1.00) | same |

The upgraded definition: emergence = endogenous, selective contraction
of the counterfactual reachable joint-future set, irreducible to
environment + independent adaptation + all lower-order interactions,
carrying macro causal load. Strong vs weak is graded by reducibility
(C_irr\|E > 0 and D load-bearing), not by contraction magnitude.

## Collective-constraint certificate: same outcome, different mechanism

Answers the sharpest framing objection: possibility must be defined on
the JOINT-ACTION branch set, not on {success/failure} (every success
would "collapse") nor on single-agent choices. The load-bearing test is
the interaction-broken counterfactual P_broken (cut agent-agent coupling
while keeping marginals, environment and any controller). Two-part
certificate: GENERATION (C = H(P_broken)-H(P_real); G = JSD(P_real,
P_broken); M = P(Z\|real)-P(Z\|broken)) and PRODUCT (N = C_irr\|E;
R = persistence). Macro structure Z: a1+a2+a3 = 0 (mod 3), order-3
irreducible. Thresholds C >= 0.5 bits, G >= 0.05, N >= 0.3, R >= 0.6.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CC-1 | Matched confound: central-script, common-cause and local-feedback have IDENTICAL joint-action distributions, IDENTICAL single-agent marginals, and P(Z)=1 -- no joint statistic (nor the outcome-collapse of Z) separates them | PASS (joint identical, marginals uniform 1/3, P(Z)=1 all three) | collective_constraint.py, collective_constraint.json |
| CC-2 | Endogeneity separates same-outcome mechanisms: under the interaction-broken (keep-external) counterfactual, C >= 0.5 AND G >= 0.05 AND M > 0 ONLY for local feedback; central-script and common-cause give C = G = M = 0 | PASS (local feedback C 1.585, G 0.459, M +0.667; both external mechanisms 0/0/0) | same |
| CC-3 | Micro-down / macro-up: for local feedback H(P_real) < H(P_broken) (joint freedom pruned) yet endogenous macro gain M > 0 -- emergence is not a total-entropy decrease | PASS (H_real 3.170 < H_broken 4.755; M +0.667) | same |
| CC-4 | Persistence rejects coincidence: independent coincidence has R < 0.6 (transient) while local feedback has R >= 0.6 | PASS (independent R 0.30; local feedback R 1.00) | same |
| CC-5 | Four-quadrant certificate accepts ONLY local feedback, and each imposter fails on the PREDICTED component (central-script & common-cause on endogeneity C/M/N\|E; independent on persistence R and constraint C) | PASS (only local feedback accepted; failure routes exactly as predicted) | same |

Definition, in its tested form: emergence = endogenous formation of
collective constraints that reorganize independently-composable
joint-action possibilities into a stable, causally efficacious
macro-organization. Possibility collapse (C) and reorganization (G)
certify genesis; irreducibility (N\|E), persistence (R) and macro
autonomy (A) certify the product; usefulness V is a signed annotation,
not a gate; abruptness is separate and non-necessary; philosophical
strong emergence stays outside the empirical scale. This is the
concrete answer to "same bridge, four mechanisms -- which is emergence?"

## Canonical possibility-collapse validation matrix

Unification layer, not another private generator: already-frozen outputs
from classical exemplars, the coordinate battery, ant bridge,
Overcooked, grokking and induction-head runs are projected into one
schema. Each row records (i) the domain, (ii) the relevant possibility
space, (iii) the expected status, (iv) the evidence route or predicted
failure dimension, and (v) the provenance tier. Analytic truth,
constructed mechanism truth, canonical convergent validity, and external
empirical evidence are not conflated.

Core interpretive rule: macro structure is the product/projection; the
candidate universal mechanism is endogenous collapse or reorganization
and stabilization of a reachable possibility space. Thus Boids uses
heading futures; Kuramoto uses relative phase futures; Schelling uses
spatial configuration futures; ant bridges use route futures;
Overcooked uses context-conditioned role futures; grokking and induction
heads use output/computation futures. Success/failure alone is never the
possibility space.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| CPC-1 | Publicly canonical positive cases show the expected possibility-collapse-compatible profile under stored tests: Kuramoto supercritical, Boids high coupling, Schelling high tau, Game-of-Life glider, and ant trail | PASS 5/5 (convergent validity, not analytic truth) | canonical_possibility_collapse.py, canonical_possibility_collapse.json |
| CPC-2 | Canonical/constructed negatives fail on the predicted reason: Kuramoto subcritical, ant solo, common driver, central controller, metric-artifact jump, scripted/BC Overcooked controls | PASS >= 6/6 (all listed rows match) | same |
| CPC-3 | Capability-emergence cases fit the same schema on computation/output futures: grokking, transformer grokking and induction heads pass; memorizer/no-structure/1-layer controls fail on usefulness, burstiness or architectural possibility | PASS >= 6/6 (all capability rows match) | same |
| CPC-4 | The real public Overcooked bridge supports role-future collapse for accepted learned seeds (8/8) and rejects external high-score scripted/clone controls (24/24), while retaining the limitation that full agent-channel replay is unavailable | PASS with limitation retained | same |
| CPC-5 | The benchmark exports a complete ground-truth matrix: every row has domain, possibility_space, expected_status, evidence_route, and either acceptance coordinates or predicted failure dimension | PASS (19/19 rows complete and matched) | same |
| CPC-6 | Evidence provenance is explicit: canonical-consensus/external rows are not mislabeled analytic ground truth; Life is marked as an observer-contract boundary | PASS (all rows tiered; Life exact-state C=0 versus perturbation-ensemble D/R stated) | same |

Reading: this battery is the direct answer to "prove possibility
collapse is really emergence using public/classic examples." It does
not claim that literature agreement is ground truth or that every row
has the same estimator; it standardizes the ontology of the possibility
space and failure route. The remaining frontier is a single frozen
estimator across families and full intervention replays where
checkpoints/trajectories are available.

## Live blind-accuracy demonstration (reviewer walkthrough)

24 knob vectors drawn uniformly at random (off the calibration grid),
hidden from the instrument, measured through the standard
finite-sample pipeline, then unsealed.

| ID | Prediction | Outcome | Recorded in |
|---|---|---|---|
| LD-1 | Every matched dimension recovers its hidden truth at Spearman >= 0.9 | PASS (s->S 0.995, b->M 0.999, v->V 0.999, q->Q 0.976, r->R 0.959) | live_demonstration.py, live_demonstration.json |
| LD-2 | Raw cross-correlations <= 0.35 for undeclared pairs | **FAIL, retained** -- and diagnosed in a disclosed follow-up: the excess is (a) the two declared structural couplings and (b) sampling correlation between the random draws themselves (\|rho\| ~ 0.35 at n=24 between SET knobs). Partial correlations controlling for the matched knob leave worst undeclared leakage 0.39, BELOW the permutation null median 0.43 (p = 0.65): undeclared leakage is statistically indistinguishable from the n=24 noise floor | same |
| LD-3 | Live-trained fresh policy and four known-identity controls receive the correct verdicts with the correct failure routes | PASS (learned landed in the selective basin and is the only acceptance; switcher/clone fail exactly endogeneity+acquisition; twin fails selectivity/usefulness/acquisition; blanket democracy fails potential) | same |

## Theory (machine-verified, not predictions)

| ID | Statement | Outcome | Recorded in |
|---|---|---|---|
| TH-0..8 | Props 0--5 plus observer-refinement, rollout-model error and conjunction-margin guarantees; bounded burst lemma | All algebraically proved; 20,000 random-distribution checks and 27 measured bounded-burst checks pass | verify_theory_bounds.py, verify_observer_bounds.py, exact_prior_formalisms.py, THEORY.md |
| TH-S | Proposition S: spatial collapse = total correlation (Watanabe/McGill); independence gives exactly linear N-growth of the open space; coupling monotone; scripts attain the maximum (provenance blindness derived, not asserted) | S-A..S-D all pass (20,000 random joint laws, exact enumerated families) | verify_spatial_bridge.py, spatial_bridge_verification.json, THEORY.md |

## Tally

- Frozen predictions and checks indexed above: **~259 individual
  verdicts/checks** (counting each battery system, seed replication,
  grid cell group, and lettered prediction once).
- **Registered misses: 46**, all kept, none re-thresholded:
  GW-6 (G=7 cell), GW-7 (agreement check), E-4 (2/5 seeds), SC-1,
  MB-R3, MB-R5, MB-T2, MB-T3, MB-T6, CH-C5, DM-D2, LBF-DET-1,
  PY-T2 (route only; the rejection verdict itself was correct),
  SC-S1 (2.8B burstiness), SC-S5 (head half), SC-S7 (thinning
  agreement), LC-3 (route), PS-1, PS-4, the reframing-revision misses
  DO-1/DO-3 (counted as one contract miss with its agreement
  corollary) and RO-1, the story-alignment misses ST-3
  (suddenness ordering, grid-resolution route) and TG-1 (weak
  different-lineage referee), IR-2 (diffuse rollout model dilutes
  the behaving system), and P2D-1 (one 2-D cell where low context
  frequency makes discovery stochastic), WM-3 (ensemble disagreement
  is blind to shared bias in data-starved world models), CAL-1's
  original rank-based cross-term rule (rank of jitter on constant
  series), RS-2 (structure-only score cannot separate scripted from
  learned -- the layering's own claim), PV-1..3 (early causal
  magnitude does not predict boundary-cell acceptance; the value axis
  does, reported descriptively), GC-2 (frozen exemption list omitted
  the matched (Q_raw, q) pair; the R<-b coupling is a stated
  structural coupling), and CV-2/CV-3 (near the phase boundary early
  performance predicts the value endpoint as well as early usefulness;
  no incremental LOO R^2 at n=20 -- the record's predictive content is
  axis-specific, not a universal performance improvement), and
  CR-1/CR-5 (crowd domain: three seeds learn blanket democracy and are
  correctly rejected as context-blind competence; the convention's
  selectivity is graded), BF-4 (the convention basin is decided
  late in training; population-level forecastable, seed-level not at
  the 25% snapshot), MC-2/3/4 (the hard-signature commitment measure
  tracks Q jitter, and the population convention space never
  collapses -- a stable bifurcation), and MC-F-1 (margins grow
  steadily; no step-like moat), U-4c (semantics-free basins trade
  sensitivity for objectivity, always conservatively), LD-2 (the raw
  cross-leakage rule confounds draw-sampling correlation; undeclared
  partial leakage sits below the permutation null), and the EC
  run-1..3 specification errors (majority-truth arithmetic, A-chain
  equality, two interaction-breaker bugs, R-truth misassignment --
  every one caught by the battery's own exact computations before any
  blind system was scored).
- The CLBF protocol's registered target was at least 9/10 learned passes
  and was met exactly. The seed-1104 component miss remains visible in
  outputs, figures and text; it is not counted as a failed registered
  group prediction.
- Two misses carry the paper's main methodological finding (CH-C5 +
  DM-D2: under a converged/endogenous behaving policy, P_t(win) is a
  martingale; useful collapse must be read from do-contrasts). Three
  misses are informative about metric/grid dependence (MB-T2/T3/T6,
  confirming the Schaeffer-side point from the mechanism side). One
  (LBF-DET-1) was a control-set composition gap, fixed with a frozen
  round-2 prediction that then passed.
- No registered miss was converted into a success or removed from the
  ledger. When a miss motivated a criterion or estimator refinement, the
  original miss remains counted here; the revised rule contributes only
  through later re-frozen tests on fresh seeds, public checkpoints, or new
  task families. Pilot-stage estimator changes are documented in script
  docstrings with logs in `outputs/` (see REPRODUCIBILITY.md).
