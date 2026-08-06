# NMI readiness audit

Date: 2026-07-13

## Bottom line

The paper now has a defensible methodological core, but it is not possible to
guarantee acceptance at *Nature Machine Intelligence*. The strongest supported
claim is a scoped, intervention-aware framework with complementary empirical
instruments. The evidence does not support saying that one universal definition
has already been fully validated in every domain.

Current heuristic readiness: **7.5--8/10 for NMI**. This is an expert-risk
estimate, not a calibrated acceptance model. A direct submission would still
face a high desk-reject/reviewer-reject risk, but the previous single-domain
six-component bottleneck is now materially reduced.

## Fatal or major issues corrected in this audit

1. **Conditioning was conflated with causation.**
   `C(m)=KL(P(tau|M=m)||P(tau))` is specific information for a declared
   macro-readout. It does not make `M` causal. The manuscript now reserves
   causal claims for do-contrasts.
2. **Rarity was incorrectly used to derive endogeneity.**
   A designer can hardwire behavior that is rare because its context is rare.
   Endogeneity and acquisition now remain separate audited components.
3. **The data-processing claim was too broad.**
   Coarse-grained KL/JS lower-bound trajectory KL/JS; entropy differences do
   not inherit that bound. Potential and entropy-drop summaries are now
   explicitly partition-scale dependent.
4. **Pinsker's inequality was interpreted in the wrong direction.**
   Collapse upper-bounds a bounded representation jump; an observed jump
   lower-bounds required distribution movement.
5. **A dependent product of per-run p values was invalid.**
   The `1.3e-16` omnibus claim was removed. Only per-run empirical window ranks
   are reported.
6. **MARL evaluation episodes were treated too much like independent training
   replications.**
   Episode-level intervals and p values are now explicitly conditional on
   three trained policies; policy-seed medians are reported separately.
7. **The internal battery was mislabeled as a full six-component
   confirmation.**
   It scores five informative components because acquisition is undefined for
   its forced constructions. Full six-component confirmation is now carried by
   Contextual LBF: 9/10 registered fresh seeds pass, and a post-confirmation
   five-seed extension gives 4/5 learned full passes with all controls rejected.
8. **The phase-boundary claim overemphasized the primary run.**
   The paper now reports 11/12 scored replication points and one false
   acceptance at `G=7`.
9. **Internal project protocols were called third-party preregistrations.**
   They are now described as author-maintained prospectively frozen records
   without an external timestamp.
10. **Published-form rival audits were described as globally exact.**
    The equations are computed without Monte Carlo error, but only within
    declared candidate coarse-graining/feature families.
11. **Public-checkpoint probes were mislabeled as held-out tests.**
    MultiBERTs and Pythia score one fixed templated evaluation battery and
    store the same value in compatibility columns named `train_acc` and
    `test_acc`; the manuscript now states that no probe train/test split exists.
12. **The chess 99.2% ranking diagnostic conflicted with the martingale
    lesson.**
    Its preferred score subtracts a common within-position baseline, which
    cannot change move ranking. The diagnostic is retained for audit but
    removed from main text, figures, tables and cover-letter claims.
13. **Burst-alignment ranks were vulnerable to multiplicity and grid
    resolution.**
    A post-hoc bounded/window/thinning audit now reports both strengths and
    failures. The bounded transform preserves 27/27 full-grid verdicts, but
    Holm-adjusted alignment values are all at least 0.252 and thinning
    agreement is 93.8%, with misses concentrated in Pythia tail controls.
    Alignment is therefore descriptive, not a family-wise significance claim.
14. **Seed-aware MARL uncertainty was previously only verbal.**
    Two-stage seed--episode bootstraps are now implemented. LBF remains
    positive (mean 95% interval [0.072, 0.153]); simple_spread's mean interval
    crosses zero ([-0.005, 0.187]). With three positive seeds the exact
    one-sided seed sign test is p=0.125 in both domains.

## New load-bearing evidence completed

1. **A second end-to-end six-component domain.**
   - Contextual Level-Based Foraging retains the recognized benchmark's
     dynamics, observations, actions and sparse reward while balancing two
     unlabeled finite-horizon geometry contexts.
   - The protocol and ten fresh seeds were frozen after two disclosed,
     excluded design pilots.
   - Learned policies pass all six components on 9/10 seeds; the retained miss
     is selectivity 0.4875 versus the frozen 0.5 threshold.
   - All 40 initialization/scripted controls reject. The competent
     team-nearest controller passes all behavioral components and fails exactly
     endogeneity and acquisition on 10/10 seeds.
   - Seed-bootstrap lower 95% bounds are positive for usefulness (0.053) and
     acquisition (0.634). All six registered predictions pass.
2. **Post-confirmation Contextual LBF extension.**
   - Five additional fresh training seeds were run without changing thresholds
     or evaluation logic.
   - Learned policies pass the full rule on 4/5 seeds; all 20 controls reject.
   - Every learned seed has positive usefulness and acquisition; seed-bootstrap
     lower bounds over the five extension seeds remain positive for usefulness
     (0.049) and acquisition (0.490).
   - This is supportive robustness evidence, not a new preregistered
     significance claim.
3. **Single-signal audit on the full six-component domain.**
   - On the registered CLBF confirmation, behavior-only single signals reach at
     most 0.86 accuracy under hindsight-optimal thresholds.
   - On the five-seed extension, behavior-only single signals reach at most
     0.88 accuracy.
   - Acquisition alone can reproduce the full verdict in these runs, but it is
     a definition-internal learning/provenance component using the same-seed
     initialization twin, not a traditional prior detector.
4. **Seed-powered deep-MARL replication (previous "must do" item closed).**
   - A post-hoc seed extension trained three further simple_spread and five
     further LBF policies with unchanged code and probes; every new seed has
     a positive mean do-contrast.
   - Combined seed-level inference over 6 and 8 independent policies:
     exact one-sided sign tests p=0.016 (simple_spread) and p=0.004 (LBF);
     cluster-bootstrap mean intervals [0.032, 0.156] and [0.159, 0.390],
     both excluding zero (`hierarchical_marl_analysis_combined.json`).
   - The simple_spread win-shift reading (D2) failed again on the extension
     while the do-contrast reading stayed positive on 3/3 new seeds --
     replicating the martingale diagnosis on fresh data.
5. **Contextual LBF threshold-sensitivity rescoring.**
   - One-at-a-time sweeps around the frozen cutoffs: potential stable over
     0.1--1.0 bits, specificity over 0.05--0.7 bits, acquisition over
     0.1--0.4; all controls rejected at every grid point of every sweep.
   - Honestly reported binding component: conditional selectivity. At 0.45
     the registered miss (seed 1104) would pass (10/10); at 0.55 three more
     seeds would fail. This is stated in the manuscript rather than hidden.
6. **Manuscript-number consistency audit.**
   - `verify_manuscript_numbers.py` cross-checks 26 headline numbers
     (CLBF confirmation/extension, Pythia 160m/1B, deep MARL, chess) against
     the stored JSON outputs: 26/26 pass.
7. **Held-out Pythia scaling protocol completed (2026-07-16).**
   - 1B and 1.4B: agreement accepted with the same anchored window (step
     1000), burstiness 11.9/7.3, usefulness 0.47/0.43; all controls rejected.
   - 2.8B: registered S1 FAILURE kept -- window again step 1000 and
     usefulness passes (0.42, final accuracy 0.95), but burstiness 3.2 < 5:
     collapse is spread over several early intervals at the published grid.
   - S5 head-facts half FAILED (only 160m accepts; 1B/1.4B/2.8B reject via
     burstiness at final accuracy 1.0); both frequency-tail families are
     rejected at every scale. S2/S3/S4/S6 pass.
   - Totals across 160m--2.8B: 4/5 agreement acceptances, 10/10 control
     rejections, 8/8 tail rejections. Outcomes recorded in
     `PYTHIA_SCALING_PREREGISTRATION.md`; consistency audit now 34/34.
   - The hash audit detected two upstream repository defects at 2.8B
     (step64000 duplicates step143000 in both weight formats -- excluded;
     step32000's single-file safetensors is a stale copy of the final
     weights -- rebuilt from the authentic per-revision pytorch_model.bin and
     tensor-identity verified; step96000/step128000 bin-cross-checked clean).
     This finding itself strengthens the paper's reproducibility contribution.
8. **Pythia scaling integrity fix.**
   - The old 2.8B outputs are quarantined as mirror-invalid because early
     checkpoints reused a single-file weight object.
   - The downloader now uses official Hugging Face revisions and prefers
     revision-specific safetensors shards.
   - A 2.8B smoke check confirms real checkpoint separation: step0 has
     initialization-like agreement (about 0.50) and step1000 rises to about
     0.77 with large collapse from step0.
   - Official 1.4B/2.8B collapse and tail scaling runs are in progress; no
     manuscript claim should cite them until the queue finishes and the scaling
     summary passes integrity checks.

## New load-bearing evidence completed (2026-07-16, discovery pass)

1. **Prospective discovery on uncurated chess positions (the former
   top-priority open item).** Protocol frozen before any data was
   downloaded (`CHESS_DISCOVERY_PREREGISTRATION.md`): 400 positions sampled
   with a frozen RNG from an uncurated public monthly dump (population
   filters only; no theme tags exist in the source), scored with the
   unchanged shallow observer, labelled by an independent depth-18 referee
   never fed back into scores. 4/4 registered predictions pass: AUROC 0.730
   (>= 0.70), frozen-flag precision 0.24 = 2.5x base rate 0.095 (recall
   0.47), beats tactical/material baselines, flagged median potential 1.92
   bits. The same-family deep-eval-gap baseline (AUROC 0.762) is reported
   head-to-head and honestly framed. The framework now demonstrates
   discovery, not only recovery.
2. **Leave-one-component-out audit** across all 75 Contextual LBF systems:
   only conditional selectivity is non-redundant there (admits the two
   borderline learned seeds, no controls); every control fails >= 2
   components. Stated in the manuscript as the three-layer structure
   (definition / observer contract / identification protocol) instead of
   implying six separate necessity proofs.
3. **Non-triviality bridge and resolution theory in THEORY.md**: measured
   exclusions for ordinary decisions (potential), gradual learning
   (burstiness), exogenous injection (endogeneity/acquisition) and useless
   contraction (do-contrast); the 2.8B S1/S7 registered failures upgraded
   into a measured grid-dependence statement.
4. Consistency audit extended to 40/40 checks.

## New load-bearing evidence completed (2026-07-16, reviewer-hardening pass)

1. **Discovery replication across years**: 2016-03 frozen addendum, 4/4
   predictions pass again; the engine-native baseline is unstable across
   months (0.762 -> 0.652) while the collapse do-gap is stable
   (0.730 -> 0.725) -- the mechanism-level score generalizes, the
   same-family baseline does not.
2. **Referee sensitivity**: AUROC monotone-increasing with label strictness;
   the frozen 150 cp outcome is interior, not a selected peak.
3. **Exact trajectory-basin coupling**: closes the "ontology is decoration"
   objection -- controls show huge path-space contrasts with ~0 basin
   retention; the emergent system retains 10x more; DPI/rarity exact.
4. Consistency audit 42/42.

## New load-bearing evidence completed (2026-07-16, external-validity pass)

1. **Third full six-component domain in the SEQUENCE modality** (the main
   condition previously attached to reaching 8.5+): next-token-trained
   transformer, latent contexts never labelled, thresholds copied unchanged.
   10/10 fresh seeds pass, 40/40 controls rejected, one informative
   registered failure (the oracle router is intervention-inert in sequence
   space and fails four components instead of the predicted provenance
   pair). The full criterion now spans embodied RL and sequence generation.
2. **Cross-engine-family discovery referee**: classical (pre-NNUE)
   Stockfish 11 labels preserve the discovery signal on both months
   (AUROC 0.743/0.669) -- the same-engine circularity objection is closed.
3. **Adversarial observer audit**: random basin partitions cannot hide the
   do-contrast (80% of random observers still pass specificity) nor rescue
   any control (four components are partition-independent); the declared
   partition is the most informative of all 1000 tested (100th percentile).
4. **Ordinary-learner boundary probe**: the process proxy accepts a strong
   smooth learner 6/6 -- disclosed as a measured scope statement (the proxy
   is an acquisition-shape instrument; emergence verdicts require the
   episode-level components), pre-empting the "your proxy calls any fast
   learner emergent" attack by measuring it ourselves.
5. Ledger retains all misses; consistency audit is now 109/109 after the
   separately frozen capability-novelty, burst-boundary, Overcooked
   transition-scaffold and learned-pilot boundary tests; manuscript 36 pages
   at this audit stage.

## Reframing revision (2026-07-17, external-review triage pass)

Driven by a strong external mock review. Accepted and executed:

1. **Theory reframed as substrate + layered criteria.** Eq. 1 is now
   explicitly positioned as the specific-information / Bayesian-surprise
   substrate (with the missing literature: Itti-Baldi, DeWeese-Meister,
   Kolchinsky-Wolpert semantic information, empowerment, causal entropic
   forces). The definition is split into structural collapse (potential,
   selectivity, specificity) and adaptive qualification (usefulness,
   endogeneity, acquisition); the headline claim is narrowed to an
   intervention-aware identification framework for adaptive emergence
   acquired during learning. Harmful/prewired collapse are representable
   categories, not denied ones.
2. **Fair multivariate baselines** (`fair_baseline_comparison.py`):
   prior signals given equal/greater degrees of freedom; frozen transfer
   to the fresh battery caps at 0.9 and always misses the true positive;
   six-component reference 10/10 (stored). Closes the "one scalar vs six
   thresholds" asymmetry objection with data.
3. **Dual plausible observer contracts** (`dual_observer_contracts.py`):
   60/60 controls rejected under both contracts, structural layer agrees
   14/15, value layer conservatively flips 5 learned seeds -- recorded as
   registered misses DO-1/DO-3 and reported as measured evidence for the
   layered definition's predicted asymmetry.
4. **Engine-free realized-outcome referee** (`chess_realized_outcome.py`):
   directionally consistent, registered interaction null (RO-1 miss,
   power-limited). Honest scope sentence in the manuscript.
5. **Figure hierarchy aligned with claim hierarchy**: main figures are now
   concept, battery+fair baselines, three-domain full-criterion
   confirmation, chess discovery (uncurated months + referee families),
   and MARL seed-level + phase boundary. Mechanism, public checkpoints,
   episode-level MARL detail and the chess grid moved to Extended Data.
   Main display items: 5 figures + 1 table.
6. **Writing**: main text compressed from ~6,700-word equivalent to
   ~3,900 words before Methods; rebuttal-style phrases removed;
   Discussion subheadings flattened; "registered" defined once as
   internally frozen; "exact audit" scoped to declared candidate
   families; Prop./ED cross-reference errors fixed; tables widened to
   stop hyphenation breaks.
7. Ledger now ~161 predictions with 21 retained misses; consistency audit
   57/57; manuscript 33 pages.

Explicitly rejected from the review: dropping usefulness from the
criterion (kept, but layered), and claiming third-party registration we
do not have (stated as future hardening instead).

## Story-alignment pass (2026-07-17 evening)

Driven by the author's narrative (strong emergence = sudden stabilizing
collapse of a non-prespecified macro-structure in an open possibility
space, with a strength gradient prescribed < shaped < discovered):

1. **Emergence-strength gradient** (`strength_gradient_battery.py` +
   `strength_gradient_fine.py`): the rarity identity's graded prediction
   tested with one macro-pattern under three provenances. ST-1/2/4 pass
   (seed-mean provenance rarity 0 < 0.39 < 0.65 bits at matched
   competence; open-space rarity identical 6.28 bits); ST-3 suddenness
   ordering FAILED and is retained (both acquisitions step-like at the
   measured grids; fine-grid follow-up shows discovery ~2x later and
   pre-discovery rarity higher for outcome-only). This turns the
   non-emergent / weak / strong folk gradient into measured numbers and
   answers the "substrate must produce a new prediction" objection.
2. **Different-lineage engine referee** (Toga II / Fruit): frozen 150 cp
   rule fails through referee-scale mismatch (TG-1 retained miss);
   quantile-matched follow-up reported separately.
3. **Mover-cluster inference** for the discovery AUROC: CIs
   [0.615, 0.829] / [0.613, 0.834], excluding chance.
4. **Figure 1 redesigned**: schematic four-regime row plus a measured
   six-component walkthrough (CLBF seed 1101) with the layered checklist.
5. Methods: terminology + declared system boundary; seed-extension power
   planning disclosed. Consistency audit 62/62; ED figures now 13.

The story's "possibility space" question is answered by the observer
contract plus its audits (dual plausible contracts, adversarial
partitions, refinement theorem); the story's retrospective phrasing
(past trajectories re-constrained by the macro-pattern) is the
bookkeeping identity, with all measurements prospective.

## Consolidation round (2026-07-18, second external-review triage)

Executed from the round-3 mock review (accepted items only):

1. **Cross-fitted low-level basin discovery** (XF-1..3 all pass): the
   partition is learned from per-step coordinates/actions/rewards only,
   fitted on a train half, k by silhouette, four clustering methods;
   mean verdict agreement with hand basins 0.957, controls 60/60 under
   every method. Together with the summary-feature audit this closes the
   "possibility space must be hand-crafted" attack to the extent a
   single domain can.
2. **Learned harmful emergence** (HE-1..4, 5/5 seeds): a policy LEARNS a
   context-selective structure that is +7.5 under its beneficiary's
   value and -2.0 under the team value -- harmful emergence is now a
   measured category, not a forced construction; value dependence is
   expressive content.
3. **Matched-behaviour provenance** (MP-1..5 pass, after a disclosed
   pilot): script / BC-clone / shaped / outcome-only with matched natural
   behaviour; the do-contrast separates the clone from its source
   (0.38-0.45 vs 1.0 bits); acquisition separates static from trained;
   provenance rarity orders the trained routes. Endogeneity/acquisition
   read the system, not the experiment log.
4. **Prospective 2-D phase surface**: non-rectangular acceptance region
   over (goal reward, context weight) derived before training; 14/15
   non-fragile cells match by seed majority; P2D-1 one-cell miss
   retained (low context frequency makes discovery stochastic).
5. **Observer-contract ensemble**: R_contract over seven stored frozen
   contracts; controls 420/420 invariant negative; learned mean R 0.87;
   relative band = exactly the known borderline seeds.
6. Title narrowed to "Useful possibility collapse identifies adaptive
   emergent structure in learning systems"; strength decomposed into
   outcome rarity / discovery difficulty / structural magnitude;
   epistemic-vs-behavioural rollout distinction declared in Methods.
7. Ledger ~187 checks, 25 retained misses; consistency audit 72/72;
   manifest 35 claims / 207 outputs; manuscript 35 pages.

Remaining top gaps (unchanged in kind, reduced in number):
- third-party/public environment full-criterion preregistered
  validation (Overcooked-AI installs and loads on this server; training
  pipeline + preregistration is the next big work item);
- external timestamps (OSF/public repo) -- requires the author's account;
- an equally strong independent-lineage chess referee.

## Public-environment round (2026-07-18/19, Overcooked-AI + profile audits)

The two top gaps above are now CLOSED simultaneously:

1. **Overcooked-AI externally timestamped confirmation (OC-1..5 all
   pass).** Public, unmodified benchmark; preregistration pushed to
   `github.com/FanYixiang2000/emergence-prereg` (tag
   `v1.0-overcooked-prereg`, commit `8415e45`) BEFORE any confirmatory
   seed; 12 self-play PPO seeds, 5M steps each, four controls per seed.
   Learned accepted 8/12 (registered line exactly 8/12; every rejection
   routes through selectivity -- context-blind competence correctly
   rejected); controls 48/48 rejected with the layered routes; trigger
   direction 12/12; contract-B twin rejections 12/12; usefulness
   do-contrast positive 12/12 (p = 2.4e-4). This kills the
   "author-designed worlds + no external timestamp" objection in one
   experiment.
2. **Continuous-profile ranking stability** (RS-1 pass, mean Spearman
   0.76 across five stored contracts; RS-2 retained miss with the
   E_adapt follow-up 15/15) and **predictive validity** (PV-1..3
   retained misses; early value axis descriptively 0.81) -- the
   continuous record is calibrated and honestly bounded.
3. Ledger ~208 checks, 31 retained misses; verification 82/82;
   manifest 43 claims / 229 outputs; manuscript 37 pages including the
   new ED Overcooked figure.
4. A silent file-reversion incident on 2026-07-19 (IDE checkpoint
   restore) was detected by the verification scripts and fully
   repaired; all restored numbers re-verified against stored outputs.

## Construct-validity round (2026-07-19, GPT round-6 triage)

Executed from the round-6 external review (accepted items):

1. **Six-knob ground-truth generator calibration** (GC-1..5): diagonal
   dominance with zero violations; nullity/value/provenance
   separability pass; GC-2 retained miss (specification error in the
   frozen exemption list + one stated structural coupling). This is
   the "construct calibration" the review demanded -- the record
   measures what it names, verified against known generative truth.
2. **Record axioms A1-A8 machine-verified** (verify_record_axioms.py):
   nullity, boundedness, monotonicity, data processing, context
   sensitivity, value separability, provenance separability,
   abstention -- all pass on 20,000-sample fresh ensembles.
3. **Admissible observer contract family + identification interval**
   formalized in THEORY.md; measured instance: median per-seed
   E_struct interval width 0.14 across five stored contracts, ranking
   stable, no control interval touches the adaptive layer.
4. **Convergent validity** (component -> matching endpoint): CV-1 pass
   (0.56 vs -0.09); CV-2/3 retained misses -- predictive content is
   axis-specific, exactly the mature claim the review asked for.
5. **Overcooked round-2 held-out replication**: pilots running on
   candidate pairs (coordination_ring + counter_circuit / bottleneck /
   centre_pots / large_room); a NEW preregistration
   (OVERCOOKED_ROUND2_PREREGISTRATION.md) will be frozen and
   externally timestamped before ~20 fresh seeds launch. Round-1 data
   is never pooled with round 2.
6. The primary output is now formally the multi-dimensional record
   G_chi = (Y, M, V, A, Q, R, U); scalar summaries are declared
   secondary (ordering/visualization only).

## Remaining load-bearing experiments

### Must do before an NMI submission

1. **Seed-powered hierarchical replication of the non-contextual mechanism
   probes. -- DONE (2026-07-14).**
   - Three further simple_spread and five further LBF policies were trained
     under unchanged frozen probes; combined seed-level sign tests are
     p=0.016 and p=0.004 with positive cluster-bootstrap mean intervals.
   - Episode resampling remains conditional measurement uncertainty only.
   - The full six-component extension of LBF exists separately as the
     Contextual LBF domain; the non-contextual probes stay mechanism-level.

### High-value strengthening

2. **Bounded process-proxy robustness.**
   Freeze a bounded burst-concentration statistic, checkpoint-thinning grid,
   alternative basin maps and ability-window rules. Test on held-out public
   checkpoints. The current median-denominator burst ratio can become
   numerically extreme when background bursts are zero.
   - Existing-data re-analysis is complete; a prospectively frozen >=1B
     Pythia scaling sweep is in progress using official sharded revisions to
     supply the held-out-size test.

3. **Prospective external candidate discovery.**
   Chess currently measures recovery conditional on curated sacrifice puzzles.
   Add an unselected or prospectively assembled corpus and lock candidate
   scoring before revealing labels.

4. **Larger or independent public decoder family.**
   A >=1B public series would improve NMI relevance, but it is secondary to the
   causal/full-criterion and seed-level gaps.

## Recommended story

1. A declared future-distribution/trajectory-law family is the common
   descriptive substrate.
2. A six-component episode criterion adds selectivity, intervention,
   usefulness, provenance and acquisition where all are measurable.
3. Controlled counterexamples show why individual signatures are insufficient
   on the audited testbed.
4. Deep MARL and chess test scoped mechanism/event components.
5. Public checkpoints test a separate four-component acquisition-process
   proxy, not the full causal criterion.
6. The discussion states exactly which claims transfer and which do not.

Recommended central claim:

> Useful possibility collapse provides a common future-distribution substrate
> for several audited emergence signatures. A six-component intervention
> criterion separates named imitations where all components are measurable,
> while scoped event-level and process-level instruments transfer parts of the
> account to strategic behavior, deep MARL and public model checkpoints.

## Verification completed

- Regenerated all individual, composite and Extended Data figures.
- Re-ran the burst-alignment audit without an omnibus product.
- Added bounded-burst, alternate-window, checkpoint-thinning and multiplicity
  sensitivity over 27 stored process runs.
- Added seed-aware hierarchical MARL inference and prospective power planning.
- Added observer-refinement, rollout-model error and conjunction-margin
  guarantees; 20,000 random-distribution checks and all measured bounded-burst
  checks passed.
- Added a prospectively frozen ten-seed Contextual LBF confirmation:
  9/10 learned full passes, 40/40 controls rejected, all six registered
  predictions passed.
- Added a post-confirmation five-seed Contextual LBF extension:
  4/5 learned full passes, 20/20 controls rejected, all learned usefulness and
  acquisition effects positive.
- Added Contextual LBF single-signal audits showing behavior-only signals cannot
  replace the full six-component verdict under hindsight-optimal thresholds.
- Quarantined old Pythia-2.8B mirror-invalid outputs and verified the official
  sharded-revision path with a step0/step1000 smoke check.
- Recompiled `paper/main.pdf` twice (30 pages) with no undefined references.
- Recompiled `paper/cover_letter.pdf`.
- Python compilation and IDE lint checks passed for edited scripts.

