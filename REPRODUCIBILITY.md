# Reproducibility map

Every figure and headline number maps to a script, its outputs, and (where
applicable) a preregistration whose outcome section was written only after
the run. All randomness is seeded in-script. Runtimes are wall-clock on a
128-core CPU machine; the held-out Pythia 1B+ extension uses one RTX 6000 Ada.

## Environment

- Python 3.13, PyTorch (CPU/CUDA), numpy, matplotlib, python-chess, zstandard,
  wordfreq, mpe2/pettingzoo, tensorflow (only to read MultiBERTs
  checkpoints).
- Three RTX 6000 Ada GPUs are available. Earlier experiments and Pythia
  160m/410m are CPU runs; prospectively frozen 1B+ Pythia inference uses
  float32 on one GPU, with probabilities converted to float32 before scoring.
- External binaries: Stockfish 14.1 and Stockfish 11 extracted from
  Ubuntu-archive debs into `external_chess/` (no install needed).
- External data, downloaded by the scripts themselves:
  - MultiBERTs checkpoints: `gs://multiberts/public/intermediates/*`
    (~440 MB per checkpoint, deleted after each evaluation).
  - Lichess puzzle database: `https://database.lichess.org/lichess_db_puzzle.csv.zst`
    (~300 MB, kept in `external_chess/`).

## Figures -> scripts -> outputs

All figures regenerate with `python3 generate_paper_figures.py` (plus
`python3 generate_figure1_concept.py` for the concept figure) from files
already in `outputs/`. To regenerate the underlying outputs:

| Figure | Underlying script(s) | Key outputs | Preregistration | Runtime |
|---|---|---|---|---|
| figure1_concept | generate_figure1_concept.py | (schematic) | -- | seconds |
| fig1-6 (analytic core) | possibility_preservation_tree.py, possibility_ablation.py, planning_horizon_ablation.py, ptc_ground_truth_validation.py | possibility_tree_*, closure_*, horizon_* | -- | minutes |
| fig7-12 (benchmarks) | run_spatial_sweep.py, run_contextual_sweep.py, performance_closure_benchmark.py, performance_robustness_sweep.py | *_summary.csv/json | -- | ~1 h |
| fig13-14 (representation bridge) | representation_jump_bridge.py, learned_representation_jump_probe.py, contextual_learned_representation_probe.py | learned_representation_* | -- | minutes |
| fig15-18 (mechanism) | within_episode_collapse_probe.py, unsupervised_basin_discovery.py, run_within_episode_sweep.py, neural_within_episode_probe.py | within_episode_*, unsupervised_* | -- | ~1 h |
| fig19 (battery) | criterion_ablation_battery.py | criterion_battery_* | -- | ~30 min |
| fig20 (estimator robustness) | estimator_robustness_check.py | estimator_robustness_* | -- | ~1 h |
| fig21/24/25 (external transfer + refinement) | external_swarm_criterion_transfer.py, run_external_transfer_sweep.py, refined_criterion_confirmation.py | external_transfer_* | EXTERNAL_TRANSFER_PREREGISTRATION.md | ~2 h |
| fig22 (phase boundary) | phase_boundary_prediction.py | phase boundary outputs | prediction table in-script, frozen pre-training | ~1 h |
| fig23/27 (grokking + thresholds) | grokking_collapse_bridge.py, threshold_sensitivity_analysis.py | grokking_collapse_*, threshold_* | thresholds frozen in-script | ~1 h |
| fig26 (scale decomposition) | scale_emergence_decomposition.py | scale decomposition outputs | -- | ~2 h |
| fig28 (prior detectors, internal) | prior_metrics_comparison.py | prior detector outputs | -- | ~30 min |
| fig29 (induction heads) | induction_head_emergence.py, run_induction_seed_sweep.py | induction_head_* | task revision documented in docstring; pilot log kept | ~2 h |
| fig30 (transformer grokking) | transformer_grokking_replication.py | transformer grokking outputs | optimizer retune documented; pilot log kept | ~1 h |
| fig31 (MultiBERTs) | multiberts_collapse_probe.py (--model_seed 0..4) | multiberts_collapse_* | MULTIBERTS_PREREGISTRATION.md (P1-P4, G1-G5) | ~1 h/seed, download-bound |
| fig32 (phenomena battery + alignment) | multiberts_phenomena_battery.py, burst_alignment_test.py | multiberts_phenomena_*, burst_alignment_test.json | same file (R1-R5) | ~30 min |
| fig33 (chess main) | chess_collapse_probe.py --workers 60 | chess_collapse_main_* | CHESS_PREREGISTRATION.md (C1-C5) | ~3 min + pilots |
| fig34 (gradualism + robustness + detectors) | multiberts_tail_gradualism.py, chess_robustness_grid.py, chess_prior_detectors.py | multiberts_tail_*, chess_robustness_grid.json, chess_prior_detectors.json | MULTIBERTS_PREREGISTRATION.md (T1-T7); success criteria in grid docstring | ~15 min + ~25 min |
| fig35 (deep MARL) | deep_marl_collapse_probe.py (--seeds 11/22/33), deep_marl_aggregate.py | deep_marl_collapse_mappo_seed*.json, deep_marl_collapse_aggregate.json | DEEP_MARL_PREREGISTRATION.md (D1-D4) | ~20 min/seed |
| Box 1 / theory (Props 0-4) | verify_theory_bounds.py | theory_bounds_verification.json | propositions stated with proofs in THEORY.md before verification | seconds |
| LBF cross-task deep MARL | lbf_collapse_probe.py (--seeds 11 22 33 --train_episodes 8000) | lbf_collapse_main.json, lbf_net_seed*.pt | LBF_PREREGISTRATION.md (L1-L4) | ~15 min/seed |
| headline CIs | bootstrap_intervals.py | bootstrap_intervals.json | re-analysis of stored outputs, no new measurements | ~1 min |
| fig37 (exact rival formalisms, Prop 5) | exact_prior_formalisms.py | exact_prior_formalisms.json | blind spots predicted in prior_metrics_comparison.py docstring before the exact run | ~1 min + 4x policy training |
| LBF estimator robustness | lbf_robustness_grid.py (saved nets, no retraining) | lbf_robustness_grid.json | G1/G2 frozen in module docstring | ~4 h |
| LBF prior detectors | lbf_prior_detectors.py (saved nets) | lbf_prior_detectors.json (+ _round1.json archive) | predictions frozen in module docstring; round-1 failure archived, round-2 prediction frozen before rerun | ~2 h |
| fig38 (Pythia decoder) | pythia_collapse_probe.py, pythia_tail_gradualism.py | pythia_collapse_*, pythia_tail_* | PYTHIA_PREREGISTRATION.md (Y1-Y4, T1-T3; tokenization amendment A1 recorded) | ~10 min each, download-bound (public mirror) |
| Pythia 410m scale check | pythia_collapse_probe.py --size 410m | pythia_collapse_*_410m | PYTHIA_PREREGISTRATION.md (S1-S3) | ~2 h, download-bound |
| Extended Data robustness/hierarchy | process_proxy_robustness.py, hierarchical_marl_analysis.py, verify_observer_bounds.py, generate_robustness_figures.py | process_proxy_robustness.json, hierarchical_marl_analysis.json, observer_bounds_verification.json, ed_fig7 | exploratory stored-data re-analysis | seconds |
| Contextual LBF full criterion | contextual_lbf_transfer.py --seeds 1101..1110, contextual_lbf_analysis.py | contextual_lbf_confirmation*.json, contextual_lbf_net_seed*.pt, ed_fig8 | CONTEXTUAL_LBF_PREREGISTRATION.md (C1-C6; exact code hash frozen) | ~11 min total on 16 CPU threads |
| composite main figures | assemble_main_figures.py (after generate_paper_figures.py) | paper/main_fig1..6.png for the manuscript; figures/ed_fig1..6.png for Extended Data | layout in MANUSCRIPT.md; no new measurements | seconds |
| manuscript PDF | pdflatex paper/main.tex (twice; TeX Live in ~/texlive/2026) | paper/main.pdf (29 pp) | full draft: theory Props 0-4 + scoped coverage, instrument-scope map, 6 main + 6 Extended Data figures, unification table, Methods | seconds |
| cover letter | pdflatex paper/cover_letter.tex | paper/cover_letter.pdf | NMI cover letter; referees to be filled in | seconds |

## Registered misses (all kept, none re-thresholded)

Complete indexed ledger of every frozen prediction and outcome:
`PREDICTION_LEDGER.md` (~191 verdicts/checks, 26 registered misses, grouped
into four tiers). The ledger count is a transparency record, not a sample size
or combined significance claim.
The misses:

| ID | Domain | File recording the outcome |
|---|---|---|
| marginal selectivity accepts untrained net (2/5 seeds) | external swarm | EXTERNAL_TRANSFER_PREREGISTRATION.md |
| phase boundary G=7 cell (3-seed pooling) | gridworld | EVIDENCE_AUDIT.md item 18 |
| unsupervised-basin effective-mode agreement check | gridworld | EVIDENCE_AUDIT.md item 17 |
| R2 normalization check | scale decomposition | README.md (scale section) |
| R3 facts gradualism | MultiBERTs | MULTIBERTS_PREREGISTRATION.md |
| R5 NPI gradualism | MultiBERTs | MULTIBERTS_PREREGISTRATION.md |
| T2/T3 pair-metric tail families | MultiBERTs | MULTIBERTS_PREREGISTRATION.md |
| T6 tail-facts top-1 | MultiBERTs | MULTIBERTS_PREREGISTRATION.md |
| C5 effect-size margin | chess | CHESS_PREREGISTRATION.md |
| D2 win-shift prediction | deep MARL | DEEP_MARL_PREREGISTRATION.md |
| LBF detector round-1 (performance separated a set with no competent imitation) | deep MARL | lbf_prior_detectors.py, _round1.json archive |
| PY-T2 tail-facts route (verdict correct, rejected via burstiness not usefulness) | Pythia | PYTHIA_PREREGISTRATION.md |

## Current freeze and one-command checks

The current NMI manuscript candidate is frozen in `ANALYSIS_FREEZE.md`.
The machine-readable claim manifest is `manifest.json`; invalid/quarantined
artifacts are listed in `INVALID_DATA_REGISTRY.md`; package versions are in
`requirements-lock.txt`.

Use the Makefile:

```bash
make audit
make figures
make paper
make small-reproduction
```

At freeze:

- `verify_manuscript_numbers.py`: 109/109 headline checks pass.
- `manifest.json`: 21 claim records, 13 protocol hashes, 193 output hashes.
- `paper/main.pdf`: 36 pages.
- New results after the older figure table above include: latent-context
  sequence-model full criterion, chess prospective discovery and replication,
  classical-versus-NNUE cross-evaluation, CLBF persistence and sequence
  generalization boundaries, adversarial observer audits, ordinary-learner
  boundary probe, held-out Pythia scaling through 2.8B, and trajectory--basin
  coupling cross-implementation checks.

## Pilot logs kept in `outputs/`

induction_head_log_fixed_offset_pilot.txt,
transformer_grokking_log_wd1_pilot.txt, chess_pilot_log_initial_params.txt,
chess_pilot_log_depth4_temp300.txt, chess_pilot_log_depth4_temp500.txt,
deep_marl_pilot_log.txt, deep_marl_train_pilot2_log.txt,
deep_marl_mappo_pilot_log.txt, multiberts_tail_log_pair_only.txt,
lbf_pilot_log.txt, lbf_pilot2_log.txt, lbf_pilot3_log.txt,
lbf_debug_do.txt (probe-temperature sweep + do-release fix).
