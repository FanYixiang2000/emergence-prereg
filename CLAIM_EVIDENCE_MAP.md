# Claim--evidence map for the NMI manuscript

This is the writing-stage map. It separates load-bearing claims from
diagnostic and engineering evidence so the main text does not read like a
project log.

## Tier 1: Load-bearing scientific claims

1. **Useful possibility collapse can be operationalized as a full
   six-component protocol where all components are measurable.**
   - Evidence: external swarm confirmation, Contextual LBF confirmation,
     latent-context sequence confirmation.
   - Outputs: `refined_confirmation_summary.json`,
     `contextual_lbf_confirmation_analysis.json`,
     `latent_context_lm_confirmation.json`.
   - Main-text role: controlled confirmation across three domains / two
     modalities.

2. **Single prior signatures are lossy projections, not equivalent verdicts.**
   - Evidence: internal battery, exact Hoel EI and Rosas Psi, CLBF
     single-signal audit, component witness matrix.
   - Outputs: `exact_prior_formalisms.json`,
     `contextual_lbf_single_signal_audit.json`,
     `component_witness_matrix.json`.
   - Main-text role: why a new framework is needed.

3. **The future-distribution score performs prospective discovery, not only
   recovery.**
   - Evidence: uncurated chess discovery, second-month replication,
     classical-versus-NNUE referee.
   - Outputs: `chess_discovery_main.json`,
     `chess_discovery_replication_2016_03.json`,
     `chess_discovery_cross_engine.json`.
   - Main-text role: external validation in unselected data.

4. **Emergent structure is stable under a declared perturbation family, but
   not claimed to generalize without bound.**
   - Evidence: Contextual LBF persistence, sequence-domain generalization.
   - Outputs: `persistence_retention_curves.json`,
     `latent_context_generalization.json`.
   - Main-text role: non-triviality bridge and scope boundary.

## Tier 2: Replication and boundary claims

5. **Deep-MARL mechanism evidence is seed-powered after extension.**
   - Output: `hierarchical_marl_analysis_combined.json`.
   - Key number: 6/6 and 8/8 positive policy-seed mean do-contrasts;
     exact sign tests p=0.016 and p=0.004.

6. **Public checkpoint process proxy transfers within scope but fails at
   grid-resolution boundaries.**
   - Outputs: `pythia_scaling_summary.json`,
     `held_out_scaling_robustness.json`,
     `pythia_1.4b_checkpoint_hashes.json`,
     `pythia_2.8b_checkpoint_hashes.json`.
   - Key message: 4/5 agreement acceptances, 10/10 control rejections,
     8/8 tail rejections; 2.8B S1/S7 failures retained and interpreted as
     velocity/grid-resolution limits.

7. **The framework predicts phase regions, not only post hoc labels.**
   - Output: `phase_boundary_multiseed.json`.
   - Key number: 11/12 non-tie replication points match.

## Tier 3: Diagnostic and audit claims

8. **Basin measurements are not decorative approximations of trajectory KL;
   they are task-relevant filters.**
   - Outputs: `trajectory_basin_coupling.json`,
     `trajectory_kl_implementation_check.json`.

9. **Observer choice affects distributional components but cannot manufacture
   a full verdict.**
   - Outputs: `adversarial_observer_audit.json`,
     `adversarial_observer_controls.json`.

10. **The process proxy is a velocity/acquisition-shape instrument, not a
    stand-alone emergence verdict.**
    - Output: `ordinary_learner_control.json`.

## Tier 4: Reproducibility and data-truth safeguards

11. **Every headline manuscript number is backed by stored data.**
    - Output/script: `verify_manuscript_numbers.py`; current result 109/109.

12. **External-data and checkpoint integrity are audited.**
    - Files: `manifest.json`, `INVALID_DATA_REGISTRY.md`,
      `verify_pythia_checkpoints.py`.

13. **The analysis freeze is explicit.**
    - File: `ANALYSIS_FREEZE.md`.

## Main-text discipline

The main text should emphasize Tiers 1--2. Tiers 3--4 belong in methods,
Extended Data, supplement and reviewer response material. The manuscript
should never imply that the ledger count is a sample size or that process
proxy acceptance equals a full emergence verdict.
