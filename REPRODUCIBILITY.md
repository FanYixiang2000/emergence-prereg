# Reproducibility map (current manuscript)

This file documents the pipeline behind the manuscript in the repository
root (`main.tex`, `si.tex`, compiled to `main.pdf` and `si.pdf`). The
repository also contains the complete history of the wider research
program (earlier pipelines, their outputs and preregistrations); that
history is documented in `README.md` and is not required to reproduce
the manuscript.

All randomness is seeded in-script. Pinned package versions are in
`requirements-lock.txt`. Runtimes are wall-clock on a 128-core CPU
machine; no GPU is required for anything in the current manuscript.

## One-command checks

```bash
make audit                # verify_paper_numbers.py: every number cited in the
                          # manuscript recomputed from stored outputs;
                          # manuscript_numbers.py: prints each cited number;
                          # generate_manifest.py: regenerates manifest.json
make figures              # make_figures.py (Figs 2-6, ED Figs 1-2) and
                          # make_si_tables.py (all Supplementary Tables),
                          # from stored outputs only
make paper                # latexmk main.tex and si.tex
make small-reproduction   # re-runs the deterministic analysis layer from
                          # scratch (no training, no downloads, ~90 s) and
                          # verifies the audit still passes
```

Figure 1 (`figures/fig0_overview.pdf`) is a data-free schematic; every
data panel is drawn by `make_figures.py` from files in `outputs/`.

## Experiment families -> scripts -> outputs

Output files are named after the script that writes them
(`X.py` writes `outputs/X.json`); the one exception is
`emergence_certificate.py`, which writes
`outputs/emergence_certificates.json` and refreshes the per-system
breakpoint files it reads. Preregistration documents are named in each
script's module docstring; the registered-outcome sections of those
documents were written only after the corresponding run.

| Family | Scripts | Type | Runtime |
|---|---|---|---|
| Ground-truth factorial (72 cells) | bench72_factorial.py | exact enumeration | seconds |
| Source-decomposition calibration | collapse_source_decomposition.py, sd_audit.py | exact enumeration | seconds |
| Breakpoint-detector held-out validation | detector_validation.py | synthetic curves, seeded | seconds |
| Sample-complexity bound | learn_n_exact.py | exact enumeration | seconds |
| Grip transport (REINFORCE) | learn_grip_transport.py, learn_grip_transport_b5.py, learn_grip_ext.py | training | minutes-hours |
| Grip replications and formation | learn_grip_a2c.py, learn_grip_formation.py, learn_grip_formation_fine.py | training | minutes-hours |
| Grip interventions and confound tests | learn_grip_policy.py, learn_grip_confound.py, learn_grip_utility.py | training | minutes-hours |
| Convention, roles, neural resolution | learn_convention.py, learn_roles.py, learn_nn_resolution.py, learn_stance_sticky.py, learn_stance_control.py, learn_stance_transport.py, learn_transport_equivariant_slow.py | training | minutes-hours |
| Cross-play barrier (stored policies, replay) | barrier_xplay.py | deterministic replay | ~1 min |
| Higher-order triad | triad_highorder_cue.py, tri_c_breakpoint.py, tri_c_breakpoint_ext.py | training | minutes |
| Gradual-collapse negative control | learn_quorum_breakpoint.py | training | minutes |
| Ant colony | ant_conditional_leverage.py, ant_fss.py | agent-based simulation | minutes |
| Kuramoto | kuramoto_breakpoint.py, kuramoto_breakpoint_r2.py, kuramoto_scale.py, kuramoto_scale_n10.py, kuramoto_offdesign_ladder.py | ensemble simulation | minutes-hours |
| Potts, Swift-Hohenberg, Schelling, Vicsek | regime_discovery_audit.py, regime_discovery_audit2.py, regime_ensemble_audit.py, regime_ensemble_audit2.py, ceb_vicsek_dense.py | simulation | minutes-hours |
| Overcooked (official layouts, PPO) | overcooked_profile_confirmatory.py, overcooked_ring_convention.py, oc_ring_ext.py, oc_ring_realization.py, overcooked_state_breakpoint.py, overcooked_occupancy_breakpoint.py | training | hours-days |
| Overcooked causal intervention | oc_ring_intervention.py, oci_seed_level.py, oc_ring_fixed_time.py | resumed training from stored checkpoints (`outputs/overcooked_genesis_*.pt`) | hours |
| Overcooked counter_circuit pilots (competence gate) | oc_cc_pilot.py, oc_cc_pilot2.py | training | hours |
| Method-baseline battery | bench_baselines.py | re-analysis of stored outputs | seconds |
| Discovered-regime controllability race | learn_grip_discovery_utility.py | training + re-analysis | hours |
| Seed-level statistics for the grip race | learn_grip_stat_unit.py | training + re-analysis | ~30 min |
| Contract-violation injections | semi_inject.py | re-analysis of stored curves | seconds |
| Representation robustness | repr_equiv_grip.py, repr_equiv_convention.py, repr_robustness.py | re-analysis + training | minutes |
| Emergence certificates (all systems) | emergence_certificate.py | re-analysis of stored outputs | seconds |

The deterministic analysis layer (first four rows plus `semi_inject.py`,
`emergence_certificate.py` and `barrier_xplay.py`) is exactly what
`make small-reproduction` re-runs; each of those scripts regenerates its
stored JSON byte-identically, and `verify_paper_numbers.py` then
re-checks every cited number.

Training pipelines are re-run by invoking the scripts in the table;
they need no external data. The Overcooked experiments require the
public `overcooked_ai` package (pinned in `requirements-lock.txt`) and
use its official layouts unmodified.

## Registered misses

Predictions that failed are retained and reported, not re-thresholded.
The registered outcomes (passes and misses) for the current manuscript
live in the preregistration documents named in each script docstring --
primarily `V2_ALIGNMENT_PREREGISTRATION.md` and
`COLLAPSE_SOURCE_PREREGISTRATION.md` -- and every miss cited in the
manuscript is checked against the stored outputs by
`verify_paper_numbers.py`.

## Machine-readable manifest

`manifest.json` (regenerated by `generate_manifest.py`) records, for
every stored output the manuscript depends on: its SHA-256 hash, the
generating script, and which consumers (number audit, figures,
Supplementary Tables) read it, plus hashes of every protocol document
and every file in `outputs/`. Quarantined artifacts are listed in
`INVALID_DATA_REGISTRY.md` and are excluded from all statistics.
