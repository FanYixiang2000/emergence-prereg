# Invalid / quarantined artifact registry

Every artifact excluded from the final statistics, with the exclusion
reason and when it was decided. Nothing in this registry feeds any reported
number; the machine-readable copy lives in `manifest.json`
(`invalid_data_registry`).

| Artifact | Reason | Decided | In final stats |
|---|---|---|---|
| `outputs/pythia_*_2.8b_mirror_invalid.*` | Mirror served a single weight object for early 2.8B revisions (identical metrics across steps 0-256 exposed it) | 2026-07-13, before any manuscript use | No |
| `external_pythia_2.8b/step64000.invalid_duplicates_step143000` | Upstream repository defect: both published weight formats duplicate step143000 | 2026-07-16, on hash audit, before the reported 2.8B run | No (revision excluded; 19 intermediate checkpoints evaluated) |
| `external_pythia_2.8b/step32000/model.safetensors.stale_final_weights` | Upstream stale single-file object (contains final weights); authentic per-revision `pytorch_model.bin` verified and converted in its place | 2026-07-16, on bin cross-check, before the reported 2.8B run | No (rebuilt file used; tensor-identity verified) |
| `outputs/contextual_lbf_persistence_layoutbug.json` | First P1 perturbation layouts violated the benchmark's lexicographic food-identity convention (mechanics bug in the perturbation, not in any policy) | 2026-07-16, amendment recorded in PERSISTENCE_PREREGISTRATION.md before the reported run | No |
| `outputs/ordinary_learner_control_attempt1_failed_design.json` | 97-class ordinal task unlearnable by the frozen architecture (final acc 0.07): failed design, not an informative control | 2026-07-16, disclosed in the probe docstring | No |
| `outputs/lbf_prior_detectors_round1.json` | Registered round-1 set-composition failure (no competent imitation in the system set); round-2 prediction frozen before rerun | 2026-07-07, recorded in the script docstring | No (archived; round 2 reported) |
| `outputs/pythia_local_smoke/`, `outputs/pythia_revision_smoke/` | Infrastructure smoke tests (download-path validation), never analysis inputs | 2026-07-13/14 | No |
| CLBF pilots (seeds 101, 102), LM pilot (seeds 2001, 2002), chess discovery pilot (16 positions) | Disclosed design pilots, excluded from confirmatory counts by their preregistrations | per protocol | No |
| `outputs/crowd_vote_domain_pilot1_contextblind.json` | Trigger definition conditioned on the context label, manufacturing selectivity 1.0 for every hazard-band voter; feature map carried no context observable | 2026-07-19, caught by the run's own controls, disclosed in the script docstring | No |
| `outputs/crowd_vote_domain_pilot2_ortrigger.json` | Trigger (any democracy step) mismatched the basin definition (majority mode), zeroing the do-block fall shift | 2026-07-19, disclosed in the script docstring | No |
| `outputs/universal_observer_run1_crosspolicy.json` | Run 1 compared freshly retrained battery policies against the STORED battery's hand verdicts (different training runs); corrected run scores both verdict types side by side on the same policies | 2026-07-21, disclosed in the script docstring | No |
| Overcooked round-2 layout pilots (`outputs/oc2_pilot_*.log`, seeds 8951-8961) | Disclosed design pilots for the round-2 held-out pair choice, excluded from confirmation by OVERCOOKED_ROUND2_PREREGISTRATION.md | 2026-07-19, per protocol | No |
