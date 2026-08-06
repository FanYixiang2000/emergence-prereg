"""Generate the reproducibility manifest.

Records, for every headline claim: the backing output file and its SHA-256,
the generating script, and the claim tier. Also hashes every protocol
document, every output JSON/CSV, and lists the invalid/quarantined artifacts
with exclusion reasons. Read-only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


CLAIMS = [
    # (claim_id, tier, description, output file, script)
    ("CLBF-CONF", 1, "Second full six-component domain: 9/10 learned, 40/40 controls",
     "contextual_lbf_confirmation_analysis.json", "contextual_lbf_transfer.py"),
    ("LM-CONF", 1, "Third full domain (sequence): 10/10 learned, 40/40 controls",
     "latent_context_lm_confirmation.json", "latent_context_lm.py"),
    ("SWARM-CONF", 1, "External swarm confirmation 25/25",
     "refined_confirmation_summary.json", "refined_criterion_confirmation.py"),
    ("DISC-MAIN", 1, "Prospective discovery 4/4 (AUROC 0.730)",
     "chess_discovery_main.json", "chess_discovery_probe.py"),
    ("DISC-REP", 2, "Discovery replication, second year, 4/4 (AUROC 0.725)",
     "chess_discovery_replication_2016_03.json", "chess_discovery_probe.py"),
    ("DISC-XEVAL", 2, "Classical-vs-NNUE cross-evaluation referee",
     "chess_discovery_cross_engine.json", "chess_discovery_cross_engine.py"),
    ("SCALE", 2, "Pythia scaling 160m-2.8B with registered S1/S5/S7 failures",
     "pythia_scaling_summary.json", "summarize_pythia_scaling.py"),
    ("MARL-SEEDS", 2, "Combined seed-level MARL inference p=0.016/0.004",
     "hierarchical_marl_analysis_combined.json", "hierarchical_marl_analysis.py"),
    ("PERSIST", 2, "Persistence: dual retention curves + boundary",
     "persistence_retention_curves.json", "contextual_lbf_persistence.py"),
    ("LM-GEN", 2, "Sequence generalization audit (LG1-LG3)",
     "latent_context_generalization.json", "latent_context_generalization.py"),
    ("CHESS-EVENT", 1, "Curated chess event-level do-gap 0.539",
     "chess_collapse_main_summary.json", "chess_collapse_probe.py"),
    ("PHASE", 2, "Phase-boundary forecast 11/12",
     "phase_boundary_multiseed.json", "phase_boundary_prediction.py"),
    ("OBS-AUDIT", 3, "Adversarial observer audit, learned side",
     "adversarial_observer_audit.json", "adversarial_observer_audit.py"),
    ("OBS-NULL", 3, "Full-verdict null, control side (0/13000)",
     "adversarial_observer_controls.json", "adversarial_observer_controls.py"),
    ("ORD-LEARN", 3, "Ordinary-learner boundary probe (proxy scope)",
     "ordinary_learner_control.json", "ordinary_learner_control.py"),
    ("CAP-NOVELTY", 2, "Capability novelty boundary: ordinary rejected, "
     "grokking/induction accepted",
     "capability_novelty_boundary.json", "capability_novelty_boundary.py"),
    ("BURST-BOUND", 2, "Burst-collapse boundary: evidence channel, not "
     "definition",
     "burst_boundary_audit.json", "burst_boundary_audit.py"),
    ("COUPLING", 3, "Exact trajectory-basin coupling",
     "trajectory_basin_coupling.json", "trajectory_basin_coupling.py"),
    ("KL-XCHECK", 4, "Chain rule vs enumeration cross-implementation",
     "trajectory_kl_implementation_check.json",
     "verify_trajectory_kl_implementation.py"),
    ("ABLATION", 3, "Leave-one-component-out + witness matrix",
     "component_witness_matrix.json", "component_witness_matrix.py"),
    ("HASH-14B", 4, "Pythia 1.4B revision hash audit (21 distinct)",
     "pythia_1.4b_checkpoint_hashes.json", "verify_pythia_checkpoints.py"),
    ("HASH-28B", 4, "Pythia 2.8B hash audit (upstream defects found)",
     "pythia_2.8b_checkpoint_hashes.json", "verify_pythia_checkpoints.py"),
    ("NUM-AUDIT", 4, "Manuscript-number consistency audit",
     None, "verify_manuscript_numbers.py"),
    ("FAIR-BASE", 2, "Fair multivariate baselines: frozen transfer <= 0.9, "
     "all miss latent_conditional",
     "fair_baseline_comparison.json", "fair_baseline_comparison.py"),
    ("DUAL-OBS", 2, "Dual plausible observer contracts: 60/60 controls, "
     "14/15 structural agreement, 5 conservative value flips (DO-1/DO-3 "
     "misses)",
     "dual_observer_contracts.json", "dual_observer_contracts.py"),
    ("REALIZED", 3, "Engine-free realized-outcome referee: direction "
     "consistent, registered interaction null (RO-1 miss)",
     "chess_realized_outcome.json", "chess_realized_outcome.py"),
    ("STRENGTH", 2, "Emergence-strength gradient: provenance rarity "
     "0 < 0.39 < 0.65 bits at matched competence (ST-3 suddenness miss "
     "retained)",
     "strength_gradient_battery.json", "strength_gradient_battery.py"),
    ("STRENGTH-FINE", 3, "Fine-grid follow-up: discovery ~2x later for "
     "outcome-only provenance",
     "strength_gradient_fine.json", "strength_gradient_fine.py"),
    ("CLUSTER", 4, "Mover-cluster bootstrap of discovery AUROC excludes "
     "0.5 both months",
     "chess_clustered_inference.json", "chess_clustered_inference.py"),
    ("LEARNED-BASIN", 2, "Machine-discovered basins reproduce CLBF "
     "verdicts: 60/60 controls, 14/15 learned, 74/75 agreement",
     "learned_basin_clbf.json", "learned_basin_clbf.py"),
    ("ROLLOUT", 2, "Rollout-model audit: openness survives near-greedy "
     "decoding 15/15; diffuse-model IR-2 miss retained",
     "independent_rollout_audit.json", "independent_rollout_audit.py"),
    ("TOGA", 3, "Different-lineage referee TG-1 retained miss "
     "(weak-referee label noise)",
     "chess_discovery_toga_referee.json", "chess_discovery_toga_referee.py"),
    ("XFIT-BASIN", 1, "Cross-fitted low-level basin discovery: 4 methods, "
     "mean verdict agreement 0.957, controls 60/60 each",
     "crossfit_lowlevel_basins.json", "crossfit_lowlevel_basins.py"),
    ("HARMFUL", 2, "Learned harmful emergence: structural pass, "
     "U_private +7.5 / U_team -2.0, 5/5 seeds",
     "learned_harmful_emergence.json", "learned_harmful_emergence.py"),
    ("MATCHED-PROV", 2, "Matched-behaviour provenance: clone "
     "counterfactually distinguishable; rarity 0.33<0.39<0.65",
     "matched_provenance.json", "matched_provenance.py"),
    ("PHASE-2D", 1, "Prospective 2-D phase surface: 14/15 non-fragile "
     "cells match (P2D-1 one-cell miss retained); non-rectangular as "
     "derived",
     "phase_2d_prediction.json", "phase_2d_prediction.py"),
    ("ENSEMBLE", 2, "Contract ensemble: controls 420/420 invariant "
     "negative; learned mean R 0.87",
     "contract_ensemble_analysis.json", "contract_ensemble_analysis.py"),
    ("WM-CLOSURE", 2, "World-model closure: WM1/WM2 pass; WM3 "
     "ensemble-spread proxy bias-blind (retained miss)",
     "world_model_closure.json", "world_model_closure.py"),
    ("WM-FOLLOWUP", 3, "Coverage-augmented rule: 20/20 mismatches "
     "caught, conservative at K=20000",
     "world_model_closure_followup.json", "world_model_closure_followup.py"),
    ("BRIDGE", 4, "Collapse-bridge identity (Prop. B) machine-verified",
     "bridge_identity_verification.json", "verify_bridge_identity.py"),
    ("CALIBRATION", 2, "Continuous-profile construct calibration: "
     "orthogonality (corrected rule; original rank rule retained miss) "
     "+ dose-response Spearman 1.0",
     "profile_calibration.json", "profile_calibration.py"),
    ("PROFILES", 2, "Taxonomy continuous profiles separate along "
     "declared axes (5/5 checks)",
     "profile_existing_systems.json", "profile_existing_systems.py"),
    ("RANK-STABILITY", 3, "Cross-contract E_struct ranking rho 0.76; "
     "RS-2 structure-only separation retained miss; E_adapt 15/15",
     "contract_ranking_stability.json", "contract_ranking_stability.py"),
    ("PRED-VALIDITY", 3, "Early-magnitude prediction battery PV-1..3 "
     "retained misses; early value axis 0.81 (descriptive)",
     "predictive_validity.json", "predictive_validity.py"),
    ("OVERCOOKED", 1, "Public third-party environment, externally "
     "timestamped preregistration: OC-1..5 all passed (learned 8/12, "
     "controls 48/48 rejected, useful+ 12/12 p=2.4e-4)",
     "overcooked_confirmation_pooled.json", "overcooked_aggregate.py"),
    ("GENERATOR-CAL", 2, "Six-knob ground-truth generator: sensitivity "
     "matrix diagonally dominant (GC-1), nullity/value/provenance "
     "separability pass; GC-2 retained miss with disclosed follow-up",
     "generator_calibration.json", "generator_calibration.py"),
    ("AXIOMS", 4, "Continuous-record axioms A1-A8 machine-verified",
     "record_axioms_verification.json", "verify_record_axioms.py"),
    ("CONVERGENT", 3, "Convergent validity: CV-1 pass (matched "
     "structure prediction 0.56 vs -0.09); CV-2/3 retained misses",
     "convergent_validity.json", "convergent_validity.py"),
    ("OC-PROFILES", 3, "Overcooked continuous profiles: rejected "
     "learned seeds rank strictly below accepted (0.653 < 0.695), "
     "controls all zero, read-only",
     "overcooked_profiles.json", "overcooked_profiles.py"),
    ("OC-TRANSITION", 3, "Overcooked real-vs-ghost transition scaffold: "
     "state-level smoke passes; no learned flagship claim",
     "overcooked_transition_certificate_smoke_scripted.json",
     "overcooked_transition_certificate.py"),
    ("OC-TRANS-PILOT", 3, "Overcooked learned transition pilots: 40k/500k "
     "boundary, 2M single-seed pilot positive",
     "overcooked_transition_pilot_audit.json",
     "overcooked_transition_pilot_audit.py"),
    ("CROWD", 2, "Collective-control domain (crowd voting): 50/50 "
     "controls rejected with declared routes; do-block reproduces the "
     "historical counterfactual 10/10; CR-1 (7/10) and CR-5 retained "
     "misses; two design pilots quarantined",
     "crowd_vote_domain.json", "crowd_vote_domain.py"),
    ("BIFURCATION", 2, "Convention bifurcation: selective fraction "
     "0.1->0.9 across the measured value-gap sign change; gap sign "
     "predicts majority basin 4/5; BF-4 retained miss (basin decided "
     "late)", "convention_bifurcation.json", "convention_bifurcation.py"),
    ("META-COLLAPSE", 3, "Population convention space never collapses "
     "(stable 26/22/2 bifurcation); MC-2/3/4 retained misses; hard "
     "signature exposed as jitter artifact",
     "meta_collapse_commitment.json", "meta_collapse_commitment.py"),
    ("META-MARGINS", 3, "Soft-margin follow-up: basin decided in the "
     "open-ground state 24/24 (no overlap), late and stochastic; F-1 "
     "retained miss", "meta_collapse_margins.json",
     "meta_collapse_margins.py"),
    ("SPATIAL-BRIDGE", 4, "Proposition S machine-verified: spatial "
     "collapse = total correlation; linear N-growth of the open "
     "space; scripts attain the maximum (provenance-blind)",
     "spatial_bridge_verification.json", "verify_spatial_bridge.py"),
    ("EXEMPLARS", 2, "Canonical exemplars (Boids/Schelling/Life): "
     "CE-1..5 all pass; structural-not-adaptive classification with "
     "measured reasons; Life at the substrate boundary exactly",
     "canonical_exemplars.json", "canonical_exemplars.py"),
    ("UNIVERSAL-OBS", 1, "One semantics-free possibility-space recipe, "
     "identical code across domains: battery 8/9 (positives accepted), "
     "crowd 18/18 + 15/15 controls, CLBF stored 95.7%; two disclosed "
     "run-1 corrections", "universal_observer.json",
     "universal_observer.py"),
    ("UNIVERSAL-OC", 2, "Universal recipe on public Overcooked: 19/20 "
     "agreement, 16/16 controls; U-4c retained miss (coarser "
     "partitions are conservative, never manufacture acceptance)",
     "overcooked_universal_observer.json",
     "overcooked_universal_observer.py"),
    ("LIVE-DEMO", 2, "Blind accuracy: 24 hidden random knob vectors "
     "recovered at Spearman 0.96-0.999 per matched dimension; live "
     "five-system verdict walkthrough correct; LD-2 raw rule retained "
     "miss (undeclared partial leakage below permutation null, "
     "p=0.65)", "live_demonstration.json", "live_demonstration.py"),
    ("PROMOTE", 2, "Emergence-promoting selection: optimizing the "
     "record raises convention acquisition 0.39->0.78 at no value "
     "cost, beating value-selection 0.53 (PE-1..3 all pass)",
     "emergence_promoting_selection.json",
     "emergence_promoting_selection.py"),
    ("COORDINATES", 1, "Emergence coordinates: analytic per-dimension "
     "truths recovered (EC-1); frozen thresholds transfer blind to "
     "Kuramoto/Life/learned conventions matching literature labels "
     "4/4 (EC-2); adversarial matrix 8/8 on predicted dimensions "
     "(EC-3); three runs of spec errors caught by the battery's own "
     "exact computations, retained",
     "emergence_coordinates.json", "emergence_coordinates.py"),
]

INVALID = [
    {"artifact": "outputs/pythia_collapse_summary_2.8b_mirror_invalid.json",
     "reason": "mirror served one weight object for early revisions "
               "(detected 2026-07-13 before any manuscript use)",
     "in_final_statistics": False},
    {"artifact": "outputs/pythia_tail_summary_2.8b_mirror_invalid.json",
     "reason": "same mirror defect", "in_final_statistics": False},
    {"artifact": "external_pythia_2.8b/step64000.invalid_duplicates_step143000",
     "reason": "upstream repo defect: both weight formats duplicate "
               "step143000 (detected by hash audit before the reported run)",
     "in_final_statistics": False},
    {"artifact": "external_pythia_2.8b/step32000/model.safetensors"
                 ".stale_final_weights",
     "reason": "upstream stale single-file object (= final weights); "
               "rebuilt from bin-verified pytorch_model.bin before the "
               "reported run", "in_final_statistics": False},
    {"artifact": "outputs/contextual_lbf_persistence_layoutbug.json",
     "reason": "perturbation layout spec violated lexicographic food-identity "
               "convention; fixed and rerun (amendment recorded)",
     "in_final_statistics": False},
    {"artifact": "outputs/ordinary_learner_control_attempt1_failed_design.json",
     "reason": "97-class ordinal task unlearnable by the frozen architecture "
               "(final acc 0.07): failed design, not an informative control",
     "in_final_statistics": False},
    {"artifact": "outputs/lbf_prior_detectors_round1.json",
     "reason": "registered round-1 set-composition failure, archived; "
               "round 2 frozen before rerun", "in_final_statistics": False},
]


def main() -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "claims": [],
        "invalid_data_registry": INVALID,
        "protocols": {},
        "outputs": {},
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                                capture_output=True, text=True, timeout=10)
        manifest["git_commit"] = (commit.stdout.strip()
                                  if commit.returncode == 0 else None)
    except Exception:
        manifest["git_commit"] = None

    for cid, tier, desc, fname, script in CLAIMS:
        entry = {"claim_id": cid, "tier": tier, "description": desc,
                 "script": script}
        if fname:
            p = OUTPUTS / fname
            entry["output"] = fname
            entry["output_sha256"] = sha256(p) if p.exists() else "MISSING"
        manifest["claims"].append(entry)

    for proto in sorted(HERE.glob("*PREREGISTRATION*.md")) + [
            HERE / "PREDICTION_LEDGER.md",
            HERE / "THEORY.md",
            HERE / "ANALYSIS_FREEZE.md",
            HERE / "CLAIM_EVIDENCE_MAP.md",
            HERE / "REPRODUCIBILITY.md",
            HERE / "INVALID_DATA_REGISTRY.md",
            HERE / "NMI_READINESS_AUDIT.md",
            HERE / "EVIDENCE_AUDIT.md",
    ]:
        if proto.exists():
            manifest["protocols"][proto.name] = sha256(proto)

    for out_file in sorted(OUTPUTS.glob("*.json")) + sorted(
            OUTPUTS.glob("*.csv")):
        manifest["outputs"][out_file.name] = sha256(out_file)

    path = HERE / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing = [c["claim_id"] for c in manifest["claims"]
               if c.get("output_sha256") == "MISSING"]
    print(f"claims: {len(manifest['claims'])}  "
          f"protocols: {len(manifest['protocols'])}  "
          f"outputs hashed: {len(manifest['outputs'])}  "
          f"missing: {missing or 'none'}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
