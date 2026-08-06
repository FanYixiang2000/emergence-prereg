# Reproduction targets for the possibility-collapse project.
#
#   make audit               read-only: re-check every headline number against
#                            stored outputs and regenerate the manifest
#   make figures             rebuild every figure from stored outputs
#   make paper               compile the manuscript and cover letter
#   make small-reproduction  re-run the cheap exact/synthetic analyses from
#                            scratch (no training, no downloads)
#   make all                 audit + figures + paper
#
# Full training/download pipelines are documented per experiment script and
# in the preregistration files; they are deliberately not part of `all`.

PY      ?= python
TEXPATH ?= /home/Yixiang/texlive/2026/bin/x86_64-linux

.PHONY: all audit figures paper small-reproduction

all: audit figures paper

audit:
	$(PY) verify_manuscript_numbers.py
	$(PY) generate_manifest.py

figures:
	$(PY) generate_figure1_concept.py
	$(PY) generate_paper_figures.py
	$(PY) generate_robustness_figures.py
	$(PY) generate_contextual_lbf_figure.py
	$(PY) generate_scaling_figure.py
	$(PY) generate_reframe_figures.py
	$(PY) generate_overcooked_figure.py
	$(PY) generate_calibration_figure.py
	$(PY) assemble_main_figures.py

paper:
	cd paper && PATH="$(TEXPATH):$$PATH" latexmk -pdf \
		-interaction=nonstopmode main.tex
	cd paper && PATH="$(TEXPATH):$$PATH" latexmk -pdf \
		-interaction=nonstopmode cover_letter.tex

small-reproduction:
	$(PY) contract_ranking_stability.py
	$(PY) overcooked_aggregate.py
	$(PY) overcooked_profiles.py
	$(PY) verify_record_axioms.py
	$(PY) generator_calibration.py
	$(PY) crowd_vote_domain.py
	$(PY) fair_baseline_comparison.py --skip_fresh
	$(PY) chess_realized_outcome.py
	$(PY) chess_clustered_inference.py
	$(PY) strength_gradient_battery.py
	$(PY) strength_gradient_fine.py
	$(PY) matched_provenance.py
	$(PY) learned_harmful_emergence.py
	$(PY) contract_ensemble_analysis.py
	$(PY) verify_trajectory_kl_implementation.py
	$(PY) verify_theory_bounds.py
	$(PY) verify_observer_bounds.py
	$(PY) contextual_lbf_threshold_sensitivity.py
	$(PY) contextual_lbf_single_signal_audit.py
	$(PY) component_ablation_witnesses.py
	$(PY) component_witness_matrix.py
	$(PY) persistence_retention_analysis.py
	$(PY) chess_discovery_referee_sensitivity.py
	$(PY) summarize_pythia_scaling.py
	$(PY) held_out_scaling_robustness.py
	$(PY) verify_manuscript_numbers.py
