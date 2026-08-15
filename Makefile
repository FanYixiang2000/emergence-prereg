# Reproduction targets for the possibility-collapse project.
#
#   make audit               read-only: re-check every number cited in the
#                            manuscript against stored outputs and regenerate
#                            the output manifest
#   make figures             rebuild every manuscript figure from stored outputs
#   make paper               compile the manuscript and Supplementary Information
#   make small-reproduction  re-run the cheap exact/synthetic analyses from
#                            scratch (no training, no downloads)
#   make all                 audit + figures + paper
#
# Full training/download pipelines are documented per experiment script and
# in the preregistration files; they are deliberately not part of `all`.

PY ?= python

.PHONY: all audit figures paper small-reproduction

all: audit figures paper

audit:
	$(PY) verify_paper_numbers.py
	$(PY) manuscript_numbers.py
	$(PY) generate_manifest.py

figures:
	$(PY) make_figures.py
	$(PY) make_si_tables.py

paper:
	latexmk -pdf -interaction=nonstopmode main.tex
	latexmk -pdf -interaction=nonstopmode si.tex

small-reproduction:
	$(PY) bench72_factorial.py
	$(PY) collapse_source_decomposition.py
	$(PY) sd_audit.py
	$(PY) detector_validation.py
	$(PY) emergence_certificate.py
	$(PY) semi_inject.py
	$(PY) learn_n_exact.py
	$(PY) barrier_xplay.py
	$(PY) verify_paper_numbers.py
