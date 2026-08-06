# Latent-context sequence model: six-component confirmation protocol

Status: author-maintained protocol frozen before training or evaluating any
fresh confirmation seed listed below. Frozen: 2026-07-16.

Frozen implementation: `latent_context_lm.py` (TinyLM, 2-layer causal
transformer, d=64; task, observer, thresholds and systems as in the module
docstring). Thresholds are copied unchanged from the previously frozen
Contextual LBF criterion: potential >= 0.5 bits; conditional selectivity
>= 0.5; specificity >= 0.2 bits JS; usefulness > 0; acquisition >= 0.3;
endogeneity by provenance flag.

## Pilot disclosure

One design pilot (seeds 2001, 2002; tag `pilot`) was run before this freeze
to verify training feasibility and estimator sanity. Both pilot seeds passed
all six components and all controls were rejected. Pilot seeds are excluded
from confirmatory counts. No threshold was tuned at any point.

## Why this domain

The two existing full six-component domains are multi-agent RL tasks. This
domain instantiates the full criterion in the sequence/language modality: a
small causal transformer trained by next-token prediction on synthetic
sequences whose two latent contexts (long-range marker present or absent)
are never labelled. The valued continuation rule differs by context
(long-range copy R1 vs local copy R0); the trigger is the long-range
commitment on the first generated token; do-interventions force or forbid
that token; the initialization twin supplies the acquisition reference.

## Systems and expected labels

1. `learned` (expected emergent).
2. `initial_twin` (expected non-emergent).
3. `router` -- oracle context router, scripted (expected non-emergent via
   endogeneity/acquisition).
4. `fixed_R0`, `fixed_R1` -- unconditional rules (expected non-emergent).

## Evaluation

80 prefixes per context per condition; natural, do_trigger and
do_non_trigger modes; frozen evaluation-seed block 30,000,000 + seed*100,000.
Training: 3,000 steps, batch 128, AdamW lr 3e-4 (pilot-fixed).

## Fresh confirmation seeds

`2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110`

No seed may be retrained or replaced because of an outcome.

## Prospective predictions

- LC1: the learned model passes the full six-component rule on at least
  9/10 fresh seeds (exact one-sided sign test p <= 11/1024).
- LC2: all 40 non-learned verdicts (twin + three scripted, 10 seeds) are
  rejected.
- LC3: the oracle router passes all four behavioural components and fails
  exactly {endogeneity, acquisition} on at least 9/10 seeds.
- LC4: every learned seed has positive acquisition and every twin fails
  acquisition.
- LC5: the learned natural trigger rate is higher in context 1 than
  context 0 on every seed; usefulness is positive on at least 9/10 seeds.
- LC6: seed-bootstrap lower 95% bounds over the ten learned seeds are
  positive for usefulness and acquisition.

## Failure handling

Failed predictions are recorded as registered failures with routes; no
threshold, task, observer or seed change is allowed after this freeze.

## Outcomes

(recorded 2026-07-16 after the ten-seed run; nothing above edited;
`outputs/latent_context_lm_confirmation.json`)

- LC1 PASS: 10/10 learned models pass the full six-component rule
  (selectivity 0.950--1.000; usefulness 0.475--0.500; acquisition
  0.888--0.988).
- LC2 PASS: 40/40 non-learned verdicts rejected.
- LC3 **REGISTERED FAILURE, informative route**: the oracle router is
  rejected on every seed, but 0/10 seeds show the predicted exact
  {endogeneity, acquisition} route. Being fully deterministic given the
  latent context, the router's continuations are unchanged by the first-token
  interventions, so it also fails specificity (JS = 0) and usefulness
  (gap = 0) -- it is rejected on four components, more broadly than
  predicted. The contrast with Contextual LBF's team_nearest (which does
  respond to physical interventions and fails exactly the provenance pair)
  is itself informative: in sequence space a scripted router is
  intervention-inert, in embodied space it is intervention-responsive.
- LC4 PASS: 10/10 positive acquisition; 10/10 twins fail acquisition.
- LC5 PASS: 10/10 context ordering; 10/10 positive usefulness.
- LC6 PASS: seed-bootstrap lower 95% bounds positive
  (usefulness [0.481, 0.494]; acquisition [0.918, 0.955]).

5/6 registered predictions pass; LC3 is kept as a registered failure. This
is the third full six-component domain and the first in the sequence/
language modality: the full criterion now spans embodied multi-agent RL
(swarm, Contextual LBF) and next-token-trained transformers.
