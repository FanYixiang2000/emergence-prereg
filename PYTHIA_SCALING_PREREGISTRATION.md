# Pythia held-out scaling protocol

Status: author-maintained protocol frozen before downloading or evaluating any
Pythia-1b, Pythia-1.4b or Pythia-2.8b checkpoint.

Frozen: 2026-07-11. This is not a third-party timestamped registered report.
It is retained to separate prospective predictions from later exploratory
analysis.

## Motivation

The existing process proxy was developed on synthetic grokking/induction
systems, transferred to MultiBERTs, and then evaluated on Pythia-160m and
Pythia-410m. The larger Pythia scales below are held out from all threshold,
window and probe decisions. The experiment asks whether the frozen instrument
transfers across unseen parameter scales; it does not instantiate the full
six-component episode criterion.

## Frozen models and checkpoints

- `EleutherAI/pythia-1b`
- `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-2.8b`
- checkpoints:
  `{0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,
  64000,96000,128000,143000}`

All weights are the public EleutherAI revisions fetched from the same public
mirror used for the completed 160m/410m runs. Inference dtype/device may change
for feasibility; any such change must be recorded, and thresholds may not.

## Frozen evaluation batteries

1. Subject--verb agreement: the exact 288 templated pairs in
   `pythia_collapse_probe.py`.
2. Random-target and shuffled-vocabulary controls generated with the existing
   fixed seed.
3. High-frequency facts, tail facts and tail words: the exact item lists and
   normalized completion scoring in `pythia_tail_gradualism.py`.

These are fixed evaluation batteries, not separate probe-training/test splits.

## Frozen process proxy

Imported unchanged from `grokking_collapse_bridge.py`:

- potential: pre-window entropy >= 1.0 bit;
- burstiness: window burst / median burst >= 5;
- usefulness: evaluation gain across the anchored window >= 0.2;
- endogeneity: no task-specific supervision or authored training intervention.

The ability anchor is the largest positive evaluation-accuracy jump. The
window is anchor +/- one interval. No scale-specific threshold or window may be
introduced for the confirmatory verdict.

The bounded robustness statistic

`q = window_burst / (window_burst + median_burst)`

is frozen as an equivalent reporting transform. Its threshold is exactly
`5/6`; it may not replace a failed registered verdict.

## Prospective predictions

S1. Agreement passes all four process-proxy components at every held-out scale
(3/3).

S2. Random-target and shuffled-vocabulary controls fail usefulness at every
held-out scale (6/6 control verdicts). Other failure routes are descriptive.

S3. Agreement pair accuracy exceeds 0.8 by checkpoint step 4,000 at every
held-out scale.

S4. The largest positive collapse increment occurs no later than one published
checkpoint interval after the largest agreement-accuracy jump in at least two
of three scales.

S5. High-frequency facts are accepted at at least two of three scales. Tail
facts and tail words are both rejected at at least two of three scales; the
component carrying rejection is not prescribed.

S6. Replacing the unbounded burst ratio by the frozen bounded transform leaves
all primary process-proxy verdicts unchanged.

S7. Across checkpoint-thinning factors 2--4 and offsets, at least 90% of
condition-level verdicts agree with the full-grid verdict. This is a robustness
prediction, not a new definition.

## Primary summaries

- per-scale component values and verdicts;
- 3/3 agreement transfer count and paired control outcomes;
- ability-crossing checkpoint;
- burst lead/lag in published checkpoint intervals;
- bounded-transform equivalence;
- thinning-cell agreement;
- all failures retained without threshold changes.

No p values will be multiplied across scales, tasks or controls. Models share
training data and architecture, so scale outcomes are not independent trials.

## Failure handling

- Download or hardware failure: retry and log; missing checkpoints remain
  missing.
- Numerical OOM: use bfloat16 and record it; do not change the probe.
- Tokenization mismatch: report dropped items; if fewer than 10 items remain in
  a family, that family-scale prediction is void.
- A failed prediction remains failed. Any new metric or probe proposed after
  inspecting these scales is exploratory and requires a new held-out family.

## Outcomes (recorded 2026-07-16 after the 1B/1.4B/2.8B runs; nothing above
edited)

Data provenance notes (mechanics, not threshold changes): all weights are the
public EleutherAI revisions. 1.4B: all 21 sharded revisions cached and SHA-256
verified pairwise distinct. 2.8B: revisions step0--16000 use sharded
safetensors; late revisions publish only single-file weights, and a hash audit
found two upstream repository defects: (a) `step64000`'s weight files
duplicate `step143000` in both formats -- the revision is unusable and was
excluded (19 evaluated intermediate checkpoints plus step0); (b) `step32000`'s
`model.safetensors` is a stale copy of the final weights, while its
`pytorch_model.bin` is authentic -- the safetensors was rebuilt from the
bin (tensor-identity verified) before the reported run. `step96000` and
`step128000` were bin-cross-checked (0 mismatching tensors). Byte-range spot
checks confirmed the download source serves content identical to
huggingface.co for locally verified shards.

- S1 **FAILED at 2.8B** (2/3 held-out scales pass). Agreement passes all four
  components at 1B and 1.4B (burstiness 11.9 and 7.3, gains 0.47/0.43, window
  step 1000 in both). At 2.8B the anchored window is again step 1000 and
  usefulness passes (gain 0.42), but burstiness is 3.18 < 5: on the published
  checkpoint grid the larger model's collapse is spread across several
  early intervals instead of one dominant burst. Registered failure, kept.
- S2 PASS 6/6. Every random-target and shuffled-vocabulary control fails
  usefulness at every held-out scale.
- S3 PASS 3/3. Agreement crosses 0.8 by step 1000 (1B, 1.4B) and step 2000
  (2.8B), within the registered step-4000 bound.
- S4 PASS 3/3. The largest collapse increment precedes the largest agreement
  jump by two published intervals at every scale (never later than one
  interval after).
- S5 **head-facts half FAILED** (1/4 scales accept: 160m only; 1B/1.4B/2.8B
  reject via burstiness despite final accuracy 1.0). Tail-facts and
  tail-words halves PASS: both frequency-tail families are rejected at every
  scale (4/4 including 160m).
- S6 PASS. The bounded transform leaves every primary verdict unchanged at
  all scales.
- S7 **FAILED** (scored 2026-07-16, `held_out_scaling_robustness.json`).
  Condition-level thinning agreement across the three held-out scales is
  130/162 = 80.2% < the registered 90%. The disagreements are concentrated
  and diagnostic: (a) the 2.8B agreement condition flips to ACCEPT in all
  nine thinning cells -- coarser grids re-aggregate the spread-out collapse
  and push burstiness back over threshold, confirming that the S1 failure is
  a grid-resolution effect rather than an absent transition; (b) borderline
  tail/head families at 1B--2.8B (burstiness near 5) flip under thinning for
  the same reason. Radius sensitivity shows the same pattern (1.4B agreement
  rejects at radius 0; 2.8B agreement accepts at radius 2). The 160m/410m
  stored-run audit had 93.8% agreement; the held-out scales show that
  borderline-burstiness verdicts are grid-relative, which we report as the
  measured scope of the process proxy rather than adjusting any threshold.

Reading (exploratory, post-hoc): the S1/S5 burstiness failures at larger
scales share one route -- abilities that larger models acquire faster and
earlier are no longer concentrated in a single dominant burst at the
published log-spaced grid resolution. This is the declared grid-resolution
scope of the process proxy, now measured across a 17.5x parameter range,
not a scale-invariance claim.

## S8 extension: Pythia-6.9B (frozen 2026-07-20T00:35+08:00, BEFORE any
## 6.9B data was obtainable)

Freeze context, stated for the record: at the time of this freeze the
public weight hosts (huggingface.co and its mirror) are unreachable
from this machine (connection resets logged in the session terminals),
so no 6.9B checkpoint, revision list or metric could have been seen.
The predictions below are therefore frozen strictly before data.

Everything above is imported unchanged: checkpoint grid, evaluation
batteries, thresholds, windows, the bounded q transform, the hash
audit (per-revision SHA-256 with upstream-defect quarantine as at
2.8B), and the failure-handling rules. Inference on cuda/bfloat16 is
recorded as the feasibility setting. Checkpoints are deleted after
evaluation; ~13 GB peak disk per revision.

The grid-resolution law measured at 1B-2.8B now makes directional
predictions one scale up (2.5x parameters). Frozen:

    S8a  Both frequency-tail families (tail facts, tail words) are
         REJECTED at 6.9B (double-dissociation transfer).
    S8b  Agreement usefulness passes: final agreement accuracy >= 0.9
         and windowed gain >= 0.2.
    S8c  Agreement burstiness on the full published grid FAILS at
         6.9B (the spread-collapse route measured at 2.8B, predicted
         to persist or worsen with scale), AND flips to accept under
         2x checkpoint thinning for at least one offset (the law's
         second half). Both halves are scored; either half failing is
         a registered miss.
    S8d  Both controls (random target, shuffled vocabulary) are
         rejected at every evaluated checkpoint.
    S8e  Integrity: any upstream duplicate/stale revision detected by
         the hash audit is quarantined and disclosed, and the run
         proceeds on the remaining revisions (as at 2.8B).

S8c is the sharpest item: it risks the grid-resolution law itself. If
6.9B agreement is bursty on the full grid, the law's monotone-in-scale
reading is falsified and will be reported as such.

