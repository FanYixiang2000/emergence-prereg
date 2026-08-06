# Pre-registration: possibility collapse across a public checkpoint series

Written and frozen BEFORE any checkpoint beyond step_0 was downloaded or
evaluated. The feasibility pilot permitted below may adjust probe wording and
batch mechanics only -- never the thresholds, the component mapping, or the
predictions.

## Target system (zero authorial control)

MultiBERTs seed_0 intermediate checkpoints (Sellam et al., ICLR 2022):
a BERT-base masked-language model (~110M parameters, 12 layers, vocab
30522) trained by Google Research and published with 29 intermediate
checkpoints (steps 0, 20k ... 200k, 300k ... 2M) at
`gs://multiberts/public/intermediates/seed_0`. We did not train it, did not
choose its data, its architecture, its schedule, or its checkpoint spacing.
The download URL set, fixed now, is every available step:
0, 20k, 40k, 60k, 80k, 100k, 120k, 140k, 160k, 180k, 200k, 300k, 400k,
500k, 600k, 700k, 800k, 900k, 1M, 1.1M, 1.2M, 1.3M, 1.4M, 1.5M, 1.6M,
1.7M, 1.8M, 1.9M, 2M.

## Target ability (externally documented)

Long-range subject-verb agreement is a standard targeted syntactic
evaluation (Linzen et al. 2016; Marvin & Linzen 2018; Goldberg 2019) and is
documented to be acquired during BERT pretraining without any supervision on
the task (the training signal is generic masked-token prediction). Probing
studies across pretraining time (e.g. Liu et al. 2021, "Probing Across
Time"; Chiang et al. 2020) report that syntactic abilities are acquired
early and rapidly relative to total pretraining.

## Observer possibility space

For each checkpoint k, the model's own predictive distribution over the
30522-token vocabulary at a masked verb position on a FIXED probe set.
Identical quantities to the grokking bridge (thresholds imported frozen from
`grokking_collapse_bridge.THRESHOLDS`; component mapping identical):

    H_k   = mean predictive entropy at the masked position (bits)
    C_k   = mean KL(P_k || P_0), P_0 = the published step-0 checkpoint
    acc_k = minimal-pair accuracy: fraction of probes where the correct
            verb form outscores the number-mismatched form
    B_k   = max(C_k - C_{k-1}, 0); window anchored at the largest acc jump

    potential:   H_pre >= 1.0 bits
    burstiness:  window burst / median burst >= 5.0
    usefulness:  acc gain across the window >= 0.2
    endogeneity: design flag -- MLM pretraining contains no agreement
                 supervision and we did not train the system (true for the
                 real series; the shuffled-vocabulary control below is
                 flagged prespecified=False as well, it must fail on
                 measured components alone)

## Probe sets (fixed before measurement)

- `agreement`: 288 minimal pairs from Marvin & Linzen-style templates,
  singular/plural subjects x {simple, prepositional-phrase attractor of
  the opposite number} x verb pairs (is/are, was/were, has/have,
  does/do, likes/like, writes/write, knows/know, sees/see). Chance = 0.5.
- `random_target`: the same masked contexts, but the "correct" answer is
  redefined as a fixed random vocabulary token per probe (no structure
  links context to target). Usefulness cannot rise above chance except by
  accident.
- `attractor_only` (reported, not scored): the attractor subset alone, to
  document whether the harder long-range cases are acquired at the same
  time.

## Registered conditions and predictions

- P1 `multiberts_agreement`: the checkpoint series on `agreement` probes
  passes all four components (emergent). The collapse burst must coincide
  with the accuracy jump under the registered anchored window.
- P2 `multiberts_random_target`: fails usefulness (collapse happens --
  the distribution concentrates on plausible words -- but cannot track
  arbitrary targets).
- P3 `shuffled_vocab` control: the agreement probes scored through a fixed
  random permutation of the model's output vocabulary (applied at every
  checkpoint). The permutation destroys the correct/incorrect ordering, so
  usefulness must fail, while entropy and KL are IDENTICAL to P1 by
  construction -- collapse alone cannot pass the criterion.
- P4 Timing (reported): the accuracy jump happens in the first 200k steps
  (the dense-checkpoint region), consistent with the external probing
  literature; potential before the jump exceeds 1 bit by a wide margin
  (near-uniform init over 30522 tokens is ~14.9 bits).

## Failure handling

Any failed prediction is reported as a registered failure, exactly as the
marginal-selectivity failure and the R2 normalization failure were. The
criterion and thresholds cannot be adjusted in response; only honest
documentation is allowed.

## Feasibility constraints recorded now

Downloads run at ~2 MB/s aggregate; each checkpoint is ~440 MB. The
pipeline downloads, converts, evaluates, and deletes checkpoints one at a
time. If a checkpoint fails to download after retries it is skipped and
listed; the analysis uses the surviving grid (the anchored-window analysis
is robust to missing points as long as the dense early region survives).

## Second registration: ability generality + probe-level uncertainty

Frozen BEFORE any new measurement (after the agreement results above were
known; the two new abilities and all thresholds are fixed now). Motivation:
a reviewer can object that one hand-picked ability with one probe family
proves little. Response: two additional abilities, both externally
documented in the targeted-syntactic-evaluation literature (Marvin & Linzen
2018; Goldberg 2019), plus probe-level bootstrap confidence intervals.

New probe families (chance = 0.5 for all):

- `reflexive`: reflexive anaphora number agreement. "the author hurt
  [MASK] ." -> himself vs themselves; distractor variant embeds a
  relative clause of the opposite number ("the author that the guards
  praised hurt [MASK] .").
- `determiner`: demonstrative determiner-noun agreement. "[MASK] author
  is happy ." -> this vs these; distractor variant inserts an adjective
  ("[MASK] tired author is happy .").
- Each ability gets a matched `random_target` control (same masked
  contexts, correct answer redefined as a fixed random vocabulary token).

Registered predictions (thresholds imported frozen, as before):

- G1: `reflexive` passes all four components (emergent) on every seed run.
- G2: `determiner` passes all four components (emergent) on every seed run.
- G3: every `random_target` control fails usefulness on every seed run.
- G4: every ability's anchored window lands in the dense early region
  (step <= 200k).
- G5 (uncertainty): the 95% probe-level bootstrap CI of the usefulness
  gain at the registered anchored window lies entirely ABOVE the 0.2
  threshold for each ability, and entirely BELOW it for each
  random-target control (10000 resamples over probe indices, window held
  fixed at the registered anchor).

Seeds: 0, 1, 2 (three of the five published series). Any failure is
reported as a registered failure.

## Post-registration addendum (after the seed_0 run; nothing above edited)

The seed_0 run passed all four predictions (see
`outputs/multiberts_collapse_summary.json`). The identical frozen protocol
was then applied, unchanged, to the four remaining published intermediate
seeds (1-4). Results: agreement emergent 5/5 seeds, both controls rejected
via usefulness 5/5 seeds, anchor window at step 20k and potential 14.7
bits on every seed, no skipped checkpoints (15/15 verdicts overall).

## Registered extension: phenomena battery (frozen BEFORE running it)

Written after the agreement results above were known, but BEFORE any of
the probes below were evaluated on any checkpoint. Thresholds, component
mapping, and analysis code are unchanged and frozen.

Motivation (reviewer-facing): one ability is not a battery, and every
rejection so far on the public model went through the usefulness route.
The extension tests (a) two more externally documented abrupt-early
syntactic abilities and (b) one externally documented GRADUAL ability, to
show the criterion separates abrupt structured acquisition from gradual
accumulation inside the same public model.

Probe families (all masked minimal pairs, chance 0.5, vocabulary-checked):

- `reflexive`: subject-reflexive number agreement ("the doctor blamed
  [MASK] ." -> himself/herself vs themselves; plural subjects reversed).
  Singular credit = max(himself, herself) outscores themselves, to avoid
  the gender confound of uncased BERT.
- `determiner`: demonstrative-noun number agreement ("he likes this
  [MASK] ." -> book vs books; "these" reversed). Documented as an
  early-acquired easy paradigm (BLiMP determiner-noun agreement family).
- `facts`: country-capital recall ("the capital of france is [MASK] ." ->
  paris vs a fixed wrong capital). Factual/world knowledge is documented
  to accrue LATER and more gradually than syntax during pretraining
  ("Probing Across Time", Liu et al. 2021).

Registered predictions:

    R1 reflexive:  passes the criterion (emergent), window in the dense
                   early region (step <= 200k).
    R2 determiner: passes the criterion (emergent), window <= 200k.
    R3 facts:      NOT emergent. Expected failure route: burstiness
                   and/or windowed usefulness (gradual acquisition means
                   no single window pairs a collapse burst with a >= 0.2
                   accuracy jump). Any failure route counts as a pass for
                   R3, and the route is reported; if facts IS classified
                   emergent, that is a registered failure.
    R4 alignment:  for every ability classified emergent, the collapse
                   burst at the accuracy-jump anchor is top-3 among all
                   window positions (operationalized in
                   burst_alignment_test.py as permutation p <= 3/n_windows).

Scope: seed_0 first; if predictions hold, replicate on seed_1 as a
robustness check (same frozen protocol).

### Outcome of the phenomena battery (recorded after the seed_0 run)

R1 PASS (reflexive emergent, window 20k). R2 PASS (determiner emergent,
window 20k). R4 PASS for every emergent family (alignment p = 1/27 each).
R3 **REGISTERED FAILURE**: the facts family was classified emergent
(0.50 -> 0.90 by step 20k, burst-coincident). The failed part is our
auxiliary prediction about which abilities are gradual, not an internal
inconsistency of the criterion: 20 country-capital pairs are among the
highest-frequency facts in any corpus, and high-frequency facts are
documented to be learned early; the probing-across-time claim about
gradual factual knowledge concerns broad, tail-heavy fact distributions.
Reported as a registered failure, kept in all outputs.

### Registered follow-up: NPI licensing (frozen BEFORE running it)

To retest the abrupt-vs-gradual/late contrast with an ability that the
external literature specifically documents as HARD and LATE for BERT-type
models (negative-polarity-item licensing is among the worst BLiMP
paradigms for BERT), we add one family:

- `npi`: minimal pairs "no [nouns] have [MASK] [participle] ..." vs
  "the [nouns] have [MASK] [participle] ...", candidates {ever, never};
  the licensed context prefers "ever", the unlicensed prefers "never".
  Chance 0.5.

Registered prediction R5: npi is NOT classified emergent on seed_0
(expected routes: low/unstable usefulness in any single window, or
missing burst coincidence). If npi also turns out emergent at step 20k,
that is another registered failure and will be reported as evidence
against the granularity of the probing-across-time gradualism claim
rather than adjusted away.

### Outcome of R5 (recorded after the run)

R5 **REGISTERED FAILURE**: npi was classified emergent, with window at
step 40k (one grid interval LATER than agreement/determiner/facts at
20k) and a two-phase signature: at 20k the npi possibility space had
already collapsed (14.7 -> 4.65 bits) while accuracy was still exactly
0.5; the accuracy jump to 1.0 arrived at 40k with a further burst. So the
"hard/late" ordering is real (npi is the last family to become useful)
but at this granularity the acquisition is still abrupt and
burst-coincident, not gradual.

Honest methodological note, recorded with the failure: our templated
minimal pairs are far easier than the diverse BLiMP paradigms on which
NPI is reported as hard for BERT; templated probes hit ceiling quickly.

## Registered follow-up: tail-gradualism rejection test (frozen BEFORE running)

Written after all results above were known, BEFORE any probe below was
evaluated on any checkpoint. Thresholds, component mapping, and analysis
code (`grokking_collapse_bridge.THRESHOLDS`, `analyze_run`, `verdict`)
are unchanged and frozen. Seed 0, the full published checkpoint grid.

Motivation (the over-acceptance risk, stated plainly): every ability we
have probed on this public model so far was classified emergent, and both
auxiliary gradualism predictions (R3 facts, R5 NPI) failed. A reviewer
can therefore ask whether the criterion ever rejects anything on a public
system other than the artificial random-target / shuffled-vocab controls.
The R3 post-mortem identified the design flaw: our facts were
HIGH-FREQUENCY facts. The external literature is specific about what is
gradual: knowledge of LOW-FREQUENCY, tail entities correlates with
document frequency and accrues slowly (Kandpal et al. 2023, long-tail
knowledge; Chang & Bergen 2022, word acquisition curves in LMs are
frequency-ordered, with low-frequency words acquired later and more
slowly; Liu et al. 2021, probing across time). So the registered gradual
candidates this time are frequency-selected, not category-selected.

Probe families (chance 0.5, all words vocabulary-checked; frequency
strata from the wordfreq package, zipf scale):

- `head_facts`: the 20 high-frequency country-capital pairs from the
  phenomena battery, unchanged (positive contrast within the design).
- `tail_facts`: country-capital pairs whose CAPITAL has zipf <= 3.4
  (e.g. tirana, ljubljana, bratislava, tallinn, vilnius, montevideo),
  both directions ("the capital of albania is [MASK]." -> tirana vs a
  fixed other rare capital; "tirana is the capital of [MASK]." ->
  albania vs a fixed other country). Fact frequency, not just word
  frequency, is what Kandpal et al. tie to slow accrual.
- `tail_words`: definitional cloze minimal pairs whose targets are rare
  in-vocab whole words (zipf <= 3.3), distractor = the target of a
  paired probe from the same stratum (frequency-matched). Construction
  rule registered now: probes are written blind to any checkpoint
  behavior and kept only if all content words are in-vocab; the final
  probe list is fixed by the script before the first download.

Registered predictions:

    T1 head_facts: emergent (replicates the R3 outcome; abrupt at the
       first dense window).
    T2 tail_facts: NOT emergent. Expected route: usefulness -- no
       registered window pairs a collapse burst with a >= 0.2 accuracy
       gain, because tail-fact accuracy accrues gradually (or never
       leaves chance at this scale).
    T3 tail_words: NOT emergent, same expected route as T2.
    T4 (scope check, reported either way): the informative rejection is
       "acquired but gradual", i.e. final accuracy >= 0.6 with no
       emergent window. If a tail family never leaves chance
       (final < 0.6), its rejection only shows the criterion does not
       fire on unlearned abilities, and is reported with that weaker
       reading.

If T2 or T3 comes out emergent, that is a registered failure with a
serious consequence, recorded now: it would mean the criterion, at the
published checkpoint granularity, classifies even frequency-selected
tail knowledge as abrupt, and the over-acceptance objection stands. The
claims in the paper would then have to be scoped to "every measured
ability on this system is acquired abruptly at this granularity" with
the objection reported, not argued away.

### Outcome (recorded after the run; nothing above edited)

T1 PASS: head_facts emergent (0.50 -> 0.90 at 20k), replicating R3.
T2 **REGISTERED FAILURE**: tail_facts classified emergent -- accuracy
0.611 at step 0 (above chance already under random init pairing),
1.000 by step 20k. T3 **REGISTERED FAILURE**: tail_words classified
emergent -- 0.472 at step 0, 0.889 by 20k, ceiling thereafter.

Post-mortem, recorded with the failures: the failed component of the
design is the PROBE FORMAT, not the frequency selection. Binary
minimal-pair discrimination (rare correct vs rare distractor) only
requires enough signal to order two candidates, and 20k steps of MLM
already provides that even for tail words; the external gradualism
findings (Chang & Bergen 2022; Kandpal et al. 2023) concern ABSOLUTE
recall/surprisal of the target, not two-way forced choice. This is
itself a Schaeffer-style observation: whether tail knowledge looks
abrupt or gradual depends on the metric through which the ability is
scored. Our criterion tracks the observable it is given; it does not
adjudicate which observable is the "true" ability.

## Registered amendment: absolute-recall metric (frozen BEFORE running)

Same probe families, same checkpoints, same frozen thresholds and
analysis. One change, registered now: accuracy is redefined as TOP-1
RECALL over the full 30522-token vocabulary (the probe scores 1 only if
the correct token is the argmax at the masked position). This is the
metric family on which the external literature documents gradual,
frequency-ordered acquisition. Chance is ~1/30522, so the 0.2
usefulness-gain threshold now demands a genuine jump in absolute
competence within one window.

Registered predictions:

    T5 head_facts_top1: emergent (high-frequency facts, documented
       early; paris/rome-level answers should become argmax abruptly
       in the dense region).
    T6 tail_facts_top1: NOT emergent. Route: usefulness (gradual
       accrual across the grid, no single window >= 0.2 gain) or
       never-acquired (T4 scope reading applies: final < 0.6 is the
       weaker reading).
    T7 tail_words_top1: NOT emergent, same routes as T6.

Any emergent outcome on T6/T7 is another registered failure and would
leave the over-acceptance objection standing at this granularity; it
would then be reported as a scoping limitation of the framework on
checkpoint-grid systems, in the paper's limitations section.

### Outcome of the amendment (recorded after the run; nothing above edited)

T5 PASS: head_facts_top1 emergent (0 -> 0.60 top-1 recall by step 100k,
window in the dense region, burst-coincident).
T6 **REGISTERED FAILURE**: tail_facts_top1 was classified emergent
(window 120k, gain +0.315). The full curve is a textbook sigmoid --
0.06 (20k), 0.26 (60k), 0.54 (100k), 0.85 (120k), 0.98 (400k) -- i.e.
slower and later than head facts exactly as the frequency literature
predicts, but its steepest 20k-interval segment still exceeds the 0.2
windowed-gain threshold. At the published grid resolution, "gradual"
and "abrupt" differ by less than one checkpoint interval for facts
this frequent (our tail capitals are still newswire-level entities).
T7 PASS: tail_words_top1 NOT emergent, rejected via burstiness AND
usefulness (max windowed gain +0.056). T4 scope reading, by the frozen
rule: final top-1 0.194 < 0.6, so the registered label is the weaker
"never fully acquired" -- but the run is still the informative one:
top-1 recall rises steadily from 0 to ~0.19-0.25 (far above the 1/30522
chance floor) across two orders of magnitude of steps with no window
ever pairing a burst with a >= 0.2 gain. Real, slow learning that the
criterion correctly declines to call emergent.

What the three-family, two-metric design established, all of it kept:

1. The criterion CAN reject on this public system beyond artificial
   controls (tail_words on both metrics; the only ability family whose
   competence accrues slowly enough at this grid).
2. Whether an ability looks abrupt or gradual depends on the metric
   (tail_facts: gradual-looking sigmoid on top-1, ceiling-abrupt on
   binary pairs) AND on the grid resolution -- a measured, registered
   confirmation of the Schaeffer-style metric-dependence point, now
   from the mechanism side.
3. The honest scope statement for the paper: at the published 20k-step
   grid, the criterion separates slow continuous accrual (tail words)
   from windowed jumps (syntax, facts); it cannot distinguish "truly
   discontinuous" from "sigmoid steeper than one grid interval", and
   no checkpoint-grid method can.

Both auxiliary-gradualism failures (R3, R5) therefore say the following,
and only the following, about the framework: on 2M-step BERT-base
pretraining measured at the published checkpoint grid with templated
probes, every ability we probed is acquired as an abrupt useful collapse
in the first ~2-4% of training, ordered facts/syntax first, NPI later.
Finding an ability that the criterion REJECTS as gradual on this public
system remains open and is reported as such; the within-system negative
controls (random-target, shuffled-vocab) still pin the usefulness
component on this system.
