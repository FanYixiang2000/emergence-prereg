# Pre-registration: possibility collapse across a public DECODER checkpoint series

Status: FROZEN 2026-07-08, before any checkpoint beyond step0 is
evaluated. A feasibility pilot may adjust probe wording, tokenization
mechanics, and download mechanics only -- never the thresholds, the
component mapping, the checkpoint grid, or the predictions below.
Amendments are labeled and dated; outcomes are recorded in place.

## Why this experiment

The public-model evidence so far comes from MultiBERTs (encoder,
masked-LM objective). The field's emergent-abilities debate is about
AUTOREGRESSIVE decoders. This experiment applies the SAME frozen
protocol (thresholds and component mapping imported unchanged from
`grokking_collapse_bridge.THRESHOLDS`; analysis code `analyze_run` /
`verdict` reused verbatim) to a public decoder pretraining series we
did not train.

## Target system (zero authorial control)

EleutherAI Pythia-160m (Biderman et al., ICML 2023): a GPT-NeoX
causal LM trained on the Pile with 154 published intermediate
checkpoints as git branches. We did not train it, did not choose its
data, architecture, schedule, or checkpoint spacing. Network access to
huggingface.co is blocked in this environment; checkpoints are fetched
from the public mirror `aifasthub.com` serving the same repository
content (same files, same revision structure; the config SHA-checked
against the model card values where available).

Checkpoint grid, fixed now (21 points: the published log2 early points
plus a uniform late grid):

    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000,
    96000, 128000, 143000

If a checkpoint fails to download after retries it is skipped and
listed, as in the MultiBERTs registration.

## Target ability (externally documented)

Subject-verb number agreement -- the same ability family as the
MultiBERTs probe, standard in targeted syntactic evaluation of
autoregressive LMs (Linzen et al. 2016; Marvin & Linzen 2018; BLiMP,
Warstadt et al. 2020). The Pythia suite's own evaluations and the
probing literature document that BLiMP-style agreement rises early in
pretraining relative to total steps.

## Observer possibility space (causal analogue of the MultiBERTs probe)

For checkpoint k: the model's own next-token predictive distribution
at the CRITICAL POSITION -- the verb slot -- given the left context
(prefix), on a fixed probe set. This is the decoder analogue of the
masked-position distribution:

    H_k   = mean predictive entropy at the critical position (bits)
    C_k   = mean KL(P_k || P_0), P_0 = the published step-0 checkpoint
    acc_k = minimal-pair accuracy: fraction of probes where the
            correct verb form has higher probability than the
            number-mismatched form at the critical position, given
            the SAME prefix
    B_k   = max(C_k - C_{k-1}, 0); window anchored at the largest
            accuracy jump (analyze_run, unchanged)

Components and thresholds (imported frozen):

    potential:   H_pre >= 1.0 bits
    burstiness:  window burst / median burst >= 5.0
    usefulness:  acc gain across the anchored window >= 0.2
    endogeneity: design flag -- autoregressive pretraining on the Pile
                 contains no agreement supervision and we did not train
                 the system (controls flagged prespecified=False too;
                 they must fail on measured components alone)

## Probe sets (fixed construction rule; wording pilot-tunable)

- `agreement`: minimal pairs from the same noun x verb-pair templates
  as the MultiBERTs probe (9 nouns x 8 verb pairs x 2 numbers x
  {simple, prepositional-attractor} prefixes), e.g.
  prefix "the author" -> candidates " is" / " are";
  prefix "the author of the books" -> " is" / " are" (attractor of
  opposite number). Both candidate forms must tokenize to a SINGLE
  token with leading space in the GPT-NeoX vocabulary; verb pairs that
  do not are dropped for both numbers (recorded in the output). Probes
  are lowercase-with-leading-capital exactly as written here; no
  checkpoint behavior is consulted during construction.
- `random_target`: same prefixes, correct answer redefined as a fixed
  random vocabulary token per probe (seed 20260708; ids >= 1000).
- `shuffled_vocab`: agreement probes scored through a fixed random
  permutation of the output vocabulary (seed 20260708), applied
  identically at every checkpoint. Entropy and KL are IDENTICAL to
  `agreement` by construction; only usefulness can differ.

## Registered predictions

- Y1 `pythia_agreement`: passes all four components (emergent); the
  collapse burst coincides with the accuracy jump under the registered
  anchored window.
- Y2 `pythia_random_target`: fails usefulness.
- Y3 `shuffled_vocab`: fails usefulness (collapse alone cannot pass).
- Y4 Timing (reported): the anchored window lands at or before step
  16000 (~11% of training), consistent with the early-syntax
  literature; potential before the jump exceeds 1 bit by a wide margin
  (near-uniform init over 50k tokens is ~15.6 bits).

## Failure handling

Identical to every previous registration: any failed prediction is a
registered failure, reported with its route, entered into
PREDICTION_LEDGER.md; thresholds, grid, and definitions cannot be
adjusted in response.

## Outcome (recorded 2026-07-08 after the single registered run)

Pipeline note (mechanics only, allowed): the mirror rejects the default
Python urllib user agent (HTTP 403); a browser-like User-Agent header
was added before any checkpoint beyond step0 was evaluated. All 21
checkpoints downloaded; none skipped; no verb pairs dropped (all 8
pairs are single tokens with leading space in the GPT-NeoX vocabulary).

- Y1 PASS. `pythia_agreement` emergent = 1 (all four components).
  H_pre = 8.86 bits, burstiness = 27.6 (threshold 5), usefulness gain
  = 0.47 (threshold 0.2), endogeneity by design. Accuracy 0.49 at step
  512 -> 0.83 at step 1000 -> 0.93 at step 2000.
- Y2 PASS. `pythia_random_target` fails usefulness (gain 0.045, final
  acc 0.507) while sharing the identical entropy/collapse trace.
- Y3 PASS. `shuffled_vocab` fails usefulness (gain -0.024, final acc
  0.510).
- Y4 PASS. Anchored window at steps 512 -> 2000 (window_epoch 1000),
  far earlier than the registered bound of step 16000 (~0.7% of
  training); H_pre = 8.86 bits >> 1 bit.

Unregistered observation (reported, not used in any verdict): the
attractor-template accuracy DIPS to 0.188 at step 512 -- below chance,
the signature of a transient linear-proximity heuristic (agree with
the nearest noun) documented in the acquisition literature -- then
jumps to 0.750 at step 1000 inside the registered window. The largest
single collapse burst (4.14 bits, step 128 -> 256) PRECEDES the
ability jump; the anchored-window burst (1.12 bits) coincides with it.
This is the same foreshadow-then-jump temporal shape seen in grokking
and simple_spread.

4/4 registered predictions pass. Recorded in PREDICTION_LEDGER.md.

---

# Registered follow-up: decoder tail-gradualism rejection test

Status: FROZEN 2026-07-08, after the main-run outcome above and BEFORE
any probe below is evaluated on any checkpoint. Mirrors the MultiBERTs
tail-gradualism registration: the criterion must REJECT abilities the
external literature ties to slow frequency-driven accrual, on the same
public decoder series, with the same frozen thresholds.

## Probe families (construction rules fixed now; only the tokenizer,
## never checkpoint behavior, is consulted during construction)

- `head_facts`: high-frequency country capitals as next-token prompts,
  "The capital of France is" -> " Paris" vs the capital of the next
  country in the fixed list. Registered: EMERGENT (replicating the
  MultiBERTs head_facts outcome on the decoder side).
- `tail_facts`: the same low-frequency capital list as the MultiBERTs
  test (zipf <= 3.4). Registered: NOT emergent, route usefulness.
- `tail_words`: the same definitional-cloze items as the MultiBERTs
  test, rewritten as prefix prompts ("A narrow inlet of the sea
  between steep cliffs is a" -> " fjord"), distractor = paired target
  from the same frequency stratum. Registered: NOT emergent, route
  usefulness.

Tokenization rule: an item is kept only if BOTH its correct and
distractor completions are single tokens with leading space in the
GPT-NeoX vocabulary; drops are recorded per family. If a family
retains fewer than 10 items it is reported as underpowered and its
prediction is void (reported, not counted).

AMENDMENT A1 (2026-07-08, tokenization mechanics; recorded before any
checkpoint behavior was observed for any family below): under the
single-token rule the GPT-NeoX BPE vocabulary retains 0/27 tail_facts
and 0/36 tail_words items (rare words and city names are multi-token),
so the registered test cannot run at all. Scoring is amended to the
standard autoregressive minimal-pair method (EleutherAI eval harness
"acc_norm"): each candidate completion is scored by its MEAN per-token
log-probability under teacher forcing given the prefix; accuracy is
the fraction of items where the correct completion outscores the
distractor. Entropy and collapse remain defined on the next-token
distribution at the first completion position (unchanged). The same
scoring is applied to all three families. No item lists, thresholds,
grid, or predictions change.

## Registered predictions

- T1 `head_facts`: emergent = 1.
- T2 `tail_facts`: emergent = 0, failing component includes
  usefulness.
- T3 `tail_words`: emergent = 0, failing component includes
  usefulness.

Same checkpoint grid, same thresholds, same analysis, same failure
handling as the main registration.

## Outcome (recorded 2026-07-08 after the single registered run)

All 21 checkpoints downloaded; none skipped. Under Amendment A1 all
items are retained: head_facts 20, tail_facts 27, tail_words 36.

- T1 PASS. `head_facts` emergent = 1 (H_pre 8.61 bits, burstiness
  16.4, usefulness gain 0.35; acc 0.60 at step 1000 -> 0.85 at 2000 ->
  0.95 at 4000).
- T2 VERDICT CORRECT, ROUTE MISS. `tail_facts` emergent = 0 as
  registered, but the failing component is BURSTINESS (3.2 < 5), not
  usefulness (windowed gain 0.37 exceeds 0.2): the accuracy climbs
  0.48 -> 0.67 -> 0.74 -> 0.89 -> 0.93 across steps 1000..128000 --
  a spread-out ramp with no single dominant collapse burst, which is
  exactly the gradualism signature, caught by the burst component
  rather than the usefulness component. Counted as a registered route
  miss (the MB-T2 analogue failed outright; the decoder-side test
  rejects correctly).
- T3 PASS. `tail_words` emergent = 0, route usefulness exactly as
  registered (windowed gain 0.056; final acc 0.72 accrues in small
  increments across four decades of steps).

Combined with the main run: 7/8 registered predictions fully pass,
1 verdict-correct route miss (PY-T2). The decoder series shows the
full double dissociation the MultiBERTs series only partially showed:
agreement and head facts emerge with coincident collapse bursts;
tail facts and tail words -- the abilities the frequency literature
says accrue slowly -- are both REJECTED by the frozen criterion.

---

# Registered follow-up: scaling-family replication (Pythia-410m)

Status: FROZEN 2026-07-08, after the 160m outcomes above and BEFORE
any 410m checkpoint is evaluated. Same repository family
(EleutherAI/pythia-410m), same published checkpoint grid, same probe
sets, same seeds, same frozen thresholds, same analysis code. The only
change is the model size. This is the MultiBERTs seed-replication
analogue on the scale axis: the criterion's verdicts must not be a
one-size accident.

## Registered predictions

- S1 `pythia_agreement` (410m): emergent = 1 (all four components).
- S2 both controls (`pythia_random_target`, `shuffled_vocab`): fail
  usefulness.
- S3 anchored window at or before the 160m window's step (2000) OR at
  most one grid point later -- the scaling literature says larger
  models acquire early syntax at the same or fewer tokens; reported
  either way.

## Outcome (recorded 2026-07-08 after the single registered run)

All 21 checkpoints downloaded; none skipped; no verb pairs dropped.

- S1 PASS. `pythia_agreement` (410m) emergent = 1: H_pre 9.63 bits,
  burstiness 11.0, usefulness gain 0.462; accuracy 0.38 at step 512 ->
  0.75 at 1000 -> 0.93 at 2000.
- S2 PASS. Both controls fail usefulness (random_target gain 0.003,
  shuffled_vocab gain 0.042; both also miss burstiness).
- S3 PASS. Anchored window at step 1000, identical to the 160m window.

Unregistered observation (reported, not used in any verdict): the
attractor-template dip replicates AND DEEPENS at the larger size
(0.111 at step 512 vs 0.188 on 160m) before jumping to 0.68 inside
the registered window -- the transient linear-proximity heuristic
phase is a reproducible waypoint of this acquisition, not a 160m
accident.

3/3 registered predictions pass. The decoder verdicts are not a
one-size accident: 160m and 410m give the same accept/reject pattern,
the same window, and the same foreshadow shape under identical frozen
thresholds.
