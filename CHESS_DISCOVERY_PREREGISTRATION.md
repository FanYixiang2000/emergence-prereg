# Prospective discovery on uncurated chess positions

Status: author-maintained protocol frozen BEFORE downloading game data or
scoring any position. Frozen: 2026-07-16. This is not a third-party
timestamped registered report.

## Why this experiment

Every existing chess result in the project measures *recovery*: externally
annotated sacrifice puzzles are re-scored by the frozen observer. The
predictable objection is that recovery on curated positions cannot show the
framework *finds* useful-collapse events. This experiment asks the framework
to perform *discovery*: on uncurated rated games, using only shallow/local
information, predict which positions contain a uniquely value-critical
decision, then have an independent deeper referee assign labels.

## Population (declared before any data is seen)

- Source: the public Lichess open database, standard rated games, monthly
  dump `lichess_db_standard_rated_2015-08.pgn.zst`, read from the start of
  the stream (chronological; no theme or tactic tags exist in this data).
- Game filter (population declaration, not outcome curation): both players
  rated 1800--2400 (the curated study's band); base time control >= 300
  seconds; game length >= 30 plies; standard variant; no abandoned games.
- Position sampling: one position per eligible game. Ply drawn uniformly
  from [16, min(60, n_plies - 8)] with the per-game RNG seeded by
  `GLOBAL_SEED=20260716` and the game's index. Positions where the side to
  move is in check or has fewer than 6 legal moves are re-drawn (max 5
  attempts, else the game is skipped) -- declared to keep the multipv-6
  observer well-defined.
- Main sample size: 400 scored positions. A feasibility pilot of up to 16
  positions may tune WORKER COUNT ONLY; scores from pilot positions are
  discarded from the main analysis.

## Predictor (frozen; identical shallow observer as the curated study)

Instrument constants imported unchanged from `chess_collapse_probe.py`:
MULTIPV=6, PLAYOUT_DEPTH=4, TEMPERATURE_CP=300, HORIZON_PLIES=12,
N_ROLLOUTS=32, basins {win,adv,equal,disadv,loss} at 300/100 cp.

Per position, using only shallow (depth-4) information:

- `potential` = H(P_0(B)) from natural softmax playouts;
- candidate move m* = multipv rank-1 move at depth 4;
- alternative a* = multipv rank-2 move at depth 4;
- `do_gap` = [P(win)+P(adv) | do m*] - [P(win)+P(adv) | do a*], forced first
  move then natural playouts (the curated study's do-operator);
- continuous discovery score = `do_gap`;
- binary discovery flag = (`potential` >= 1.0 bits) AND (`do_gap` >= 0.15).
  Both cutoffs are the curated registration's frozen C4/C3 values, reused
  unchanged.

## Independent referee label (computed after scoring; never fed back)

Deep engine analysis at depth 18, multipv 4, of the same position. Label
POSITIVE ("uniquely value-critical decision") iff

    eval(best) - eval(second best) >= 150 cp   AND   eval(best) >= -50 cp

from the side to move, mate scores clipped to +/-1000 cp. All other
positions are NEGATIVE.

## Baselines (each given the hindsight-optimal threshold, i.e. best case)

1. shallow eval gap: depth-4 multipv rank-1 minus rank-2 centipawns
   (the natural cheap alternative -- the hard baseline);
2. tactical density: number of legal captures plus checking moves;
3. absolute material imbalance;
4. absolute shallow eval of the position.

## Registered predictions

- CD1: AUROC(do_gap) >= 0.70 against the referee label.
- CD2: at the frozen binary flag, precision >= 2x the sample base rate.
- CD3: AUROC(do_gap) exceeds AUROC(tactical density) and AUROC(material).
  The shallow eval gap is reported head-to-head without a registered
  direction: it shares the depth-4 engine and is the honest strong baseline.
- CD4: among flag-positive positions, median potential >= 1.0 bits
  (discovered events sit in genuinely open positions, as in the curated
  study).

## Failure handling

Failed predictions are recorded as registered failures with routes; no
threshold, filter, sample or label rule may change after this freeze. If
fewer than 300 eligible positions can be scored (data or compute failure),
the run is reported as underpowered with whatever N was completed.

## Outcomes

(recorded 2026-07-16 after the main run; nothing above edited)

Pilot: 16 positions, worker count fixed at 24 for the main run; pilot scores
discarded. Main run: 400 uncurated positions (eligible indices 16-415),
referee base rate 0.095.

- CD1 PASS. AUROC(do_gap) = 0.730 >= 0.70.
- CD2 PASS. Frozen flag: 75/400 positions flagged, precision 0.24 =
  2.53x the 0.095 base rate; recall 0.47.
- CD3 PASS. AUROC(do_gap) 0.730 > tactical density 0.635 > material 0.589
  and > absolute shallow eval 0.565.
- CD4 PASS. Flagged median potential 1.92 bits >= 1.0: discovered events sit
  in genuinely open positions, replicating the curated-study structure
  without curation.
- Head-to-head (no registered direction, reported honestly): the shallow
  eval gap baseline scores AUROC 0.762, slightly above the collapse do-gap.
  This is expected by construction -- the referee label is itself a deep
  eval gap, so the same-family shallow eval gap is structurally the closest
  predictor. The registered claim was never that the collapse score beats a
  same-depth engine at predicting engine evaluations; it is that a
  mechanism-level quantity (future-basin distribution shift under forced
  moves, plus openness) prospectively discovers value-critical decisions in
  uncurated play at usable precision, which 4/4 registered predictions
  confirm.

4/4 registered predictions pass (`outputs/chess_discovery_main.json`).

Post-hoc referee sensitivity (labels only; scores untouched;
`chess_discovery_referee_sensitivity.json`): AUROC rises monotonically with
referee strictness (0.612 / 0.690 / 0.730 / 0.771 / 0.788 at
100/125/150/200/250 cp), and flag lift stays >= 2.27x at 125 cp and above
(1.78x at the loosest 100 cp label). The instrument specializes in strongly
value-critical decisions; the frozen 150 cp registered outcomes are interior
to this pattern, not a selected peak. Top-decile do-gap hit rate is 0.425 =
4.5x base rate.

## Replication on a second month (frozen 2026-07-16, BEFORE downloading any
2016-03 data; nothing above edited)

Identical protocol, filters, thresholds, referee and predictions CD1--CD4,
applied to `lichess_db_standard_rated_2016-03.pgn.zst` (a different year and
a post-dump-format-change month). GLOBAL_SEED for sampling: 20260717.
Sample: 400 positions after a 16-position worker-count pilot is skipped
(skip=16), exactly as in the primary run. Outcomes recorded below.

### Replication outcomes

(recorded 2026-07-16 after the run; nothing above edited)

400 positions from 2016-03, referee base rate 0.1175. 4/4 registered
predictions PASS again (`outputs/chess_discovery_replication_2016_03.json`):

- CD1 PASS: AUROC(do_gap) = 0.725 (primary: 0.730 -- stable across years).
- CD2 PASS: precision 0.347 = 2.95x base rate; recall 0.55.
- CD3 PASS: do_gap 0.725 > tactical density 0.708 > material 0.500.
- CD4 PASS: flagged median potential 1.98 bits.
- Head-to-head reversal worth reporting: the same-family shallow eval-gap
  baseline DROPS to 0.652 on this month (primary: 0.762), while the
  collapse do-gap is essentially unchanged. The mechanism-level score is
  stable across data distributions; the shallow engine-eval baseline is
  not. The primary run's baseline advantage was therefore not a stable
  property of the baseline.
