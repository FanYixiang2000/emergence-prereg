# Pre-registration: within-state useful collapse in a real strategic system (chess)

Written and frozen BEFORE any puzzle position was measured. A feasibility
pilot (12 positions) may adjust ESTIMATOR parameters only -- playout depth,
multipv width, sampling temperature, horizon, rollout count, worker count --
never the selection rules, the candidate-move definitions, the basin
definition boundaries, the thresholds, or the predictions below.

## Why chess (and why not Go, recorded honestly)

The roadmap lists Go/KataGo search trees as the external-system target. In
this environment KataGo is unavailable (binaries are GitHub-hosted and
GitHub is unreachable; no usable GPU). Chess with Stockfish 14.1 (Ubuntu
archive build, NNUE, verified tactically sound here) provides the same
epistemic content:

- a real strategic system we did not design, with a structured possibility
  space produced by human play;
- externally annotated key moves: the lichess puzzle database
  (`database.lichess.org`, ~4M puzzles extracted from real rated human
  games, theme-tagged by an external pipeline and community, including a
  `sacrifice` theme);
- a search tree from which future-outcome distributions can be estimated.

Neither the positions, nor the key-move labels, nor the engine are ours:
zero authorial control over the possibility space.

## What this experiment instantiates

This is the WITHIN-STATE (mechanism-level) instantiation of the framework,
the chess analogue of `within_episode_collapse_probe.py`: potential,
selectivity, specificity and usefulness are measured at a decision point of
a real system. The acquisition component does not apply (no training
series here); endogeneity holds by design (the positions arose in real
games; the legal-move possibility space is fixed by the rules of chess,
not by us) and is recorded as a design flag exactly as in the MultiBERTs
registration.

The target claim, in the language of the framework: an externally
annotated sacrificial key move is a trigger that is LOCALLY COSTLY
(immediate material loss, so no greedy account explains it) yet produces a
SELECTIVE, USEFUL collapse of the future-outcome distribution, which
near-alternative moves (including the shallow-greedy move and the best
deep alternative) do not produce. This is the "locally suboptimal,
retrospectively necessary" signature of the possibility-preservation tree,
measured in a system we did not construct.

## Observer possibility space (frozen definition, estimator params tunable)

For a position `s` with the puzzle side to move, the future-outcome
distribution `P(B | s)` is estimated by N stochastic playouts under an
imperfect-play observer: both sides sample each ply from a softmax over
the engine's multipv candidate scores (temperature in centipawns). After a
horizon of H plies (or earlier termination), the reached position is
scored by the engine at classification depth and mapped to one of five
outcome basins from the puzzle side's point of view:

    win    : eval >= +300 cp, or delivers mate
    adv    : +100 <= eval < +300
    equal  : -100 < eval < +100
    disadv : -300 < eval <= -100
    loss   : eval <= -300, or is mated

`P(B | s, a)` is estimated the same way after forcing move `a` first.
Initial estimator parameters (pilot may tune): multipv 4, playout depth 6,
temperature 150 cp, horizon 12 plies, N = 32 rollouts, classification
depth 12. The imperfect-play observer plays the same role as the
raised-temperature probe policy in the gridworld probes: it keeps
aleatoric openness visible instead of letting a deterministic oracle
collapse everything in advance.

## Position selection (frozen, deterministic)

Source file: `lichess_db_puzzle.csv.zst`, iterated in file order.

Sacrifice set (target n = 240): first 240 rows passing ALL of
  - `sacrifice` in Themes;
  - 1800 <= Rating <= 2400; NbPlays >= 1000;
  - len(Moves) >= 4 plies (a real combination, not a one-shot capture);
  - the key move does not deliver immediate checkmate;
  - position after the setup move is legal, puzzle side to move.
The position measured is FEN + Moves[0] (the setup/blunder move applied);
the KEY move is Moves[1], the externally annotated solution start.

Quiet control set (target n = 120): first 120 rows (disjoint from above)
passing ALL of
  - `middlegame` in Themes; NbPlays >= 1000;
  - the PRE-blunder position (FEN as published, before Moves[0]) has
    engine eval |cp| <= 60 at depth 12 (balanced), is not in check, and
    has >= 20 legal moves.
The position measured is the FEN itself. These are real, balanced,
non-critical states from the same game distribution.

## Candidate moves per position (frozen definitions)

- `key`     (sacrifice set only): the annotated solution move Moves[1].
- `deep_alt`: engine best move at depth 16 EXCLUDING `key` (the strongest
  deep alternative; on the quiet set, where there is no key, the engine
  best move itself is measured as `best`).
- `greedy`  : engine best move at depth 2 excluding `key` (the shallow /
  immediate-value move; a formalization of local greed).
- `random`  : a uniform random legal move excluding `key` (seed 20260706).

## Registered quantities

    potential(s)      = H(P(B | s)) in bits (5 basins, max ~2.32)
    collapse(s, a)    = H(P(B | s)) - H(P(B | s, a))
    useful_shift(s,a) = P(win | s, a) - P(win | s)
    specificity(s)    = JS(P(B | s, key), P(B | s, deep_alt)) in bits
    local_cost(a)     = material delta (pawn units, puzzle side POV,
                        standard values 1/3/3/5/9) after `a` and the
                        opponent's engine-best reply at depth 12,
                        relative to the pre-move material count.

## Registered predictions (thresholds frozen now)

- C1 LOCAL COST: on the sacrifice set, median `local_cost(key)` < 0
  (the annotated move really gives up material against best reply), and
  median `local_cost(greedy)` > median `local_cost(key)`. The trigger is
  not explainable as immediate greed.
- C2 USEFUL COLLAPSE: `useful_shift(key)` > `useful_shift(greedy)` on a
  majority of sacrifice positions, one-sided sign test p < 1e-3; same
  versus `random`.
- C3 SELECTIVITY: mean `P(win | key)` exceeds mean `P(win | deep_alt)` by
  >= 0.15 on the sacrifice set, with a one-sided sign test p < 1e-3
  (lichess solutions are unique-winning by construction, so even the best
  deep alternative should fail to reach the win basin; this is
  conditional selectivity measured in the wild).
- C4 POTENTIAL: median `potential(s)` >= 1.0 bits on the sacrifice set
  under the frozen observer (same 1.0-bit threshold as
  `grokking_collapse_bridge.THRESHOLDS['potential']`): before the key
  move, the future is genuinely multimodal for an imperfect player --
  the win is present but not yet inevitable.
- C5 QUIET CONTROL: on the quiet set, the best move produces no
  comparable useful collapse: mean `useful_shift(best)` < 0.10, and the
  sacrifice-set mean `useful_shift(key)` exceeds the quiet-set mean
  `useful_shift(best)` by >= 0.25. Balanced non-critical states offer no
  trigger whose collapse creates decisive value -- possibility collapse
  is a property of (state, action) structure, not of chess itself.

## Failure handling

Identical to previous registrations: any failed prediction is a
registered failure, reported with its route; thresholds and definitions
cannot be adjusted in response. Estimator-parameter changes made during
the pilot are documented in the script docstring with the pilot log kept
in `outputs/`.

## Outcome (recorded after the main run; nothing above edited)

Main run: 240 sacrifice + 120 quiet positions,
`outputs/chess_collapse_main_summary.json`, positions CSV alongside.

- C1 PASS. Median `local_cost(key)` = -3.0 pawns against best reply
  (80% of key moves strictly costly); median `local_cost(greedy)` = 0.0.
  The annotated triggers really are locally costly.
- C2 PASS. `useful_shift(key)` beats greedy on 238/240 positions
  (p = 1.6e-68) and random on 239/240 (p = 1.1e-72).
- C3 PASS. Mean `P(win | key)` = 0.777 vs `P(win | deep_alt)` = 0.238,
  gap 0.539 >= 0.15, key wins the sign test 240/240 (p = 5.7e-73).
  Mean specificity JS = 0.37 bits.
- C4 PASS. Median potential = 1.19 bits >= 1.0 under the frozen
  observer.
- C5 **REGISTERED FAILURE** on the frozen effect-size margin, via
  exactly the route recorded in the pilot note BEFORE the main run.
  The failed quantity is `useful_shift(key)` = P(win|s,key) - P(win|s):
  because the observer plays the key move endogenously in the base
  rollouts (that is what makes the trigger endogenous), P(win|s)
  already contains the key move's effect (mean 0.720 on the sacrifice
  set), so the SHIFT is small (0.057) even though the do-contrast is
  huge. The first half of C5 held (quiet best shift -0.021 < 0.10);
  the 0.25 margin between the two shifts did not (gap 0.078).
  The exogenous contrast that C3 froze -- forcing the key vs forcing
  the best alternative -- is 0.539 on sacrifice positions and
  essentially zero on quiet positions (quiet best move leaves
  P(win) at 0.27 vs base 0.29). Lesson recorded: in systems where the
  observer's policy already contains the trigger, useful collapse must
  be measured as a do-contrast between triggering and the best
  non-trigger alternative (as C3 does), not as a shift against the
  endogenous base. This mirrors the gridworld design, where
  `do_trigger` vs `do_non_trigger` is the registered contrast.

Net: 4/5 predictions pass; the C5 margin failure is a measurement-
definition lesson, not a counterexample to the framework, and is
reported wherever the chess result is cited.

## Robustness addendum (run after the main result; success criteria
stated in the script docstring before running)

`chess_robustness_grid.py`, same 360 positions, 12 perturbation cells:
observer temperature {200, 300, 450 cp} x playout depth {3, 4, 6},
basin thresholds {300/100 registered, 400/150, 500/200 cp}, and a
cross-engine cell (Stockfish 11, classical handcrafted evaluation,
pre-NNUE) replacing Stockfish 14.1 NNUE everywhere.

Result: the core conclusions (C1 cost order, C2 sign tests at p < 1e-3,
C3 do-contrast gap >= 0.15 with sign test) hold in 12/12 cells,
including the classical-evaluation engine (gap 0.430) and both
alternative basin cuts (gaps 0.531, 0.524). The C3 gap varies smoothly
with observer strength (0.398 at the softest observer to 0.678 at the
strongest) and never approaches the 0.15 threshold. C4 potential shows
exactly the monotone estimator dependence recorded in the pilot note:
median potential falls below 1.0 bits in the three temperature-200
cells (0.53-0.80 bits; stronger observers see less openness) and passes
in the other nine. The potential COMPONENT is therefore
estimator-scale-dependent in absolute value -- as every entropy
estimate under an observer model must be -- while the key-move
contrasts, the quiet-control contrast, and all sign conclusions are
estimator-, threshold-, and engine-robust.
