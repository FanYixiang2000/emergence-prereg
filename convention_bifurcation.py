"""Convention bifurcation battery: why do some seeds learn context-blind
conventions?

The anomaly (observed in two independent domains, retained as CR-1 and
the four rejected Overcooked seeds): roughly a third of training seeds
converge to a context-blind convention (same aggregation mode / same
role allocation in both contexts) instead of the context-selective
one, and fail the certificate through selectivity -- correctly. This
battery asks WHY, with the framework's own value algebra
(Proposition 3): the two conventions are attractor basins of training,
and the basin odds should be governed by the VALUE GAP between the
selective convention and the best context-blind one.

Manipulation: the democracy time cost d (ticks per democratic move) in
the crowd-vote domain, swept over {1.0, 1.5, 2.0, 2.5, 3.0}; the
confirmatory runs of crowd_vote_domain.py used d = 2. Small d makes
blanket democracy nearly free (the blind basin's value handicap
vanishes); large d makes context-blind democracy expensive in the
field context. Ten fresh seeds per cost (6201..6210 + 100*index).

Reference value gap, measured not assumed: for each d, the mean
natural value of the SCRIPTED selective switcher minus the scripted
always-democracy policy (200 episodes each, both contexts). This is
computable without any learner and predicts the learners' basin odds.

Registered predictions (frozen before running):
    BF-1  The fraction of selective seeds (final conditional
          selectivity >= 0.5) is non-decreasing in d across the grid
          (allowing ties; no decrease by more than 1 seed).
    BF-2  At d = 1.0 the selective fraction is <= 3/10; at d = 3.0 it
          is >= 7/10.
    BF-3  The sign of the measured reference value gap predicts the
          majority basin at >= 4/5 grid points.
    BF-4  Meta-collapse: the convention basin is decided early -- a
          snapshot at 25% of training predicts the final basin
          (hazard-band democracy preference in the ledge-visible
          state) on >= 70% of seeds, pooled over the grid.

Misses are retained. This battery EXPLAINS the retained CR-1 miss
mechanistically; it does not replace or re-score it.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np

import crowd_vote_domain as cv

OUTPUTS = Path(__file__).resolve().parent / "outputs"

COSTS = (1.0, 1.5, 2.0, 2.5, 3.0)
N_SEEDS = 10
SEED_BASE = 6200
SNAP_FRAC = 0.25
N_EVAL_REF = 200


class CostedEpisode(cv.Episode):
    """Episode with parametrized democracy tick cost."""

    def __init__(self, context: str, seed: int, dcost: float):
        super().__init__(context, seed)
        self.dcost = dcost

    def step(self, mode: str) -> None:
        self.mode_log.append((self.pos, mode))
        vs = cv.votes(self.lane, self.rng)
        if mode == "democracy":
            dl = cv.majority(vs)
            self.ticks += self.dcost
        else:
            dl = vs[self.rng.randrange(cv.N_VOTERS)]
            self.ticks += 1
        new_lane = min(2, max(0, self.lane + dl))
        if (self.context == "ledge" and self.pos in cv.HAZARD
                and new_lane != 1):
            self.done, self.outcome = True, "fall"
            return
        self.lane = new_lane
        if self.lane == 1:
            self.pos += 1
        if self.pos >= cv.LENGTH:
            self.done = True
            self.outcome = ("success_fast"
                            if self.ticks <= cv.FAST_TICKS
                            else "success_slow")
        elif self.ticks >= cv.TICK_LIMIT:
            self.done, self.outcome = True, "timeout"


def run_episode(policy, context: str, seed: int, dcost: float) -> Dict:
    ep = CostedEpisode(context, seed, dcost)
    while not ep.done:
        f = cv.features(ep.pos, ep.lane, context)
        ep.step(policy(f, ep.rng))
    return {"value": ep.value(), "outcome": ep.outcome}


def train(seed: int, dcost: float):
    """Q-learning as in the confirmatory domain, with a 25% snapshot."""
    rng = random.Random(seed)
    q: Dict = {}
    snapshot = None
    snap_at = int(cv.TRAIN_EPISODES * SNAP_FRAC)
    for episode in range(cv.TRAIN_EPISODES):
        if episode == snap_at:
            snapshot = copy.deepcopy(q)
        context = "ledge" if rng.random() < 0.5 else "field"
        ep = CostedEpisode(context, rng.randrange(10 ** 9), dcost)
        eps = max(0.05, 0.5 * (1 - episode / cv.TRAIN_EPISODES))
        history = []
        while not ep.done:
            f = cv.features(ep.pos, ep.lane, context)
            if rng.random() < eps:
                a = rng.choice(cv.ACTIONS)
            else:
                qa = q.get(f, {m: 0.0 for m in cv.ACTIONS})
                a = max(qa, key=qa.get)
            history.append((f, a))
            ep.step(a)
        g = ep.value()
        for f, a in history:
            qa = q.setdefault(f, {m: 0.0 for m in cv.ACTIONS})
            qa[a] += 0.1 * (g - qa[a])
    return snapshot, q


def selectivity_of(q: Dict, dcost: float, seed: int) -> float:
    pol = cv.policy_from_q(q)
    trig = {}
    for context in ("field", "ledge"):
        hits = []
        for k in range(60):
            ep = CostedEpisode(context, seed + k, dcost)
            while not ep.done:
                f = cv.features(ep.pos, ep.lane, context)
                ep.step(pol(f, ep.rng))
            modes = [m for pos, m in ep.mode_log
                     if pos in cv.HAZARD_BAND]
            hits.append(1.0 if modes and modes.count("democracy")
                        >= len(modes) / 2 else 0.0)
        trig[context] = float(np.mean(hits))
    return abs(trig["ledge"] - trig["field"])


def democracy_pref(q: Dict) -> int:
    """Snapshot signature: does the policy prefer democracy in the
    cliff-visible state (dist<=1, on-lane, cliff visible)?"""
    qa = q.get((1, 0, 1)) or q.get((0, 0, 1)) or {}
    if not qa:
        return 0
    return int(qa.get("democracy", 0.0) > qa.get("anarchy", 0.0))


def main() -> None:
    report = {"status": ("convention bifurcation battery; BF-1..BF-4 "
                         "frozen in the docstring"), "costs": {}}
    for ci, d in enumerate(COSTS):
        rows = []
        gap_vals = []
        for k in range(N_EVAL_REF):
            ctx = ("field", "ledge")[k % 2]
            sel = run_episode(cv.scripted_switcher, ctx,
                              900_000 + k, d)["value"]
            bl = run_episode(cv.always_democracy, ctx,
                             900_000 + k, d)["value"]
            gap_vals.append(sel - bl)
        ref_gap = float(np.mean(gap_vals))
        for k in range(N_SEEDS):
            seed = SEED_BASE + 100 * ci + k + 1
            snap, final = train(seed, d)
            sel_final = selectivity_of(final, d, seed + 50_000)
            selective = int(sel_final >= 0.5)
            early_pref = democracy_pref(snap) if snap else 0
            final_pref = democracy_pref(final)
            # early basin signature: prefers democracy at cliff AND
            # anarchy in the no-cliff state
            qa0 = (snap or {}).get((4, 0, 0), {})
            early_blind = int(qa0.get("democracy", 0.0)
                              > qa0.get("anarchy", 0.0))
            rows.append({"seed": seed, "selectivity": sel_final,
                         "selective": selective,
                         "early_cliff_dem": early_pref,
                         "early_field_dem": early_blind,
                         "final_cliff_dem": final_pref})
            print(f"d={d} seed {seed}: sel {sel_final:.2f} "
                  f"({'selective' if selective else 'blind'})",
                  flush=True)
        frac = sum(r["selective"] for r in rows) / N_SEEDS
        report["costs"][str(d)] = {
            "reference_value_gap_selective_minus_blanket": ref_gap,
            "selective_fraction": frac,
            "rows": rows,
        }
        out = OUTPUTS / "convention_bifurcation.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"cost {d}: ref gap {ref_gap:+.2f}, "
              f"selective {frac:.0%}", flush=True)

    fracs = [report["costs"][str(d)]["selective_fraction"] for d in COSTS]
    bf1 = all(fracs[i + 1] >= fracs[i] - 0.1 for i in range(len(fracs) - 1))
    bf2 = fracs[0] <= 0.3 and fracs[-1] >= 0.7
    bf3_hits = sum(
        1 for d in COSTS
        if (report["costs"][str(d)]
            ["reference_value_gap_selective_minus_blanket"] > 0)
        == (report["costs"][str(d)]["selective_fraction"] > 0.5))
    pooled = [r for d in COSTS for r in report["costs"][str(d)]["rows"]]
    # early signature: cliff-democracy AND NOT field-democracy => selective
    correct = sum(
        1 for r in pooled
        if int(r["early_cliff_dem"] and not r["early_field_dem"])
        == r["selective"])
    bf4_rate = correct / len(pooled)
    report["registered_outcomes"] = {
        "BF1_monotone_nondecreasing": bool(bf1),
        "BF2_endpoints": bool(bf2),
        "BF3_gap_sign_predicts_majority": f"{bf3_hits}/5 -> {bf3_hits >= 4}",
        "BF4_early_snapshot_predicts_basin":
            f"{bf4_rate:.2f} -> {bf4_rate >= 0.7}",
        "selective_fractions_by_cost": dict(zip(map(str, COSTS), fracs)),
    }
    out = OUTPUTS / "convention_bifurcation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
