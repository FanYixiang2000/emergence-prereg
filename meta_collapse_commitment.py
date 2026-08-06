"""Meta-collapse: WHEN and HOW is the convention basin decided?

Follow-up to the retained BF-4 miss (a 25%-training snapshot predicts
the final convention at chance). The recursive question: convention
selection is itself a collapse of a possibility space -- the space of
conventions the learner could end up with. If so, our own instruments
should measure it: a per-seed COMMITMENT TIME (after which the
convention signature never changes), a population-level convention
entropy H_t that contracts as seeds commit, and an experience-based
mechanism for which basin wins.

Setup: crowd-vote domain at the confirmatory cost d = 2.0 (the cell
where 8/10 seeds went selective and 2/10 blind), 50 fresh seeds
(6501..6550), 30 checkpoints (every 1,000 of 30,000 episodes). The
convention signature at a checkpoint is deterministic from the Q
table: greedy mode at the cliff-visible state (dist<=1, on-lane,
cliff) x greedy mode at the far-field state (dist 4, on-lane, no
cliff). Classes: selective (democracy, anarchy), blind_dem
(democracy, democracy), blind_anarchy (anarchy, anarchy), inverse
(anarchy, democracy). Commitment time t_c = first checkpoint after
which the class never changes.

Per-episode experience counters up to each checkpoint (recorded during
training, no extra rollouts): ledge falls, field successes under
majority-anarchy hazard traversal, field successes under
majority-democracy.

Registered predictions (frozen before running):
    MC-1  Median commitment time > 25% of training (explains BF-4:
          most seeds are uncommitted at the snapshot).
    MC-2  The commitment-time distribution is broad, not clock-like:
          interquartile range >= 20% of training.
    MC-3  Mechanism: by their own commitment time, blind_dem seeds
          have accumulated FEWER field successes under
          majority-anarchy than selective seeds (median comparison) --
          the blind basin wins when the learner has not yet
          experienced that anarchy suffices in the field.
    MC-4  Population convention entropy H_t (over the four classes,
          across seeds) falls by more than half of its total decline
          AFTER the 25% checkpoint (late collapse at the population
          level too).

Misses are retained.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

import crowd_vote_domain as cv

OUTPUTS = Path(__file__).resolve().parent / "outputs"

N_SEEDS = 50
SEED_BASE = 6500
TRAIN_EPISODES = 30_000
CKPT_EVERY = 1_000
D_COST = 2.0

CLIFF_STATES = ((0, 0, 1), (1, 0, 1))
FAR_STATE = (4, 0, 0)
CLASSES = ("selective", "blind_dem", "blind_anarchy", "inverse")


def greedy_mode(q: Dict, states) -> str:
    for s in states if isinstance(states, tuple) and isinstance(
            states[0], tuple) else (states,):
        qa = q.get(s)
        if qa:
            return max(qa, key=qa.get)
    return "anarchy"


def signature(q: Dict) -> str:
    cliff = greedy_mode(q, CLIFF_STATES)
    far = greedy_mode(q, FAR_STATE)
    if cliff == "democracy" and far == "anarchy":
        return "selective"
    if cliff == "democracy" and far == "democracy":
        return "blind_dem"
    if cliff == "anarchy" and far == "anarchy":
        return "blind_anarchy"
    return "inverse"


def train_traced(seed: int) -> Dict:
    rng = random.Random(seed)
    q: Dict = {}
    sigs: List[str] = []
    counters = {"ledge_falls": 0, "field_success_anarchy": 0,
                "field_success_dem": 0}
    counter_trace: List[Dict] = []
    for episode in range(TRAIN_EPISODES):
        context = "ledge" if rng.random() < 0.5 else "field"
        ep = cv.Episode(context, rng.randrange(10 ** 9))
        eps = max(0.05, 0.5 * (1 - episode / TRAIN_EPISODES))
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
        hazard_modes = [m for pos, m in ep.mode_log
                        if pos in cv.HAZARD_BAND]
        dem_major = (hazard_modes and hazard_modes.count("democracy")
                     >= len(hazard_modes) / 2)
        if context == "ledge" and ep.outcome == "fall":
            counters["ledge_falls"] += 1
        if context == "field" and ep.outcome in ("success_fast",
                                                 "success_slow"):
            counters["field_success_dem" if dem_major
                     else "field_success_anarchy"] += 1
        if (episode + 1) % CKPT_EVERY == 0:
            sigs.append(signature(q))
            counter_trace.append(dict(counters))
    return {"sigs": sigs, "counters": counter_trace,
            "final_class": sigs[-1]}


def commitment_index(sigs: List[str]) -> int:
    """First checkpoint index after which the class never changes."""
    final = sigs[-1]
    idx = len(sigs) - 1
    for i in range(len(sigs) - 1, -1, -1):
        if sigs[i] != final:
            return i + 1
        idx = i
    return idx


def entropy(labels: List[str]) -> float:
    n = len(labels)
    return -sum((labels.count(c) / n) * math.log2(labels.count(c) / n)
                for c in set(labels))


def main() -> None:
    runs = []
    for k in range(N_SEEDS):
        seed = SEED_BASE + k + 1
        r = train_traced(seed)
        r["seed"] = seed
        r["t_commit"] = commitment_index(r["sigs"])
        runs.append(r)
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{N_SEEDS} seeds; last: "
                  f"{r['final_class']} commit at ckpt {r['t_commit']}",
                  flush=True)

    n_ckpt = TRAIN_EPISODES // CKPT_EVERY
    commits = [r["t_commit"] / n_ckpt for r in runs]
    med = float(np.median(commits))
    iqr = float(np.percentile(commits, 75) - np.percentile(commits, 25))

    # MC-3: experience at own commitment time
    fs_anarchy = {"selective": [], "blind_dem": []}
    for r in runs:
        cls = r["final_class"]
        if cls in fs_anarchy:
            c = r["counters"][max(0, r["t_commit"] - 1)]
            fs_anarchy[cls].append(c["field_success_anarchy"])
    mc3_applicable = (len(fs_anarchy["selective"]) >= 5
                      and len(fs_anarchy["blind_dem"]) >= 3)
    mc3 = (mc3_applicable
           and float(np.median(fs_anarchy["blind_dem"]))
           < float(np.median(fs_anarchy["selective"])))

    # MC-4: population entropy trace
    H = [entropy([r["sigs"][t] for r in runs]) for t in range(n_ckpt)]
    total_decline = H[0] - H[-1]
    q25 = n_ckpt // 4
    decline_after = H[q25] - H[-1]
    mc4 = total_decline > 0 and decline_after > 0.5 * total_decline

    classes = {c: sum(r["final_class"] == c for r in runs)
               for c in CLASSES}
    report = {
        "status": ("meta-collapse commitment battery; MC-1..MC-4 "
                   "frozen in the docstring; d = 2.0, 50 fresh seeds"),
        "final_classes": classes,
        "commitment_fractions": commits,
        "median_commitment": med,
        "iqr_commitment": iqr,
        "population_entropy_trace": H,
        "field_success_anarchy_at_commit": {
            k: sorted(v) for k, v in fs_anarchy.items()},
        "registered_outcomes": {
            "MC1_median_gt_0.25": f"{med:.2f} -> {med > 0.25}",
            "MC2_iqr_ge_0.20": f"{iqr:.2f} -> {iqr >= 0.20}",
            "MC3_blind_fewer_anarchy_field_successes":
                (f"{mc3}" if mc3_applicable
                 else "not applicable (class counts too small)"),
            "MC4_entropy_decline_mostly_after_25pct":
                f"{decline_after:.2f}/{total_decline:.2f} -> {mc4}",
        },
    }
    out = OUTPUTS / "meta_collapse_commitment.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print("final classes:", classes)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
