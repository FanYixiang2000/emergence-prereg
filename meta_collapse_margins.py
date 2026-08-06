"""Disclosed follow-up to the meta-collapse battery (MC-F).

The registered MC battery mostly failed, and the failure pattern
suggests its commitment measure was wrong: the hard-argmax signature
flips whenever near-tied Q values jitter, so "commitment" lands
trivially at the last jitter (median 0.97, IQR 0.03). This follow-up
distinguishes DECISION from JITTER with a soft signature: the Q-value
margin m_t = Q(democracy) - Q(anarchy) at the cliff state and at the
far-field state, traced every 250 episodes on 24 fresh seeds
(6601..6624, d = 2.0).

Declared analyses (fixed before running; descriptive follow-up, not a
re-score of MC-1..4, whose misses are retained):

    F-1  Jitter check: among checkpoints AFTER the last sign flip of
         the cliff margin, the median |margin| is at least 3x the
         median |margin| among checkpoints BEFORE it (a real decision
         grows a moat; jitter does not).
    F-2  Soft commitment time: first checkpoint where the cliff
         margin's sign equals its final sign AND |margin| exceeds the
         seed's own median |margin|, never returning below half that
         level. Report the distribution; declared expectation
         (median in [0.2, 0.8] of training, IQR >= 0.15) -- i.e.,
         with the jitter removed, commitment is mid-training and
         stochastic, unlike the clock-like artifact.
    F-3  The far-field margin separates the two basins: final
         far-field margin < 0 for selective seeds and > 0 for
         blind_dem seeds on >= 90% of seeds (the basin difference
         lives in the FIELD state, not the cliff state -- both
         conventions vote democracy at the cliff).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

import crowd_vote_domain as cv
from meta_collapse_commitment import CLIFF_STATES, FAR_STATE, signature

OUTPUTS = Path(__file__).resolve().parent / "outputs"

N_SEEDS = 24
SEED_BASE = 6600
TRAIN_EPISODES = 30_000
TRACE_EVERY = 250


def margin(q: Dict, states) -> float:
    if isinstance(states[0], tuple):
        for s in states:
            if s in q:
                qa = q[s]
                return qa.get("democracy", 0.0) - qa.get("anarchy", 0.0)
        return 0.0
    qa = q.get(states, {})
    return qa.get("democracy", 0.0) - qa.get("anarchy", 0.0)


def train_margins(seed: int) -> Dict:
    rng = random.Random(seed)
    q: Dict = {}
    cliff_m: List[float] = []
    far_m: List[float] = []
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
        if (episode + 1) % TRACE_EVERY == 0:
            cliff_m.append(margin(q, CLIFF_STATES))
            far_m.append(margin(q, FAR_STATE))
    return {"cliff": cliff_m, "far": far_m,
            "final_class": signature(q)}


def soft_commit(series: List[float]) -> float:
    """F-2 rule, returns fraction of training (1.0 = never commits)."""
    final_sign = 1.0 if series[-1] >= 0 else -1.0
    med = float(np.median(np.abs(series)))
    if med == 0:
        return 1.0
    n = len(series)
    for t in range(n):
        if (series[t] * final_sign > 0 and abs(series[t]) >= med
                and all(s * final_sign > 0.5 * med * final_sign
                        * (1 if final_sign > 0 else -1)
                        or s * final_sign > 0.5 * med
                        for s in series[t:])):
            return t / n
    return 1.0


def main() -> None:
    runs = []
    for k in range(N_SEEDS):
        seed = SEED_BASE + k + 1
        r = train_margins(seed)
        r["seed"] = seed
        runs.append(r)
        print(f"{k + 1}/{N_SEEDS}: {r['final_class']} "
              f"cliff_final {r['cliff'][-1]:+.2f} "
              f"far_final {r['far'][-1]:+.2f}", flush=True)

    # F-1 jitter check on the cliff margin
    ratios = []
    for r in runs:
        s = r["cliff"]
        final_sign = 1.0 if s[-1] >= 0 else -1.0
        flips = [t for t in range(1, len(s))
                 if (s[t] >= 0) != (s[t - 1] >= 0)]
        last_flip = flips[-1] if flips else 0
        before = [abs(x) for x in s[:last_flip]] or [0.0]
        after = [abs(x) for x in s[last_flip:]]
        med_b = float(np.median(before))
        med_a = float(np.median(after))
        ratios.append(med_a / med_b if med_b > 0 else float("inf"))
    f1_frac = float(np.mean([x >= 3 for x in ratios]))

    # F-2 soft commitment distribution
    commits = [soft_commit(r["cliff"]) for r in runs]
    med_c = float(np.median(commits))
    iqr_c = float(np.percentile(commits, 75)
                  - np.percentile(commits, 25))

    # F-3 far-field margin separates the basins
    ok, applicable = 0, 0
    for r in runs:
        if r["final_class"] == "selective":
            applicable += 1
            ok += int(r["far"][-1] < 0)
        elif r["final_class"] == "blind_dem":
            applicable += 1
            ok += int(r["far"][-1] > 0)
    f3_rate = ok / applicable if applicable else float("nan")

    classes = {}
    for r in runs:
        classes[r["final_class"]] = classes.get(r["final_class"], 0) + 1

    report = {
        "status": ("disclosed follow-up to MC-1..4 (misses retained); "
                   "soft-signature margins, F-1..F-3 declared in "
                   "docstring"),
        "final_classes": classes,
        "f1_moat_ratio_ge3_fraction": f1_frac,
        "f2_soft_commit_median": med_c,
        "f2_soft_commit_iqr": iqr_c,
        "f2_commits": commits,
        "f3_far_margin_separates": f"{ok}/{applicable} = {f3_rate:.2f}",
        "declared_outcomes": {
            "F1_moat_after_last_flip": f"{f1_frac:.2f} of seeds >= 3x",
            "F2_midtraining_stochastic":
                f"median {med_c:.2f} in [0.2,0.8]: "
                f"{0.2 <= med_c <= 0.8}; IQR {iqr_c:.2f} >= 0.15: "
                f"{iqr_c >= 0.15}",
            "F3_basin_lives_in_field_state":
                f"{f3_rate:.2f} >= 0.9: {f3_rate >= 0.9}",
        },
        "per_seed": [{"seed": r["seed"], "class": r["final_class"],
                      "cliff_final": r["cliff"][-1],
                      "far_final": r["far"][-1]} for r in runs],
    }
    out = OUTPUTS / "meta_collapse_margins.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["declared_outcomes"], indent=2))
    print("classes:", classes)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
