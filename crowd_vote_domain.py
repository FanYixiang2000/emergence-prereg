"""Crowd-vote aggregation domain: a fifth full six-component family
(collective control), inspired by the Twitch-Plays-Pokemon phenomenon.

Tens of thousands of humans once beat a video game by streaming raw
button presses through one shared avatar; the crowd invented a
context-dependent aggregation convention (chaotic throughput on open
ground, majority voting at precision passages). The historical event is
observational-scale only -- no do-operators, no initialization twin.
This domain reconstructs the PHENOMENON in a system where every
component is measurable.

Environment (episodic, two contexts drawn uniformly):
    field    a 12-step corridor, 3 lanes; stepping off-lane wastes a
             step but is harmless;
    ledge    the same corridor along a cliff: any off-lane step at
             positions 4..8 falls (episode fails).
Avatar actions come from N = 40 voters: 32 noisy goal-seekers (per-
voter lane noise 0.25) and 8 adversarial trolls (vote off-lane). The
SYSTEM's own action, per step, is the aggregation mode:
    anarchy     execute one uniformly sampled voter's action;
                1 tick per move (fast, noisy);
    democracy   execute the majority action over all voters;
                2 ticks per move (slow, precise).
Value = 10 * success - 0.25 * ticks used. Timeout at 40 ticks.

Learned system: a tabular Q meta-controller over (context-free local
features: distance-to-hazard band, lane offset) choosing the mode per
step; trained on value with no context label (the ledge is only
visible through the hazard-distance feature, as in Contextual LBF).

Trigger: majority-democracy aggregation within the hazard band
(positions 3..8; context-free, matching the basin definition). Basins over episode futures resolve the
convention: (hazard-band aggregation mode) x (success, fall, timeout),
six cells.
Interventions: do_commit = force democracy inside the hazard band;
do_block = forbid democracy (always anarchy). Contract: value above,
horizon = episode, rollout T = policy as trained.

Systems per seed (6): learned; initialization twin (untrained Q);
always_democracy (scripted); always_anarchy (scripted); scripted
context-switcher (hand-coded switch at the hazard band -- competent,
non-learned); BC clone of the scripted switcher (trained on its
demonstrations, provenance = imitation).

Thresholds copied unchanged from the frozen criterion: potential
>= 0.5 bits; selectivity >= 0.5; specificity JS >= 0.2 bits;
usefulness > 0; endogeneity (learned, not scripted/imitative);
acquisition >= 0.3 selectivity gain over the twin.

Registered predictions (frozen before the confirmatory run):
    CR-1 learned passes all six components on >= 8/10 seeds;
    CR-2 all 50 control verdicts (5 controls x 10 seeds) are
         rejections; scripted switcher and its BC clone fail exactly
         via endogeneity/acquisition; always_democracy fails
         selectivity; always_anarchy fails usefulness (via falls);
         twin fails selectivity and acquisition;
    CR-3 learned usefulness do-contrast positive on >= 9/10 seeds;
    CR-4 do-block reproduces the historical counterfactual: fall rate
         in ledge episodes rises by >= 0.3 absolute on every seed;
    CR-5 in field episodes the learned policy stays in anarchy >= 70%
         of steps (the convention is context-selective, not blanket
         caution).
Misses are retained.

DISCLOSED PILOT (design failure, quarantined): the first frozen run
(crowd_vote_domain_pilot1_contextblind.json) had two design errors
found by its own controls: (i) the trigger definition conditioned on
the context label, making selectivity trivially 1.0 for ANY policy
that votes in the hazard band (always_democracy scored 1.0 -- the
observer manufactured the selectivity); (ii) the feature map carried
no context-correlated observable, so no policy could genuinely
differentiate contexts. Redesign: the avatar SEES the cliff when
within 2 cells of a hazard position in a ledge episode (an observable,
like the CLBF context marker), and the trigger is context-free
(democracy in the hazard band, any context). Predictions CR-1..CR-5
re-frozen for the redesigned run; the pilot is retained as a
registered design failure (CR-0).

SECOND DISCLOSED PILOT (design failure, quarantined): the redesigned
run (crowd_vote_domain_pilot2_ortrigger.json) still defined the
trigger as ANY democracy step in the hazard band while the basin used
the MAJORITY hazard-band mode -- the mismatch made blanket-democracy
policies trigger at rate 1.0 in both contexts and zeroed the measured
do-block fall shift. Final spec: trigger = the basin's own
majority-mode event (definitional alignment, not a threshold change).
Outcomes of the final run are recorded as-is below; CR-1 and CR-5
misses are retained.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"

LENGTH = 12
HAZARD = range(4, 9)
HAZARD_BAND = range(3, 9)
N_VOTERS = 40
N_TROLLS = 8
VOTER_NOISE = 0.25
TICK_LIMIT = 40
FAST_TICKS = 18
SEEDS = list(range(6101, 6111))
TRAIN_EPISODES = 30_000
N_EVAL = 120

THRESHOLDS = {"potential_bits": 0.5, "selectivity": 0.5,
              "specificity_js": 0.2, "usefulness": 0.0,
              "acquisition": 0.3}

BASINS = tuple(f"{m}_{o}" for m in ("anarchy", "democracy")
               for o in ("success", "fall", "timeout"))
ACTIONS = ("anarchy", "democracy")


def votes(lane: int, rng: random.Random) -> List[int]:
    """Per-voter lane moves: -1, 0, +1 relative moves toward lane 1."""
    out = []
    for v in range(N_VOTERS):
        if v < N_TROLLS:
            out.append(rng.choice((-1, 1)))
        else:
            target = 0 if lane == 1 else (1 if lane < 1 else -1)
            if rng.random() < VOTER_NOISE:
                out.append(rng.choice((-1, 0, 1)))
            else:
                out.append(target)
    return out


def majority(vs: List[int]) -> int:
    counts = {a: vs.count(a) for a in (-1, 0, 1)}
    return max(counts, key=lambda a: (counts[a], -abs(a)))


def features(pos: int, lane: int, context: str) -> Tuple[int, int, int]:
    dist = min((abs(pos - h) for h in HAZARD), default=9)
    cliff_visible = int(context == "ledge" and dist <= 2)
    return (min(dist, 4), 0 if lane == 1 else 1, cliff_visible)


class Episode:
    def __init__(self, context: str, seed: int):
        self.context = context
        self.rng = random.Random(seed)
        self.pos = 0
        self.lane = 1
        self.ticks = 0
        self.done = False
        self.outcome: Optional[str] = None
        self.mode_log: List[Tuple[int, str]] = []

    def step(self, mode: str) -> None:
        self.mode_log.append((self.pos, mode))
        vs = votes(self.lane, self.rng)
        if mode == "democracy":
            dl = majority(vs)
            self.ticks += 2
        else:
            dl = vs[self.rng.randrange(N_VOTERS)]
            self.ticks += 1
        new_lane = min(2, max(0, self.lane + dl))
        if self.context == "ledge" and self.pos in HAZARD and new_lane != 1:
            self.done, self.outcome = True, "fall"
            return
        self.lane = new_lane
        if self.lane == 1:
            self.pos += 1
        if self.pos >= LENGTH:
            self.done = True
            self.outcome = ("success_fast" if self.ticks <= FAST_TICKS
                            else "success_slow")
        elif self.ticks >= TICK_LIMIT:
            self.done, self.outcome = True, "timeout"

    def value(self) -> float:
        return (10.0 if self.outcome in ("success_fast", "success_slow")
                else 0.0) - 0.25 * self.ticks


def run_episode(policy, context: str, seed: int,
                intervention: Optional[str] = None) -> Dict:
    ep = Episode(context, seed)
    triggered = False
    while not ep.done:
        f = features(ep.pos, ep.lane, context)
        mode = policy(f, ep.rng)
        if intervention == "do_commit" and ep.pos in HAZARD_BAND:
            mode = "democracy"
        if intervention == "do_block":
            mode = "anarchy"
        ep.step(mode)
    field_anarchy = (sum(1 for _, m in ep.mode_log if m == "anarchy")
                     / max(1, len(ep.mode_log)))
    hazard_modes = [m for pos, m in ep.mode_log if pos in HAZARD_BAND]
    hazard_mode = ("democracy" if hazard_modes
                   and hazard_modes.count("democracy")
                   >= len(hazard_modes) / 2 else "anarchy")
    triggered = hazard_mode == "democracy"
    outcome_class = ("success" if ep.outcome in ("success_fast",
                                                 "success_slow")
                     else ep.outcome)
    return {"basin": f"{hazard_mode}_{outcome_class}",
            "trigger": int(triggered),
            "value": ep.value(), "field_anarchy_frac": field_anarchy}


# ---------------------------------------------------------------- systems

def train_learned(seed: int) -> Dict:
    rng = random.Random(seed)
    q: Dict = {}
    for episode in range(TRAIN_EPISODES):
        context = "ledge" if rng.random() < 0.5 else "field"
        ep = Episode(context, rng.randrange(10 ** 9))
        eps = max(0.05, 0.5 * (1 - episode / TRAIN_EPISODES))
        history = []
        while not ep.done:
            f = features(ep.pos, ep.lane, context)
            if rng.random() < eps:
                a = rng.choice(ACTIONS)
            else:
                qa = q.get(f, {m: 0.0 for m in ACTIONS})
                a = max(qa, key=qa.get)
            history.append((f, a))
            ep.step(a)
        g = ep.value()
        for f, a in history:
            qa = q.setdefault(f, {m: 0.0 for m in ACTIONS})
            qa[a] += 0.1 * (g - qa[a])
    return q


def policy_from_q(q: Dict):
    def pol(f, rng):
        qa = q.get(f)
        if not qa:
            return "anarchy"
        return max(qa, key=qa.get)
    return pol


def scripted_switcher(f, rng):
    _dist, _lane, cliff = f
    return "democracy" if cliff else "anarchy"


def always_democracy(f, rng):
    return "democracy"


def always_anarchy(f, rng):
    return "anarchy"


def bc_clone(seed: int):
    """Clone of the scripted switcher trained on its demonstrations."""
    rng = random.Random(seed)
    table: Dict = {}
    for _ in range(4000):
        context = "ledge" if rng.random() < 0.5 else "field"
        ep = Episode(context, rng.randrange(10 ** 9))
        while not ep.done:
            f = features(ep.pos, ep.lane, context)
            a = scripted_switcher(f, rng)
            counts = table.setdefault(f, {m: 0 for m in ACTIONS})
            counts[a] += 1
            ep.step(a)

    def pol(f, rng2):
        counts = table.get(f)
        if not counts:
            return "anarchy"
        return max(counts, key=counts.get)
    return pol


# ---------------------------------------------------------------- metrics

def entropy_bits(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((k / total) * math.log2(k / total)
                for k in counts.values() if k > 0)


def js_bits(p: Dict[str, float], q: Dict[str, float]) -> float:
    m = {b: (p.get(b, 0) + q.get(b, 0)) / 2 for b in BASINS}

    def kl(x, y):
        return sum(x.get(b, 0) * math.log2(x.get(b, 0) / y[b])
                   for b in BASINS if x.get(b, 0) > 0 and y[b] > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist(rows: List[Dict]) -> Dict[str, float]:
    counts = {b: 0 for b in BASINS}
    for r in rows:
        counts[r["basin"]] += 1
    total = len(rows)
    return {b: counts[b] / total for b in BASINS}


def evaluate(policy, seed_offset: int) -> Dict:
    rows = {}
    for mode in (None, "do_commit", "do_block"):
        rows[mode] = []
        for ctx_i, context in enumerate(("field", "ledge")):
            for k in range(N_EVAL):
                r = run_episode(policy, context,
                                seed_offset + 10_000 * ctx_i + k, mode)
                r["context"] = context
                rows[mode].append(r)
    nat = rows[None]
    trig = {c: float(np.mean([r["trigger"] for r in nat
                              if r["context"] == c]))
            for c in ("field", "ledge")}
    counts = {b: 0 for b in BASINS}
    for r in nat:
        counts[r["basin"]] += 1
    mean_v = lambda rs: float(np.mean([r["value"] for r in rs]))
    ledge = [r for r in rows["do_block"] if r["context"] == "ledge"]
    nat_ledge = [r for r in nat if r["context"] == "ledge"]
    fall = lambda rs: float(np.mean([r["basin"].endswith("fall")
                                     for r in rs]))
    field_nat = [r for r in nat if r["context"] == "field"]
    return {
        "potential_bits": entropy_bits(counts),
        "trigger_rates": trig,
        "conditional_selectivity": abs(trig["ledge"] - trig["field"]),
        "specificity_js_bits": js_bits(dist(rows["do_commit"]),
                                       dist(rows["do_block"])),
        "usefulness_gap": mean_v(nat) - mean_v(rows["do_block"]),
        "natural_value": mean_v(nat),
        "fall_rate_shift_ledge": fall(ledge) - fall(nat_ledge),
        "field_anarchy_frac": float(np.mean(
            [r["field_anarchy_frac"] for r in field_nat])),
    }


def verdict(m: Dict, endogenous: bool, acquisition: float) -> Dict:
    passes = {
        "potential": m["potential_bits"] >= THRESHOLDS["potential_bits"],
        "conditional_selectivity":
            m["conditional_selectivity"] >= THRESHOLDS["selectivity"],
        "specificity":
            m["specificity_js_bits"] >= THRESHOLDS["specificity_js"],
        "usefulness": m["usefulness_gap"] > THRESHOLDS["usefulness"],
        "endogeneity": endogenous,
        "acquisition": acquisition >= THRESHOLDS["acquisition"],
    }
    return {"passes": passes, "emergent": int(all(passes.values())),
            "failed": [k for k, ok in passes.items() if not ok]}


def main() -> None:
    report = {"status": ("crowd-vote aggregation domain; CR-1..CR-5 "
                         "frozen in the docstring"), "seeds": {}}
    for seed in SEEDS:
        print(f"=== crowd seed {seed} ===", flush=True)
        q = train_learned(seed)
        twin_q: Dict = {}
        offset = 40_000_000 + seed * 100_000
        systems = {
            "learned": (policy_from_q(q), True),
            "initial_twin": (policy_from_q(twin_q), True),
            "always_democracy": (always_democracy, False),
            "always_anarchy": (always_anarchy, False),
            "scripted_switcher": (scripted_switcher, False),
            "bc_clone": (bc_clone(seed + 77), False),
        }
        metrics = {name: evaluate(pol, offset)
                   for name, (pol, _) in systems.items()}
        acq = (metrics["learned"]["conditional_selectivity"]
               - metrics["initial_twin"]["conditional_selectivity"])
        entry = {}
        for name, (pol, endo) in systems.items():
            a = acq if name == "learned" else 0.0
            entry[name] = {"metrics": metrics[name], "acquisition": a,
                           "verdict": verdict(metrics[name], endo, a)}
        report["seeds"][str(seed)] = entry
        v = entry["learned"]["verdict"]
        print(f"  learned verdict {v['emergent']} "
              f"failed {';'.join(v['failed']) or '-'} "
              f"useful {metrics['learned']['usefulness_gap']:.2f} "
              f"fall_shift {metrics['learned']['fall_rate_shift_ledge']:.2f}",
              flush=True)
        out = OUTPUTS / "crowd_vote_domain.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # registered tallies
    seeds = report["seeds"]
    learned_pass = sum(s["learned"]["verdict"]["emergent"]
                       for s in seeds.values())
    controls = ("initial_twin", "always_democracy", "always_anarchy",
                "scripted_switcher", "bc_clone")
    control_rej = sum(1 for s in seeds.values() for c in controls
                      if not s[c]["verdict"]["emergent"])
    useful_pos = sum(s["learned"]["metrics"]["usefulness_gap"] > 0
                     for s in seeds.values())
    fall_ok = sum(s["learned"]["metrics"]["fall_rate_shift_ledge"] >= 0.3
                  for s in seeds.values())
    field_ok = sum(s["learned"]["metrics"]["field_anarchy_frac"] >= 0.7
                   for s in seeds.values())
    report["registered_outcomes"] = {
        "CR1_learned_ge_8_of_10": f"{learned_pass}/10 -> {learned_pass >= 8}",
        "CR2_all_50_controls_rejected":
            f"{control_rej}/50 -> {control_rej == 50}",
        "CR3_useful_positive_ge_9": f"{useful_pos}/10 -> {useful_pos >= 9}",
        "CR4_doblock_fall_shift_ge_0.3_all":
            f"{fall_ok}/10 -> {fall_ok == 10}",
        "CR5_field_anarchy_ge_0.7_all":
            f"{field_ok}/10 -> {field_ok == 10}",
    }
    out = OUTPUTS / "crowd_vote_domain.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
