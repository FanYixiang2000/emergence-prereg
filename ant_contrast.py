"""Ant double-bridge contrast: individual navigation vs collective
stigmergy, with a MEASURED gradual possibility collapse.

Answers two sharp reviewer questions with frozen predictions and
numbers, using the SAME coordinates and thresholds as
emergence_coordinates.py (copied, never retuned here):

  Q1  \"An ant bridge is built bit by bit -- where is the possibility
      collapse? Must collapse be abrupt?\"
  Q2  \"Is an ant finding food around an obstacle not emergence?\"

DESIGN (Deneubourg double-bridge, the textbook swarm exemplar):
  A nest and a food source are joined by TWO equal-length routes
  around a central obstacle (route A = go around one side, route B =
  the other). Ants make repeated nest->food->nest trips; on each trip
  an ant commits to route A or B at the branch. Identical individual
  rules in both regimes EXCEPT the interaction channel:

    SOLO   no pheromone. Each ant navigates around the obstacle on its
           own; with two equal detours it has no individual preference
           (p = 1/2). This IS \"an ant finding food around an
           obstacle\" -- individual navigation/adaptation.
    TRAIL  ants deposit and follow pheromone (Deneubourg rule
           p_A = (k+ph_A)^a / [(k+ph_A)^a + (k+ph_B)^a]). Positive
           feedback breaks the A/B symmetry: the colony CONSOLIDATES
           onto one route. Classic collective decision.

MACRO STRUCTURE Z (the possibility space of interest = distribution
over which route the COLONY commits to):
  Z = 1 iff, in the second half of the episode, one route carries a
      dominant share (consolidation = |f_B - f_A| >= 0.6).
  This is a consensus/symmetry-breaking observable an individual
  colony CAN or CANNOT reach -- SOLO can reach it in principle, it
  simply does not, which is what makes its low score meaningful rather
  than definitional.

COORDINATES (frozen thresholds N>=0.3, D>=0.5, R>=0.6):
  D  relative loss of consolidation under a marginal-preserving
     surrogate decoupling of the pheromone read (deposition kept, the
     current-choice<->current-trail coupling broken) -- the do-operator
     on the coupling, exactly as in emergence_coordinates.run_units.
  R  recovery ratio of consolidation after an IRRELEVANT micro
     perturbation (a burst of forced-random individual choices at
     mid-episode; the trail persists, the attractor should reform).
  N  co-information of three individual ants' route preferences about
     Z (reported descriptively; for many-body swarms N is redundancy-
     dominated and, exactly as in emergence_coordinates.py for the
     Kuramoto/Life held-out families, the weak-emergence VERDICT is
     taken on D and R, not N).
  C(t)  windowed Shannon entropy of route choice over trips -- the
     gradual-collapse series (1 bit = routes still open, 0 = collapsed).

REGISTERED PREDICTIONS (frozen before running):
  ANT-1 (Q2)  SOLO: consolidation rate < 0.3 and D < 0.5 -> individual
        obstacle navigation is NOT collective weak emergence under the
        frozen coordinates. (It is still individual adaptation; the
        coordinates separate the two.)
  ANT-2       TRAIL: D >= 0.5 and R >= 0.6 -> collective stigmergy IS
        weak emergence, reproducing the double-bridge literature and
        matching the held-out-family verdict rule weak(D,R).
  ANT-3 (Q1)  Gradual, non-abrupt collapse: track the colony's route-
        commitment dev(t) = |p_A(t) - 1/2| * 2 (0 = both routes open,
        1 = fully committed), where p_A is the model's actual choice
        probability. An ABRUPT collapse would happen in a single step:
        10%-90% span ~ 0 and one trip carrying ~all of it
        (max_step_frac ~ 1). The GATE for "gradual" is therefore the
        span: in TRAIL the colony commits (final dev >= 0.5) and the
        10%-90% collapse is SPREAD OVER >= 10% of the foraging horizon.
        max_step_frac is reported as corroboration (a true step -> ~1).
        SOLO never collectively commits (final dev < 0.3).

Misses are retained.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"

N_ANTS = 12
N_TRIPS = 500
WINDOW = 40
# Deneubourg double-bridge dynamics (standard form): small additive
# constant K, alpha=2 nonlinearity, slow evaporation so the trail
# accumulates to K << pheromone and the colony locks in. These are
# model parameters, NOT the frozen coordinate thresholds/prediction
# bounds below.
K, ALPHA, RHO, Q = 5.0, 2.0, 0.01, 1.0
TH = {"N": 0.3, "D": 0.5, "R": 0.6}   # copied from emergence_coordinates.py
N_EP = 60


def binary_entropy(f: float) -> float:
    if f <= 0.0 or f >= 1.0:
        return 0.0
    return -(f * math.log2(f) + (1 - f) * math.log2(1 - f))


def mi_from_counts(joint: Dict) -> float:
    total = sum(joint.values())
    if total == 0:
        return 0.0
    px, pz = {}, {}
    for (x, z), c in joint.items():
        px[x] = px.get(x, 0) + c
        pz[z] = pz.get(z, 0) + c
    out = 0.0
    for (x, z), c in joint.items():
        p = c / total
        out += p * math.log2(p / ((px[x] / total) * (pz[z] / total)))
    return out


def run_bridge(mode: str, seed: int, shuffle: bool = False,
               perturb: bool = False) -> Dict:
    """One colony episode on the double bridge.

    shuffle: marginal-preserving decoupling of the pheromone read
             (randomly swap the two route readings each trip) -> breaks
             positive feedback while keeping deposition marginals.
    perturb: irrelevant micro perturbation -- a burst of forced-random
             individual choices around mid-episode; the trail is left
             intact, so a true collective attractor recovers.
    """
    rng = np.random.default_rng(seed)
    phA, phB = 1.0, 1.0
    choices: List[int] = []
    ent_series: List[float] = []
    p_series: List[float] = []
    per_ant = [[0, 0] for _ in range(N_ANTS)]  # [A_count, B_count]
    pwin = range(N_TRIPS // 2, N_TRIPS // 2 + 40)

    for t in range(N_TRIPS):
        ant = t % N_ANTS
        forced_random = perturb and t in pwin
        if mode == "SOLO" or forced_random:
            p = 0.5
        else:
            a = (K + phA) ** ALPHA
            b = (K + phB) ** ALPHA
            if shuffle and rng.random() < 0.5:
                a, b = b, a
            p = a / (a + b)
        p_series.append(p)
        c = 0 if rng.random() < p else 1
        choices.append(c)
        per_ant[ant][c] += 1
        phA *= (1 - RHO)
        phB *= (1 - RHO)
        if c == 0:
            phA += Q
        else:
            phB += Q
        if t >= WINDOW:
            fB = sum(choices[-WINDOW:]) / WINDOW
            ent_series.append(binary_entropy(fB))

    second = choices[N_TRIPS // 2:]
    consolidation = abs(sum(second) / len(second) - 0.5) * 2.0
    tail = choices[int(N_TRIPS * 0.75):]
    tail_cons = abs(sum(tail) / len(tail) - 0.5) * 2.0
    # per-ant preference sector: 0 mostly-A, 1 mixed, 2 mostly-B
    prefs = []
    for a_c, b_c in per_ant:
        tot = a_c + b_c
        fb = b_c / tot if tot else 0.5
        prefs.append(0 if fb < 0.33 else (2 if fb > 0.67 else 1))
    return {
        "Z": int(consolidation >= 0.6),
        "consolidation": float(consolidation),
        "tail_recovered": int(tail_cons >= 0.6),
        "ent_series": ent_series,
        "p_series": p_series,
        "prefs": tuple(prefs[:3]),
    }


def measure_consolidation_rate(mode: str) -> float:
    return float(np.mean([run_bridge(mode, 1000 + k)["Z"]
                          for k in range(N_EP)]))


def measure_D(mode: str) -> float:
    nat = np.mean([run_bridge(mode, 2000 + k)["consolidation"]
                   for k in range(N_EP)])
    brk = np.mean([run_bridge(mode, 2000 + k, shuffle=True)["consolidation"]
                   for k in range(N_EP)])
    return float((nat - brk) / nat) if nat > 0.05 else 0.0


def measure_R(mode: str) -> float:
    base = np.mean([run_bridge(mode, 3000 + k)["Z"] for k in range(N_EP)])
    if base < 0.05:
        return 0.0
    rec = np.mean([run_bridge(mode, 3000 + k, perturb=True)["tail_recovered"]
                   for k in range(N_EP)])
    return float(min(1.0, rec / base))


def measure_N(mode: str) -> float:
    joint: Dict = {}
    singles = [dict() for _ in range(3)]
    for k in range(200):
        ep = run_bridge(mode, 5000 + k)
        z = ep["Z"]
        x = ep["prefs"]
        joint[(x, z)] = joint.get((x, z), 0) + 1
        for i in range(3):
            singles[i][(x[i], z)] = singles[i].get((x[i], z), 0) + 1
    return mi_from_counts(joint) - sum(mi_from_counts(s) for s in singles)


def gradualism(mode: str) -> Dict:
    """Route-commitment dev(t)=|p_A-1/2|*2 averaged over episodes.

    dev rises from 0 (both routes open) toward 1 (colony committed).
    A gradual collapse is spread over many trips with no single
    dominant step; an abrupt one is a near-step jump.
    """
    devs = []
    ents = []
    for k in range(N_EP):
        ep = run_bridge(mode, 4000 + k)
        devs.append([abs(p - 0.5) * 2.0 for p in ep["p_series"]])
        ents.append(ep["ent_series"])
    dev = np.mean(devs, axis=0)
    ent = np.mean(ents, axis=0)
    total = float(dev[-1] - dev[0])
    if total > 1e-6:
        lo = dev[0] + 0.1 * total
        hi = dev[0] + 0.9 * total
        t10 = int(np.argmax(dev >= lo))
        t90 = int(np.argmax(dev >= hi))
        span_frac = (t90 - t10) / len(dev)
        max_step_frac = float(np.max(np.diff(dev)) / total)
    else:
        span_frac = 0.0
        max_step_frac = 0.0
    return {
        "dev0": float(dev[0]),
        "dev_final": float(dev[-1]),
        "total_collapse": total,
        "span_frac": float(span_frac),
        "max_step_frac": float(max_step_frac),
        "entropy_C0": float(ent[0]),
        "entropy_Cfinal": float(ent[-1]),
        "dev_series": dev.tolist(),
        "entropy_series": ent.tolist(),
    }


def main() -> None:
    report = {"status": ("ant double-bridge contrast (Deneubourg); "
                         "ANT-1..3 frozen in the docstring; N/D/R "
                         "thresholds copied from emergence_coordinates.py, "
                         "verdict rule weak(D,R) matches the held-out "
                         "families; N reported descriptively"),
              "thresholds": TH}
    for mode in ("SOLO", "TRAIL"):
        print(f"=== {mode} ===", flush=True)
        rate = measure_consolidation_rate(mode)
        d = measure_D(mode)
        r = measure_R(mode)
        n = measure_N(mode)
        g = gradualism(mode)
        weak = int(d >= TH["D"] and r >= TH["R"])   # held-out rule
        report[mode] = {"consolidation_rate": rate, "N": n, "D": d,
                        "R": r, "gradualism": g, "weak_emergence": weak}
        print(f"  rate={rate:.3f} N={n:+.3f} D={d:.3f} R={r:.3f} "
              f"weak(D,R)={weak} collapse={g['total_collapse']:.3f} "
              f"span_frac={g['span_frac']:.2f} "
              f"max_step_frac={g['max_step_frac']:.3f}", flush=True)

    solo, trail = report["SOLO"], report["TRAIL"]
    ant1 = (solo["consolidation_rate"] < 0.3 and solo["D"] < TH["D"])
    ant2 = (trail["D"] >= TH["D"] and trail["R"] >= TH["R"])
    tg = trail["gradualism"]
    ant3 = (tg["total_collapse"] >= 0.5
            and tg["span_frac"] >= 0.10
            and solo["gradualism"]["total_collapse"] < 0.3)
    report["registered_outcomes"] = {
        "ANT1_solo_not_collective_emergence": bool(ant1),
        "ANT2_trail_is_weak_emergence": bool(ant2),
        "ANT3_collapse_is_gradual_not_abrupt": bool(ant3),
    }
    out = OUTPUTS / "ant_contrast.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
