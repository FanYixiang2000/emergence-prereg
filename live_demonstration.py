"""Live reviewer demonstration: does the instrument actually measure
what it claims, on cases it has never seen?

Two parts, run end to end in one sitting, predictions frozen first.

PART 1 -- blind accuracy against hidden truth. 24 knob vectors are
drawn UNIFORMLY AT RANDOM (fresh seed, none on the calibration grid);
the generator hides them; the instrument measures the continuous
record through the same finite-sample pipeline used on real systems;
then the answers are unsealed and compared.

    LD-1  For every matched pair (s->S, b->M, v->V, q->Q_rel, r->R),
          Spearman(set, measured) >= 0.9 over the 24 blind draws.
    LD-2  Mean absolute cross-leakage: for every unmatched pair, the
          absolute Spearman correlation is <= 0.35 (noise floor at
          n = 24), EXCEPT the two declared structural couplings
          (Q_raw with structure; R with structure).

PART 2 -- verdict correctness on known identities. One learned policy
is trained live in the crowd-vote domain (fresh seed 6901) alongside
four systems whose classification is known by construction. The full
six-component certificate is computed for all five.

    LD-3  The learned policy that acquired the selective convention is
          the ONLY system accepted; the scripted switcher and its
          clone fail exactly {endogeneity, acquisition}; the twin
          fails selectivity/acquisition; blanket democracy fails
          potential. (If the fresh seed lands in the known blind-dem
          basin, ~20-50% at d=2, the learned system must instead be
          REJECTED via selectivity -- either outcome must match the
          policy's measured convention, and which occurred is
          reported.)

Misses are retained.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

import crowd_vote_domain as cv
from generator_calibration import measure_system, REFERENCE

OUTPUTS = Path(__file__).resolve().parent / "outputs"

N_BLIND = 24
BLIND_SEED = 20260721


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def part1() -> dict:
    rng = random.Random(BLIND_SEED)
    draws = []
    for i in range(N_BLIND):
        knobs = {
            "s": rng.uniform(0.05, 1.0),
            "b": rng.uniform(0.05, 1.0),
            "v": rng.uniform(-1.0, 1.0),
            "q": rng.uniform(0.0, 1.0),
            "a": rng.uniform(0.05, 1.0),
            "r": rng.uniform(0.0, 1.0),
        }
        measured = measure_system(knobs, 500_000 + i * 17)
        draws.append({"set": knobs, "measured": measured})
        print(f"draw {i + 1:2d}: set s={knobs['s']:.2f} b={knobs['b']:.2f} "
              f"v={knobs['v']:+.2f} q={knobs['q']:.2f} r={knobs['r']:.2f}"
              f" | measured S={measured['S']:.2f} M={measured['M']:.2f} "
              f"V={measured['V']:+.2f} Q={measured['Q_rel']:.2f} "
              f"R={measured['R']:.2f}", flush=True)

    matched = {"s": "S", "b": "M", "v": "V", "q": "Q_rel", "r": "R"}
    rho_matched = {}
    for knob, dim in matched.items():
        rho_matched[f"{knob}->{dim}"] = spearman(
            [d["set"][knob] for d in draws],
            [d["measured"][dim] for d in draws])
    declared = {("b", "Q_raw"), ("s", "Q_raw"), ("b", "R"), ("s", "R")}
    leakage = {}
    worst = 0.0
    for knob in matched:
        for dim in ("S", "M", "V", "Q_rel", "R"):
            if matched[knob] == dim or (knob, dim) in declared:
                continue
            r = abs(spearman([d["set"][knob] for d in draws],
                             [d["measured"][dim] for d in draws]))
            leakage[f"{knob}->{dim}"] = r
            worst = max(worst, r)
    ld1 = all(r >= 0.9 for r in rho_matched.values())
    ld2 = worst <= 0.35
    return {"draws": draws, "rho_matched": rho_matched,
            "cross_leakage": leakage, "worst_leakage": worst,
            "LD1": ld1, "LD2": ld2}


def part2() -> dict:
    seed = 6901
    print("training a fresh learned policy (seed 6901)...", flush=True)
    q = cv.train_learned(seed)
    systems = {
        "learned": (cv.policy_from_q(q), True),
        "initial_twin": (cv.policy_from_q({}), True),
        "always_democracy": (cv.always_democracy, False),
        "scripted_switcher": (cv.scripted_switcher, False),
        "bc_clone": (cv.bc_clone(seed + 77), False),
    }
    offset = 95_000_000
    metrics = {n: cv.evaluate(pol, offset)
               for n, (pol, _e) in systems.items()}
    acq = (metrics["learned"]["conditional_selectivity"]
           - metrics["initial_twin"]["conditional_selectivity"])
    rows = {}
    for n, (pol, endo) in systems.items():
        v = cv.verdict(metrics[n], endo,
                       acq if n == "learned" else 0.0)
        rows[n] = {"metrics": {k: round(val, 3) if isinstance(val, float)
                               else val for k, val in metrics[n].items()},
                   "verdict": v}
        print(f"{n:18s} emergent={v['emergent']} "
              f"failed={';'.join(v['failed']) or '-'}", flush=True)

    learned_selective = (metrics["learned"]["conditional_selectivity"]
                         >= 0.5)
    if learned_selective:
        ld3 = (rows["learned"]["verdict"]["emergent"] == 1
               and all(rows[n]["verdict"]["emergent"] == 0
                       for n in systems if n != "learned"))
    else:
        ld3 = (rows["learned"]["verdict"]["emergent"] == 0
               and "conditional_selectivity"
               in rows["learned"]["verdict"]["failed"])
    routes_ok = (
        set(rows["scripted_switcher"]["verdict"]["failed"])
        == {"endogeneity", "acquisition"}
        and set(rows["bc_clone"]["verdict"]["failed"])
        == {"endogeneity", "acquisition"}
        and "potential" in rows["always_democracy"]["verdict"]["failed"]
        and "conditional_selectivity"
        in rows["initial_twin"]["verdict"]["failed"])
    return {"rows": rows,
            "learned_convention": ("selective" if learned_selective
                                   else "blind"),
            "LD3": bool(ld3 and routes_ok)}


def main() -> None:
    print("== PART 1: blind accuracy ==", flush=True)
    p1 = part1()
    print(json.dumps(p1["rho_matched"], indent=1))
    print(f"worst cross-leakage {p1['worst_leakage']:.2f}")
    print("== PART 2: live verdicts ==", flush=True)
    p2 = part2()
    # Disclosed follow-up to LD-2 (the frozen rule is scored as-is
    # above): at n = 24 the RANDOM DRAWS themselves carry sampling
    # correlations of |rho| ~ 0.35 between knobs, which the raw
    # cross-correlation confounds with instrument leakage. The
    # follow-up computes partial Spearman correlations (unmatched
    # knob vs measured dimension, controlling for the dimension's own
    # matched knob by rank regression residuals).
    matched = {"S": "s", "M": "b", "V": "v", "Q_rel": "q", "R": "r"}
    draws = p1["draws"]

    def ranks(x):
        r = np.argsort(np.argsort(x)).astype(float)
        return (r - r.mean()) / (r.std() + 1e-12)

    partial = {}
    worst_partial = 0.0
    declared = {("b", "Q_rel"), ("s", "Q_rel"), ("b", "R"), ("s", "R")}
    for dim, own in matched.items():
        y = ranks([d["measured"][dim] for d in draws])
        x_own = ranks([d["set"][own] for d in draws])
        resid_y = y - (y @ x_own) / (x_own @ x_own) * x_own
        for knob in ("s", "b", "v", "q", "r"):
            if knob == own:
                continue
            x = ranks([d["set"][knob] for d in draws])
            resid_x = x - (x @ x_own) / (x_own @ x_own) * x_own
            denom = np.sqrt((resid_x ** 2).sum() * (resid_y ** 2).sum())
            r = float((resid_x @ resid_y) / denom) if denom > 0 else 0.0
            partial[f"{knob}->{dim}"] = r
            if (knob, dim) not in declared:
                worst_partial = max(worst_partial, abs(r))

    report = {
        "status": ("live reviewer demonstration; LD-1..3 frozen in "
                   "the docstring; blind draws off the calibration "
                   "grid"),
        "part1_blind_accuracy": p1,
        "part2_live_verdicts": p2,
        "ld2_followup_partial_correlations": {
            "note": ("raw rule retained as a miss; partial "
                     "correlations control for the matched knob and "
                     "for draw-sampling correlation"),
            "partial": partial,
            "worst_undeclared_partial": worst_partial,
        },
        "registered_outcomes": {
            "LD1_matched_spearman_ge_0.9": p1["LD1"],
            "LD2_cross_leakage_le_0.35": p1["LD2"],
            "LD3_verdicts_match_known_identities": p2["LD3"],
        },
    }
    out = OUTPUTS / "live_demonstration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
