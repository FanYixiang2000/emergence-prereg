"""CEB-POTTS: q=2 vs q=10 continuous/first-order contrast.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Tests
whether the collapse profile distinguishes generic ordering from a
sharper first-order regime conversion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
L = 48
N_REP = 6
THERM_SWEEPS = 900
SAMPLE_SWEEPS = 300
TEMP_FACTORS = (1.35, 1.20, 1.10, 1.05, 1.02, 1.00, 0.98, 0.95, 0.90, 0.80, 0.70)
GATE = 0.1
SEED = 98_001


def tc_potts(q: int) -> float:
    return 1.0 / math.log(1.0 + math.sqrt(q))


def color_stats(spins: np.ndarray, q: int):
    flat = spins.reshape(spins.shape[0], -1)
    counts = np.stack([(flat == k).sum(axis=1) for k in range(q)], axis=1)
    probs = counts / flat.shape[1]
    max_frac = probs.max(axis=1)
    order = (q * max_frac - 1.0) / (q - 1.0)
    ent = []
    for row in probs:
        nz = row[row > 0]
        ent.append(float(-(nz * np.log2(nz)).sum() / math.log2(q)))
    return order, np.array(ent)


def checkerboard_sweep(spins: np.ndarray, q: int, temp: float,
                       rng: np.random.Generator) -> None:
    yy, xx = np.indices((L, L))
    for parity in (0, 1):
        current = spins
        prop = (current + rng.integers(1, q, size=current.shape)) % q
        cur_matches = (
            (current == np.roll(current, 1, axis=1)).astype(np.int8)
            + (current == np.roll(current, -1, axis=1)).astype(np.int8)
            + (current == np.roll(current, 1, axis=2)).astype(np.int8)
            + (current == np.roll(current, -1, axis=2)).astype(np.int8)
        )
        prop_matches = (
            (prop == np.roll(current, 1, axis=1)).astype(np.int8)
            + (prop == np.roll(current, -1, axis=1)).astype(np.int8)
            + (prop == np.roll(current, 1, axis=2)).astype(np.int8)
            + (prop == np.roll(current, -1, axis=2)).astype(np.int8)
        )
        d_e = -(prop_matches - cur_matches)
        mask = ((yy + xx) % 2 == parity)[None, :, :]
        accept = (d_e <= 0) | (rng.random(current.shape) < np.exp(-d_e / temp))
        spins[mask & accept] = prop[mask & accept]


def run_scan(q: int, temps: List[float], seed: int, start: str) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    if start == "hot":
        spins = rng.integers(0, q, size=(N_REP, L, L), dtype=np.int16)
    elif start == "cold":
        spins = np.zeros((N_REP, L, L), dtype=np.int16)
    else:
        raise ValueError(start)

    orders, openness = [], []
    for temp in temps:
        for _ in range(THERM_SWEEPS):
            checkerboard_sweep(spins, q, temp, rng)
        sample_o, sample_h = [], []
        for _ in range(SAMPLE_SWEEPS):
            checkerboard_sweep(spins, q, temp, rng)
            o, h = color_stats(spins, q)
            sample_o.append(o)
            sample_h.append(h)
        orders.append(float(np.median(np.mean(np.array(sample_o), axis=0))))
        openness.append(float(np.median(np.mean(np.array(sample_h), axis=0))))
        print(f"q={q} {start} T={temp:.4f}: m={orders[-1]:.4f} O={openness[-1]:.4f}",
              flush=True)
    return {"order": orders, "openness": openness}


def adjudicate(openness: List[float]) -> Dict[str, object]:
    y = np.array(openness)
    drop = float(y[0] - y[-1])
    out: Dict[str, object] = {"drop": round(drop, 4),
                              "max_adjacent_drop": round(float(np.max(-np.diff(y))), 4),
                              "gate_passed": bool(drop >= GATE)}
    if out["gate_passed"]:
        h = hinge_linear(np.arange(len(y), dtype=float), y)
        out["hinge"] = h
        out["b5_control"] = bool(h["delta_bic"] >= 10)
    else:
        out["b5_control"] = False
    return out


def main() -> None:
    report = {"status": "CEB-POTTS q=2/q=10 contrast; preregistered",
              "config": {"L": L, "n_rep": N_REP,
                         "therm_sweeps": THERM_SWEEPS,
                         "sample_sweeps": SAMPLE_SWEEPS,
                         "temp_factors": TEMP_FACTORS},
              "systems": {}}
    for q in (2, 10):
        tc = tc_potts(q)
        cool_temps = [tc * f for f in TEMP_FACTORS]
        heat_temps = list(reversed(cool_temps))
        cool = run_scan(q, cool_temps, SEED + q * 100, "hot")
        heat_raw = run_scan(q, heat_temps, SEED + q * 1000, "cold")
        heat = {
            "order": list(reversed(heat_raw["order"])),
            "openness": list(reversed(heat_raw["openness"])),
        }
        hyst = float(np.max(np.abs(np.array(cool["openness"]) - np.array(heat["openness"]))))
        adj = adjudicate(cool["openness"])
        report["systems"][str(q)] = {
            "tc": round(tc, 6),
            "temps_high_to_low": [round(v, 6) for v in cool_temps],
            "cool_order": [round(v, 5) for v in cool["order"]],
            "cool_openness": [round(v, 5) for v in cool["openness"]],
            "heat_order_high_to_low": [round(v, 5) for v in heat["order"]],
            "heat_openness_high_to_low": [round(v, 5) for v in heat["openness"]],
            "hysteresis_max": round(hyst, 5),
            "adj": adj,
        }

    q2 = report["systems"]["2"]
    q10 = report["systems"]["10"]
    outcomes = {
        "POTTS1_both_order": bool(
            q2["cool_openness"][0] > 0.8 and q2["cool_openness"][-1] < 0.4
            and q10["cool_openness"][0] > 0.8 and q10["cool_openness"][-1] < 0.4
        ),
        "POTTS2_q10_sharper": bool(
            q10["adj"]["max_adjacent_drop"] > q2["adj"]["max_adjacent_drop"]
            and q10["adj"].get("hinge", {}).get("delta_bic", -999)
            > q2["adj"].get("hinge", {}).get("delta_bic", -999)
        ),
        "POTTS3_q10_more_hysteresis": bool(
            q10["hysteresis_max"] > 2.0 * q2["hysteresis_max"]
        ),
    }
    report["registered_outcomes"] = outcomes
    out = OUTPUTS / "ceb_potts.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
