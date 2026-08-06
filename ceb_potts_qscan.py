"""CEB-POTTS-QSCAN: Potts transition-order calibration curve.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Extends
q=2 vs q=10 into q={2,3,4,5,8,10}, L={32,48}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
QS = (2, 3, 4, 5, 8, 10)
LS = (32, 48)
N_REP = 4
THERM_SWEEPS = 600
SAMPLE_SWEEPS = 180
TEMP_FACTORS = (1.30, 1.18, 1.10, 1.05, 1.02, 1.00, 0.98, 0.95, 0.90, 0.82, 0.74)
GATE = 0.1
SEED = 106_001


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


def sweep(spins: np.ndarray, q: int, temp: float, rng: np.random.Generator) -> None:
    l = spins.shape[1]
    yy, xx = np.indices((l, l))
    for parity in (0, 1):
        current = spins
        prop = (current + rng.integers(1, q, size=current.shape)) % q
        cur = (
            (current == np.roll(current, 1, axis=1)).astype(np.int8)
            + (current == np.roll(current, -1, axis=1)).astype(np.int8)
            + (current == np.roll(current, 1, axis=2)).astype(np.int8)
            + (current == np.roll(current, -1, axis=2)).astype(np.int8)
        )
        new = (
            (prop == np.roll(current, 1, axis=1)).astype(np.int8)
            + (prop == np.roll(current, -1, axis=1)).astype(np.int8)
            + (prop == np.roll(current, 1, axis=2)).astype(np.int8)
            + (prop == np.roll(current, -1, axis=2)).astype(np.int8)
        )
        d_e = -(new - cur)
        mask = ((yy + xx) % 2 == parity)[None, :, :]
        accept = (d_e <= 0) | (rng.random(current.shape) < np.exp(-d_e / temp))
        spins[mask & accept] = prop[mask & accept]


def run_scan(q: int, l: int, temps: List[float], seed: int, start: str):
    rng = np.random.default_rng(seed)
    if start == "hot":
        spins = rng.integers(0, q, size=(N_REP, l, l), dtype=np.int16)
    elif start == "cold":
        spins = np.zeros((N_REP, l, l), dtype=np.int16)
    else:
        raise ValueError(start)
    openness, order = [], []
    for temp in temps:
        for _ in range(THERM_SWEEPS):
            sweep(spins, q, temp, rng)
        hs, ms = [], []
        for _ in range(SAMPLE_SWEEPS):
            sweep(spins, q, temp, rng)
            m, h = color_stats(spins, q)
            hs.append(h)
            ms.append(m)
        openness.append(float(np.median(np.mean(np.array(hs), axis=0))))
        order.append(float(np.median(np.mean(np.array(ms), axis=0))))
        print(f"q={q} L={l} {start} T={temp:.4f}: O={openness[-1]:.4f} m={order[-1]:.4f}",
              flush=True)
    return {"openness": openness, "order": order}


def adjudicate(openness: List[float]) -> Dict[str, object]:
    y = np.array(openness)
    drop = float(y[0] - y[-1])
    out: Dict[str, object] = {
        "drop": round(drop, 4),
        "max_adjacent_drop": round(float(np.max(-np.diff(y))), 4),
        "gate_passed": bool(drop >= GATE),
    }
    if out["gate_passed"]:
        h = hinge_linear(np.arange(len(y), dtype=float), y)
        out["hinge"] = h
        out["b5_control"] = bool(h["delta_bic"] >= 10)
    else:
        out["b5_control"] = False
    return out


def run_cell(q: int, l: int):
    tc = tc_potts(q)
    cool_t = [tc * f for f in TEMP_FACTORS]
    heat_t = list(reversed(cool_t))
    base = SEED + q * 10_000 + l
    cool = run_scan(q, l, cool_t, base, "hot")
    heat_raw = run_scan(q, l, heat_t, base + 5_000, "cold")
    heat = {
        "openness": list(reversed(heat_raw["openness"])),
        "order": list(reversed(heat_raw["order"])),
    }
    hyst = float(np.max(np.abs(np.array(cool["openness"]) - np.array(heat["openness"]))))
    return {
        "tc": round(tc, 6),
        "temps_high_to_low": [round(v, 6) for v in cool_t],
        "cool_openness": [round(v, 5) for v in cool["openness"]],
        "cool_order": [round(v, 5) for v in cool["order"]],
        "heat_openness_high_to_low": [round(v, 5) for v in heat["openness"]],
        "heat_order_high_to_low": [round(v, 5) for v in heat["order"]],
        "hysteresis_max": round(hyst, 5),
        "adj": adjudicate(cool["openness"]),
    }


def main() -> None:
    systems = {}
    for l in LS:
        systems[str(l)] = {}
        for q in QS:
            systems[str(l)][str(q)] = run_cell(q, l)

    def group_vals(l: int, qs):
        return [systems[str(l)][str(q)] for q in qs]

    outcomes = {"per_L": {}}
    for l in LS:
        cont = group_vals(l, (2, 3, 4))
        first = group_vals(l, (5, 8, 10))
        cont_h = [r["hysteresis_max"] for r in cont]
        first_h = [r["hysteresis_max"] for r in first]
        cont_b = [r["adj"].get("hinge", {}).get("delta_bic", -999) for r in cont]
        first_b = [r["adj"].get("hinge", {}).get("delta_bic", -999) for r in first]
        outcomes["per_L"][str(l)] = {
            "mean_hysteresis_q_le_4": round(float(np.mean(cont_h)), 5),
            "mean_hysteresis_q_gt_4": round(float(np.mean(first_h)), 5),
            "mean_delta_bic_q_le_4": round(float(np.mean(cont_b)), 3),
            "mean_delta_bic_q_gt_4": round(float(np.mean(first_b)), 3),
            "PQS2_boundary": bool(np.mean(first_h) > np.mean(cont_h)
                                  and np.mean(first_b) > np.mean(cont_b)),
            "PQS3_high_q_stronger": bool(
                systems[str(l)]["8"]["hysteresis_max"] > systems[str(l)]["2"]["hysteresis_max"]
                and systems[str(l)]["10"]["hysteresis_max"] > systems[str(l)]["3"]["hysteresis_max"]
            ),
        }

    qgt4_gain = (
        outcomes["per_L"]["48"]["mean_hysteresis_q_gt_4"]
        - outcomes["per_L"]["32"]["mean_hysteresis_q_gt_4"]
    )
    qle4_gain = (
        outcomes["per_L"]["48"]["mean_hysteresis_q_le_4"]
        - outcomes["per_L"]["32"]["mean_hysteresis_q_le_4"]
    )
    outcomes["PQS1_all_order"] = all(
        systems[str(l)][str(q)]["cool_openness"][0] > systems[str(l)][str(q)]["cool_openness"][-1]
        for l in LS for q in QS
    )
    outcomes["PQS4_size_sharpening"] = bool(qgt4_gain > qle4_gain)
    outcomes["qgt4_hysteresis_gain_32_to_48"] = round(float(qgt4_gain), 5)
    outcomes["qle4_hysteresis_gain_32_to_48"] = round(float(qle4_gain), 5)

    report = {
        "status": "CEB-POTTS-QSCAN transition-order calibration; preregistered",
        "config": {"qs": QS, "Ls": LS, "n_rep": N_REP,
                   "therm_sweeps": THERM_SWEEPS,
                   "sample_sweeps": SAMPLE_SWEEPS,
                   "temp_factors": TEMP_FACTORS},
        "systems": systems,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_potts_qscan.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
