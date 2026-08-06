"""CEB-SCHELLING: boundary pressure test for Schelling segregation.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Tests
whether a classic weak/social emergence model shows macro-organization
without onset-type B5 under the neighborhood-openness object.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

from kuramoto_breakpoint_r2 import truncate_at_saturation
from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
L = 50
N_REP = 20
T_SWEEPS = 300
SAVE_EVERY = 5
GRID = list(range(0, T_SWEEPS + 1, SAVE_EVERY))
GATE = 0.1
SEED = 92_001


def init_grid(rng: np.random.Generator) -> np.ndarray:
    n = L * L
    vals = np.array([1] * int(0.45 * n) + [-1] * int(0.45 * n))
    vals = np.concatenate([vals, np.zeros(n - len(vals), dtype=int)])
    rng.shuffle(vals)
    return vals.reshape(L, L)


def neighbor_counts(grid: np.ndarray):
    occ = grid != 0
    same = np.zeros_like(grid, dtype=int)
    total = np.zeros_like(grid, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
            nb_occ = nb != 0
            total += nb_occ
            same += nb_occ & (nb == grid)
    return same, total, occ


def stats(grid: np.ndarray):
    same, total, occ = neighbor_counts(grid)
    valid = occ & (total > 0)
    frac = np.zeros_like(grid, dtype=float)
    frac[valid] = same[valid] / total[valid]
    native = float(frac[valid].mean()) if valid.any() else 0.0
    counts = np.bincount(same[valid], minlength=9)
    p = counts / max(counts.sum(), 1)
    q = p[p > 0]
    openness = float(-(q * np.log2(q)).sum() / math.log2(9))
    return native, openness


def step(grid: np.ndarray, tau: float, rng: np.random.Generator):
    same, total, occ = neighbor_counts(grid)
    frac = np.zeros_like(grid, dtype=float)
    valid = occ & (total > 0)
    frac[valid] = same[valid] / total[valid]
    unhappy = np.argwhere(valid & (frac < tau))
    vacancies = np.argwhere(grid == 0)
    if len(unhappy) == 0 or len(vacancies) == 0:
        return
    rng.shuffle(unhappy)
    for x, y in unhappy:
        vacancies = np.argwhere(grid == 0)
        if len(vacancies) == 0:
            break
        j = rng.integers(len(vacancies))
        vx, vy = vacancies[j]
        grid[vx, vy] = grid[x, y]
        grid[x, y] = 0


def simulate(tau: float, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    natives, opens = [], []
    reps = [init_grid(rng) for _ in range(N_REP)]
    for sweep in range(T_SWEEPS + 1):
        if sweep % SAVE_EVERY == 0:
            vals = [stats(g) for g in reps]
            natives.append(float(np.median([v[0] for v in vals])))
            opens.append(float(np.median([v[1] for v in vals])))
        if sweep == T_SWEEPS:
            break
        for g in reps:
            step(g, tau, rng)
    return {
        "native": natives,
        "openness": opens,
        "native_first_last": [round(natives[0], 4), round(natives[-1], 4)],
        "openness_first_last": [round(opens[0], 4), round(opens[-1], 4)],
    }


def adjudicate(openness) -> Dict:
    x = np.array(GRID, dtype=float)
    y = np.array(openness)
    drop = float(y[0] - y[-1])
    out = {"drop": round(drop, 4), "gate_passed": bool(drop >= GATE)}
    if not out["gate_passed"]:
        out["verdict"] = "no_collapse_B5_not_applicable"
        out["b5_onset"] = False
        return out
    xw, yw, t_sat = truncate_at_saturation(x, y)
    out["t_sat"] = t_sat
    if len(yw) < 8:
        out["verdict"] = "window_too_short"
        out["b5_onset"] = False
        return out
    full = hinge_linear(xw, yw)
    span = xw[-1] - xw[0]
    thin_ok = True
    thin = {}
    for parity in (0, 1):
        if len(xw[parity::2]) < 5:
            t = {"verdict": "too_few_points", "ok": False}
            ok = False
        else:
            t = hinge_linear(xw[parity::2], yw[parity::2])
            ok = (t["delta_bic"] >= 2.0 and t["onset_type"]
                  and abs(t["t_star"] - full["t_star"]) <= 0.10 * span)
        t["ok"] = bool(ok)
        thin[f"parity{parity}"] = t
        thin_ok = thin_ok and ok
    out.update({
        "hinge": full,
        "thinning": thin,
        "b5_onset": bool(full["delta_bic"] >= 10
                         and full["onset_type"] and thin_ok),
    })
    return out


def main() -> None:
    rows = {}
    for name, tau in (("main", 0.35), ("control", 0.0)):
        row = simulate(tau, SEED + len(rows) * 101)
        adj = adjudicate(row["openness"])
        row["adj"] = adj
        rows[name] = row
        h = adj.get("hinge", {})
        print(f"{name}: native {row['native_first_last']} "
              f"O {row['openness_first_last']} "
              f"b5={adj.get('b5_onset')} t*={h.get('t_star')} "
              f"dBIC={h.get('delta_bic')} "
              f"slopes={h.get('slope_before')}->{h.get('slope_after')}",
              flush=True)

    main_row = rows["main"]
    ctrl = rows["control"]
    sch1 = bool(main_row["native"][-1] - main_row["native"][0] >= 0.2)
    sch2 = bool(not main_row["adj"].get("b5_onset"))
    sch3 = bool(abs(ctrl["native"][-1] - ctrl["native"][0]) < 0.05
                and not ctrl["adj"].get("b5_onset"))
    outcomes = {
        "SCH1_organization": sch1,
        "SCH2_no_false_onset_expected": sch2,
        "SCH3_control_null": sch3,
    }
    report = {
        "status": "CEB-SCHELLING boundary pressure test; preregistered",
        "config": {"L": L, "n_rep": N_REP, "sweeps": T_SWEEPS},
        "grid": GRID,
        "conditions": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_schelling.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
