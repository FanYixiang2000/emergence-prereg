"""CEB-LIFE: Game of Life perturbation-ensemble boundary test.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Life is
deterministic, so this uses a declared single-cell perturbation
ensemble to estimate future macro-outcome openness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SIZE = 60
SAVE_GRID = list(range(0, 121, 15))
H_CONT = 60
N_CONT = 60
MARGIN = 5
GATE = 0.1
SEED = 103_001
CLASSES = ("empty", "bounded_small", "translator", "complex", "other")


def step(x: np.ndarray) -> np.ndarray:
    neigh = sum(
        np.roll(np.roll(x, dx, axis=-2), dy, axis=-1)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)
    )
    return ((neigh == 3) | (x & (neigh == 2)))


def place(cells: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    x = np.zeros((SIZE, SIZE), dtype=bool)
    off = SIZE // 2
    for r, c in cells:
        x[off + r, off + c] = True
    return x


def init_state(name: str) -> np.ndarray:
    if name == "block":
        return place(((0, 0), (0, 1), (1, 0), (1, 1)))
    if name == "blinker":
        return place(((0, -1), (0, 0), (0, 1)))
    if name == "glider":
        return place(((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)))
    if name == "r_pentomino":
        return place(((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)))
    raise ValueError(name)


def bbox(x: np.ndarray):
    pts = np.argwhere(x)
    if len(pts) == 0:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return lo, hi


def com(x: np.ndarray) -> np.ndarray:
    pts = np.argwhere(x)
    if len(pts) == 0:
        return np.array([np.nan, np.nan])
    return pts.mean(axis=0)


def perturb_batch(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    batch = np.repeat(state[None, :, :], N_CONT, axis=0)
    b = bbox(state)
    if b is None:
        rows = rng.integers(0, SIZE, size=N_CONT)
        cols = rng.integers(0, SIZE, size=N_CONT)
    else:
        lo, hi = b
        r0 = max(0, int(lo[0]) - MARGIN)
        r1 = min(SIZE - 1, int(hi[0]) + MARGIN)
        c0 = max(0, int(lo[1]) - MARGIN)
        c1 = min(SIZE - 1, int(hi[1]) + MARGIN)
        rows = rng.integers(r0, r1 + 1, size=N_CONT)
        cols = rng.integers(c0, c1 + 1, size=N_CONT)
    batch[np.arange(N_CONT), rows, cols] ^= True
    return batch


def classify(start: np.ndarray, final: np.ndarray) -> str:
    live = int(final.sum())
    if live == 0:
        return "empty"
    b = bbox(final)
    if b is None:
        return "empty"
    lo, hi = b
    extent = hi - lo + 1
    area = int(extent[0] * extent[1])
    disp = np.linalg.norm(com(final) - com(start))
    if live <= 8 and area <= 30 and disp > 4:
        return "translator"
    if live <= 20 and area <= 80:
        return "bounded_small"
    if live > 40 or area > 250:
        return "complex"
    return "other"


def openness_from_counts(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(len(CLASSES)))


def continue_classes(state: np.ndarray, rng: np.random.Generator):
    batch = perturb_batch(state, rng)
    start = batch.copy()
    for _ in range(H_CONT):
        batch = step(batch)
    counts = {k: 0 for k in CLASSES}
    for i in range(N_CONT):
        counts[classify(start[i], batch[i])] += 1
    arr = np.array([counts[k] for k in CLASSES])
    return counts, openness_from_counts(arr)


def adjudicate(openness):
    y = np.array(openness)
    drop = float(y[0] - y[-1])
    out: Dict[str, object] = {
        "drop": round(drop, 4),
        "max_drop_from_peak": round(float(np.max(y) - y[-1]), 4),
        "gate_passed": bool(abs(drop) >= GATE or (float(np.max(y) - y[-1]) >= GATE)),
    }
    if out["gate_passed"]:
        h = hinge_linear(np.array(SAVE_GRID, dtype=float), y)
        out["hinge"] = h
        out["b5_onset"] = bool(h["delta_bic"] >= 10 and h["onset_type"])
    else:
        out["b5_onset"] = False
    return out


def run_case(name: str, seed: int):
    rng = np.random.default_rng(seed)
    x = init_state(name)
    states = {}
    for t in range(max(SAVE_GRID) + 1):
        if t in SAVE_GRID:
            states[t] = x.copy()
        x = step(x)
    openness, class_counts = [], {}
    for t in SAVE_GRID:
        counts, o = continue_classes(states[t], rng)
        openness.append(o)
        class_counts[str(t)] = counts
        print(f"{name} t={t}: O={o:.4f} counts={counts}", flush=True)
    return {
        "openness": [round(v, 5) for v in openness],
        "class_counts": class_counts,
        "adj": adjudicate(openness),
    }


def main() -> None:
    cases = {}
    for i, name in enumerate(("block", "blinker", "glider", "r_pentomino")):
        cases[name] = run_case(name, SEED + i * 101)
    outcomes = {
        "LIFE1_seeded_low_G_caveat": True,
        "LIFE2_r_pentomino_b5": bool(cases["r_pentomino"]["adj"]["b5_onset"]),
        "LIFE2_r_pentomino_drop": cases["r_pentomino"]["adj"]["drop"],
        "LIFE3_no_posthoc_classifier_change": True,
    }
    report = {
        "status": "CEB-LIFE perturbation-ensemble boundary test; preregistered",
        "config": {"size": SIZE, "save_grid": SAVE_GRID, "h_cont": H_CONT,
                   "n_cont": N_CONT, "margin": MARGIN,
                   "classes": CLASSES},
        "cases": cases,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_life.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
