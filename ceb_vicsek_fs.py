"""CEB-VICSEK-FS: finite-size / hysteresis Vicsek re-test.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Uses a
cell-list implementation to scan the noise axis at increasing N.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ceb_vicsek import entropy_from_counts
from tri_c_breakpoint import hinge_linear

OUTPUTS = Path(__file__).resolve().parent / "outputs"
RHO = 2.0
SPEED = 0.03
RADIUS = 1.0
NBINS = 24
N_REP = 3
T_STEPS = 350
ETAS = (3.0, 2.4, 1.8, 1.3, 0.9, 0.6, 0.35, 0.2, 0.1)
NS = (100, 400, 1600)
GATE = 0.1
SEED = 99_001


def stats(theta: np.ndarray) -> Tuple[float, float]:
    hs, phis = [], []
    for r in range(theta.shape[0]):
        bins = np.floor(((theta[r] + np.pi) / (2 * np.pi)) * NBINS).astype(int) % NBINS
        counts = np.bincount(bins, minlength=NBINS)
        hs.append(entropy_from_counts(counts) / math.log2(NBINS))
        phis.append(abs(np.exp(1j * theta[r]).mean()))
    return float(np.median(hs)), float(np.median(phis))


def step_cell(x: np.ndarray, theta: np.ndarray, box: float,
              rng: np.random.Generator, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    n_rep, n, _ = x.shape
    ncell = max(3, int(box / RADIUS))
    cell_size = box / ncell
    out_theta = np.empty_like(theta)
    for r in range(n_rep):
        cells: List[List[int]] = [[] for _ in range(ncell * ncell)]
        ij = np.floor(x[r] / cell_size).astype(int) % ncell
        for i in range(n):
            cells[ij[i, 0] * ncell + ij[i, 1]].append(i)
        for i in range(n):
            cx, cy = ij[i]
            neigh_idx = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neigh_idx.extend(cells[((cx + dx) % ncell) * ncell + ((cy + dy) % ncell)])
            arr = np.array(neigh_idx, dtype=int)
            d = x[r, arr] - x[r, i]
            d = (d + box / 2) % box - box / 2
            arr = arr[(d ** 2).sum(axis=1) <= RADIUS ** 2]
            z = np.exp(1j * theta[r, arr]).sum()
            out_theta[r, i] = np.angle(z) + rng.uniform(-eta / 2, eta / 2)
    out_theta = (out_theta + np.pi) % (2 * np.pi) - np.pi
    x = (x + SPEED * np.stack([np.cos(out_theta), np.sin(out_theta)], axis=-1)) % box
    return x, out_theta


def scan(n: int, etas: List[float], seed: int, start: str) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    box = math.sqrt(n / RHO)
    x = rng.uniform(0, box, size=(N_REP, n, 2))
    if start == "hot":
        theta = rng.uniform(-np.pi, np.pi, size=(N_REP, n))
    elif start == "cold":
        base = rng.uniform(-np.pi, np.pi, size=(N_REP, 1))
        theta = base + rng.normal(0, 0.02, size=(N_REP, n))
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
    else:
        raise ValueError(start)

    openness, phi = [], []
    for eta in etas:
        for _ in range(T_STEPS):
            x, theta = step_cell(x, theta, box, rng, eta)
        o, p = stats(theta)
        openness.append(o)
        phi.append(p)
        print(f"N={n} {start} eta={eta:.2f}: O={o:.4f} phi={p:.4f}",
              flush=True)
    return {"openness": openness, "phi": phi}


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


def main() -> None:
    systems = {}
    for n in NS:
        cool = scan(n, list(ETAS), SEED + n, "hot")
        heat_raw = scan(n, list(reversed(ETAS)), SEED + n * 3, "cold")
        heat = {
            "openness": list(reversed(heat_raw["openness"])),
            "phi": list(reversed(heat_raw["phi"])),
        }
        hyst = float(np.max(np.abs(np.array(cool["openness"]) - np.array(heat["openness"]))))
        systems[str(n)] = {
            "box": round(math.sqrt(n / RHO), 5),
            "cool_openness": [round(v, 5) for v in cool["openness"]],
            "cool_phi": [round(v, 5) for v in cool["phi"]],
            "heat_openness_high_to_low": [round(v, 5) for v in heat["openness"]],
            "heat_phi_high_to_low": [round(v, 5) for v in heat["phi"]],
            "hysteresis_max": round(hyst, 5),
            "adj": adjudicate(cool["openness"]),
        }

    max_drops = [systems[str(n)]["adj"]["max_adjacent_drop"] for n in NS]
    dbics = [systems[str(n)]["adj"].get("hinge", {}).get("delta_bic", -999) for n in NS]
    hysts = [systems[str(n)]["hysteresis_max"] for n in NS]
    outcomes = {
        "VFS1_ordering_all_N": all(
            systems[str(n)]["cool_openness"][0] > systems[str(n)]["cool_openness"][-1]
            and systems[str(n)]["cool_phi"][0] < systems[str(n)]["cool_phi"][-1]
            for n in NS
        ),
        "VFS2_finite_size_sharpening": bool(
            max_drops[-1] > max_drops[0] or dbics[-1] > dbics[0]
        ),
        "VFS3_hysteresis_increases": bool(hysts[-1] > hysts[0]),
        "max_adjacent_drops": max_drops,
        "delta_bics": dbics,
        "hysteresis": hysts,
    }
    report = {
        "status": "CEB-VICSEK-FS finite-size/hysteresis re-test; preregistered",
        "config": {"rho": RHO, "speed": SPEED, "radius": RADIUS,
                   "n_rep": N_REP, "steps_per_eta": T_STEPS,
                   "etas_high_to_low": ETAS, "Ns": NS},
        "systems": systems,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_vicsek_fs.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
