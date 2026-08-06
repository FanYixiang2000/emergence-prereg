"""Off-design generator test for the collapse-source ladder (KUR).

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. The
generators are Kuramoto phase oscillators -- a mechanism vocabulary
(continuous phases, sinusoidal coupling, a periodic driver) that the
ladder was NOT designed around. The ladder sees only discretized
phase bins. Predictions KUR-1..4 (KUR-4 registered as may-miss).

Conditions (three oscillators, Euler-Maruyama, T=15, dt=0.05,
noise sigma=0.1, 100k samples each):
  uncoupled : omega=(1.0,1.35,1.7), no coupling, no driver
  driven    : same omegas, no coupling, driver amplitude 3.0 at
              Omega=1.2; driver initial phase phi0 in {0, pi} with
              equal probability; declared E = phi0 index (binary)
  pair12    : omega=(1.0,1.05,1.7), coupling 2.0 on the (1,2) edge
              only, no driver
  allcoupled: omega=(1.0,1.05,1.1), all-to-all coupling 2.0, no driver

Phases are binned into 10 bins; joint tables are 10x10x10; the ladder
code is imported unchanged from the analytic battery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from collapse_source_decomposition import NA, ladder

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_SAMPLES = 100_000
T_STEPS = 300
DT = 0.05
SIGMA = 0.1
OMEGA_DRIVER = 1.2
TWO_PI = 2 * np.pi


def simulate(omegas, coupling: np.ndarray, driver_amp: float,
             seed: int):
    """Returns (bins, env) with bins shape (N,3) in 0..9; env in {0,1}."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, size=(N_SAMPLES, 3))
    env = rng.integers(0, 2, size=N_SAMPLES)
    phi0 = env * np.pi
    om = np.asarray(omegas)
    for step in range(T_STEPS):
        t = step * DT
        drive = np.zeros((N_SAMPLES, 3))
        if driver_amp > 0:
            phi = (OMEGA_DRIVER * t + phi0)[:, None]
            drive = driver_amp * np.sin(phi - theta)
        # pairwise coupling term: sum_j K[i,j] * sin(theta_j - theta_i)
        diff = theta[:, None, :] - theta[:, :, None]  # (N, i, j)
        coup = (coupling[None, :, :] * np.sin(diff)).sum(axis=2)
        noise = SIGMA * np.sqrt(DT) * rng.standard_normal((N_SAMPLES, 3))
        theta = theta + DT * (om[None, :] + drive + coup) + noise
    bins = np.floor((theta % TWO_PI) / (TWO_PI / NA)).astype(int) % NA
    return bins, env


def joint_tables(bins: np.ndarray, env: np.ndarray):
    pe = {}
    for e in (0, 1):
        sel = bins[env == e]
        table = np.zeros((NA, NA, NA))
        np.add.at(table, (sel[:, 0], sel[:, 1], sel[:, 2]), 1.0)
        pe[e] = table / table.sum()
    return pe


def main() -> None:
    k_none = np.zeros((3, 3))
    k_pair = np.zeros((3, 3))
    k_pair[0, 1] = k_pair[1, 0] = 2.0
    k_all = np.full((3, 3), 2.0)
    np.fill_diagonal(k_all, 0.0)

    conditions = {
        "uncoupled": ((1.0, 1.35, 1.7), k_none, 0.0, True),
        "driven": ((1.0, 1.35, 1.7), k_none, 3.0, True),
        "pair12": ((1.0, 1.05, 1.7), k_pair, 0.0, False),
        "allcoupled": ((1.0, 1.05, 1.1), k_all, 0.0, False),
    }
    rows = {}
    for i, (name, (om, K, amp, declare_env)) in enumerate(
            conditions.items()):
        bins, env = simulate(om, K, amp, seed=77_001 + i)
        pe = joint_tables(bins, env)
        # conditions without a driver have no real E; both env halves
        # are iid samples of the same law, so C_env ~= 0 by symmetry
        res = ladder(pe, declare_env=True)
        comp = {k: round(v, 5) for k, v in res["components"].items()}
        rows[name] = {"declared_E": "driver phi0 (binary)"
                      if declare_env and amp > 0 else
                      "binary split of iid samples (E carries nothing)",
                      "components_bits": comp}
        print(name, comp, flush=True)

    c = {n: rows[n]["components_bits"] for n in rows}
    small = 0.05
    outcomes = {
        "KUR1_uncoupled_all_null": all(
            c["uncoupled"][k] < small
            for k in ("C_env", "C_pair", "C_high")),
        "KUR2_driven_env_dominant":
            c["driven"]["C_env"] > 3 * max(c["driven"]["C_pair"], 1e-9),
        "KUR3_pair12_pair_dominant": (
            c["pair12"]["C_pair"] > 3 * max(c["pair12"]["C_env"], 1e-9)
            and c["pair12"]["C_pair"] > 3 * max(c["pair12"]["C_high"],
                                                1e-9)),
        "KUR4_allcoupled_highorder_beyond_pairwise_MAY_MISS":
            c["allcoupled"]["C_high"] > small,
    }
    report = {
        "status": ("off-design generator test (Kuramoto) for the "
                   "collapse-source ladder; registered in "
                   "V2_ALIGNMENT_PREREGISTRATION.md (KUR); KUR-4 "
                   "preregistered as may-miss"),
        "n_samples": N_SAMPLES,
        "conditions": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "kuramoto_offdesign_ladder.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
