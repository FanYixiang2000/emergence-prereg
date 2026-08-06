"""DETECTOR-VALIDATION: frozen held-out benchmark for the B5 detector.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Addresses
reviewer critique #6 (detector engineered through its own failures,
lacking independent validation). The detector is FROZEN: we import
`adjudicate` from ant_fine_onset exactly as used for every B5 verdict in
the paper. This script only scores NEW synthetic curves with known
labels; it does not modify the detector.

We report, per curve family: onset-detection rate, and across the
control families the false-positive onset rate, plus power/FPR as a
function of grid density and noise. A standard change-point method
(ruptures binary segmentation, RBF cost) is run on the same curves for a
location comparison.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import ruptures as rpt

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"

# Curves are expressed as an OPENNESS curve on [0, log2(3)] because
# `adjudicate` divides by log2(3) internally (its native scale). We build
# openness in [0,1] and multiply by log2(3) so the drop is preserved.
LOG2_3 = math.log2(3)
GATE = 0.1                       # matches ant_fine_onset.GATE
N_PER_FAMILY = 200               # curves per family per condition
REF_DENSITY = 80
REF_SIGMA = 0.02
DENSITIES = [12, 20, 40, 80]
SIGMAS = [0.0, 0.01, 0.02, 0.04, 0.08]
SEED = 4_040_001


def _clip01(y):
    return np.clip(y, 0.0, 1.0)


def make_onset(n, rng, sigma):
    """Slow plateau then fast collapse (logistic knee in first half)."""
    x = np.linspace(0, 1, n)
    t0 = rng.uniform(0.30, 0.55)
    k = rng.uniform(18, 40)
    hi = rng.uniform(0.9, 1.0)
    lo = rng.uniform(0.0, 0.12)
    o = lo + (hi - lo) / (1 + np.exp(k * (x - t0)))
    return _clip01(o + rng.normal(0, sigma, n))


def make_knee(n, rng, sigma):
    """Fast-then-slow deceleration (immediate onset, decelerating)."""
    x = np.linspace(0, 1, n)
    rate = rng.uniform(5, 10)
    hi = rng.uniform(0.9, 1.0)
    lo = rng.uniform(0.0, 0.12)
    o = lo + (hi - lo) * np.exp(-rate * x)
    return _clip01(o + rng.normal(0, sigma, n))


def make_gradual(n, rng, sigma):
    """Single constant-rate linear decline."""
    x = np.linspace(0, 1, n)
    hi = rng.uniform(0.9, 1.0)
    lo = rng.uniform(0.0, 0.15)
    o = hi - (hi - lo) * x
    return _clip01(o + rng.normal(0, sigma, n))


def make_scurve(n, rng, sigma):
    """Symmetric logistic with two knees (onset knee + saturation knee)."""
    x = np.linspace(0, 1, n)
    t0 = rng.uniform(0.4, 0.6)
    k = rng.uniform(8, 16)
    hi = rng.uniform(0.9, 1.0)
    lo = rng.uniform(0.0, 0.1)
    o = lo + (hi - lo) / (1 + np.exp(k * (x - t0)))
    return _clip01(o + rng.normal(0, sigma, n))


def make_flat(n, rng, sigma):
    """No collapse: openness stays high (drop below gate)."""
    hi = rng.uniform(0.9, 1.0)
    o = np.full(n, hi)
    return _clip01(o + rng.normal(0, sigma, n))


FAMILIES = {
    "onset": (make_onset, True),      # label: onset TRUE
    "knee": (make_knee, False),
    "gradual": (make_gradual, False),
    "scurve": (make_scurve, None),    # onset acceptable if truncation isolates onset knee
    "flat": (make_flat, False),
}


def ours_onset(openness_curve, density):
    grid = list(range(density))
    y = np.asarray(openness_curve) * LOG2_3     # adjudicate divides by log2(3)
    adj = adjudicate(grid, y)
    tstar = adj.get("hinge", {}).get("t_star")
    return bool(adj["b5_onset"]), tstar, adj


def ruptures_cp(openness_curve):
    """Standard change-point: binary segmentation, RBF cost, 1 bkp."""
    sig = np.asarray(openness_curve, dtype=float).reshape(-1, 1)
    try:
        algo = rpt.Binseg(model="rbf").fit(sig)
        bkps = algo.predict(n_bkps=1)
        return float(bkps[0])
    except Exception:
        return None


def eval_condition(density, sigma, rng):
    """Return per-family detection stats at a given grid/noise setting."""
    stats = {}
    for fam, (gen, _label) in FAMILIES.items():
        n_onset = 0
        loc_err = []
        for _ in range(N_PER_FAMILY):
            o = gen(density, rng, sigma)
            fired, tstar, _ = ours_onset(o, density)
            n_onset += int(fired)
            if fired and tstar is not None:
                cp = ruptures_cp(o)
                if cp is not None:
                    loc_err.append(abs(tstar - cp) / max(density - 1, 1))
        stats[fam] = {
            "onset_rate": n_onset / N_PER_FAMILY,
            "median_loc_err_frac": (round(float(np.median(loc_err)), 4)
                                    if loc_err else None),
            "n_loc_compared": len(loc_err),
        }
    return stats


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- reference operating point ---
    ref = eval_condition(REF_DENSITY, REF_SIGMA, rng)
    onset_power = ref["onset"]["onset_rate"]
    # false-positive rate across the explicit control families
    control_fams = ["knee", "gradual", "flat"]
    fpr = float(np.mean([ref[f]["onset_rate"] for f in control_fams]))

    # --- grid-density sweep (fixed reference sigma) ---
    by_density = {}
    for d in DENSITIES:
        s = eval_condition(d, REF_SIGMA, rng)
        by_density[str(d)] = {
            "onset_power": s["onset"]["onset_rate"],
            "fpr": float(np.mean([s[f]["onset_rate"] for f in control_fams])),
        }

    # --- noise sweep (fixed reference density) ---
    by_sigma = {}
    for sg in SIGMAS:
        s = eval_condition(REF_DENSITY, sg, rng)
        by_sigma[str(sg)] = {
            "onset_power": s["onset"]["onset_rate"],
            "fpr": float(np.mean([s[f]["onset_rate"] for f in control_fams])),
        }

    powers = [by_density[str(d)]["onset_power"] for d in DENSITIES]
    fprs_density = [by_density[str(d)]["fpr"] for d in DENSITIES]
    sigma_powers = [by_sigma[str(s)]["onset_power"] for s in SIGMAS]
    sigma_fprs = [by_sigma[str(s)]["fpr"] for s in SIGMAS]

    def nondecreasing(a):
        return all(a[i + 1] >= a[i] - 1e-9 for i in range(len(a) - 1))

    def nonincreasing(a):
        return all(a[i + 1] <= a[i] + 1e-9 for i in range(len(a) - 1))

    outcomes = {
        "DV1_specificity_fpr_le_0.05": bool(fpr <= 0.05),
        "DV2_power_ge_0.80": bool(onset_power >= 0.80),
        "DV3_power_monotone_in_density": bool(nondecreasing(powers)),
        "DV3_fpr_not_increasing_in_density": bool(nonincreasing(fprs_density)
                                                  or max(fprs_density) <= 0.05),
        "DV4_power_graceful_in_noise": bool(nonincreasing(sigma_powers)),
        "DV4_fpr_le_0.10_all_noise": bool(max(sigma_fprs) <= 0.10),
        "ref_onset_power": round(onset_power, 4),
        "ref_fpr": round(fpr, 4),
    }

    report = {
        "status": ("DETECTOR-VALIDATION frozen held-out benchmark; "
                   "detector imported unchanged from ant_fine_onset; "
                   "registered before run"),
        "config": {"n_per_family": N_PER_FAMILY, "ref_density": REF_DENSITY,
                   "ref_sigma": REF_SIGMA, "densities": DENSITIES,
                   "sigmas": SIGMAS, "gate": GATE, "seed": SEED,
                   "external_method": "ruptures.Binseg(model=rbf), n_bkps=1"},
        "reference_point": ref,
        "by_density": by_density,
        "by_sigma": by_sigma,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "detector_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== reference (density=80, sigma=0.02) ===")
    for fam, s in ref.items():
        print(f"  {fam:8s} onset_rate={s['onset_rate']:.3f} "
              f"loc_err={s['median_loc_err_frac']} (n={s['n_loc_compared']})")
    print(f"  -> onset power={onset_power:.3f}  control FPR={fpr:.3f}")
    print("=== density sweep ===")
    for d in DENSITIES:
        b = by_density[str(d)]
        print(f"  n={d:3d}: power={b['onset_power']:.3f} fpr={b['fpr']:.3f}")
    print("=== noise sweep ===")
    for s in SIGMAS:
        b = by_sigma[str(s)]
        print(f"  sigma={s:.2f}: power={b['onset_power']:.3f} fpr={b['fpr']:.3f}")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
