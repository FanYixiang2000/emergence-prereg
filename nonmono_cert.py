"""NONMONO-CERT: settled-openness certification for non-monotone commitment.

Preregistered in V2_ALIGNMENT_PREREGISTRATION.md (2026-08-04) before this
file was written.  The settled openness O~(t) = max_{s>=t} O(s) records the
irrevocably closed part of the possibility space; the frozen B5 detector is
applied to O~ with all thresholds unchanged.  Two new frozen constants only:
END_GUARD = 10 points, PERSIST_TOL = 0.1.

Stage V: synthetic validation library (power / FPR), run first.
Stage R: one-shot application to the 8 stored ring seeds + 2 cramped
controls, only if V1 and V2 pass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LOG2_3 = math.log2(3)

END_GUARD = 10          # frozen (preregistered)
PERSIST_TOL = 0.1       # frozen (preregistered, inherited from certificate)


def median_filter(o: np.ndarray, w: int = 5) -> np.ndarray:
    half = w // 2
    padded = np.pad(o, half, mode="edge")
    return np.array([np.median(padded[i:i + w]) for i in range(len(o))])


def settled(openness: np.ndarray) -> np.ndarray:
    """Future-max envelope: O~(t) = max_{s>=t} O(s)."""
    return np.maximum.accumulate(openness[::-1])[::-1]


def certify_settled(grid, openness) -> dict:
    o = median_filter(np.asarray(openness, dtype=float))
    # resolution-matched quantization: differences below the persistence
    # tolerance are not meaningful possibility distinctions
    q = np.round(o / (PERSIST_TOL / 2)) * (PERSIST_TOL / 2)
    env = settled(q)
    # refuse if the irrevocable closure completes only inside the guard
    if env[-END_GUARD] - env[-1] > PERSIST_TOL:
        return {"verdict": "commitment_not_persistent_within_window",
                "b5_onset": False}
    adj = adjudicate(np.asarray(grid, dtype=float), env * LOG2_3)
    adj["settled_envelope"] = [round(v, 4) for v in env]
    return adj


# Stage V

def _osc(rng, n, lo, hi, period):
    ph = rng.uniform(0, 2 * math.pi)
    t = np.arange(n)
    base = (hi + lo) / 2 + (hi - lo) / 2 * np.sin(2 * math.pi * t / period + ph)
    return base


def gen_positive(rng, n=100):
    """Oscillating with re-openings, permanent lock at T_lock."""
    frac = rng.choice([0.3, 0.5, 0.7])
    t_lock = int(frac * n)
    sigma = rng.choice([0.02, 0.05])
    period = rng.uniform(8, 20)
    dip = rng.uniform(0.4, 0.7)
    committed = rng.uniform(0.05, 0.25)
    pre = _osc(rng, t_lock, dip, 1.0, period)
    # decay from the pre-lock level to the committed level over ~8 points
    n_post = n - t_lock
    decay = committed + (pre[-1] - committed) * np.exp(-np.arange(n_post) / 3.0)
    o = np.concatenate([pre, decay]) + rng.normal(0, sigma, n)
    return np.clip(o, 0, 1), t_lock


def gen_negative(rng, kind, n=100):
    sigma = rng.choice([0.02, 0.05])
    if kind == "stationary_osc":
        o = _osc(rng, n, rng.uniform(0.4, 0.7), 1.0, rng.uniform(8, 20))
    elif kind == "gradual_osc":
        env_hi = np.linspace(1.0, 0.2, n)
        o = env_hi - rng.uniform(0.0, 0.3) * (0.5 + 0.5 * np.sin(
            2 * math.pi * np.arange(n) / rng.uniform(8, 20)))
        o = np.clip(o, 0.02, None)
    elif kind == "recovering_dip":
        o = _osc(rng, n, 0.75, 1.0, rng.uniform(8, 20))
        d0 = int(rng.uniform(0.4, 0.7) * n)
        w = int(rng.uniform(5, 12))
        o[d0:d0 + w] = rng.uniform(0.05, 0.2)
    elif kind == "late_dip":
        o = _osc(rng, n, 0.75, 1.0, rng.uniform(8, 20))
        d0 = n - int(rng.uniform(2, END_GUARD - 1))
        o[d0:] = rng.uniform(0.05, 0.2)
    else:
        raise ValueError(kind)
    return np.clip(o + rng.normal(0, sigma, n), 0, 1)


def stage_v() -> dict:
    rng = np.random.default_rng(97001)
    grid = np.arange(100, dtype=float)
    span = grid[-1] - grid[0]

    n_pos_ok = 0
    pos_records = []
    for i in range(300):
        o, t_lock = gen_positive(rng)
        adj = certify_settled(grid, o)
        # ground truth for the settled object: last return to the plateau
        f = median_filter(o)
        plateau = float(np.max(f))
        t_true = int(np.max(np.where(f >= plateau - 0.1)[0]))
        ok = bool(adj.get("b5_onset")) and abs(
            adj["hinge"]["t_star"] - t_true) <= 0.10 * span
        n_pos_ok += ok
        pos_records.append({"t_lock": t_lock, "t_true": t_true,
                            "onset": bool(adj.get("b5_onset")),
                            "t_star": adj.get("hinge", {}).get("t_star"),
                            "ok": bool(ok)})
    power = n_pos_ok / 300

    neg = {}
    n_fp = n_neg = 0
    for kind in ("stationary_osc", "gradual_osc", "recovering_dip", "late_dip"):
        fp = 0
        for i in range(100):
            o = gen_negative(rng, kind)
            adj = certify_settled(grid, o)
            fp += bool(adj.get("b5_onset"))
        neg[kind] = fp / 100
        n_fp += fp
        n_neg += 100
    fpr = n_fp / n_neg

    return {"power": float(power), "V1_power_ge_090": bool(power >= 0.90),
            "fpr_pooled": float(fpr), "fpr_by_family": neg,
            "V2_fpr_le_005": bool(fpr <= 0.05),
            "n_pos": 300, "n_neg": n_neg,
            "pos_sample": pos_records[:5]}


# Stage R

def stage_r() -> dict:
    ring, cramped = {}, {}
    orig = json.load(open(OUTPUTS / "overcooked_ring_convention.json"))
    ext = json.load(open(OUTPUTS / "oc_ring_ext.json"))

    def pull(rec):
        grid = np.asarray(rec["grid"], dtype=float)
        o = np.asarray([c["circulation_openness"] for c in rec["curves"]])
        return grid, o, rec["capability_crossing"]

    for s, rec in orig["systems"]["ring"].items():
        ring[s] = pull(rec)
    for s, rec in ext["ext_seeds"].items():
        ring[s] = pull(rec)
    for s, rec in orig["systems"]["cramped"].items():
        cramped[s] = pull(rec)

    out = {"ring": {}, "cramped": {}}
    n_cert = 0
    leads = []
    for s, (grid, o, cross) in ring.items():
        adj = certify_settled(grid, o)
        cert = bool(adj.get("b5_onset"))
        t_star = adj.get("hinge", {}).get("t_star") if cert else None
        n_cert += cert
        if cert:
            leads.append(bool(t_star <= cross) if cross is not None else None)
        out["ring"][s] = {
            "certified": cert, "t_star": t_star,
            "capability_crossing": cross,
            "verdict": adj.get("verdict", "hinge_tested"),
            "dBIC": adj.get("hinge", {}).get("delta_bic"),
            "drop": adj.get("drop")}
    n_cramped = 0
    for s, (grid, o, cross) in cramped.items():
        adj = certify_settled(grid, o)
        n_cramped += bool(adj.get("b5_onset"))
        out["cramped"][s] = {"certified": bool(adj.get("b5_onset")),
                             "verdict": adj.get("verdict", "hinge_tested"),
                             "drop": adj.get("drop")}
    out["registered_outcomes"] = {
        "R1_ge_4of8_certified": n_cert >= 4,
        "n_certified": n_cert,
        "R2_all_certified_lead_capability": bool(leads) and all(leads),
        "R3_cramped_zero": n_cramped == 0,
        "n_cramped_certified": n_cramped}
    return out


def main() -> None:
    v = stage_v()
    print("Stage V:", {k: v[k] for k in
                       ("power", "V1_power_ge_090", "fpr_pooled",
                        "V2_fpr_le_005", "fpr_by_family")})
    result = {"stage_v": v}
    if v["V1_power_ge_090"] and v["V2_fpr_le_005"]:
        r = stage_r()
        result["stage_r"] = r
        print("Stage R:", json.dumps(r["registered_outcomes"], indent=1))
        for s, rec in r["ring"].items():
            print(f"  ring {s}: cert={rec['certified']} t*={rec['t_star']} "
                  f"cross={rec['capability_crossing']} dBIC={rec['dBIC']} "
                  f"verdict={rec['verdict']}")
        for s, rec in r["cramped"].items():
            print(f"  cramped {s}: cert={rec['certified']} "
                  f"verdict={rec['verdict']}")
    else:
        result["stage_r"] = "NOT RUN (validation failed, preregistered stop)"
        print("Validation failed; ring adjudication not performed.")
    def _san(x):
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        raise TypeError(type(x))

    with open(OUTPUTS / "nonmono_cert.json", "w") as f:
        json.dump(result, f, indent=1, default=_san)
    print("written outputs/nonmono_cert.json")


if __name__ == "__main__":
    main()
