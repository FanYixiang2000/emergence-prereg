"""KUR-N10: KUR-SCALE seed extension, 2 -> 10 seeds per coupling.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Simulation and detector are
byte-identical to KUR-SCALE; only the seed set grows. Bootstrap
confidence intervals and Spearman monotonicity tests are added as
registered.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
KS = (0.9, 1.1, 1.5, 2.0, 2.5)
SEEDS = tuple(range(82_001, 82_011))
N_BOOT = 10_000
N_PERM = 100_000


def one(args):
    k, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from kuramoto_breakpoint import simulate
    from kuramoto_breakpoint_r2 import adjudicate
    adj = adjudicate(simulate(k, seed))
    return k, seed, adj


def seed_pass(adj) -> bool:
    return bool(adj.get("onset_pass", False))


def main() -> None:
    jobs = [(k, s) for k in KS for s in SEEDS]
    per_k: dict[str, dict[str, dict]] = {str(k): {} for k in KS}
    with ProcessPoolExecutor(max_workers=25) as ex:
        for k, seed, adj in ex.map(one, jobs):
            per_k[str(k)][str(seed)] = adj
            h = adj.get("hinge", {})
            print(f"K={k} seed {seed}: gate={adj['gate_passed']} "
                  f"onset_pass={adj.get('onset_pass')} "
                  f"t*={h.get('t_star')} "
                  f"slope_after={h.get('slope_after')}", flush=True)

    rng = np.random.default_rng(2026_08_15)

    def values(k, field):
        return [abs(per_k[str(k)][str(s)]["hinge"][field]) for s in SEEDS
                if seed_pass(per_k[str(k)][str(s)])]

    def boot_ci(vals):
        if len(vals) < 2:
            return None
        arr = np.asarray(vals, dtype=float)
        idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
        means = arr[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return [float(lo), float(hi)]

    summary = {}
    for k in KS:
        ts, sl = values(k, "t_star"), values(k, "slope_after")
        n_pass = sum(seed_pass(per_k[str(k)][str(s)]) for s in SEEDS)
        summary[str(k)] = {
            "n_pass": n_pass,
            "mean_t_star": float(np.mean(ts)) if ts else None,
            "ci_t_star": boot_ci(ts),
            "mean_post_slope": float(np.mean(sl)) if sl else None,
            "ci_post_slope": boot_ci(sl),
        }

    kn1 = all(seed_pass(per_k[str(k)][str(s)])
              for k in KS if k >= 1.1 for s in SEEDS)
    k09 = per_k["0.9"]
    k09_gated_null = all(not k09[str(s)]["gate_passed"] for s in SEEDS)
    k09_ts = [k09[str(s)]["hinge"]["t_star"] for s in SEEDS
              if seed_pass(k09[str(s)])]
    higher_means = [summary[str(k)]["mean_t_star"] for k in KS if k >= 1.1]
    higher_means = [t for t in higher_means if t is not None]
    k09_slower = bool(k09_ts and higher_means
                      and min(k09_ts) > max(higher_means))
    kn1_nearcrit = bool(k09_gated_null or k09_slower)

    passing = [k for k in KS
               if all(seed_pass(per_k[str(k)][str(s)]) for s in SEEDS)]
    tmeans = [summary[str(k)]["mean_t_star"] for k in passing]
    smeans = [summary[str(k)]["mean_post_slope"] for k in passing]
    kn2_strict = bool(len(passing) >= 2
                      and all(a > b for a, b in zip(tmeans, tmeans[1:])))
    kn3_strict = bool(len(passing) >= 2
                      and all(a < b for a, b in zip(smeans, smeans[1:])))

    def spearman_perm(pairs, sign):
        """One-sided permutation p for Spearman rho having the
        registered sign; seeded, N_PERM permutations."""
        from scipy.stats import spearmanr
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        rho = spearmanr(x, y).statistic
        prng = np.random.default_rng(7_2026)
        count = 0
        for _ in range(N_PERM):
            r = spearmanr(x, prng.permutation(y)).statistic
            if sign * r >= sign * rho:
                count += 1
        return float(rho), (count + 1) / (N_PERM + 1)

    pooled_t = [(k, abs(per_k[str(k)][str(s)]["hinge"]["t_star"]))
                for k in KS for s in SEEDS
                if seed_pass(per_k[str(k)][str(s)])]
    pooled_s = [(k, abs(per_k[str(k)][str(s)]["hinge"]["slope_after"]))
                for k in KS for s in SEEDS
                if seed_pass(per_k[str(k)][str(s)])]
    rho_t, p_t = spearman_perm(pooled_t, sign=-1)
    rho_s, p_s = spearman_perm(pooled_s, sign=+1)

    outcomes = {
        "KN1_existence": bool(kn1),
        "KN1_nearcritical_consistent": kn1_nearcrit,
        "KN2_strict_mean_monotone": kn2_strict,
        "KN2_spearman_rho": rho_t,
        "KN2_perm_p": p_t,
        "KN2_pass": bool(kn2_strict and p_t < 0.05),
        "KN3_strict_mean_monotone": kn3_strict,
        "KN3_spearman_rho": rho_s,
        "KN3_perm_p": p_s,
        "KN3_pass": bool(kn3_strict and p_s < 0.05),
        "passing_K": passing,
        "per_K_summary": summary,
    }
    report = {"status": ("KUR-N10 seed extension of KUR-SCALE; detector "
                         "and simulation unchanged; registered in "
                         "METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md "
                         "before run"),
              "config": {"Ks": KS, "seeds": SEEDS, "n_boot": N_BOOT,
                         "n_perm": N_PERM},
              "per_K": per_k,
              "registered_outcomes": outcomes}
    out = OUTPUTS / "kuramoto_scale_n10.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
