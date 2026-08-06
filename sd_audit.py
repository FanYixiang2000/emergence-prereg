"""SD-AUDIT: stress audit of the collapse-source decomposition ladder.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Reuses the
exact-enumeration ladder of collapse_source_decomposition.py unchanged.

  SDA-1 mixed sources: two knobs HIGH jointly, own components top-2.
  SDA-2 off-family generators: modular-sum triple (pure high-order),
        Markov copy chain (pure pairwise), 50 random Dirichlet joints
        (non-negativity + exact telescoping identity).
  SDA-3 nesting order: the ladder stages are NESTED constraint sets
        (mixture marginals subset per-env marginals subset per-env
        pairwise subset full joint), so the individual/environment swap
        is not definable within the filtration; the genuine order
        freedom -- declaring the environment before vs after the
        pairwise stage -- is computed across the 81-cell grid, with the
        individual and high-order components invariant by construction.
  SDA-4 sample complexity: component estimates from finite samples of a
        mixed-source truth, bias/error curves and finite-sample
        negativity rates (no clipping).
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from collapse_source_decomposition import (BASE, HIGH, KNOB_TO_COMPONENT,
                                           KNOBS, NA, components_at, entropy,
                                           ipf_pairwise, joint_with_env,
                                           ladder, mixture,
                                           product_of_marginals)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
COMPS = ("C_individual", "C_env", "C_pair", "C_high")


# ------------------------------------------------------------ SDA-1

def sda1_mixed_sources():
    ref = components_at(**{k: BASE for k in KNOBS})
    detail = {}
    ok = True
    for ka, kb in itertools.combinations(KNOBS, 2):
        kw = {k: BASE for k in KNOBS}
        kw[ka] = HIGH
        kw[kb] = HIGH
        comp = components_at(**kw)
        deltas = {c: comp[c] - ref[c] for c in COMPS}
        own = {KNOB_TO_COMPONENT[ka], KNOB_TO_COMPONENT[kb]}
        top2 = set(sorted(deltas, key=deltas.get, reverse=True)[:2])
        passed = top2 == own
        ok = ok and passed
        detail[f"{ka}+{kb}"] = {
            "deltas_bits": {c: round(d, 5) for c, d in deltas.items()},
            "own": sorted(own), "top2": sorted(top2), "pass": passed}
    return ok, detail


# ------------------------------------------------------------ SDA-2

def modular_sum_joint():
    P = np.zeros((NA, NA, NA))
    for a1 in range(NA):
        for a2 in range(NA):
            P[a1, a2, (a1 + a2) % NA] = 1.0
    return P / P.sum()


def markov_chain_joint(copy_p: float = 0.7):
    P = np.zeros((NA, NA, NA))
    u = 1.0 / NA
    for a1 in range(NA):
        for a2 in range(NA):
            p2 = copy_p * (a2 == a1) + (1 - copy_p) * u
            for a3 in range(NA):
                p3 = copy_p * (a3 == a2) + (1 - copy_p) * u
                P[a1, a2, a3] = u * p2 * p3
    return P / P.sum()


def sda2_off_family():
    detail = {}
    # (a) modular sum: pure high-order
    pe = {0: modular_sum_joint(), 1: modular_sum_joint()}
    comp_a = ladder(pe)["components"]
    a_ok = comp_a["C_pair"] <= 0.02 and comp_a["C_high"] >= 1.0
    detail["modular_sum"] = {
        "components": {c: round(comp_a[c], 5) for c in COMPS},
        "pass": bool(a_ok)}
    # (b) Markov copy chain: pure pairwise
    pe = {0: markov_chain_joint(), 1: markov_chain_joint()}
    comp_b = ladder(pe)["components"]
    b_ok = comp_b["C_high"] <= 0.02 and comp_b["C_pair"] >= 0.2
    detail["markov_chain"] = {
        "components": {c: round(comp_b[c], 5) for c in COMPS},
        "pass": bool(b_ok)}
    # (c) 50 random Dirichlet(0.3) env-conditional joints
    rng = np.random.default_rng(20260804)
    min_comp = math.inf
    max_identity_gap = 0.0
    neg_count = 0
    for _ in range(50):
        pe = {e: rng.dirichlet(np.full(NA ** 3, 0.3)).reshape(NA, NA, NA)
              for e in (0, 1)}
        res = ladder(pe)["components"]
        gap = abs(res["C_total"] - sum(res[c] for c in COMPS))
        max_identity_gap = max(max_identity_gap, gap)
        m = min(res[c] for c in COMPS)
        min_comp = min(min_comp, m)
        if m < -1e-9:
            neg_count += 1
    c_ok = neg_count == 0 and max_identity_gap <= 1e-9
    detail["dirichlet_50"] = {
        "min_component_bits": float(min_comp),
        "n_with_negative_component": neg_count,
        "max_identity_gap_bits": float(max_identity_gap),
        "pass": bool(c_ok)}
    return bool(a_ok and b_ok and c_ok), detail


# ------------------------------------------------------------ SDA-3

def ladder_env_after_pair(pe):
    """Alternative chain: Q0 -> QI -> Qpair(mixture) -> Qpair(per-env) -> P.

    The environment is declared AFTER the pairwise stage; C_individual
    and C_high are shared with the published chain by construction.
    """
    p_mix = mixture(pe)
    h_p = entropy(p_mix)
    h_q0 = 3 * math.log2(NA)
    h_qi = entropy(product_of_marginals(p_mix))
    h_qpair_mix = entropy(ipf_pairwise(p_mix))
    h_qpair_e = entropy(mixture({e: ipf_pairwise(pe[e]) for e in (0, 1)}))
    return {
        "C_individual": h_q0 - h_qi,
        "C_pair": h_qi - h_qpair_mix,
        "C_env": h_qpair_mix - h_qpair_e,
        "C_high": h_qpair_e - h_p,
        "C_total": h_q0 - h_p,
    }


def sda3_nesting_order():
    grid = [0.0, 0.4, 0.8]
    max_env_shift = 0.0
    max_ind_shift = 0.0
    max_high_shift = 0.0
    worst = None
    for lam, rho, kap, gam in itertools.product(grid, repeat=4):
        pe = joint_with_env(lam, rho, kap, gam)
        a = ladder(pe)["components"]
        b = ladder_env_after_pair(pe)
        env_shift = abs(a["C_env"] - b["C_env"])
        max_ind_shift = max(max_ind_shift, abs(a["C_individual"]
                                               - b["C_individual"]))
        max_high_shift = max(max_high_shift, abs(a["C_high"] - b["C_high"]))
        if env_shift > max_env_shift:
            max_env_shift = env_shift
            worst = {"knobs": [lam, rho, kap, gam],
                     "published": {c: round(a[c], 5) for c in COMPS},
                     "env_after_pair": {c: round(b[c], 5) for c in COMPS}}
    invariant_ok = max_ind_shift <= 1e-6 and max_high_shift <= 1e-6
    return bool(invariant_ok), {
        "note": ("individual/environment stages cannot be literally "
                 "swapped: mixture marginals are a subset of per-env "
                 "marginals, so the published chain is a filtration; the "
                 "genuine order freedom is declaring the environment "
                 "before vs after the pairwise stage, quantified here"),
        "max_individual_shift_bits": float(max_ind_shift),
        "max_high_shift_bits": float(max_high_shift),
        "max_env_pair_split_shift_bits": float(max_env_shift),
        "worst_case": worst,
    }


# ------------------------------------------------------------ SDA-4

TRUTH_KNOBS = {k: 0.4 for k in KNOBS}
NS = (300, 1_000, 3_000, 10_000, 30_000, 100_000)
N_REPS = 20


def sample_counts(pe, n, rng):
    counts = {}
    ne = rng.binomial(n, 0.5)
    for e, m in ((0, ne), (1, n - ne)):
        flat = pe[e].ravel()
        counts[e] = rng.multinomial(m, flat).reshape(NA, NA, NA)
    return counts


def sda4_sample_complexity():
    pe_true = joint_with_env(TRUTH_KNOBS["lambda_ind"],
                             TRUTH_KNOBS["rho_env"],
                             TRUTH_KNOBS["kappa_pair"],
                             TRUTH_KNOBS["gamma_high"])
    truth = ladder(pe_true)["components"]
    rng = np.random.default_rng(20260805)
    table = {}
    neg_stats = {}
    for n in NS:
        errs = {c: [] for c in COMPS}
        negs = []
        for _ in range(N_REPS):
            cnt = sample_counts(pe_true, n, rng)
            pe_hat = {e: cnt[e] / max(cnt[e].sum(), 1) for e in (0, 1)}
            est = ladder(pe_hat)["components"]
            for c in COMPS:
                errs[c].append(est[c] - truth[c])
            negs.append(min(est[c] for c in COMPS))
        table[str(n)] = {
            c: {"median_abs_err": round(float(np.median(np.abs(errs[c]))), 5),
                "median_bias": round(float(np.median(errs[c])), 5)}
            for c in COMPS}
        neg_vals = [v for v in negs if v < 0]
        neg_stats[str(n)] = {
            "negativity_rate": round(len(neg_vals) / N_REPS, 3),
            "worst_negative_bits": (round(float(min(neg_vals)), 5)
                                    if neg_vals else 0.0)}
    at_target = table[str(30_000)]
    ok = all(at_target[c]["median_abs_err"] <= 0.05 for c in COMPS)
    return bool(ok), {
        "truth_components": {c: round(truth[c], 5) for c in COMPS},
        "error_table": table,
        "negativity": neg_stats,
    }


def main() -> None:
    results = {}
    ok1, d1 = sda1_mixed_sources()
    results["SDA1_mixed_sources"] = {"pass": ok1, "detail": d1}
    print("SDA1", "PASS" if ok1 else "FAIL", flush=True)
    ok2, d2 = sda2_off_family()
    results["SDA2_off_family"] = {"pass": ok2, "detail": d2}
    print("SDA2", "PASS" if ok2 else "FAIL", flush=True)
    ok3, d3 = sda3_nesting_order()
    results["SDA3_nesting_order"] = {"pass": ok3, "detail": d3}
    print("SDA3", "PASS" if ok3 else "FAIL", flush=True)
    ok4, d4 = sda4_sample_complexity()
    results["SDA4_sample_complexity"] = {"pass": ok4, "detail": d4}
    print("SDA4", "PASS" if ok4 else "FAIL", flush=True)

    report = {
        "status": ("SD-AUDIT stress audit of the collapse-source ladder; "
                   "mixed sources, off-family generators, nesting order, "
                   "finite samples; ladder imported unchanged; registered "
                   "before run"),
        "results": results,
        "registered_outcomes": {"SDA1": ok1, "SDA2": ok2,
                                "SDA3": ok3, "SDA4": ok4},
    }
    out = OUTPUTS / "sd_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
