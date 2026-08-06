"""Collapse-source decomposition battery (analytic ground truth).

Registered in COLLAPSE_SOURCE_PREREGISTRATION.md (frozen before run).

Implements the multi-source possibility-collapse ladder exactly on a
three-agent, ten-action, binary-environment system:

    C_total = C_individual + C_env + C_pair + C_high

with nested references Q0 (uniform), QI (independent marginals),
QE (conditionally independent given declared E), Qpair (per-E pairwise
maxent via iterative proportional fitting), P (true joint).

Everything is enumerated over the 10^3 joint action space; there is no
sampling noise anywhere in this battery.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
NA = 10  # actions per agent
EPS = 1e-15


# ------------------------------------------------------ generative model

def base_dist(lambda_ind: float) -> np.ndarray:
    """Mix uniform(10) with concentration on actions {0,1}."""
    uni = np.full(NA, 1.0 / NA)
    conc = np.zeros(NA)
    conc[0] = conc[1] = 0.5
    return (1 - lambda_ind) * uni + lambda_ind * conc


def env_tilt(p: np.ndarray, e: int, rho_env: float) -> np.ndarray:
    w = np.array([1.0 + 4.0 * rho_env if (a % 2) == e else 1.0
                  for a in range(NA)])
    q = p * w
    return q / q.sum()


def joint_given_e(e: int, lam: float, rho: float, kappa: float,
                  gamma: float) -> np.ndarray:
    """Exact P(a1,a2,a3 | E=e) as a (10,10,10) array."""
    p = env_tilt(base_dist(lam), e, rho)
    # agent 1
    p1 = p
    # agent 2: copy a1 with prob kappa else independent draw from p
    # agent 3: draw from p; with prob gamma resample restricted to the
    # parity class making b1^b2^b3 = 0.
    even_mass = p[np.arange(NA) % 2 == 0].sum()
    odd_mass = 1.0 - even_mass
    P = np.zeros((NA, NA, NA))
    for a1 in range(NA):
        for a2 in range(NA):
            pa2 = kappa * (1.0 if a2 == a1 else 0.0) + (1 - kappa) * p[a2]
            if pa2 <= 0:
                continue
            need_parity = (a1 % 2) ^ (a2 % 2)  # b3 must equal this
            class_mass = even_mass if need_parity == 0 else odd_mass
            for a3 in range(NA):
                in_class = (a3 % 2) == need_parity
                p_restricted = (p[a3] / class_mass
                                if (in_class and class_mass > 0) else 0.0)
                pa3 = (1 - gamma) * p[a3] + gamma * p_restricted
                P[a1, a2, a3] = p1[a1] * pa2 * pa3
    return P / P.sum()


def joint_with_env(lam: float, rho: float, kappa: float,
                   gamma: float) -> Dict[int, np.ndarray]:
    return {e: joint_given_e(e, lam, rho, kappa, gamma) for e in (0, 1)}


# ------------------------------------------------------------- entropies

def entropy(p: np.ndarray) -> float:
    q = p[p > EPS]
    return float(-(q * np.log2(q)).sum())


def mixture(pe: Dict[int, np.ndarray]) -> np.ndarray:
    return 0.5 * pe[0] + 0.5 * pe[1]


def marginal(p: np.ndarray, axis: int) -> np.ndarray:
    axes = tuple(i for i in range(3) if i != axis)
    return p.sum(axis=axes)


def product_of_marginals(p: np.ndarray) -> np.ndarray:
    m = [marginal(p, i) for i in range(3)]
    return np.einsum("i,j,k->ijk", m[0], m[1], m[2])


def ipf_pairwise(p: np.ndarray, iters: int = 400) -> np.ndarray:
    """Maximum-entropy joint matching the three pairwise marginals of p
    (iterative proportional fitting on the 10x10x10 table)."""
    targets = {
        (0, 1): p.sum(axis=2),
        (0, 2): p.sum(axis=1),
        (1, 2): p.sum(axis=0),
    }
    q = np.full_like(p, 1.0 / p.size)
    for _ in range(iters):
        for (i, j), tgt in targets.items():
            axis = ({0, 1, 2} - {i, j}).pop()
            cur = q.sum(axis=axis)
            ratio = np.where(cur > EPS, tgt / np.maximum(cur, EPS), 0.0)
            shape = [1, 1, 1]
            shape[i] = NA
            shape[j] = NA
            q = q * ratio.reshape(shape)
        s = q.sum()
        if s > 0:
            q = q / s
    return q


def ladder(pe: Dict[int, np.ndarray], declare_env: bool = True) -> Dict:
    """Nested reference ladder and the source components (bits)."""
    p_mix = mixture(pe)
    h_p = entropy(p_mix)  # H of true joint (marginalized over E)
    h_q0 = 3 * math.log2(NA)
    qi = product_of_marginals(p_mix)
    h_qi = entropy(qi)
    if declare_env:
        qe = mixture({e: product_of_marginals(pe[e]) for e in (0, 1)})
        h_qe = entropy(qe)
        qpair = mixture({e: ipf_pairwise(pe[e]) for e in (0, 1)})
        h_qpair = entropy(qpair)
        # numerical guard: enforce ladder monotonicity records
        comp = {
            "C_individual": h_q0 - h_qi,
            "C_env": h_qi - h_qe,
            "C_pair": h_qe - h_qpair,
            "C_high": h_qpair - h_p,
        }
        hs = {"H_Q0": h_q0, "H_QI": h_qi, "H_QE": h_qe,
              "H_Qpair": h_qpair, "H_P": h_p}
    else:
        # hidden environment: ladder skips QE; env correlation flows
        # downstream into pair/high attribution.
        qpair = ipf_pairwise(p_mix)
        h_qpair = entropy(qpair)
        comp = {
            "C_individual": h_q0 - h_qi,
            "C_env": 0.0,
            "C_pair": h_qi - h_qpair,
            "C_high": h_qpair - h_p,
        }
        hs = {"H_Q0": h_q0, "H_QI": h_qi, "H_Qpair": h_qpair, "H_P": h_p}
    comp["C_total"] = h_q0 - h_p
    return {"entropies": hs, "components": comp}


# ------------------------------------------------------------- batteries

BASE = 0.2
HIGH = 0.8
KNOBS = ("lambda_ind", "rho_env", "kappa_pair", "gamma_high")
KNOB_TO_COMPONENT = {
    "lambda_ind": "C_individual",
    "rho_env": "C_env",
    "kappa_pair": "C_pair",
    "gamma_high": "C_high",
}


def components_at(**kw) -> Dict[str, float]:
    pe = joint_with_env(kw.get("lambda_ind", 0.0), kw.get("rho_env", 0.0),
                        kw.get("kappa_pair", 0.0),
                        kw.get("gamma_high", 0.0))
    return ladder(pe)["components"]


def sd1_nesting() -> Tuple[bool, list]:
    grid = [0.0, 0.4, 0.8]
    violations = []
    for lam, rho, kap, gam in itertools.product(grid, repeat=4):
        pe = joint_with_env(lam, rho, kap, gam)
        hs = ladder(pe)["entropies"]
        seq = [hs["H_Q0"], hs["H_QI"], hs["H_QE"], hs["H_Qpair"], hs["H_P"]]
        for a, b in zip(seq, seq[1:]):
            if b > a + 1e-6:
                violations.append({"knobs": [lam, rho, kap, gam],
                                   "pair": [a, b]})
    return len(violations) == 0, violations


def sd2_diagonal() -> Tuple[bool, Dict]:
    detail = {}
    ok = True
    for knob in KNOBS:
        lo = {k: BASE for k in KNOBS}
        hi = {k: BASE for k in KNOBS}
        lo[knob] = BASE
        hi[knob] = HIGH
        c_lo = components_at(**lo)
        c_hi = components_at(**hi)
        deltas = {c: c_hi[c] - c_lo[c]
                  for c in ("C_individual", "C_env", "C_pair", "C_high")}
        own = KNOB_TO_COMPONENT[knob]
        others = {c: d for c, d in deltas.items() if c != own}
        passed = deltas[own] > max(others.values())
        ok = ok and passed
        detail[knob] = {"deltas_bits": deltas, "own": own,
                        "own_delta": deltas[own],
                        "max_other_delta": max(others.values()),
                        "pass": passed}
    return ok, detail


def sd3_dissociations() -> Tuple[bool, Dict]:
    cases = {
        "pure_env": dict(rho_env=HIGH),
        "pure_pair": dict(kappa_pair=HIGH),
        "pure_high": dict(gamma_high=HIGH),
    }
    comps = {name: components_at(**kw) for name, kw in cases.items()}
    checks = {
        "env_no_pair": comps["pure_env"]["C_pair"] < 0.02,
        "env_no_high": comps["pure_env"]["C_high"] < 0.02,
        "pair_no_high": comps["pure_pair"]["C_high"] < 0.02,
        "pair_positive": comps["pure_pair"]["C_pair"] > 0.2,
        "high_no_pair": comps["pure_high"]["C_pair"] < 0.02,
        "high_positive": comps["pure_high"]["C_high"] > 0.2,
    }
    return all(checks.values()), {"components": comps, "checks": checks}


def sd4_hidden_env() -> Tuple[bool, Dict]:
    pe = joint_with_env(0.0, HIGH, 0.0, 0.0)
    declared = ladder(pe, declare_env=True)["components"]
    hidden = ladder(pe, declare_env=False)["components"]
    misattributed = hidden["C_pair"] + hidden["C_high"]
    return misattributed > 0.1, {
        "declared": declared, "hidden": hidden,
        "misattributed_bits": misattributed,
    }


def profile_value(kind: str, s: float) -> float:
    if kind == "linear":
        return s
    if kind == "sigmoid":
        return 1.0 / (1.0 + math.exp(-12.0 * (s - 0.5)))
    if kind == "step":
        return 0.0 if s < 0.5 else 1.0
    raise ValueError(kind)


def sd5_time_phenotype(n_steps: int = 41) -> Tuple[bool, Dict]:
    widths = {}
    curves = {}
    for kind in ("linear", "sigmoid", "step"):
        cs = []
        for i in range(n_steps):
            s = i / (n_steps - 1)
            gam = HIGH * profile_value(kind, s)
            cs.append(components_at(gamma_high=gam)["C_high"])
        cs = np.array(cs)
        total = cs[-1] - cs[0]
        t10 = int(np.argmax(cs >= cs[0] + 0.1 * total))
        t90 = int(np.argmax(cs >= cs[0] + 0.9 * total))
        widths[kind] = (t90 - t10) / (n_steps - 1)
        curves[kind] = [round(float(c), 5) for c in cs]
    ok = widths["step"] < widths["sigmoid"] < widths["linear"]
    return ok, {"widths_10_90": widths, "curves_C_high": curves}


def main() -> None:
    results = {}
    sd1_ok, sd1_v = sd1_nesting()
    results["SD1_nesting"] = {"pass": sd1_ok, "n_violations": len(sd1_v),
                              "violations": sd1_v[:5]}
    sd2_ok, sd2_d = sd2_diagonal()
    results["SD2_diagonal_dominance"] = {"pass": sd2_ok, "detail": sd2_d}
    sd3_ok, sd3_d = sd3_dissociations()
    results["SD3_dissociations"] = {"pass": sd3_ok, "detail": sd3_d}
    sd4_ok, sd4_d = sd4_hidden_env()
    results["SD4_hidden_env_misattribution"] = {"pass": sd4_ok,
                                                "detail": sd4_d}
    sd5_ok, sd5_d = sd5_time_phenotype()
    results["SD5_time_phenotype"] = {"pass": sd5_ok, "detail": sd5_d}

    report = {
        "status": ("collapse-source decomposition battery on analytic "
                   "ground truth; registered in "
                   "COLLAPSE_SOURCE_PREREGISTRATION.md; instrument "
                   "validation for the multi-source possibility-collapse "
                   "ontology, not a real-system emergence claim"),
        "system": {"agents": 3, "actions": NA, "env_states": 2,
                   "exact_enumeration": True},
        "results": results,
        "registered_outcomes": {
            "SD1": sd1_ok, "SD2": sd2_ok, "SD3": sd3_ok,
            "SD4": sd4_ok, "SD5": sd5_ok,
        },
    }
    out = OUTPUTS / "collapse_source_decomposition.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    for name, res in results.items():
        print(name, "PASS" if res["pass"] else "FAIL")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
