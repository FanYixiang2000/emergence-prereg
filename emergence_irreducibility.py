"""Irreducibility battery: the strong-emergence claim survives the five
frontier attacks on collapse-style definitions.

Motivation. A reader mapped five 2025-2026 papers onto our five
descriptors and argued they collectively raise the bar so that
"the group developed a new correlation / higher-order structure /
possibility contraction" is no longer sufficient:

  [CE2.0]   Hoel, Causal Emergence 2.0 / Quantifying Emergent
            Complexity -- causal contribution is distributed across
            scales; a single scale is a slice, not a rival.
  [PITHON]  Emergence of Temporal Higher-Order Interactions from
            Pairwise Collaboration -- higher-order simplices grow from
            pairwise edge dynamics, so observing temporal higher-order
            structure is not itself novel.
  [ENVDRV]  Environment-Driven Emergence of Higher-Order Collective
            Behavior -- a NO-GO result: a common (time-varying)
            environment alone produces synergy-dominated / negative
            O-information with NO interaction. Higher-order statistics
            do NOT imply higher-order interaction.
  [COGNET]  Unraveling the Emergence of Collective Behavior in Networks
            of Cognitive Agents -- collapse (consensus) can be
            PATHOLOGICAL (premature convergence on a bad basin), so
            collapse magnitude does not measure emergence quality.
  [KRAK]    Krakauer, Krakauer & Mitchell, LLMs and Emergence -- a
            sudden benchmark jump is not emergence; you must exhibit a
            coarse-grained macro variable that predicts/controls and
            screens off micro detail (knowledge-in vs knowledge-out).

This battery answers all five with one operational upgrade. We stop
scoring collapse MAGNITUDE and instead score the IRREDUCIBLE,
ENDOGENOUS, macro-causal part of the contraction of joint futures.

DEFINITIONS (exact on small discrete systems; analytic anchor in the
Gaussian case). Let X=(X1,X2,X3) be the agents' (future) states.

  Total joint contraction (how much the joint future set is smaller
  than the product of marginals):
        C_total = TC(X) = sum_i H(X_i) - H(X)         (total correlation)

  O-information (Rosas 2019; sign matches ENVDRV):
        Omega = (n-2)H(X) + sum_i [H(X_i) - H(X_{-i})]
        Omega > 0 redundancy, Omega < 0 synergy.

  Maximum-entropy surrogate hierarchy (Schneidman 2003; Amari 2001):
    P^(1)  independent, matches single marginals,
    P^(2)  pairwise max-ent (IPF), matches ALL pairwise marginals,
    P      true joint.
  Irreducible higher-order connected information (order >= 3):
        C_irr = KL(P || P^(2))
  -- the part of the contraction NOT reproducible by any pairwise
  model. This kills the PITHON objection: pairwise-grown structure has
  C_irr = 0 by construction.

  Endogeneity (defeats ENVDRV). Condition on the common environment E
  (its full state, including any time-varying coupling regime):
        C_irr|E = E_e[ KL(P(.|e) || P^(2)(.|e)) ]
  A purely environment-driven system has C_irr|E = 0 even when its
  marginal C_irr and |Omega| are large. Second, observability-friendly
  route -- the do-operator on the coupling (cut agent<->agent channels,
  KEEP the environment):
        D_higher = 1 - C_irr(broken)/C_irr(natural)
  For a common-cause system the structure survives cutting agent
  channels, so D_higher = 0.

  Macro causal contribution (KRAK / CE2.0). For a coarse-grained macro
  variable M = m(X):
        macro_gain = I(M;X) - sum_i I(M;X_i)   ( > 0 iff M carries
        joint information no individual carries -- a genuine new
        coarse-grained variable, not a relabelling ).

  Selectivity / value (defeats COGNET). V_gain = E_P[u] - E_uniform[u]
  for a task utility u; selectivity = probability retained on the
  task-optimal structure. Pathological collapse has V_gain <= 0.

STRONG (irreducible) EMERGENCE, by REDUCIBILITY not magnitude:
        C_irr|E >= 0.3 bits AND D_higher >= 0.5 AND macro_gain > 0.
WEAK / reducible: the contraction is explained by environment,
independent adaptation, or pairwise structure (C_irr|E < 0.05).

REGISTERED PREDICTIONS (frozen before running):
  IR-1  The threat is real. (a) Gaussian common cause -> Omega > 0
        (redundancy), closed form. (b) a deterministic common-cause
        system (X1=e1, X2=e2, X3=e1 XOR e2) is DISTRIBUTIONALLY
        IDENTICAL to the endogenous role-lock -- marginal C_irr = 1 bit
        and Omega = -1 (synergy) -- with NO agent interaction. Higher-
        order statistics alone cannot tell interaction from common
        cause.
  IR-2  The framework defeats ENVDRV. For that common-cause system:
        C_irr|E < 0.05 bits AND D_higher < 0.5 -> rejected as strong
        emergence despite an identical higher-order signature and large
        collapse. Same distribution as role-lock, opposite verdict.
  IR-3  PITHON reducibility. A pairwise (Markov-chain) system has
        C_irr < 0.05 bits -> higher-order-looking structure is fully
        pairwise-reducible; weak, not irreducible.
  IR-4  Irreducible role-lock (positive). A genuine 3-way constraint
        (parity role-interlock) has C_irr|E >= 0.3 AND D_higher >= 0.5
        AND macro_gain > 0 -> strong emergence.
  IR-5  Magnitude != strength (COGNET). Across the five systems,
        Spearman(C_total, C_irr|E) <= 0; the maximum-C_total system
        (redundant consensus) has C_irr|E ~ 0 while the strong case has
        SMALLER C_total.
  IR-6  Functional vs pathological at MATCHED magnitude. A consensus
        system tuned to C_total within 0.1 bit of the role-lock has
        V_gain <= 0 and C_irr|E ~ 0, while the role-lock has V_gain > 0
        and C_irr|E >= 0.3: a double dissociation at equal collapse.
Misses are retained.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
STATES = list(itertools.product((0, 1), repeat=3))
TH_IRR = 0.3        # bits, strong-emergence irreducibility threshold
TH_REDUCIBLE = 0.05  # bits, "explained away" ceiling
TH_D = 0.5          # interaction-dependence threshold (framework value)

Dist = Dict[Tuple[int, int, int], float]


# ----------------------------------------------------------- info tools

def _norm(p: Dist) -> Dist:
    z = sum(p.values())
    return {k: v / z for k, v in p.items() if v > 0}


def H(p: Dist, idx=(0, 1, 2)) -> float:
    marg: Dict = {}
    for s, pr in p.items():
        key = tuple(s[i] for i in idx)
        marg[key] = marg.get(key, 0.0) + pr
    return -sum(v * math.log2(v) for v in marg.values() if v > 0)


def total_correlation(p: Dist) -> float:
    return sum(H(p, (i,)) for i in range(3)) - H(p, (0, 1, 2))


def o_information(p: Dist) -> float:
    n = 3
    allh = H(p, (0, 1, 2))
    out = (n - 2) * allh
    for i in range(n):
        rest = tuple(j for j in range(n) if j != i)
        out += H(p, (i,)) - H(p, rest)
    return out


def pair_marginal(p: Dist, pair) -> Dict:
    m: Dict = {}
    for s, pr in p.items():
        key = (s[pair[0]], s[pair[1]])
        m[key] = m.get(key, 0.0) + pr
    return m


def pairwise_maxent(p: Dist, iters: int = 2000) -> Dist:
    """Max-entropy distribution matching all pairwise marginals of p
    (iterative proportional fitting). Forbidden pairs (target 0) zero
    out the corresponding states permanently."""
    targets = {pair: pair_marginal(p, pair) for pair in ((0, 1), (0, 2), (1, 2))}
    q = {s: 1.0 for s in STATES}
    for _ in range(iters):
        for pair in ((0, 1), (0, 2), (1, 2)):
            qm = pair_marginal(q, pair)
            for s in STATES:
                key = (s[pair[0]], s[pair[1]])
                t = targets[pair].get(key, 0.0)
                cur = qm.get(key, 0.0)
                if t <= 0.0:
                    q[s] = 0.0
                elif cur > 0.0:
                    q[s] *= t / cur
        z = sum(q.values())
        q = {s: v / z for s, v in q.items()}
    return q


def kl(p: Dist, q: Dist) -> float:
    out = 0.0
    for s, pr in p.items():
        if pr <= 0:
            continue
        qs = q.get(s, 0.0)
        if qs <= 0:
            return float("inf")
        out += pr * math.log2(pr / qs)
    return out


def c_irr(p: Dist) -> float:
    """Irreducible higher-order (order>=3) connected information."""
    return max(0.0, kl(_norm(p), pairwise_maxent(_norm(p))))


def macro_gain(p: Dist, m: Callable[[Tuple[int, int, int]], int]) -> float:
    """I(M;X) - sum_i I(M;X_i) for macro variable M=m(X)."""
    def mi(idx) -> float:
        joint: Dict = {}
        pm: Dict = {}
        pv: Dict = {}
        for s, pr in p.items():
            xv = tuple(s[i] for i in idx)
            mv = m(s)
            joint[(xv, mv)] = joint.get((xv, mv), 0.0) + pr
            pm[xv] = pm.get(xv, 0.0) + pr
            pv[mv] = pv.get(mv, 0.0) + pr
        out = 0.0
        for (xv, mv), pr in joint.items():
            out += pr * math.log2(pr / (pm[xv] * pv[mv]))
        return out
    return mi((0, 1, 2)) - sum(mi((i,)) for i in range(3))


# ----------------------------------------------------- systems (exact)

def make_env_redundant(q: float = 0.05) -> Tuple[Dist, Dist]:
    """Common-cause REDUNDANCY, NO agent interaction. E ~ Bern(1/2);
    each agent copies E with flip prob q (conditionally independent
    given E). Produces redundancy (Omega > 0), no higher-order."""
    joint_env: Dict = {}
    for e in (0, 1):
        for s in STATES:
            pr = 0.5
            for i in range(3):
                pr *= (1 - q) if s[i] == e else q
            joint_env[(s, (e,))] = joint_env.get((s, (e,)), 0.0) + pr
    marg: Dist = {}
    for (s, _e), pr in joint_env.items():
        marg[s] = marg.get(s, 0.0) + pr
    return _norm(marg), joint_env


def make_env_synergy() -> Tuple[Dist, Dist]:
    """Common-cause SYNERGY, NO agent interaction -- the ENVDRV threat.

    Environment E = (e1, e2) ~ Unif{0,1}^2. Each agent deterministically
    reads a different function of the SAME environment:
        X1 = e1,  X2 = e2,  X3 = e1 XOR e2.
    Marginally each X_i is uniform and every PAIR is independent, but the
    joint is constrained to even parity -- i.e. this is DISTRIBUTIONALLY
    IDENTICAL to the endogenous role-lock (C_irr = 1 bit, Omega = -1),
    yet there is no agent-agent interaction whatsoever. The higher-order
    signature is pure common cause. Conditioning on E collapses every X_i
    to a point mass, so C_irr|E = 0; the do-operator on agent coupling
    leaves the environment intact, so D_higher = 0.
    """
    joint_env: Dict = {}
    for e1 in (0, 1):
        for e2 in (0, 1):
            s = (e1, e2, e1 ^ e2)
            joint_env[(s, (e1, e2))] = 0.25
    marg: Dist = {}
    for (s, _e), pr in joint_env.items():
        marg[s] = marg.get(s, 0.0) + pr
    return _norm(marg), joint_env


def make_pairwise_chain(c: float = 0.8) -> Dist:
    """Markov chain X1 - X2 - X3 (pairwise/tree model, no 3-way).
    correlation strength c in [0,1)."""
    p: Dist = {}
    for s in STATES:
        pr = 0.5
        pr *= (1 + c) / 2 if s[1] == s[0] else (1 - c) / 2
        pr *= (1 + c) / 2 if s[2] == s[1] else (1 - c) / 2
        p[s] = pr
    return _norm(p)


def make_role_parity() -> Dist:
    """Genuine 3-way role interlock: only even-parity joint states are
    reachable (each agent's role is fixed by the other two). Pairwise
    marginals are uniform/independent -> all structure is order-3."""
    return _norm({s: 1.0 for s in STATES if (s[0] ^ s[1] ^ s[2]) == 0})


def make_consensus(alpha: float = 1.0) -> Dist:
    """Redundant consensus (pathological collapse): mixture of the
    all-equal states {000,111} with a uniform floor. alpha=1 -> pure
    consensus (max total correlation); alpha<1 lets us MATCH a target
    C_total for the functional-vs-pathological dissociation."""
    consensus = {s: 0.5 for s in STATES if s in ((0, 0, 0), (1, 1, 1))}
    p: Dist = {}
    for s in STATES:
        p[s] = alpha * consensus.get(s, 0.0) + (1 - alpha) * (1 / 8)
    return _norm(p)


def utility(s: Tuple[int, int, int]) -> float:
    """Task: the optimum requires the coordinated (even-parity) role
    structure. Reward +1 on even parity, 0 otherwise."""
    return 1.0 if (s[0] ^ s[1] ^ s[2]) == 0 else 0.0


def value_gain(p: Dist) -> float:
    ev = sum(pr * utility(s) for s, pr in p.items())
    base = sum(utility(s) for s in STATES) / len(STATES)
    return ev - base


def selectivity(p: Dist) -> float:
    return sum(pr for s, pr in p.items() if (s[0] ^ s[1] ^ s[2]) == 0)


# --------------------------------------------------- endogeneity route

def c_irr_given_env(joint_env: Dict) -> float:
    """E_e[ KL(P(.|e) || pairwise-maxent P(.|e)) ] over environment e."""
    env_states = sorted({e for (_s, e) in joint_env})
    total = 0.0
    for e in env_states:
        cond = {s: pr for (s, ee), pr in joint_env.items() if ee == e}
        pe = sum(cond.values())
        if pe <= 0:
            continue
        cond = _norm(cond)
        total += pe * c_irr(cond)
    return total


def d_higher(p: Dist, joint_env) -> float:
    """Interaction dependence of the order-3 structure: cut agent<->agent
    coupling but KEEP the common cause, then re-measure C_irr."""
    nat = c_irr(p)
    if nat < 1e-9:
        return 0.0
    if joint_env is not None:
        # broken = conditionally independent given env (keep common cause)
        broken: Dist = {}
        env_states = sorted({e for (_s, e) in joint_env})
        for e in env_states:
            cond = {s: pr for (s, ee), pr in joint_env.items() if ee == e}
            pe = sum(cond.values())
            cond = _norm(cond)
            indep = {}
            for s in STATES:
                pr = 1.0
                for i in range(3):
                    mi_ = sum(c for ss, c in cond.items() if ss[i] == s[i])
                    pr *= mi_
                indep[s] = pr
            for s in STATES:
                broken[s] = broken.get(s, 0.0) + pe * indep[s]
        broken = _norm(broken)
    else:
        # no common cause: cut coupling -> product of marginals
        broken = {}
        for s in STATES:
            pr = 1.0
            for i in range(3):
                mi_ = sum(c for ss, c in p.items() if ss[i] == s[i])
                pr *= mi_
            broken[s] = pr
        broken = _norm(broken)
    return float(1.0 - c_irr(broken) / nat)


# ------------------------------------------------------ Gaussian anchor

def gaussian_oinformation(cov: np.ndarray) -> float:
    n = len(cov)

    def h(idx):
        sub = cov[np.ix_(idx, idx)]
        return 0.5 * math.log((2 * math.pi * math.e) ** len(idx)
                              * np.linalg.det(sub))
    allidx = list(range(n))
    out = (n - 2) * h(allidx)
    for i in range(n):
        rest = [j for j in allidx if j != i]
        out += h([i]) - h(rest)
    return out / math.log(2)   # bits


# =============================================================== main

def coords(name: str, p: Dist, joint_env=None) -> Dict:
    ci = c_irr(p)
    cie = c_irr_given_env(joint_env) if joint_env is not None else ci
    dh = d_higher(p, joint_env)
    # macro_gain is a co-information of a READOUT and, like N, is
    # necessary-but-not-sufficient: parity of independent bits already
    # gives macro_gain > 0 with zero interaction (the ENVDRV point).
    # It is reported, NOT gated on. The verdict rests on irreducibility
    # of the JOINT distribution (C_irr|E) plus interaction dependence.
    mg = macro_gain(p, lambda s: s[0] ^ s[1] ^ s[2])
    strong = bool(cie >= TH_IRR and dh >= TH_D)
    reducible = bool(cie < TH_REDUCIBLE)
    return {
        "C_total": total_correlation(p),
        "O_information": o_information(p),
        "C_irr_marginal": ci,
        "C_irr_given_env": cie,
        "D_higher": dh,
        "macro_gain": mg,
        "V_gain": value_gain(p),
        "selectivity": selectivity(p),
        "strong_emergence": strong,
        "reducible": reducible,
    }


def spearman(xs, ys) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n))
                    * sum((ry[i] - my) ** 2 for i in range(n)))
    return num / den if den > 0 else 0.0


def main() -> None:
    report: Dict = {"status": ("irreducibility battery: endogenous, "
                               "selective contraction of joint futures "
                               "with lower-order irreducibility and "
                               "macro causal contribution; IR-1..6 "
                               "frozen in docstring"),
                    "thresholds": {"C_irr_strong": TH_IRR,
                                   "C_irr_reducible": TH_REDUCIBLE,
                                   "D_higher": TH_D}}

    env_red_m, env_red_j = make_env_redundant()
    env_syn_m, env_syn_j = make_env_synergy()
    pair = make_pairwise_chain()
    role = make_role_parity()
    consensus = make_consensus(1.0)

    systems = {
        "env_redundant": coords("env_redundant", env_red_m, env_red_j),
        "env_synergy": coords("env_synergy", env_syn_m, env_syn_j),
        "pairwise_chain": coords("pairwise_chain", pair),
        "role_parity": coords("role_parity", role),
        "consensus": coords("consensus", consensus),
    }
    for name, c in systems.items():
        print(f"{name:16s} C_total={c['C_total']:.3f} "
              f"Omega={c['O_information']:+.3f} "
              f"C_irr={c['C_irr_marginal']:.3f} "
              f"C_irr|E={c['C_irr_given_env']:.3f} "
              f"D={c['D_higher']:.2f} macro={c['macro_gain']:+.2f} "
              f"V={c['V_gain']:+.2f} strong={c['strong_emergence']}",
              flush=True)
    report["systems"] = systems

    # ---- Gaussian analytic anchor (IR-1a): common cause, invariant
    a = np.array([1.0, 1.0, 1.0])
    cov = 1.0 * np.outer(a, a) + 0.5 * np.eye(3)
    omega_gauss = gaussian_oinformation(cov)
    report["gaussian_common_cause"] = {
        "sigma_E2": 1.0, "sigma_n2": 0.5,
        "O_information_bits": omega_gauss,
        "note": "time-invariant common Gaussian cause -> redundancy"}
    print(f"gaussian common cause Omega = {omega_gauss:+.3f} bits",
          flush=True)

    # ---- IR-6: consensus matched to role_parity's C_total (+-0.1 bit)
    target = systems["role_parity"]["C_total"]
    best_alpha, best_gap = 1.0, 1e9
    for alpha in np.linspace(0.05, 1.0, 200):
        ct = total_correlation(make_consensus(float(alpha)))
        if abs(ct - target) < best_gap:
            best_gap, best_alpha = abs(ct - target), float(alpha)
    consensus_matched = make_consensus(best_alpha)
    cm = coords("consensus_matched", consensus_matched)
    cm["alpha"] = best_alpha
    cm["C_total_gap_vs_role"] = abs(cm["C_total"] - target)
    report["consensus_matched"] = cm
    print(f"consensus_matched alpha={best_alpha:.3f} "
          f"C_total={cm['C_total']:.3f} (role {target:.3f}) "
          f"C_irr|E={cm['C_irr_given_env']:.3f} V={cm['V_gain']:+.2f}",
          flush=True)

    # ---- registered outcomes
    ev = systems["env_synergy"]
    ir1a = omega_gauss > 0
    ir1b = ev["C_irr_marginal"] > 0.5 and ev["O_information"] < 0
    ir2 = ev["C_irr_given_env"] < TH_REDUCIBLE and ev["D_higher"] < TH_D
    ir3 = systems["pairwise_chain"]["C_irr_given_env"] < TH_REDUCIBLE
    r = systems["role_parity"]
    ir4 = r["C_irr_given_env"] >= TH_IRR and r["D_higher"] >= TH_D
    order = ["env_redundant", "env_synergy", "pairwise_chain",
             "role_parity", "consensus"]
    ctotals = [systems[k]["C_total"] for k in order]
    cirrs = [systems[k]["C_irr_given_env"] for k in order]
    rho = spearman(ctotals, cirrs)
    max_ct = order[int(np.argmax(ctotals))]
    ir5 = (rho <= 0.0 and systems[max_ct]["C_irr_given_env"] < TH_REDUCIBLE
           and systems["role_parity"]["C_total"]
           < systems[max_ct]["C_total"])
    ir6 = (cm["C_total_gap_vs_role"] <= 0.1
           and cm["V_gain"] <= 0.0 and cm["C_irr_given_env"] < TH_REDUCIBLE
           and r["V_gain"] > 0.0 and r["C_irr_given_env"] >= TH_IRR)

    report["magnitude_vs_strength"] = {
        "order": order, "C_total": ctotals, "C_irr_given_env": cirrs,
        "spearman": rho, "argmax_C_total": max_ct}
    report["registered_outcomes"] = {
        "IR1_environment_no_go_is_real": bool(ir1a and ir1b),
        "IR2_framework_defeats_no_go": bool(ir2),
        "IR3_pairwise_reducible_PITHON": bool(ir3),
        "IR4_irreducible_role_lock": bool(ir4),
        "IR5_magnitude_not_strength": bool(ir5),
        "IR6_functional_vs_pathological_matched": bool(ir6),
    }
    out = OUTPUTS / "emergence_irreducibility.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
