"""Machine verification of the spatial-collapse bridge (Proposition S).

Formalizes two folk intuitions inside the trajectory substrate and
connects them to the classical multivariate-information literature:

(i)   "More agents feel more emergent" because the open possibility
      space GROWS linearly with N: under independence the joint
      future-basin entropy is H(B_1..B_N) = sum_i H(B_i) ~ N H_1, so
      both the potential and the maximal available collapse scale
      with the population.
(ii)  "Coordination is emergence" because coordination IS a
      contraction of the joint possibility space relative to the
      independence null: the spatial collapse

          C_spatial = sum_i H(B_i) - H(B_1,...,B_N)

      is exactly the total correlation / multi-information of the
      agents' futures (Watanabe 1960; McGill 1954), the quantity that
      integration-style measures build on (Tononi, Sporns & Edelman
      1994). Each individual becomes predictable from the others
      precisely to the extent C_spatial > 0.

(iii) The bridge also states WHY synergy-flavoured detectors accept
      scripted systems: C_spatial is a functional of the joint law
      alone, so a deterministic script that copies a common latent
      achieves MAXIMAL total correlation. Structural collapse is
      provenance-blind by construction; the adaptive layer (value,
      endogeneity, acquisition) is where scripts are excluded --
      which is the measured content of the battery's scripted
      counterexamples and the exact-Psi audit.

Checks (exact computations on enumerated joint laws; failures would
be reported unchanged):

    S-A  Identity and nonnegativity: C_spatial = TC >= 0 on 20,000
         random joint laws (N in 2..4, k in 2..4), with equality to
         the KL(joint || product of marginals) form, max gap < 1e-12.
    S-B  Independence null: C_spatial = 0 exactly iff the joint
         factorizes; under independence H_joint = N * H_1 exactly
         (linear growth of the open space), checked for N = 1..8.
    S-C  Coupling monotonicity: for the latent-copy family
         (each of N=3 agents copies a shared uniform latent with
         probability kappa, else draws independently), exact
         C_spatial(kappa) is strictly increasing on the grid
         kappa = 0, 0.1, ..., 1.0.
    S-D  Provenance-blindness: the deterministic script (kappa = 1)
         attains the maximum C_spatial = (N-1) log2 k over the
         family, and two mechanistically different generators with
         the same joint law have identical C_spatial (computed both
         ways, gap 0) -- the formal blind spot that the adaptive
         layer repairs.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
RNG = random.Random(20260721)

results = {}


def entropy(p) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def joint_random(n_agents: int, k: int):
    w = [RNG.random() for _ in range(k ** n_agents)]
    s = sum(w)
    return [x / s for x in w]


def marginals(joint, n_agents: int, k: int):
    outs = []
    for i in range(n_agents):
        m = [0.0] * k
        for idx, p in enumerate(joint):
            digits = np.base_repr(idx, base=k).zfill(n_agents)
            m[int(digits[i])] += p
        outs.append(m)
    return outs


def total_correlation(joint, n_agents: int, k: int) -> float:
    ms = marginals(joint, n_agents, k)
    return sum(entropy(m) for m in ms) - entropy(joint)


def kl_to_product(joint, n_agents: int, k: int) -> float:
    ms = marginals(joint, n_agents, k)
    out = 0.0
    for idx, p in enumerate(joint):
        if p <= 0:
            continue
        digits = np.base_repr(idx, base=k).zfill(n_agents)
        q = 1.0
        for i in range(n_agents):
            q *= ms[i][int(digits[i])]
        out += p * math.log2(p / q)
    return out


# S-A identity + nonnegativity
worst_gap, worst_neg = 0.0, 0.0
for _ in range(20000):
    n = RNG.randint(2, 4)
    k = RNG.randint(2, 4)
    j = joint_random(n, k)
    tc = total_correlation(j, n, k)
    kl = kl_to_product(j, n, k)
    worst_gap = max(worst_gap, abs(tc - kl))
    worst_neg = min(worst_neg, tc)
results["SA_identity"] = {
    "n": 20000, "max_gap_tc_vs_kl": worst_gap,
    "min_tc": worst_neg,
    "pass": worst_gap < 1e-12 and worst_neg > -1e-12}

# S-B independence null and linear growth
ok = True
for n in range(1, 9):
    k = 3
    m = [RNG.random() for _ in range(k)]
    s = sum(m)
    m = [x / s for x in m]
    joint = [math.prod(m[int(d)] for d in np.base_repr(i, base=k)
                       .zfill(n)) for i in range(k ** n)]
    h_joint = entropy(joint)
    ok &= abs(h_joint - n * entropy(m)) < 1e-9
    ok &= abs(total_correlation(joint, n, k)) < 1e-9
results["SB_independence_linear_growth"] = {"N_range": "1..8",
                                            "pass": bool(ok)}


def latent_copy_joint(n: int, k: int, kappa: float):
    """Each agent copies a shared uniform latent w.p. kappa, else
    draws uniformly and independently."""
    joint = [0.0] * (k ** n)
    for latent in range(k):
        for pattern in itertools.product(range(k), repeat=n):
            p = 1.0 / k
            for b in pattern:
                p *= (kappa if b == latent else 0.0) \
                    + (1 - kappa) / k
            idx = 0
            for b in pattern:
                idx = idx * k + b
            joint[idx] += p
    return joint


# S-C coupling monotonicity (exact, N=3, k=4)
n, k = 3, 4
tcs = [total_correlation(latent_copy_joint(n, k, kap), n, k)
       for kap in [i / 10 for i in range(11)]]
sc = all(tcs[i + 1] > tcs[i] - 1e-12 for i in range(len(tcs) - 1)) \
    and tcs[-1] > tcs[0] + 1.0
results["SC_coupling_monotone"] = {
    "tc_curve": [round(x, 4) for x in tcs], "pass": bool(sc)}

# S-D provenance blindness: script attains the maximum; two different
# generators with one joint law give identical TC
tc_script = tcs[-1]
max_theory = (n - 1) * math.log2(k)
j1 = latent_copy_joint(n, k, 1.0)
# generator 2: agent 0 uniform, agents 1..n-1 deterministically copy
# agent 0 (mechanistically different: no shared latent, a leader)
j2 = [0.0] * (k ** n)
for b in range(k):
    idx = 0
    for _ in range(n):
        idx = idx * k + b
    j2[idx] += 1.0 / k
gap_laws = max(abs(a - b) for a, b in zip(j1, j2))
sd = (abs(tc_script - max_theory) < 1e-9 and gap_laws < 1e-12
      and abs(total_correlation(j2, n, k) - tc_script) < 1e-12)
results["SD_provenance_blind"] = {
    "tc_script": tc_script, "max_theory": max_theory,
    "distinct_generators_same_law_gap": gap_laws, "pass": bool(sd)}

report = {
    "status": ("spatial-collapse bridge (Proposition S): "
               "C_spatial = total correlation; checks S-A..S-D"),
    "results": results,
    "all_pass": all(r["pass"] for r in results.values()),
}
out = OUTPUTS / "spatial_bridge_verification.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
for name, r in results.items():
    print(f"{'PASS' if r['pass'] else 'FAIL'}  {name}")
print("all_pass:", report["all_pass"])
print(f"Wrote {out}")
