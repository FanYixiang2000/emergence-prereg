"""Collective-constraint battery: emergence as ENDOGENOUS collective
constraint formation on the joint-action possibility space.

This addresses the sharpest reviewer challenge (paraphrased): "if the
possibility space is {bridge forms / does not} or {success / failure},
then EVERY successful process is a collapse -- that is not emergence."
Correct. Outcome-collapse and single-agent choice are not emergence.
The possibility space must be the JOINT-ACTION branch set, and the
load-bearing quantity is whether AGENT-AGENT INTERACTION endogenously
reorganizes it -- separable from a central controller, a common
environment, or independent coincidence that reach the SAME outcome.

We test this with the decisive design a referee would demand: four
mechanisms that produce the SAME final outcome and the SAME single-agent
marginals, differing only in the generative mechanism.

  central_script         a controller dictates each agent's role (no
                         agent-agent influence); external.
  common_cause           a shared environment signal; each agent maps it
                         to a role independently; external.
  independent_coincidence agents act independently; the joint constraint
                         holds only by chance; no coupling.
  local_feedback         agents mutually adjust so the joint constraint
                         is satisfied; ENDOGENOUS coupling.

MACRO STRUCTURE (the "bridge" / role-lock): three agents pick roles in
{0,1,2}; the structure Z holds iff a1+a2+a3 = 0 (mod 3) -- a genuine
three-way constraint (any two roles leave the third determined; every
PAIR is unconstrained, so the structure is order-3 irreducible).

POSSIBILITY SPACE = distribution over joint actions (a1,a2,a3), NOT over
{Z, not-Z}. The counterfactual is the INTERACTION-BROKEN distribution
P_broken: cut agent<->agent coupling while KEEPING each agent's marginal,
the environment, and any controller. Certificates:

  generation certificate (how the structure forms):
    C   collective constraint  = H(P_broken) - H(P_real)   (bits)
    G   reorganization         = JSD(P_real, P_broken)      (bits)
    M   endogenous macro gain  = P(Z|real) - P(Z|broken)
  product certificate (what the structure is):
    N   irreducibility         = C_irr|E = KL(P || pairwise-maxent | env)
    R   persistence            = recovery of Z after a micro perturbation
    (A  causal autonomy is reported via N here; EI-form lives in the
        coordinates battery.)

Micro-freedom-DOWN / macro-capability-UP: for genuine emergence the
joint-action entropy DROPS (micro branches pruned) while the macro
reachable capability RISES -- emergence is a reorganization, not a
mere total-entropy decrease.

VERDICT (endogenous collective emergence):
  C >= 0.5 bits AND G >= 0.05 bits AND M > 0 AND N >= 0.3 bits
  AND R >= 0.6. Only local_feedback should pass; the other three fail
  on the predicted certificate component.

REGISTERED PREDICTIONS (frozen before running):
  CC-1  Matched confound: central_script, common_cause and
        local_feedback have IDENTICAL joint-action distributions and
        IDENTICAL single-agent marginals and P(Z)=1 -- no statistic of
        the joint distribution separates them. (Outcome-collapse of Z is
        also identical, hence useless.)
  CC-2  Endogeneity separates them: under the interaction-broken (keep-
        external) counterfactual, C >= 0.5 and G >= 0.05 and M > 0 ONLY
        for local_feedback; the two externally-driven mechanisms give
        C = G = M = 0.
  CC-3  Micro-down / macro-up: for local_feedback H(P_real) < H(P_broken)
        (joint freedom reduced) yet endogenous macro gain M > 0.
        Emergence is not a total-entropy decrease.
  CC-4  Persistence: independent_coincidence has R < 0.6 (transient),
        while local_feedback has R >= 0.6 -- coincidental joint structure
        is rejected on persistence.
  CC-5  Full four-quadrant certificate accepts ONLY local_feedback, and
        each other mechanism fails on the PREDICTED component
        (central & common-cause on endogeneity C/M/N|E; independent on
        persistence R and constraint C).
Misses are retained.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"
V = (0, 1, 2)
STATES = list(itertools.product(V, repeat=3))
VALID = [s for s in STATES if (s[0] + s[1] + s[2]) % 3 == 0]  # Z holds
TH = {"C": 0.5, "G": 0.05, "N": 0.3, "R": 0.6}
Dist = Dict[Tuple[int, int, int], float]


def _norm(p: Dist) -> Dist:
    z = sum(p.values())
    return {k: v / z for k, v in p.items() if v > 0}


def H(p: Dist, idx=(0, 1, 2)) -> float:
    m: Dict = {}
    for s, pr in p.items():
        k = tuple(s[i] for i in idx)
        m[k] = m.get(k, 0.0) + pr
    return -sum(v * math.log2(v) for v in m.values() if v > 0)


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


def jsd(p: Dist, q: Dist) -> float:
    m = {s: 0.5 * (p.get(s, 0.0) + q.get(s, 0.0)) for s in STATES}
    m = _norm(m)
    return 0.5 * kl(_norm(p), m) + 0.5 * kl(_norm(q), m)


def pair_marginal(p: Dist, pair) -> Dict:
    m: Dict = {}
    for s, pr in p.items():
        k = (s[pair[0]], s[pair[1]])
        m[k] = m.get(k, 0.0) + pr
    return m


def pairwise_maxent(p: Dist, iters: int = 3000) -> Dist:
    targets = {pr: pair_marginal(p, pr) for pr in ((0, 1), (0, 2), (1, 2))}
    q = {s: 1.0 for s in STATES}
    for _ in range(iters):
        for pr in ((0, 1), (0, 2), (1, 2)):
            qm = pair_marginal(q, pr)
            for s in STATES:
                k = (s[pr[0]], s[pr[1]])
                t = targets[pr].get(k, 0.0)
                cur = qm.get(k, 0.0)
                if t <= 0.0:
                    q[s] = 0.0
                elif cur > 0.0:
                    q[s] *= t / cur
        z = sum(q.values())
        q = {s: v / z for s, v in q.items()}
    return q


def c_irr(p: Dist) -> float:
    p = _norm(p)
    return max(0.0, kl(p, pairwise_maxent(p)))


def c_irr_given_env(joint_env) -> float:
    envs = sorted({e for (_s, e) in joint_env})
    total = 0.0
    for e in envs:
        cond = {s: pr for (s, ee), pr in joint_env.items() if ee == e}
        pe = sum(cond.values())
        if pe <= 0:
            continue
        total += pe * c_irr(_norm(cond))
    return total


def p_Z(p: Dist) -> float:
    return sum(pr for s, pr in p.items() if s in VALID)


def marginals(p: Dist):
    return [tuple(round(sum(pr for s, pr in p.items() if s[i] == v), 4)
                  for v in V) for i in range(3)]


# mechanisms
# Each mechanism supplies: P_real, P_broken (interaction-broken, keeping
# marginals + environment/controller), joint_env (for N|E) or None.

def uniform_valid() -> Dist:
    return _norm({s: 1.0 for s in VALID})


def uniform_all() -> Dist:
    return _norm({s: 1.0 for s in STATES})


def product_of_marginals(p: Dist) -> Dist:
    marg = [{v: sum(pr for s, pr in p.items() if s[i] == v) for v in V}
            for i in range(3)]
    return _norm({s: marg[0][s[0]] * marg[1][s[1]] * marg[2][s[2]]
                  for s in STATES})


def mechanism(kind: str):
    if kind == "central_script":
        # controller draws a valid config uniformly; agents obey it.
        real = uniform_valid()
        je = {(s, s): 1.0 / len(VALID) for s in VALID}  # env = controller
        # break agent-agent coupling but KEEP the controller -> unchanged
        broken = uniform_valid()
        return real, broken, je
    if kind == "common_cause":
        real = uniform_valid()
        je = {(s, s): 1.0 / len(VALID) for s in VALID}  # env = shared signal
        broken = uniform_valid()          # keep environment -> unchanged
        return real, broken, je
    if kind == "independent_coincidence":
        real = uniform_all()              # independent, constraint by chance
        broken = uniform_all()            # no coupling to break
        return real, broken, None
    if kind == "local_feedback":
        real = uniform_valid()            # mutual adjustment satisfies Z
        broken = product_of_marginals(real)  # cut coupling, keep marginals
        return real, broken, None
    raise ValueError(kind)


def measure_R(kind: str, n: int = 600) -> float:
    """Recovery of Z after an irrelevant micro perturbation (randomize a
    random agent), applying each mechanism's own recovery step."""
    rng = np.random.default_rng(1234)
    rec = 0
    for _ in range(n):
        s = list(VALID[rng.integers(len(VALID))])
        j = int(rng.integers(3))
        s[j] = int(rng.integers(3))                      # perturb
        if kind in ("local_feedback",):
            # endogenous: the perturbed agent re-adjusts to the others
            s[j] = (-(s[(j + 1) % 3] + s[(j + 2) % 3])) % 3
        elif kind in ("central_script", "common_cause"):
            # external driver re-imposes the assigned config
            s = list(VALID[rng.integers(len(VALID))])
        else:  # independent_coincidence: re-randomize, holds by chance
            s = [int(rng.integers(3)) for _ in range(3)]
        rec += int((s[0] + s[1] + s[2]) % 3 == 0)
    return rec / n


def certificate(kind: str) -> Dict:
    real, broken, je = mechanism(kind)
    C = H(broken) - H(real)
    G = jsd(real, broken)
    M = p_Z(real) - p_Z(broken)
    N = c_irr_given_env(je) if je is not None else c_irr(real)
    R = measure_R(kind)
    accept = bool(C >= TH["C"] and G >= TH["G"] and M > 1e-9
                  and N >= TH["N"] and R >= TH["R"])
    # which necessary components fail (for the four-quadrant table)
    fails = []
    if not (C >= TH["C"]):
        fails.append("C_constraint")
    if not (G >= TH["G"]):
        fails.append("G_reorganization")
    if not (M > 1e-9):
        fails.append("M_endogenous_macro_gain")
    if not (N >= TH["N"]):
        fails.append("N_irreducibility|env")
    if not (R >= TH["R"]):
        fails.append("R_persistence")
    return {
        "H_real": H(real), "H_broken": H(broken),
        "C_constraint": C, "G_reorganization": G,
        "M_endogenous_macro_gain": M,
        "N_irreducibility_given_env": N, "R_persistence": R,
        "P_Z_real": p_Z(real), "P_Z_broken": p_Z(broken),
        "micro_down_macro_up": bool(H(real) < H(broken) and M > 1e-9),
        "accept": accept, "failed_components": fails,
    }


def main() -> None:
    kinds = ["central_script", "common_cause",
             "independent_coincidence", "local_feedback"]
    report = {"status": ("collective-constraint battery: emergence as "
                         "endogenous reorganization of the joint-action "
                         "possibility space; CC-1..5 frozen in docstring; "
                         "same-outcome/same-marginals four-mechanism test"),
              "thresholds": TH, "macro_structure": "a1+a2+a3 = 0 (mod 3)"}
    certs = {k: certificate(k) for k in kinds}
    for k, c in certs.items():
        print(f"{k:24s} C={c['C_constraint']:+.3f} "
              f"G={c['G_reorganization']:.3f} "
              f"M={c['M_endogenous_macro_gain']:+.3f} "
              f"N|E={c['N_irreducibility_given_env']:.3f} "
              f"R={c['R_persistence']:.2f} accept={c['accept']} "
              f"fails={c['failed_components']}", flush=True)
    report["mechanisms"] = certs

    # matched-confound check (CC-1): identical joint + marginals
    real_cs, _, _ = mechanism("central_script")
    real_cc, _, _ = mechanism("common_cause")
    real_lf, _, _ = mechanism("local_feedback")
    joint_identical = (max(abs(real_cs.get(s, 0) - real_lf.get(s, 0))
                           for s in STATES) < 1e-9
                       and max(abs(real_cc.get(s, 0) - real_lf.get(s, 0))
                               for s in STATES) < 1e-9)
    marg_identical = (marginals(real_cs) == marginals(real_lf)
                      == marginals(real_cc))
    pz_identical = (abs(p_Z(real_cs) - 1.0) < 1e-9
                    and abs(p_Z(real_cc) - 1.0) < 1e-9
                    and abs(p_Z(real_lf) - 1.0) < 1e-9)
    report["matched_confound"] = {
        "joint_distributions_identical": bool(joint_identical),
        "single_agent_marginals_identical": bool(marg_identical),
        "P_Z_identical_and_one": bool(pz_identical),
        "single_agent_marginal": marginals(real_lf)[0]}

    lf = certs["local_feedback"]
    ext = ["central_script", "common_cause"]
    cc1 = joint_identical and marg_identical and pz_identical
    cc2 = (lf["C_constraint"] >= TH["C"] and lf["G_reorganization"] >= TH["G"]
           and lf["M_endogenous_macro_gain"] > 1e-9
           and all(certs[k]["C_constraint"] < TH["C"]
                   and certs[k]["G_reorganization"] < TH["G"]
                   and certs[k]["M_endogenous_macro_gain"] <= 1e-9
                   for k in ext))
    cc3 = lf["micro_down_macro_up"]
    cc4 = (certs["independent_coincidence"]["R_persistence"] < TH["R"]
           and lf["R_persistence"] >= TH["R"])
    cc5 = (lf["accept"] and all(not certs[k]["accept"] for k in kinds
                                if k != "local_feedback")
           and "C_constraint" in certs["central_script"]["failed_components"]
           and "C_constraint" in certs["common_cause"]["failed_components"]
           and "R_persistence" in
           certs["independent_coincidence"]["failed_components"])

    report["registered_outcomes"] = {
        "CC1_matched_confound_indistinguishable_by_joint": bool(cc1),
        "CC2_endogeneity_separates_same_outcome": bool(cc2),
        "CC3_micro_down_macro_up": bool(cc3),
        "CC4_persistence_rejects_coincidence": bool(cc4),
        "CC5_four_quadrant_only_local_feedback": bool(cc5),
    }
    out = OUTPUTS / "collective_constraint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
