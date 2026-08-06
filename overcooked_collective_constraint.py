"""Read-only Overcooked bridge to the collective-constraint certificate.

This is deliberately NOT presented as the full flagship experiment. The
original Overcooked confirmation outputs in this checkout contain per-seed
metrics but not the saved policy checkpoints or step-by-step trajectories
needed to replay the learned policies and construct the true
agent-agent interaction-broken counterfactual. Data honesty requires saying
that plainly.

What this script can still test on the externally timestamped round-1 data:

  - the possibility space is not {success, failure}; it is a coarse
    joint branch, here (context, first-potter role);
  - learned policies should show context-conditioned role allocation:
        C_ctx = I(context; role) = H(P_broken) - H(P_real),
    where P_broken keeps the role marginal but cuts context-role dependence;
  - scripted/BC external role controllers should have C_ctx = 0 because
    they force the same role in both contexts despite high reward;
  - initialization/untrained controls may have open role entropy but should
    fail value/acquisition/persistence, preventing "random openness" from
    becoming emergence.

REGISTERED READ-ONLY PREDICTIONS (frozen before running this script):
  OCC-1  Among the 8 preregistered accepted learned seeds, >= 7 have
         C_ctx >= 0.05 bits and G_ctx >= 0.01 bits.
  OCC-2  Scripted roles and BC clones have C_ctx <= 0.01 bits in 12/12
         seeds while retaining high natural score -- external coordination
         is not endogenous collective constraint.
  OCC-3  Learned macro gain M = natural_score - do_block_score is positive
         in 12/12 seeds (same sign-test fact as OC-5, but attached to the
         generation certificate).
  OCC-4  Contract-B persistence: accepted learned seeds retain
         conditional_selectivity >= 0.5 and usefulness > 0 under the
         shorter success-indicator contract in >= 7/8 accepted seeds.
  OCC-5  Full C,G,M;N|E,R interaction-broken certificate is NOT available
         from stored round-1 artifacts in this checkout because learned
         checkpoints / trajectories are absent. This is a retained
         limitation, not a pass.

Misses and limitations are retained.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = tuple(range(77001, 77013))
CONTROLS = ("scripted_roles", "bc_clone")


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def kl(p: Dict[Tuple[int, int], float],
       q: Dict[Tuple[int, int], float]) -> float:
    out = 0.0
    for k, v in p.items():
        if v <= 0:
            continue
        if q.get(k, 0.0) <= 0:
            return float("inf")
        out += v * math.log2(v / q[k])
    return out


def jsd(p: Dict[Tuple[int, int], float],
        q: Dict[Tuple[int, int], float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist_from_trigger_rates(r0: float, r1: float):
    """P(context, role), role=1 means agent0 first-pots."""
    p = {
        (0, 1): 0.5 * r0,
        (0, 0): 0.5 * (1 - r0),
        (1, 1): 0.5 * r1,
        (1, 0): 0.5 * (1 - r1),
    }
    role1 = 0.5 * (r0 + r1)
    broken = {
        (0, 1): 0.5 * role1,
        (0, 0): 0.5 * (1 - role1),
        (1, 1): 0.5 * role1,
        (1, 0): 0.5 * (1 - role1),
    }
    # I(context;role) = H(role) - H(role|context)
    c_ctx = h2(role1) - 0.5 * (h2(r0) + h2(r1))
    return p, broken, c_ctx, jsd(p, broken)


def load_seed(seed: int) -> Dict:
    path = OUTPUTS / f"overcooked_confirm_s{seed}.json"
    data = json.loads(path.read_text())
    return data["seeds"][str(seed)]


def score_system(record: Dict) -> Dict:
    m = record["metrics"]
    r0 = float(m["trigger_rates"]["0"])
    r1 = float(m["trigger_rates"]["1"])
    p, broken, c_ctx, g_ctx = dist_from_trigger_rates(r0, r1)
    macro_gain = float(m["natural_score"] - m["do_block_score"])
    return {
        "trigger_rates": {"0": r0, "1": r1},
        "P_real_context_role": {f"{k[0]}_{k[1]}": v for k, v in p.items()},
        "P_broken_context_role": {f"{k[0]}_{k[1]}": v
                                  for k, v in broken.items()},
        "C_ctx_bits": c_ctx,
        "G_ctx_js_bits": g_ctx,
        "M_reward_gain": macro_gain,
        "natural_score": float(m["natural_score"]),
        "do_block_score": float(m["do_block_score"]),
        "verdict": int(record["verdict"]["emergent"]),
        "failed": record["verdict"]["failed"],
    }


def main() -> None:
    report = {
        "status": ("read-only Overcooked role-constraint bridge; "
                   "does not claim full interaction-broken C,G,M;N|E,R "
                   "certificate because checkpoints/trajectories are absent"),
        "limitations": {
            "has_saved_checkpoints": False,
            "has_step_trajectories": False,
            "full_interaction_broken_certificate_available": False,
            "reason": ("stored round-1 artifacts contain per-seed metrics "
                       "only; learned .pt files are absent in this checkout"),
        },
        "thresholds": {
            "C_ctx_bits": 0.05,
            "G_ctx_js_bits": 0.01,
            "contract_b_selectivity": 0.5,
        },
        "seeds": {},
    }
    accepted = []
    learned_pos_m = 0
    accepted_c = 0
    accepted_b = 0
    control_zero = {c: 0 for c in CONTROLS}

    for seed in SEEDS:
        rec = load_seed(seed)
        learned = score_system(rec["learned"])
        report["seeds"][str(seed)] = {"learned": learned}
        if learned["verdict"]:
            accepted.append(seed)
            if (learned["C_ctx_bits"] >= 0.05
                    and learned["G_ctx_js_bits"] >= 0.01):
                accepted_c += 1
            cb = rec.get("learned_contract_b", {})
            if (cb.get("conditional_selectivity", 0.0) >= 0.5
                    and cb.get("usefulness_gap", 0.0) > 0.0):
                accepted_b += 1
        if learned["M_reward_gain"] > 0:
            learned_pos_m += 1

        report["seeds"][str(seed)]["learned_contract_b"] = rec.get(
            "learned_contract_b", {})
        for ctrl in CONTROLS:
            scored = score_system(rec[ctrl])
            report["seeds"][str(seed)][ctrl] = scored
            if scored["C_ctx_bits"] <= 0.01:
                control_zero[ctrl] += 1

    report["summary"] = {
        "accepted_learned": f"{len(accepted)}/12",
        "accepted_CG_positive": f"{accepted_c}/{len(accepted)}",
        "learned_M_positive": f"{learned_pos_m}/12",
        "accepted_contractB_persistent": f"{accepted_b}/{len(accepted)}",
        "control_C_zero": {k: f"{v}/12" for k, v in control_zero.items()},
    }
    report["registered_outcomes"] = {
        "OCC1_accepted_learned_context_constraint_ge_7_of_8":
            bool(len(accepted) == 8 and accepted_c >= 7),
        "OCC2_scripted_and_clone_C_zero_12_of_12":
            bool(all(v == 12 for v in control_zero.values())),
        "OCC3_learned_macro_gain_positive_12_of_12":
            bool(learned_pos_m == 12),
        "OCC4_contractB_persistence_ge_7_of_8":
            bool(len(accepted) == 8 and accepted_b >= 7),
        "OCC5_full_certificate_unavailable_retained_limitation":
            bool(report["limitations"]
                 ["full_interaction_broken_certificate_available"] is False),
    }

    out = OUTPUTS / "overcooked_collective_constraint.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
