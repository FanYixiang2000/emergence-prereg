"""Aggregate the 12 Overcooked-AI confirmation seeds and score the
registered predictions OC-1..OC-5 (frozen in OVERCOOKED_PREREGISTRATION.md).

Reads the per-seed JSONs written by overcooked_confirmation.py
(outputs/overcooked_confirm_s770NN.json) and writes a single pooled
report plus the registered outcome tally. Misses are reported unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = list(range(77001, 77013))
CONTROLS = ("initial_twin", "scripted_roles", "bc_clone", "untrained_other")
PREFIX = "overcooked_confirm_s"
OUT_NAME = "overcooked_confirmation_pooled.json"


def sign_test_p(k: int, n: int) -> float:
    """One-sided exact sign test: P(X >= k) under Binomial(n, 0.5)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def load_seed(seed: int) -> Dict:
    p = OUTPUTS / f"{PREFIX}{seed}.json"
    data = json.loads(p.read_text())
    return data["seeds"][str(seed)]


def main() -> None:
    global SEEDS, PREFIX, OUT_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()
    if args.round == 2:
        SEEDS = list(range(78001, 78021))
        PREFIX = "overcooked_r2_s"
        OUT_NAME = "overcooked_round2_pooled.json"
    seeds = {}
    for s in SEEDS:
        try:
            seeds[s] = load_seed(s)
        except FileNotFoundError:
            print(f"WARNING: seed {s} not finished; skipping")
    n = len(seeds)
    if n == 0:
        raise SystemExit("no finished seeds yet")

    learned_accept = sum(
        1 for s in seeds.values() if s["learned"]["verdict"]["emergent"])

    control_rejections = 0
    control_total = 0
    failure_routes: Dict[str, Dict[str, int]] = {c: {} for c in CONTROLS}
    for s in seeds.values():
        for c in CONTROLS:
            if c not in s:
                continue
            control_total += 1
            v = s[c]["verdict"]
            if not v["emergent"]:
                control_rejections += 1
                for f in v["failed"]:
                    failure_routes[c][f] = failure_routes[c].get(f, 0) + 1

    twin_b_rejected = sum(
        1 for s in seeds.values()
        if not _contract_b_emergent(s.get("twin_contract_b")))

    useful_positive = sum(
        1 for s in seeds.values()
        if s["learned"]["metrics"]["usefulness_gap"] > 0)
    p_oc5 = sign_test_p(useful_positive, n)

    # OC-3: first-potter-agent-0 rate higher in asymmetric_advantages
    # (context 1) than in cramped_room (context 0).
    direction_match = sum(
        1 for s in seeds.values()
        if s["learned"]["metrics"]["trigger_rates"]["1"]
        > s["learned"]["metrics"]["trigger_rates"]["0"])

    report = {
        "status": f"pooled Overcooked-AI confirmation ({n}/12 seeds)",
        "n_seeds": n,
        "learned_accepted": learned_accept,
        "control_rejections": f"{control_rejections}/{control_total}",
        "control_failure_routes": failure_routes,
        "twin_contract_b_rejected": f"{twin_b_rejected}/{n}",
        "learned_usefulness_positive": f"{useful_positive}/{n}",
        "oc5_sign_test_p": p_oc5,
        "oc3_direction_match": f"{direction_match}/{n}",
        "registered_outcomes": {
            "OC1_learned_accepted_ge_8": bool(learned_accept >= 8),
            "OC2_all_controls_rejected": bool(
                control_rejections == control_total),
            "OC3_trigger_direction_ge_10": bool(direction_match >= 10),
            "OC4_all_twins_rejected_contractB": bool(twin_b_rejected == n),
            "OC5_usefulness_positive_ge_10_and_sig": bool(
                useful_positive >= 10 and p_oc5 < 0.05),
        },
        "per_seed": {
            str(s): {
                "learned_emergent": v["learned"]["verdict"]["emergent"],
                "learned_failed": v["learned"]["verdict"]["failed"],
                "usefulness_gap": round(
                    v["learned"]["metrics"]["usefulness_gap"], 3),
                "train_minutes": v.get("train_minutes"),
            }
            for s, v in seeds.items()
        },
    }
    out = OUTPUTS / OUT_NAME
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"learned accepted {learned_accept}/{n}; "
          f"controls rejected {control_rejections}/{control_total}; "
          f"useful+ {useful_positive}/{n} (p={p_oc5:.4f})")
    print(f"Wrote {out}")


def _contract_b_emergent(m) -> bool:
    if not m:
        return False
    from overcooked_criterion import THRESHOLDS as T
    return (m["potential_bits"] >= T["potential_bits"]
            and m["conditional_selectivity"] >= T["conditional_selectivity"]
            and m["specificity_js_bits"] >= T["specificity_js_bits"]
            and m["usefulness_gap"] > T["usefulness_gap"])


if __name__ == "__main__":
    main()
