"""Cross-contract ranking stability of the continuous scores.

Reviewer question: can an observer manipulate the CONTINUOUS record by
choosing a congenial contract, even if binary verdicts are stable?
Answer with data: across the stored frozen-threshold contracts, compute
each learned seed's structural score E_struct = (P*S*M)^(1/3) per
contract and report the Spearman rank correlation of the 15-seed
ranking between every contract pair, plus per-seed score intervals.

Contracts pooled read-only (metrics already stored):
    A  hand basins, horizon 15, T=1        (confirmation + extension)
    B  speed basins, horizon 12            (dual_observer_contracts)
    C  summary-feature k-means basins      (learned_basin_clbf)
    D  low-level cross-fitted k-means      (crossfit_lowlevel_basins)
    G  hand basins, near-greedy rollouts   (independent_rollout_audit)

Declared expectations (before computing): RS-1, mean pairwise
Spearman over the learned-seed ranking >= 0.5. RS-2, every control's
E_struct below every learned seed's within each contract -- this
expectation FAILED and is retained: the competent scripted coordinator
has high structural components BY DESIGN (it has always been the
control that passes every behavioural component and fails exactly the
adaptive layer), so a structure-only score cannot separate learned
from scripted -- which is the layering's own claim, here re-derived by
the continuous record. Disclosed follow-up RS-2b: class separation on
E_adapt (which includes acquisition) instead.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

import emergence_profile as ep

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = [str(s) for s in list(range(1101, 1111)) + list(range(1201, 1206))]
CONTROLS = ("initial_twin", "team_nearest", "fixed_food0", "fixed_food1")


def estruct(m: Dict) -> float:
    return ep.e_struct(
        ep.potential_norm(m["potential_bits"], 4),
        m["conditional_selectivity"],
        ep.magnitude_norm(m["specificity_js_bits"]))


def contract_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    hand: Dict[str, Dict[str, float]] = {}
    for path, seeds in (("contextual_lbf_confirmation.json",
                         SEEDS[:10]),
                        ("contextual_lbf_extension.json", SEEDS[10:])):
        data = json.loads((OUTPUTS / path).read_text())
        for seed in seeds:
            hand[seed] = {
                name: estruct(entry["metrics"])
                for name, entry in data["seeds"][seed]["systems"].items()}
    out["A_hand"] = hand

    dual = json.loads((OUTPUTS / "dual_observer_contracts.json").read_text())
    out["B_speed"] = {
        seed: {name: estruct(dual["seeds"][seed][name]["metrics"])
               for name in dual["seeds"][seed]}
        for seed in SEEDS}

    lb = json.loads((OUTPUTS / "learned_basin_clbf.json").read_text())
    out["C_summary_kmeans"] = {
        seed: {name: estruct(lb["seeds"][seed][name]["metrics"])
               for name in lb["seeds"][seed]}
        for seed in SEEDS}

    xf = json.loads((OUTPUTS / "crossfit_lowlevel_basins.json").read_text())
    out["D_lowlevel_kmeans"] = {
        seed: {name: estruct(xf["seeds"][seed]["kmeans"]["metrics"][name])
               for name in xf["seeds"][seed]["kmeans"]["metrics"]}
        for seed in SEEDS}

    ir = json.loads((OUTPUTS / "independent_rollout_audit.json").read_text())
    out["G_neargreedy"] = {
        seed: {name: estruct(ir["seeds"][seed]["near_greedy"][name]
                             ["metrics"])
               for name in ("learned", "initial_twin")}
        for seed in SEEDS}
    return out


def spearman(xs, ys) -> float:
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    contracts = contract_metrics()
    learned = {c: [contracts[c][s]["learned"] for s in SEEDS]
               for c in contracts}
    pairs = {}
    for c1, c2 in combinations(contracts, 2):
        pairs[f"{c1} vs {c2}"] = spearman(learned[c1], learned[c2])
    mean_rho = float(np.mean(list(pairs.values())))

    separation = {}
    for c, table in contracts.items():
        gaps = []
        for seed in SEEDS:
            names = [n for n in CONTROLS if n in table[seed]]
            if not names:
                continue
            gaps.append(table[seed]["learned"]
                        - max(table[seed][n] for n in names))
        separation[c] = {
            "n_seeds_with_controls": len(gaps),
            "min_learned_minus_best_control": float(min(gaps)),
            "mean_gap": float(np.mean(gaps)),
            "class_separated_all_seeds": bool(min(gaps) > 0),
        }

    # RS-2b disclosed follow-up: separation on E_adapt (Q included).
    # Controls have Q = 0 by construction, so E_adapt = 0; learned
    # E_adapt > 0 iff the seed has positive acquisition and value.
    adapt_sep = {}
    hand = json.loads(
        (OUTPUTS / "contextual_lbf_confirmation.json").read_text())
    ext = json.loads(
        (OUTPUTS / "contextual_lbf_extension.json").read_text())
    positive = 0
    for seed in SEEDS:
        src = hand if seed in [str(s) for s in range(1101, 1111)] else ext
        entry = src["seeds"][seed]["systems"]
        m = entry["learned"]["metrics"]
        init = entry["initial_twin"]["metrics"]
        es = estruct(m)
        q = ep.acquisition_norm(
            ep.magnitude_norm(m["specificity_js_bits"]),
            ep.magnitude_norm(init["specificity_js_bits"]),
            m["conditional_selectivity"],
            init["conditional_selectivity"])
        v = ep.value_signed(m["usefulness_gap"], 0.1)
        ea = ep.e_adapt(es, q, v)
        adapt_sep[seed] = float(ea)
        positive += int(ea > 0)
    intervals = {}
    for seed in SEEDS:
        vals = [contracts[c][seed]["learned"] for c in contracts]
        intervals[seed] = {"min": float(min(vals)),
                           "max": float(max(vals)),
                           "range": float(max(vals) - min(vals))}

    report = {
        "status": ("cross-contract ranking stability of E_struct; "
                   "read-only over stored outputs; expectation declared "
                   "in the docstring"),
        "pairwise_spearman_learned_ranking": pairs,
        "mean_pairwise_spearman": mean_rho,
        "class_separation_by_contract": separation,
        "per_seed_score_intervals": intervals,
        "E_adapt_learned_by_seed": adapt_sep,
        "declared_outcomes": {
            "RS1_mean_spearman_ge_0.5": bool(mean_rho >= 0.5),
            "RS2_class_separation_every_contract": bool(all(
                s["class_separated_all_seeds"]
                for s in separation.values())),
            "RS2_note": ("FAIL retained: the competent scripted "
                         "coordinator has high structure by design; a "
                         "structure-only score cannot separate learned "
                         "from scripted, which is the layered "
                         "definition's own claim"),
            "RS2b_followup_E_adapt_positive_learned":
                f"{positive}/{len(SEEDS)} (controls are 0 by "
                "construction: Q = 0)",
        },
    }
    out = OUTPUTS / "contract_ranking_stability.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["pairwise_spearman_learned_ranking"],
                     indent=1))
    print(json.dumps(report["declared_outcomes"], indent=1))
    print(f"mean rho {mean_rho:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
