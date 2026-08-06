"""Dual plausible observer contracts on the Contextual LBF systems.

Reviewer objection addressed: robustness to RANDOM basin partitions does not
show that two reasonable-but-different observers agree. This script re-scores
every stored Contextual-LBF system (15 policy seeds x 5 systems: learned,
initialization twin, team_nearest, fixed_food0, fixed_food1) under a second,
semantically meaningful observer contract, with all six thresholds frozen at
their registered values, and asks whether any verdict flips.

Contract A (registered; stored in contextual_lbf_confirmation/extension):
    horizon 15; basins = (win/loss) x (first food identity);
    value = discounted mean team reward; usefulness = natural minus
    do_non_trigger value.

Contract B (declared here, before running):
    horizon 12 (a shorter episode budget);
    basins = (first food identity) x (fast completion within 10 steps or
    not) -- an observer who tracks speed rather than win/loss;
    value = undiscounted success probability (both foods collected);
    fresh evaluation seed stream (a different rollout draw is part of a
    different contract).

Identifiability requirement (stated in the measurement contract): any
admissible observer must RESOLVE the declared macro-structure, i.e. its
basin map must distinguish trigger from non-trigger futures. Both contracts
satisfy it; contracts that erase the trigger identity are excluded by the
contract, not by this test.

Registered predictions (frozen before the first episode of this script):
    DO1  all 15 learned systems are accepted under contract B;
    DO2  all 60 control systems are rejected under contract B;
    DO3  contract A and contract B verdicts agree on all 75 systems.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import contextual_lbf_transfer as clbf
import lbf_collapse_probe as base

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

CONTRACT_B_HORIZON = 12
FAST_STEP_CUT = 10
N_EVAL = 80
SEED_STREAM = 9_000_000

BASINS_B = ("food0_fast", "food0_slow", "food1_fast", "food1_slow")
THRESHOLDS = clbf.THRESHOLDS  # frozen registered thresholds, unchanged

CONFIRMATION_SEEDS = (1101, 1102, 1103, 1104, 1105,
                      1106, 1107, 1108, 1109, 1110)
EXTENSION_SEEDS = (1201, 1202, 1203, 1204, 1205)


def basin_b(row: Dict[str, Any]) -> str:
    food = "food0" if row["trigger"] else "food1"
    fast = row["win"] and row["steps"] <= FAST_STEP_CUT
    return f"{food}_{'fast' if fast else 'slow'}"


def normalize_b(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {b: counts.get(b, 0) / total for b in BASINS_B}


def evaluate_contract_b(controller: clbf.TeamController,
                        seed_offset: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for context in clbf.CONTEXTS:
        for episode in range(N_EVAL):
            paired_seed = seed_offset + 10_000 * context + episode
            for mode in (None, "do_trigger", "do_non_trigger"):
                row = clbf.run_episode(controller, context, paired_seed, mode)
                row["mode"] = mode or "natural"
                rows.append(row)

    def subset(mode: str, context=None):
        return [r for r in rows if r["mode"] == mode
                and (context is None or r["context"] == context)]

    natural = subset("natural")
    do_trigger = subset("do_trigger")
    do_non = subset("do_non_trigger")

    def counts(group):
        return {b: sum(basin_b(r) == b for r in group) for b in BASINS_B}

    trigger_rates = {
        str(c): float(np.mean([r["trigger"] for r in subset("natural", c)]))
        for c in clbf.CONTEXTS
    }
    return {
        "potential_bits": clbf.entropy(normalize_b(counts(natural))),
        "conditional_selectivity": abs(
            trigger_rates["0"] - trigger_rates["1"]),
        "trigger_rates": trigger_rates,
        "specificity_js_bits": clbf.js(
            normalize_b(counts(do_trigger)), normalize_b(counts(do_non))),
        "usefulness_gap": float(
            np.mean([r["win"] for r in natural])
            - np.mean([r["win"] for r in do_non])
        ),
        "natural_success": float(np.mean([r["win"] for r in natural])),
        "do_non_trigger_success": float(np.mean([r["win"] for r in do_non])),
    }


def verdict_b(metrics: Dict[str, Any], endogenous: bool,
              acquisition: float) -> Dict[str, Any]:
    passes = {
        "potential": metrics["potential_bits"] >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": (
            metrics["conditional_selectivity"]
            >= THRESHOLDS["conditional_selectivity"]),
        "specificity": (
            metrics["specificity_js_bits"]
            >= THRESHOLDS["specificity_js_bits"]),
        "usefulness": metrics["usefulness_gap"] > THRESHOLDS["usefulness_gap"],
        "endogeneity": endogenous,
        "acquisition": acquisition >= THRESHOLDS["acquisition"],
    }
    return {
        "passes": passes,
        "emergent": int(all(passes.values())),
        "failed": [k for k, ok in passes.items() if not ok],
    }


def load_net(seed: int) -> base.PolicyNet:
    net = base.PolicyNet()
    net.load_state_dict(torch.load(
        OUTPUTS / f"contextual_lbf_net_seed{seed}.pt", map_location="cpu"))
    net.eval()
    return net


def contract_a_verdicts() -> Dict[str, Dict[str, int]]:
    stored: Dict[str, Dict[str, int]] = {}
    for path, seeds in (
        (OUTPUTS / "contextual_lbf_confirmation.json", CONFIRMATION_SEEDS),
        (OUTPUTS / "contextual_lbf_extension.json", EXTENSION_SEEDS),
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        for seed in seeds:
            systems = data["seeds"][str(seed)]["systems"]
            stored[str(seed)] = {
                name: int(sys_data["verdict"]["emergent"])
                for name, sys_data in systems.items()
            }
    return stored


def run_seed(seed: int) -> Dict[str, Any]:
    offset = SEED_STREAM + seed * 100_000
    learned_net = load_net(seed)
    twin = clbf.initial_twin(seed)

    learned = evaluate_contract_b(
        clbf.TeamController("policy", learned_net), offset)
    init = evaluate_contract_b(
        clbf.TeamController("policy", twin), offset)
    acquisition = (learned["conditional_selectivity"]
                   - init["conditional_selectivity"])
    systems = {
        "learned": (learned, True, acquisition),
        "initial_twin": (init, True, 0.0),
        "team_nearest": (evaluate_contract_b(
            clbf.TeamController("team_nearest"), offset), False, 0.0),
        "fixed_food0": (evaluate_contract_b(
            clbf.TeamController("fixed_food0"), offset), False, 0.0),
        "fixed_food1": (evaluate_contract_b(
            clbf.TeamController("fixed_food1"), offset), False, 0.0),
    }
    out: Dict[str, Any] = {}
    for name, (metrics, endo, acq) in systems.items():
        out[name] = {
            "metrics": metrics,
            "acquisition": acq,
            "verdict": verdict_b(metrics, endo, acq),
        }
    return out


def main() -> None:
    random.seed(0)
    torch.set_num_threads(8)
    clbf.HORIZON = CONTRACT_B_HORIZON  # module global read at reset time

    stored_a = contract_a_verdicts()
    expected = {"learned": 1, "initial_twin": 0, "team_nearest": 0,
                "fixed_food0": 0, "fixed_food1": 0}

    seeds = list(CONFIRMATION_SEEDS) + list(EXTENSION_SEEDS)
    results: Dict[str, Any] = {}
    agree = 0
    total = 0
    learned_accepted = 0
    controls_rejected = 0
    for seed in seeds:
        print(f"contract B, seed {seed}", flush=True)
        results[str(seed)] = run_seed(seed)
        for name, entry in results[str(seed)].items():
            b_verdict = entry["verdict"]["emergent"]
            a_verdict = stored_a[str(seed)][name]
            total += 1
            agree += int(a_verdict == b_verdict)
            if name == "learned":
                learned_accepted += b_verdict
            else:
                controls_rejected += int(b_verdict == 0)
            marker = "" if a_verdict == b_verdict else "  <-- FLIP"
            print(f"  {name:14s} A={a_verdict} B={b_verdict} "
                  f"failed={';'.join(entry['verdict']['failed']) or '-'}"
                  f"{marker}", flush=True)

    summary = {
        "status": ("dual plausible observer contracts; contract B declared "
                   "in the module docstring before running; thresholds "
                   "frozen at registered values"),
        "contract_b": {
            "horizon": CONTRACT_B_HORIZON,
            "basins": list(BASINS_B),
            "fast_step_cut": FAST_STEP_CUT,
            "value": "undiscounted success probability",
            "n_eval_per_context": N_EVAL,
        },
        "registered_outcomes": {
            "DO1_learned_accepted": f"{learned_accepted}/{len(seeds)}",
            "DO2_controls_rejected": f"{controls_rejected}/{4 * len(seeds)}",
            "DO3_contract_agreement": f"{agree}/{total}",
            "DO1_pass": learned_accepted == len(seeds),
            "DO2_pass": controls_rejected == 4 * len(seeds),
            "DO3_pass": agree == total,
        },
        "expected": expected,
        "seeds": results,
    }
    out = OUTPUTS / "dual_observer_contracts.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
