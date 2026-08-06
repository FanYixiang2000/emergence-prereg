"""Rollout-model audit: is openness a decode-temperature artifact?

Reviewer objection addressed: the future distribution is estimated by
rolling out the evaluated policy itself, so measured potential could be
nothing but the policy's own sampling stochasticity. Here every learned
Contextual-LBF policy and its initialization twin are re-scored under two
alternative rollout models with frozen thresholds:

    near-greedy  softmax temperature 0.2 (approximately deterministic
                 decoding; any remaining openness must come from genuine
                 context/layout variation, not sampling noise);
    diffuse      softmax temperature 2.0 (inflated sampling noise; a
                 stochasticity-artifact account predicts inflated
                 potential should rescue the twin or blur selectivity).

All six components are re-measured per rollout model; acquisition is the
learned-minus-twin selectivity gap within the same model.

Registered predictions (frozen before running):
    IR-1  learned potential >= 0.5 bits under near-greedy decoding on
          >= 13/15 seeds (openness survives removing sampling noise);
    IR-2  learned systems accepted under BOTH rollout models on
          >= 13/15 seeds;
    IR-3  initialization twins rejected under both models on 15/15 seeds.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import torch

import contextual_lbf_transfer as clbf
import lbf_collapse_probe as base

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

N_EVAL = 80
SEED_STREAM = 12_000_000
ROLLOUT_MODELS = {"near_greedy": 0.2, "diffuse": 2.0}
THRESHOLDS = clbf.THRESHOLDS

CONFIRMATION_SEEDS = (1101, 1102, 1103, 1104, 1105,
                      1106, 1107, 1108, 1109, 1110)
EXTENSION_SEEDS = (1201, 1202, 1203, 1204, 1205)


def policy_controller(net: base.PolicyNet,
                      temperature: float) -> clbf.TeamController:
    controller = clbf.TeamController("policy", net)
    controller.policy = base.Controller("policy", net, temperature)
    return controller


def verdict(metrics: Dict[str, Any], endogenous: bool,
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
    return {"passes": passes, "emergent": int(all(passes.values())),
            "failed": [k for k, ok in passes.items() if not ok]}


def load_net(seed: int) -> base.PolicyNet:
    net = base.PolicyNet()
    net.load_state_dict(torch.load(
        OUTPUTS / f"contextual_lbf_net_seed{seed}.pt", map_location="cpu"))
    net.eval()
    return net


def run_seed(seed: int) -> Dict[str, Any]:
    offset = SEED_STREAM + seed * 100_000
    learned_net = load_net(seed)
    twin_net = clbf.initial_twin(seed)
    out: Dict[str, Any] = {}
    for model_name, temperature in ROLLOUT_MODELS.items():
        learned = clbf.evaluate(
            policy_controller(learned_net, temperature), N_EVAL, offset)
        twin = clbf.evaluate(
            policy_controller(twin_net, temperature), N_EVAL, offset)
        acquisition = (learned["metrics"]["conditional_selectivity"]
                       - twin["metrics"]["conditional_selectivity"])
        out[model_name] = {
            "learned": {
                "metrics": learned["metrics"],
                "acquisition": acquisition,
                "verdict": verdict(learned["metrics"], True, acquisition),
            },
            "initial_twin": {
                "metrics": twin["metrics"],
                "acquisition": 0.0,
                "verdict": verdict(twin["metrics"], True, 0.0),
            },
        }
    return out


def main() -> None:
    random.seed(0)
    torch.set_num_threads(8)
    seeds = list(CONFIRMATION_SEEDS) + list(EXTENSION_SEEDS)
    results: Dict[str, Any] = {}
    ir1 = 0
    ir2 = 0
    ir3 = 0
    for seed in seeds:
        print(f"rollout models, seed {seed}", flush=True)
        results[str(seed)] = run_seed(seed)
        near = results[str(seed)]["near_greedy"]
        diff = results[str(seed)]["diffuse"]
        ir1 += int(near["learned"]["metrics"]["potential_bits"] >= 0.5)
        ir2 += int(near["learned"]["verdict"]["emergent"] == 1
                   and diff["learned"]["verdict"]["emergent"] == 1)
        ir3 += int(near["initial_twin"]["verdict"]["emergent"] == 0
                   and diff["initial_twin"]["verdict"]["emergent"] == 0)
        print(f"  near-greedy: learned pot "
              f"{near['learned']['metrics']['potential_bits']:.2f} "
              f"verdict {near['learned']['verdict']['emergent']} "
              f"(failed {';'.join(near['learned']['verdict']['failed']) or '-'}); "
              f"diffuse verdict {diff['learned']['verdict']['emergent']}",
              flush=True)

    summary = {
        "status": ("rollout-model audit (near-greedy T=0.2 and diffuse "
                   "T=2.0); frozen thresholds; predictions IR-1..IR-3 "
                   "frozen in the docstring"),
        "registered_outcomes": {
            "IR1_potential_survives_greedy": f"{ir1}/{len(seeds)}",
            "IR2_learned_accepted_both_models": f"{ir2}/{len(seeds)}",
            "IR3_twins_rejected_both_models": f"{ir3}/{len(seeds)}",
            "IR1_pass": ir1 >= 13,
            "IR2_pass": ir2 >= 13,
            "IR3_pass": ir3 == len(seeds),
        },
        "seeds": results,
    }
    out = OUTPUTS / "independent_rollout_audit.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
