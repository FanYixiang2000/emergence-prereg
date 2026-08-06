"""Out-of-sample confirmation of the REFINED criterion on fresh seeds.

Why this experiment exists
--------------------------
The multi-seed external replication exposed a registered failure: marginal
choice tension accepted `marl_untrained` on 2/5 seeds. The refinement
(conditional selectivity: per-context trigger-rate separation >= 0.5) was
therefore introduced AFTER seeing those failures. Re-scoring old data with a
component chosen to fix that data proves nothing on its own -- a reviewer
would correctly call it criterion drift. This script freezes the refined
criterion and confirms it on data it has never seen: five NEW external
training seeds and a NEW internal battery seed.

The refined criterion also upgrades endogeneity from a design flag to a
measured component:

    acquisition = separation(trained system) - separation(same system at its
                  own initialization) >= 0.3

A static system (hand rule, untrained network) is its own initialization, so
its acquisition is 0 by definition. This makes "the structure was not
prespecified" measurable rather than declared: even a random network whose
separation happens to be high (the seed-7331 failure mode, where random
weights read HP/damage features that differ across contexts) is excluded
because the structure was present at birth rather than acquired.

Frozen components (registered here, before any new seed is run):

    potential:               H0 >= 0.5 bits
    conditional_selectivity: |p_trig(ctx A) - p_trig(ctx B)| >= 0.5
    specificity:             JS(do_trigger, do_non_trigger) >= 0.2 bits
    usefulness:              counterfactual necessity > 0
    endogeneity (design):    no process reward on the trigger,
                             no hand-coded trigger semantics
    acquisition (measured):  separation gain over the SAME system at its own
                             initialization >= 0.3

Acquisition scope: it applies where an initialization twin is measurable --
the external neural systems (we re-measure each scorer with its own
initialization weights). Hand rules and untrained networks are their own
initialization, so their acquisition is 0 by definition. For the internal
tabular battery the initialization (empty Q-table) is degenerate-uniform and
the forced-behavior probe systems carry their forcing at initialization, so
acquisition is not informative there; the internal battery keeps the five
refined components with the design-flag endogeneity. This split is honest:
internally we built every system and can flag design; externally we do not
trust flags for learned systems and measure acquisition instead.

Registered predictions for the fresh external seeds (8031..8431):

    C1  marl_learned is accepted whenever its behavioral audit passes
        (per-context rates separate); expected 5/5.
    C2  marl_untrained is excluded on ALL fresh seeds, and specifically
        fails acquisition on every seed (whatever its separation happens
        to be -- this is the component that removes the seed-7331 failure
        mode without touching any other threshold).
    C3  All three hand rules are excluded on all seeds; damage_aware fails
        exactly {endogeneity, acquisition} and nothing else.
    C4  Fresh internal battery seed: all 10 systems (9 original +
        anti_selector) classified correctly; anti_selector fails exactly
        {usefulness}.

Failure counts as failure: any miss is reported, not patched.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from external_swarm_criterion_transfer import (
    THRESHOLDS as EXT_THRESHOLDS,
    audit_label,
    measure_system as measure_external_system,
    rule_controller,
    scorer_controller,
    train_marl_scorer,
    untrained_scorer,
)
from contextual_sacrifice_gridworld import MODES, train_policy
from criterion_ablation_battery import (
    THRESHOLDS as INT_THRESHOLDS,
    measure_system as measure_internal_system,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

SEPARATION_THRESHOLD = 0.5
ACQUISITION_THRESHOLD = 0.3

REFINED_COMPONENTS = (
    "potential", "conditional_selectivity", "specificity",
    "usefulness", "endogeneity", "acquisition",
)


def refined_verdict(
    h0: float, separation: float, specificity: float, usefulness: float,
    prespecified: bool, acquisition: float | None,
) -> Dict[str, Any]:
    passes = {
        "potential": h0 >= 0.5,
        "conditional_selectivity": separation >= SEPARATION_THRESHOLD,
        "specificity": specificity >= 0.2,
        "usefulness": usefulness > 0.0,
        "endogeneity": not prespecified,
    }
    if acquisition is not None:
        passes["acquisition"] = acquisition >= ACQUISITION_THRESHOLD
    return {
        "passes": passes,
        "emergent": int(all(passes.values())),
        "failed": [k for k, ok in passes.items() if not ok],
    }


def external_confirmation(seeds: List[int], iters: int, batch: int, lr: float,
                          n_eval: int) -> Dict[str, Any]:
    per_seed: Dict[str, Any] = {}
    rows_out: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    for seed in seeds:
        print(f"\n=== fresh external seed {seed} ===")
        learned, _hist = train_marl_scorer(iters, batch, lr, seed)
        init_twin = untrained_scorer(seed)  # same torch seed as learner's init
        untrained = untrained_scorer(seed + 999)

        systems = (
            ("marl_learned", scorer_controller(learned), False, init_twin),
            ("marl_untrained", scorer_controller(untrained), False, None),
            ("nearest_only", rule_controller(["nearest"]), True, None),
            ("role_oracle",
             rule_controller(["threat", "fragile", "non_decoy", "nearest"]), True, None),
            ("damage_aware", rule_controller(["damage", "nearest"]), True, None),
        )
        seed_verdicts: Dict[str, Any] = {}
        for idx, (name, controller, prespec, init_scorer) in enumerate(systems):
            row = measure_external_system(name, controller, prespec, n_eval,
                                          seed + idx * 50_000)
            separation = abs(
                float(row["aggressive_trigger_rate"]) - float(row["passive_trigger_rate"])
            )
            if init_scorer is not None:
                init_row = measure_external_system(
                    f"{name}_init", scorer_controller(init_scorer), prespec,
                    n_eval, seed + idx * 50_000 + 7,
                )
                init_sep = abs(
                    float(init_row["aggressive_trigger_rate"])
                    - float(init_row["passive_trigger_rate"])
                )
            else:
                init_sep = separation  # static system: it is its own init
            acquisition = separation - init_sep
            v = refined_verdict(
                float(row["h0_bits"]), separation, float(row["specificity_js"]),
                float(row["usefulness_gap"]), prespec, acquisition,
            )
            truth = audit_label(row) if name == "marl_learned" else 0
            seed_verdicts[name] = {
                "separation": separation,
                "init_separation": init_sep,
                "acquisition": acquisition,
                "verdict": v,
                "truth": truth,
                "correct": v["emergent"] == truth,
            }
            correct += int(v["emergent"] == truth)
            total += 1
            rows_out.append({
                "seed": seed, "system": name,
                **{k: row[k] for k in (
                    "h0_bits", "natural_trigger_rate", "passive_trigger_rate",
                    "aggressive_trigger_rate", "specificity_js", "usefulness_gap",
                )},
                "separation": separation, "acquisition": acquisition,
                "emergent": v["emergent"], "truth": truth,
            })
            print(f"  {name:15s} sep {separation:.2f} acq {acquisition:+.2f} "
                  f"verdict {v['emergent']} truth {truth} failed {';'.join(v['failed']) or '-'}")
        per_seed[str(seed)] = seed_verdicts

    checks = {
        "c1_learner_accepted_when_audited": all(
            per_seed[s]["marl_learned"]["correct"] for s in per_seed
        ),
        "c2_untrained_excluded_all_seeds_via_acquisition": all(
            per_seed[s]["marl_untrained"]["verdict"]["emergent"] == 0
            and "acquisition" in per_seed[s]["marl_untrained"]["verdict"]["failed"]
            for s in per_seed
        ),
        "c3_damage_aware_fails_exactly_endo_acq": all(
            set(per_seed[s]["damage_aware"]["verdict"]["failed"])
            == {"endogeneity", "acquisition"}
            for s in per_seed
        ),
    }
    return {
        "per_seed": per_seed, "rows": rows_out,
        "accuracy": correct / total, "correct": correct, "total": total,
        "checks": checks,
    }


def internal_confirmation(train_episodes: int, seed: int) -> Dict[str, Any]:
    print(f"\n=== fresh internal battery seed {seed} ===")
    policies = {
        regime: train_policy(regime, train_episodes, seed + idx * 10_000)
        for idx, regime in enumerate(
            ("uncertain_preference", "pure_team", "dense_shaping", "random_noise")
        )
    }
    untrained: Dict = {}
    system_specs = (
        ("latent_conditional", policies["uncertain_preference"], "uncertain_preference",
         list(MODES), None, False, 1),
        ("converged_team", policies["pure_team"], "pure_team",
         list(MODES), None, False, 0),
        ("shaped_process", policies["dense_shaping"], "dense_shaping",
         list(MODES), None, True, 0),
        ("noise_policy", policies["random_noise"], "random_noise",
         list(MODES), None, False, 1),
        ("untrained_uniform", untrained, "pure_team",
         list(MODES), None, False, 0),
        ("blind_trigger", policies["uncertain_preference"], "uncertain_preference",
         list(MODES), "do_trigger", False, 0),
        ("harmful_decoy", policies["uncertain_preference"], "uncertain_preference",
         ["bridge"], "do_trigger", False, 0),
        ("useful_habit", policies["uncertain_preference"], "uncertain_preference",
         ["rescue"], "do_trigger", False, 0),
        ("wrong_selector", policies["uncertain_preference"], "uncertain_preference",
         list(MODES), {"rescue": None, "bridge": "do_trigger"}, False, 0),
        ("anti_selector", policies["uncertain_preference"], "uncertain_preference",
         list(MODES), {"rescue": "do_non_trigger", "bridge": "do_trigger"}, False, 0),
    )
    verdicts: Dict[str, Any] = {}
    correct = 0
    for idx, (name, q_table, regime, modes, behavior, prespec, label) in enumerate(system_specs):
        row = measure_internal_system(
            name, q_table, regime, modes, behavior, prespec, label,
            probe_episodes=24, samples=36, temperature=0.25,
            probe_temperature=0.9, seed=seed + idx * 5_000,
        )
        separation = abs(
            float(row["rescue_trigger_rate"]) - float(row["bridge_trigger_rate"])
        )
        if name in ("harmful_decoy", "useful_habit"):
            separation = 0.0  # single-context systems: conditionality undefined
        # Internal battery: acquisition is not informative (see docstring);
        # the five refined components with design-flag endogeneity apply.
        v = refined_verdict(
            float(row["h0_bits"]), separation, float(row["specificity_js"]),
            float(row["usefulness_gap"]), prespec, None,
        )
        verdicts[name] = {
            "separation": separation,
            "verdict": v, "truth": label, "correct": v["emergent"] == label,
        }
        correct += int(v["emergent"] == label)
        print(f"  {name:20s} sep {separation:.2f} verdict {v['emergent']} truth {label} "
              f"failed {';'.join(v['failed']) or '-'}")
    return {"verdicts": verdicts, "accuracy": correct / len(system_specs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Out-of-sample refined-criterion confirmation.")
    parser.add_argument("--seeds", type=str, default="8031,8131,8231,8331,8431")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n_eval", type=int, default=120)
    parser.add_argument("--internal_train_episodes", type=int, default=60000)
    parser.add_argument("--internal_seed", type=int, default=7011)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    external = external_confirmation(seeds, args.iters, args.batch, args.lr, args.n_eval)
    internal = internal_confirmation(args.internal_train_episodes, args.internal_seed)

    checks = dict(external["checks"])
    checks["c4_internal_fresh_seed_all_correct"] = internal["accuracy"] == 1.0
    anti = internal["verdicts"].get("anti_selector")
    checks["c4b_anti_selector_fails_only_usefulness"] = (
        anti is not None and anti["verdict"]["failed"] == ["usefulness"]
    )

    summary = {
        "frozen_components": list(REFINED_COMPONENTS),
        "separation_threshold": SEPARATION_THRESHOLD,
        "acquisition_threshold": ACQUISITION_THRESHOLD,
        "external": {k: v for k, v in external.items() if k != "rows"},
        "internal": internal,
        "confirmation_checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "refined_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "refined_confirmation_external.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(external["rows"][0].keys()))
        writer.writeheader()
        for row in external["rows"]:
            writer.writerow(row)

    print(f"\nExternal fresh-seed accuracy: {external['correct']}/{external['total']}")
    print(f"Internal fresh-seed accuracy: {internal['accuracy']:.3f}")
    print("\nConfirmation checks:")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\nWrote {args.output_dir / 'refined_confirmation_summary.json'}")


if __name__ == "__main__":
    main()
