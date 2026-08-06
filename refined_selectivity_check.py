"""Registered-failure analysis: marginal tension vs conditional selectivity.

What happened
-------------
The multi-seed external replication (run_external_transfer_sweep.py) exposed
a genuine failure of the registered criterion: on 2 of 5 seeds the full
criterion accepted `marl_untrained`, a randomly initialized network. Its
marginal trigger rate happened to sit mid-range (tension above 0.5) and its
counterfactual-necessity gap happened to be positive, because engaging the
front enemies is helpful in about half the episodes (the aggressive context)
regardless of who engages.

Why it happened
---------------
The registered selectivity component measures marginal choice tension
4 p (1 - p). Tension asks "does the system sometimes trigger and sometimes
not?" -- it does not ask "does the trigger RESPOND to the latent context?".
A random controller can sit at mid-tension by accident. Every genuinely
emergent system in our batteries, and the pre-registered behavioral audit
rule itself, actually relies on the stronger property: per-context trigger
rates separate (the trigger is a selected response to the latent context).

The refinement
--------------
    conditional selectivity: |p_trigger(context A) - p_trigger(context B)|
                             >= 0.5
    (for single-context systems the separation is 0 by definition)

This file re-scores, without re-measuring, both the external sweep and the
internal battery under the refined component, and measures one NEW internal
system that the refinement makes necessary:

- anti_selector: perfectly conditional (triggers exactly in the WRONG
  context: never in rescue, always in bridge), open potential, trigger-
  specific, endogenous -- excluded only by usefulness. It replaces
  wrong_selector as the counterexample that uniquely pins the usefulness
  component, because wrong_selector (trigger everywhere except selective in
  rescue) loses its per-context separation under the refined component.

Expected outcomes (stated before running this script):

1. External sweep: refined criterion scores 25/25 verdicts across the 5
   seeds (untrained separation ~0.15 fails conditional selectivity).
2. Internal battery: refined criterion keeps all previous systems correct
   (their pass/fail structure is unchanged except wrong_selector, which now
   also fails selectivity) and classifies anti_selector correctly as
   non-emergent, rejected only via usefulness.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from contextual_sacrifice_gridworld import MODES, train_policy
from criterion_ablation_battery import THRESHOLDS, measure_system

OUTPUTS = Path(__file__).resolve().parent / "outputs"

SEPARATION_THRESHOLD = 0.5


def refined_passes(row: Dict[str, str | float], separation: float) -> Dict[str, bool]:
    return {
        "potential": float(row["h0_bits"]) >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": separation >= SEPARATION_THRESHOLD,
        "specificity": float(row["specificity_js"]) >= THRESHOLDS["specificity_js"],
        "usefulness": float(row["usefulness_gap"]) > THRESHOLDS["usefulness_gap"],
        "endogeneity": int(row["prespecified"]) == 0,
    }


def rescore_external(rows: List[Dict[str, str]]) -> Dict[str, object]:
    per_seed: Dict[str, Dict[str, Dict[str, object]]] = {}
    correct = 0
    total = 0
    for row in rows:
        system = row["system"]
        seed = row["seed"]
        separation = abs(
            float(row["aggressive_trigger_rate"]) - float(row["passive_trigger_rate"])
        )
        passes = refined_passes(row, separation)
        predicted = int(all(passes.values()))
        if system == "marl_learned":
            truth = int(
                float(row["aggressive_trigger_rate"]) >= 0.7
                and float(row["passive_trigger_rate"]) <= 0.3
            )
        else:
            truth = 0
        per_seed.setdefault(seed, {})[system] = {
            "separation": separation,
            "predicted": predicted,
            "truth": truth,
            "failed": [k for k, ok in passes.items() if not ok],
        }
        correct += int(predicted == truth)
        total += 1
    return {"per_seed": per_seed, "accuracy": correct / total, "correct": correct, "total": total}


def rescore_internal(rows: List[Dict[str, str]]) -> Dict[str, object]:
    verdicts: Dict[str, Dict[str, object]] = {}
    correct = 0
    for row in rows:
        separation = abs(
            float(row["rescue_trigger_rate"]) - float(row["bridge_trigger_rate"])
        )
        # Single-mode systems (harmful_decoy, useful_habit) have one rate
        # measured and the other 0 by construction; their separation is set
        # to 0 because conditionality is undefined without both contexts.
        if row["system"] in ("harmful_decoy", "useful_habit"):
            separation = 0.0
        passes = refined_passes(row, separation)
        predicted = int(all(passes.values()))
        truth = int(row["ground_truth_emergent"])
        verdicts[str(row["system"])] = {
            "separation": separation,
            "predicted": predicted,
            "truth": truth,
            "failed": [k for k, ok in passes.items() if not ok],
        }
        correct += int(predicted == truth)
    return {"verdicts": verdicts, "accuracy": correct / len(rows)}


def measure_anti_selector(train_episodes: int, seed: int) -> Dict[str, str | float]:
    print("Training uncertain_preference policy for anti_selector measurement ...")
    q_table = train_policy("uncertain_preference", train_episodes, seed)
    row = measure_system(
        "anti_selector",
        q_table,
        "uncertain_preference",
        list(MODES),
        {"rescue": "do_non_trigger", "bridge": "do_trigger"},
        prespecified=False,
        ground_truth=0,
        probe_episodes=24,
        samples=36,
        temperature=0.25,
        probe_temperature=0.9,
        seed=seed + 5_000,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Refined conditional-selectivity check.")
    parser.add_argument("--train_episodes", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=6011)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    with (OUTPUTS / "external_transfer_sweep_per_seed.csv").open(encoding="utf-8") as f:
        external_rows = list(csv.DictReader(f))
    with (OUTPUTS / "criterion_battery_measurements.csv").open(encoding="utf-8") as f:
        internal_rows = list(csv.DictReader(f))

    anti = measure_anti_selector(args.train_episodes, args.seed)
    internal_rows.append({k: str(v) for k, v in anti.items()})

    external = rescore_external(external_rows)
    internal = rescore_internal(internal_rows)

    summary = {
        "separation_threshold": SEPARATION_THRESHOLD,
        "registered_failure": {
            "description": (
                "Original marginal-tension selectivity accepted marl_untrained "
                "on 2/5 external seeds (see external_transfer_sweep_summary.json)."
            ),
        },
        "external_rescored": external,
        "internal_rescored": internal,
        "anti_selector_measurement": anti,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "refined_selectivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nExternal sweep rescored: {external['correct']}/{external['total']} "
          f"(accuracy {external['accuracy']:.3f})")
    for seed, systems in external["per_seed"].items():  # type: ignore[union-attr]
        untrained = systems["marl_untrained"]
        print(f"  seed {seed}: untrained separation {untrained['separation']:.3f} "
              f"predicted {untrained['predicted']} (truth 0)")
    print(f"\nInternal battery rescored (with anti_selector): "
          f"accuracy {internal['accuracy']:.3f}")
    for system, verdict in internal["verdicts"].items():  # type: ignore[union-attr]
        failed = ";".join(verdict["failed"]) or "-"
        print(f"  {system:20s} truth {verdict['truth']} predicted {verdict['predicted']} "
              f"sep {verdict['separation']:.3f} failed: {failed}")
    print(f"\nWrote {args.output_dir / 'refined_selectivity_summary.json'}")


if __name__ == "__main__":
    main()
