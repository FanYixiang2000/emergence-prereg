"""Performance-closure benchmark for latent possibility emergence.

This benchmark closes the loop between "new structure" and task performance.
All agents face the same finite-horizon task. The only ablated capability is
whether the agent can:

1. avoid myopic cash-out,
2. preserve the future option,
3. observe the revealed context,
4. choose the context-correct future basin.

If emergence is useful, the full capability should improve both success rate
and final return, not merely produce a high PTC score.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


CAPABILITIES = (
    "myopic_cash_out",
    "no_preserve_action",
    "preserve_no_context",
    "preserve_always_trigger",
    "preserve_always_direct",
    "preserve_context_contingent",
)


@dataclass(frozen=True)
class ClosureParams:
    p_trigger_needed: float
    cash_out: float
    preserve_cost: float
    trigger_payoff: float
    direct_payoff: float
    mismatch_payoff: float = 0.0


def evaluate_capability(params: ClosureParams, capability: str) -> Dict[str, float | str]:
    if capability not in CAPABILITIES:
        raise KeyError(f"unknown capability: {capability}")

    p = params.p_trigger_needed
    q = 1.0 - p
    if capability in ("myopic_cash_out", "no_preserve_action"):
        return {
            "capability": capability,
            "expected_return": params.cash_out,
            "success_rate": 0.0,
            "trigger_rate": 0.0,
            "option_preserved": 0.0,
            "context_used": 0.0,
            "over_trigger_rate": 0.0,
        }

    if capability == "preserve_always_trigger":
        expected = -params.preserve_cost + p * params.trigger_payoff + q * params.mismatch_payoff
        return {
            "capability": capability,
            "expected_return": expected,
            "success_rate": p,
            "trigger_rate": 1.0,
            "option_preserved": 1.0,
            "context_used": 0.0,
            "over_trigger_rate": q,
        }

    if capability == "preserve_always_direct":
        expected = -params.preserve_cost + p * params.mismatch_payoff + q * params.direct_payoff
        return {
            "capability": capability,
            "expected_return": expected,
            "success_rate": q,
            "trigger_rate": 0.0,
            "option_preserved": 1.0,
            "context_used": 0.0,
            "over_trigger_rate": 0.0,
        }

    if capability == "preserve_no_context":
        trigger_expected = p * params.trigger_payoff + q * params.mismatch_payoff
        direct_expected = p * params.mismatch_payoff + q * params.direct_payoff
        choose_trigger = trigger_expected >= direct_expected
        if choose_trigger:
            base = evaluate_capability(params, "preserve_always_trigger")
        else:
            base = evaluate_capability(params, "preserve_always_direct")
        return {
            **base,
            "capability": capability,
            "context_used": 0.0,
        }

    # Full capability: preserve the option, observe context, and choose the
    # context-correct basin.
    expected = -params.preserve_cost + p * params.trigger_payoff + q * params.direct_payoff
    return {
        "capability": capability,
        "expected_return": expected,
        "success_rate": 1.0,
        "trigger_rate": p,
        "option_preserved": 1.0,
        "context_used": 1.0,
        "over_trigger_rate": 0.0,
    }


def frange(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 10))
        value += step
    return values


def sweep(
    p_values: Sequence[float],
    cash_values: Sequence[float],
    cost_values: Sequence[float],
    trigger_payoff: float,
    direct_payoff: float,
    mismatch_payoff: float,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for p in p_values:
        for cash_out in cash_values:
            for preserve_cost in cost_values:
                params = ClosureParams(
                    p_trigger_needed=p,
                    cash_out=cash_out,
                    preserve_cost=preserve_cost,
                    trigger_payoff=trigger_payoff,
                    direct_payoff=direct_payoff,
                    mismatch_payoff=mismatch_payoff,
                )
                evaluated = [
                    evaluate_capability(params, capability)
                    for capability in CAPABILITIES
                ]
                best_return = max(float(item["expected_return"]) for item in evaluated)
                full = next(
                    item
                    for item in evaluated
                    if item["capability"] == "preserve_context_contingent"
                )
                no_context = next(
                    item for item in evaluated if item["capability"] == "preserve_no_context"
                )
                myopic = next(item for item in evaluated if item["capability"] == "myopic_cash_out")
                for item in evaluated:
                    row = {
                        **asdict(params),
                        **item,
                        "is_best_return": float(float(item["expected_return"]) == best_return),
                        "full_return_gain_vs_myopic": (
                            float(full["expected_return"]) - float(myopic["expected_return"])
                        ),
                        "full_return_gain_vs_no_context": (
                            float(full["expected_return"]) - float(no_context["expected_return"])
                        ),
                        "full_success_gain_vs_no_context": (
                            float(full["success_rate"]) - float(no_context["success_rate"])
                        ),
                        "performance_closure": float(
                            float(full["expected_return"]) > float(myopic["expected_return"])
                            and float(full["expected_return"]) > float(no_context["expected_return"])
                        ),
                    }
                    rows.append(row)
    return rows


def summarize(rows: Sequence[Mapping[str, float | str]]) -> Dict[str, float]:
    full_rows = [
        row for row in rows if row["capability"] == "preserve_context_contingent"
    ]
    no_context_rows = [row for row in rows if row["capability"] == "preserve_no_context"]
    myopic_rows = [row for row in rows if row["capability"] == "myopic_cash_out"]

    def mean(key: str, selected: Sequence[Mapping[str, float | str]]) -> float:
        return (
            sum(float(row[key]) for row in selected) / len(selected)
            if selected
            else 0.0
        )

    return {
        "n_conditions": float(len(full_rows)),
        "full_best_return_rate": mean("is_best_return", full_rows),
        "performance_closure_rate": mean("performance_closure", full_rows),
        "full_mean_return": mean("expected_return", full_rows),
        "myopic_mean_return": mean("expected_return", myopic_rows),
        "no_context_mean_return": mean("expected_return", no_context_rows),
        "full_mean_success": mean("success_rate", full_rows),
        "no_context_mean_success": mean("success_rate", no_context_rows),
        "full_return_gain_vs_myopic": mean("full_return_gain_vs_myopic", full_rows),
        "full_return_gain_vs_no_context": mean("full_return_gain_vs_no_context", full_rows),
        "full_success_gain_vs_no_context": mean("full_success_gain_vs_no_context", full_rows),
    }


def write_outputs(
    rows: Sequence[Mapping[str, float | str]],
    summary: Mapping[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "performance_closure_summary.json").write_text(
        json.dumps({"summary": summary}, indent=2),
        encoding="utf-8",
    )
    columns = list(rows[0].keys()) if rows else []
    with (output_dir / "performance_closure_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Performance-closure capability ablation.")
    parser.add_argument("--p_start", type=float, default=0.05)
    parser.add_argument("--p_stop", type=float, default=0.95)
    parser.add_argument("--p_step", type=float, default=0.05)
    parser.add_argument("--cash_start", type=float, default=0.0)
    parser.add_argument("--cash_stop", type=float, default=12.0)
    parser.add_argument("--cash_step", type=float, default=0.5)
    parser.add_argument("--cost_start", type=float, default=0.0)
    parser.add_argument("--cost_stop", type=float, default=4.0)
    parser.add_argument("--cost_step", type=float, default=0.5)
    parser.add_argument("--trigger_payoff", type=float, default=13.0)
    parser.add_argument("--direct_payoff", type=float, default=9.0)
    parser.add_argument("--mismatch_payoff", type=float, default=0.0)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = sweep(
        p_values=frange(args.p_start, args.p_stop, args.p_step),
        cash_values=frange(args.cash_start, args.cash_stop, args.cash_step),
        cost_values=frange(args.cost_start, args.cost_stop, args.cost_step),
        trigger_payoff=args.trigger_payoff,
        direct_payoff=args.direct_payoff,
        mismatch_payoff=args.mismatch_payoff,
    )
    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir)
    print("metric,value")
    for key, value in summary.items():
        print(f"{key},{value:.6f}")
    print(f"\nWrote {args.output_dir / 'performance_closure_summary.json'}")
    print(f"Wrote {args.output_dir / 'performance_closure_grid.csv'}")


if __name__ == "__main__":
    main()
