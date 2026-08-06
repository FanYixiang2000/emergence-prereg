"""Analytic controls for possibility preservation.

This file removes the RL/heuristic confound. All policies are evaluated by
closed-form expected return in the same finite-horizon tree.

Question:
    When can the locally best immediate action be globally suboptimal?

Answer:
    When the value of preserving future context-conditioned options exceeds the
    immediate cash-out value.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


POLICIES = (
    "myopic_greedy",
    "always_trigger",
    "always_direct",
    "random_preserve",
    "possibility_preserving",
)


@dataclass(frozen=True)
class TreeParams:
    p_trigger_needed: float
    cash_out: float
    preserve_cost: float
    trigger_payoff: float
    direct_payoff: float
    mismatch_payoff: float = 0.0


def expected_returns(params: TreeParams) -> Dict[str, float]:
    p = params.p_trigger_needed
    q = 1.0 - p
    cost = params.preserve_cost

    myopic = params.cash_out
    always_trigger = -cost + p * params.trigger_payoff + q * params.mismatch_payoff
    always_direct = -cost + p * params.mismatch_payoff + q * params.direct_payoff
    random_preserve = 0.5 * always_trigger + 0.5 * always_direct
    possibility_preserving = -cost + p * params.trigger_payoff + q * params.direct_payoff

    return {
        "myopic_greedy": myopic,
        "always_trigger": always_trigger,
        "always_direct": always_direct,
        "random_preserve": random_preserve,
        "possibility_preserving": possibility_preserving,
    }


def local_optimality_gap(params: TreeParams) -> float:
    """Immediate advantage of cashing out over preserving options."""

    return params.cash_out + params.preserve_cost


def option_value(params: TreeParams) -> float:
    returns = expected_returns(params)
    return returns["possibility_preserving"] - returns["myopic_greedy"]


def threshold_cash_out(params: TreeParams) -> float:
    """Largest cash-out value for which preserving options is still better."""

    p = params.p_trigger_needed
    q = 1.0 - p
    return -params.preserve_cost + p * params.trigger_payoff + q * params.direct_payoff


def winner(returns: Mapping[str, float]) -> str:
    return max(returns, key=lambda key: returns[key])


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
                params = TreeParams(
                    p_trigger_needed=p,
                    cash_out=cash_out,
                    preserve_cost=preserve_cost,
                    trigger_payoff=trigger_payoff,
                    direct_payoff=direct_payoff,
                    mismatch_payoff=mismatch_payoff,
                )
                returns = expected_returns(params)
                row: Dict[str, float | str] = {
                    **asdict(params),
                    **{f"return_{policy}": value for policy, value in returns.items()},
                    "winner": winner(returns),
                    "option_value": option_value(params),
                    "local_optimality_gap": local_optimality_gap(params),
                    "threshold_cash_out": threshold_cash_out(params),
                    "local_optimum_trap": float(
                        local_optimality_gap(params) > 0.0 and option_value(params) > 0.0
                    ),
                }
                rows.append(row)
    return rows


def summarize(rows: Sequence[Mapping[str, float | str]]) -> Dict[str, float]:
    total = len(rows)
    preserve_wins = sum(1 for row in rows if row["winner"] == "possibility_preserving")
    local_traps = sum(1 for row in rows if float(row["local_optimum_trap"]) > 0.0)
    positive_option_values = [
        float(row["option_value"]) for row in rows if float(row["option_value"]) > 0.0
    ]
    return {
        "n_conditions": float(total),
        "possibility_preserving_win_rate": preserve_wins / total if total else 0.0,
        "local_optimum_trap_rate": local_traps / total if total else 0.0,
        "mean_positive_option_value": (
            sum(positive_option_values) / len(positive_option_values)
            if positive_option_values
            else 0.0
        ),
        "max_option_value": max((float(row["option_value"]) for row in rows), default=0.0),
    }


def write_outputs(
    rows: Sequence[Mapping[str, float | str]],
    summary: Mapping[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "possibility_ablation_summary.json").write_text(
        json.dumps({"summary": summary}, indent=2),
        encoding="utf-8",
    )
    if rows:
        columns = list(rows[0].keys())
    else:
        columns = []
    with (output_dir / "possibility_ablation_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analytic possibility ablation.")
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
    print(f"\nWrote {args.output_dir / 'possibility_ablation_summary.json'}")
    print(f"Wrote {args.output_dir / 'possibility_ablation_grid.csv'}")


if __name__ == "__main__":
    main()
