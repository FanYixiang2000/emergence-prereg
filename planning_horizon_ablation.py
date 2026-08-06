"""Exact planning-horizon ablation for possibility preservation.

This experiment uses the same Bellman solver for every condition. The only
variable is planning horizon. A horizon-1 solver sees the immediate cash-out
advantage; a horizon-2 solver can value the context-conditioned option retained
by preserving possibility.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


START = "start"
CONTEXT = "context"
TERMINAL = "terminal"
START_ACTIONS = ("cash_out", "preserve_option")
CONTEXT_ACTIONS = ("trigger", "direct")


@dataclass(frozen=True)
class HorizonParams:
    p_trigger_needed: float
    cash_out: float
    preserve_cost: float
    trigger_payoff: float
    direct_payoff: float
    mismatch_payoff: float = 0.0


def immediate_reward(params: HorizonParams, state: str, action: str) -> float:
    if state == START and action == "cash_out":
        return params.cash_out
    if state == START and action == "preserve_option":
        return -params.preserve_cost
    if state == CONTEXT:
        # Before the context is observed, action value is an expectation over
        # whether trigger or direct is needed.
        p = params.p_trigger_needed
        if action == "trigger":
            return p * params.trigger_payoff + (1.0 - p) * params.mismatch_payoff
        if action == "direct":
            return p * params.mismatch_payoff + (1.0 - p) * params.direct_payoff
    return 0.0


def next_state(state: str, action: str) -> str:
    if state == START and action == "preserve_option":
        return CONTEXT
    return TERMINAL


def actions(state: str) -> Tuple[str, ...]:
    if state == START:
        return START_ACTIONS
    if state == CONTEXT:
        return CONTEXT_ACTIONS
    return ()


def bellman_value(params: HorizonParams, state: str, horizon: int) -> float:
    if horizon <= 0 or state == TERMINAL:
        return 0.0
    return max(
        immediate_reward(params, state, action)
        + bellman_value(params, next_state(state, action), horizon - 1)
        for action in actions(state)
    )


def q_value(params: HorizonParams, state: str, action: str, horizon: int) -> float:
    if horizon <= 0:
        return 0.0
    return immediate_reward(params, state, action) + bellman_value(
        params, next_state(state, action), horizon - 1
    )


def optimal_start_action(params: HorizonParams, horizon: int) -> str:
    qs = {action: q_value(params, START, action, horizon) for action in START_ACTIONS}
    return max(qs, key=lambda action: qs[action])


def start_qs(params: HorizonParams, horizon: int) -> Dict[str, float]:
    return {action: q_value(params, START, action, horizon) for action in START_ACTIONS}


def context_value(params: HorizonParams) -> float:
    return bellman_value(params, CONTEXT, horizon=1)


def exact_option_value(params: HorizonParams) -> float:
    return q_value(params, START, "preserve_option", horizon=2) - q_value(
        params, START, "cash_out", horizon=2
    )


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
                params = HorizonParams(
                    p_trigger_needed=p,
                    cash_out=cash_out,
                    preserve_cost=preserve_cost,
                    trigger_payoff=trigger_payoff,
                    direct_payoff=direct_payoff,
                    mismatch_payoff=mismatch_payoff,
                )
                q_h1 = start_qs(params, horizon=1)
                q_h2 = start_qs(params, horizon=2)
                action_h1 = optimal_start_action(params, horizon=1)
                action_h2 = optimal_start_action(params, horizon=2)
                row: Dict[str, float | str] = {
                    **asdict(params),
                    "h1_action": action_h1,
                    "h2_action": action_h2,
                    "h1_q_cash": q_h1["cash_out"],
                    "h1_q_preserve": q_h1["preserve_option"],
                    "h2_q_cash": q_h2["cash_out"],
                    "h2_q_preserve": q_h2["preserve_option"],
                    "context_value": context_value(params),
                    "option_value": exact_option_value(params),
                    "horizon_reversal": float(
                        action_h1 == "cash_out" and action_h2 == "preserve_option"
                    ),
                }
                rows.append(row)
    return rows


def summarize(rows: Sequence[Mapping[str, float | str]]) -> Dict[str, float]:
    total = len(rows)
    reversals = sum(1 for row in rows if float(row["horizon_reversal"]) > 0)
    preserve_h2 = sum(1 for row in rows if row["h2_action"] == "preserve_option")
    positive_option_values = [
        float(row["option_value"]) for row in rows if float(row["option_value"]) > 0
    ]
    return {
        "n_conditions": float(total),
        "horizon_reversal_rate": reversals / total if total else 0.0,
        "h2_preserve_rate": preserve_h2 / total if total else 0.0,
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
    (output_dir / "planning_horizon_summary.json").write_text(
        json.dumps({"summary": summary}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "planning_horizon_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        columns = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact Bellman horizon ablation.")
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
    print(f"\nWrote {args.output_dir / 'planning_horizon_summary.json'}")
    print(f"Wrote {args.output_dir / 'planning_horizon_grid.csv'}")


if __name__ == "__main__":
    main()
