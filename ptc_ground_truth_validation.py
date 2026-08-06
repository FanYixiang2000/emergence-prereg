"""Validate PTC evidence against analytic ground truth.

This experiment asks whether the proposed evidence separates four cases:

1. useful latent possibility: local optimum trap + positive option value;
2. false multimodality: structured futures but no utility gain;
3. useful but nearly single-mode futures;
4. myopic/no-option cases.

The point is to avoid claiming that "multimodal" alone means emergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from possibility_ablation import (
    TreeParams,
    expected_returns,
    frange,
    local_optimality_gap,
    option_value,
)
from ptc_metrics import effective_modes, entropy


def structured_future_score(p_trigger_needed: float) -> float:
    p = p_trigger_needed
    distribution = {
        "trigger_basin": p,
        "direct_basin": 1.0 - p,
    }
    choice_tension = 4.0 * p * (1.0 - p)
    return effective_modes(distribution) * choice_tension


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        idx = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 1)
    fp = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 1)
    tn = sum(1 for y, yhat in zip(labels, predictions) if y == 0 and yhat == 0)
    fn = sum(1 for y, yhat in zip(labels, predictions) if y == 1 and yhat == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def classify_case(
    is_trap: bool,
    option: float,
    structure: float,
    structure_threshold: float,
) -> str:
    has_structure = structure >= structure_threshold
    has_utility = option > 0.0
    if is_trap and has_structure:
        return "useful_latent_possibility"
    if has_structure and not has_utility:
        return "false_multimodality"
    if has_utility and not has_structure:
        return "useful_single_mode"
    return "no_useful_option"


def build_rows(
    p_values: Sequence[float],
    cash_values: Sequence[float],
    cost_values: Sequence[float],
    trigger_payoff: float,
    direct_payoff: float,
    mismatch_payoff: float,
    structure_threshold: float,
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
                opt = option_value(params)
                local_gap = local_optimality_gap(params)
                structure = structured_future_score(p)
                is_trap = local_gap > 0.0 and opt > 0.0
                # PTC should not be treated as utility by itself. The combined
                # signature asks for structure, utility, and local-trap status.
                combined_score = structure * max(opt, 0.0) * max(local_gap, 0.0)
                rows.append(
                    {
                        "p_trigger_needed": p,
                        "cash_out": cash_out,
                        "preserve_cost": preserve_cost,
                        "return_myopic": returns["myopic_greedy"],
                        "return_possibility": returns["possibility_preserving"],
                        "option_value": opt,
                        "local_optimality_gap": local_gap,
                        "structured_future_score": structure,
                        "combined_evidence_score": combined_score,
                        "ground_truth_useful_trap": float(is_trap),
                        "case_type": classify_case(
                            is_trap=is_trap,
                            option=opt,
                            structure=structure,
                            structure_threshold=structure_threshold,
                        ),
                    }
                )
    return rows


def summarize(rows: Sequence[Mapping[str, float | str]], structure_threshold: float) -> Dict[str, float]:
    labels = [int(float(row["ground_truth_useful_trap"])) for row in rows]
    structure_scores = [float(row["structured_future_score"]) for row in rows]
    option_scores = [float(row["option_value"]) for row in rows]
    combined_scores = [float(row["combined_evidence_score"]) for row in rows]
    predictions = [1 if score > 0.0 else 0 for score in combined_scores]
    case_counts: Dict[str, int] = {}
    for row in rows:
        case = str(row["case_type"])
        case_counts[case] = case_counts.get(case, 0) + 1

    metrics = binary_metrics(labels, predictions)
    summary = {
        "n_conditions": float(len(rows)),
        "positive_ground_truth_rate": sum(labels) / len(labels) if labels else 0.0,
        "structure_threshold": structure_threshold,
        "auc_structure_only": auc_score(labels, structure_scores),
        "auc_option_value": auc_score(labels, option_scores),
        "auc_combined_evidence": auc_score(labels, combined_scores),
        **{f"combined_{key}": value for key, value in metrics.items()},
    }
    for case, count in case_counts.items():
        summary[f"case_{case}_rate"] = count / len(rows) if rows else 0.0
    return summary


def write_outputs(
    rows: Sequence[Mapping[str, float | str]],
    summary: Mapping[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ptc_ground_truth_validation_summary.json").write_text(
        json.dumps({"summary": summary}, indent=2),
        encoding="utf-8",
    )
    columns = list(rows[0].keys()) if rows else []
    with (output_dir / "ptc_ground_truth_validation_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PTC ground-truth validation.")
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
    parser.add_argument("--structure_threshold", type=float, default=1.5)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = build_rows(
        p_values=frange(args.p_start, args.p_stop, args.p_step),
        cash_values=frange(args.cash_start, args.cash_stop, args.cash_step),
        cost_values=frange(args.cost_start, args.cost_stop, args.cost_step),
        trigger_payoff=args.trigger_payoff,
        direct_payoff=args.direct_payoff,
        mismatch_payoff=args.mismatch_payoff,
        structure_threshold=args.structure_threshold,
    )
    summary = summarize(rows, structure_threshold=args.structure_threshold)
    write_outputs(rows, summary, args.output_dir)
    print("metric,value")
    for key, value in summary.items():
        print(f"{key},{value:.6f}")
    print(f"\nWrote {args.output_dir / 'ptc_ground_truth_validation_summary.json'}")
    print(f"Wrote {args.output_dir / 'ptc_ground_truth_validation_grid.csv'}")


if __name__ == "__main__":
    main()
