"""Summary-level PTC adapter for existing sacrifice MARL experiments.

This script intentionally does not fabricate trajectories. It consumes the
existing sacrifice experiment summaries and maps them into the PTC evidence
language:

- structured conditionality: sacrifice probability changes with benefit/cost;
- useful triggering: sacrifice remains high when benefit/cost is high;
- non-blindness: sacrifice stays low when benefit/cost is low;
- factual performance: final team completion rate from trained policies.

This is summary-level external validation. It is weaker than trajectory-level
PTC, but stronger than another toy benchmark because it uses completed MARL
experiments outside GOGOGO.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SACRIFICE_DIR = WORKSPACE / "examples_6.23_sacrifice"
OUTPUTS = ROOT / "outputs"


METHODS = (
    "individual",
    "team",
    "fixed_altruism",
    "fixed_high",
    "rusp",
    "rusp_wide",
    "spc",
    "spc_anneal",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def load_final_metrics(result_dirs: Sequence[Path]) -> Dict[str, Dict[str, float]]:
    by_method: Dict[str, Dict[str, List[float]]] = {}
    for result_dir in result_dirs:
        if not result_dir.exists():
            continue
        for path in sorted(result_dir.glob("*.json")):
            try:
                data = load_json(path)
            except json.JSONDecodeError:
                continue
            method = data.get("method")
            if method is None:
                continue
            final = data.get("final_metrics")
            if final is None and data.get("eval_metrics"):
                final = data["eval_metrics"][-1]
            if not isinstance(final, dict):
                continue
            target = by_method.setdefault(method, {})
            for metric in ("ESR", "BSR", "FRR", "STS", "team_completion_rate"):
                target.setdefault(metric, []).append(float(final.get(metric, 0.0)))

    return {
        method: {metric: mean(values) for metric, values in metrics.items()}
        for method, metrics in by_method.items()
    }


def build_external_rows() -> List[Dict[str, float | str]]:
    mechanism_path = SACRIFICE_DIR / "results" / "mechanism_probe.json"
    robustness_path = SACRIFICE_DIR / "results" / "metric_robustness.json"
    permutation_path = SACRIFICE_DIR / "results" / "permutation_test.json"

    mechanism = load_json(mechanism_path)
    robustness = load_json(robustness_path) if robustness_path.exists() else {}
    permutation = load_json(permutation_path) if permutation_path.exists() else {}
    final_metrics = load_final_metrics(
        [
            SACRIFICE_DIR / "results",
            SACRIFICE_DIR / "results_spc",
            SACRIFICE_DIR / "results_spc2",
            SACRIFICE_DIR / "threshold_results",
            SACRIFICE_DIR / "threshold_results_spc",
            SACRIFICE_DIR / "threshold_results_spc2",
            SACRIFICE_DIR / "cleanup_results",
            SACRIFICE_DIR / "cleanup_results_spc",
            SACRIFICE_DIR / "cleanup_results_a05",
        ]
    )

    permutation_methods = permutation.get("methods", {})
    robustness_stability = robustness.get("regime_stability", {})

    rows: List[Dict[str, float | str]] = []
    for method in METHODS:
        stats = mechanism.get("methods", {}).get(method)
        if not stats:
            continue
        fm = final_metrics.get(method, {})
        completion = float(fm.get("team_completion_rate", 0.0))
        useful = float(stats.get("useful_sacrifice_index", 0.0))
        blind = float(stats.get("blind_sacrifice_index", 0.0))
        conditionality = max(float(stats.get("conditionality_score", 0.0)), 0.0)
        spearman = max(float(stats.get("spearman_ratio_to_sacrifice", 0.0)), 0.0)
        selectivity = float(stats.get("selectivity_high_over_low", 0.0))
        external_score = conditionality * useful * (1.0 - blind) * completion
        utility_adjusted_score = external_score * max(float(fm.get("ESR", 0.0)), 0.0)
        perm = permutation_methods.get(method, {})
        observed = perm.get("observed", {}) if isinstance(perm, dict) else {}
        p_value_gap = float(perm.get("p_value_raw_gap", 1.0)) if isinstance(perm, dict) else 1.0
        p_value_rho = float(perm.get("p_value_spearman", 1.0)) if isinstance(perm, dict) else 1.0
        stable_conditional = float(
            robustness_stability.get(method, {}).get("conditional_sacrifice", 0.0)
        )
        rows.append(
            {
                "method": method,
                "mechanism_regime": str(stats.get("regime", "unknown")),
                "low_bc_sacrifice": blind,
                "high_bc_sacrifice": useful,
                "conditionality_score": conditionality,
                "spearman_ratio_to_sacrifice": spearman,
                "selectivity_high_over_low": selectivity,
                "team_completion_rate": completion,
                "ESR": float(fm.get("ESR", 0.0)),
                "BSR": float(fm.get("BSR", 0.0)),
                "FRR": float(fm.get("FRR", 0.0)),
                "external_ptc_score": external_score,
                "utility_adjusted_ptc_score": utility_adjusted_score,
                "permutation_raw_gap": float(observed.get("raw_gap", 0.0)),
                "permutation_spearman": float(observed.get("spearman", 0.0)),
                "p_value_raw_gap": p_value_gap,
                "p_value_spearman": p_value_rho,
                "robust_conditional_regime_fraction": stable_conditional,
            }
        )
    return sorted(rows, key=lambda row: float(row["external_ptc_score"]), reverse=True)


def summarize(rows: Sequence[Mapping[str, float | str]]) -> Dict[str, Any]:
    ranked = [str(row["method"]) for row in rows]
    team_row = next((row for row in rows if row["method"] == "team"), None)
    top_row = rows[0] if rows else None
    conditional_rows = [
        row for row in rows if row["mechanism_regime"] == "conditional_sacrifice"
    ]
    significant_rows = [
        row
        for row in rows
        if float(row["p_value_raw_gap"]) < 0.01 and float(row["p_value_spearman"]) < 0.01
    ]
    return {
        "source": str(SACRIFICE_DIR),
        "evidence_level": "summary_level_external_marl",
        "ranked_by_external_ptc_score": ranked,
        "top_method": str(top_row["method"]) if top_row else None,
        "top_score": float(top_row["external_ptc_score"]) if top_row else 0.0,
        "team_external_ptc_score": (
            float(team_row["external_ptc_score"]) if team_row else 0.0
        ),
        "n_conditional_regime_methods": len(conditional_rows),
        "n_permutation_significant_methods": len(significant_rows),
        "interpretation": {
            "external_ptc_score": (
                "conditionality * high-benefit sacrifice * (1 - blind sacrifice) * completion"
            ),
            "why_team_is_penalized": "high blind-sacrifice index despite high raw sacrifice",
            "limitation": "summary-level evidence; no raw trajectory-level future distributions",
        },
    }


def write_outputs(rows: Sequence[Mapping[str, float | str]], summary: Mapping[str, Any]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "external_sacrifice_ptc_summary.json").write_text(
        json.dumps({"summary": summary, "rows": list(rows)}, indent=2),
        encoding="utf-8",
    )
    columns = list(rows[0].keys()) if rows else []
    with (OUTPUTS / "external_sacrifice_ptc_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    rows = build_external_rows()
    summary = summarize(rows)
    write_outputs(rows, summary)
    print("method,regime,completion,blind,high,conditionality,external_ptc_score")
    for row in rows:
        print(
            f"{row['method']},{row['mechanism_regime']},"
            f"{float(row['team_completion_rate']):.4f},"
            f"{float(row['low_bc_sacrifice']):.4f},"
            f"{float(row['high_bc_sacrifice']):.4f},"
            f"{float(row['conditionality_score']):.4f},"
            f"{float(row['external_ptc_score']):.6f}"
        )
    print(f"\nWrote {OUTPUTS / 'external_sacrifice_ptc_summary.json'}")
    print(f"Wrote {OUTPUTS / 'external_sacrifice_ptc_scores.csv'}")


if __name__ == "__main__":
    main()
