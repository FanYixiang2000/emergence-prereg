"""External decoy/role-aware swarm adapter for possibility-collapse evidence.

The decoy benchmark is a useful external analogue of the local-optimality trap:

- nearest_only follows an immediate local cue and attacks decoys;
- role_oracle / role_mined condition on target role and avoid the trap;
- the result is a large performance gap under the same decoy configurations.

This is summary-level evidence, but it is external to the GOGOGO toy tasks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SWARM = WORKSPACE / "examples_6.29_MARL_SWARM"
OUTPUTS = ROOT / "outputs"


CONTROLLERS = ("nearest_only", "role_oracle", "role_mined")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def controller_records(grid: Mapping[str, Any], controller: str) -> List[Mapping[str, float]]:
    rows: List[Mapping[str, float]] = []
    for record in grid.get("records", []):
        ctrl = record.get("controllers", {}).get(controller, {})
        if not ctrl:
            continue
        rows.append(
            {
                "mean_win": float(ctrl.get("mean_win", 0.0)),
                "min_win": float(ctrl.get("min_win", 0.0)),
                "mean_decoy_damage": float(ctrl.get("mean_decoy_damage", 0.0)),
            }
        )
    return rows


def target_decoy_rate(records_path: Path) -> float:
    if not records_path.exists():
        return 0.0
    data = load_json(records_path)
    records = data.get("records", data if isinstance(data, list) else [])
    if not records:
        return 0.0
    return mean(float(bool(record.get("target_is_decoy", False))) for record in records)


def build_rows() -> List[Dict[str, float | str]]:
    grid_path = SWARM / "outputs" / "decoy_robustness_grid_marl.json"
    if not grid_path.exists():
        grid_path = SWARM / "outputs" / "decoy_robustness_grid.json"
    grid = load_json(grid_path)

    decoy_rates = {
        "nearest_only": target_decoy_rate(SWARM / "outputs" / "mined_from_nearest_records.json"),
        "role_oracle": target_decoy_rate(SWARM / "outputs" / "mined_from_role_oracle_records.json"),
        "role_mined": 0.0,
    }
    nearest = controller_records(grid, "nearest_only")
    nearest_win = mean(row["mean_win"] for row in nearest)
    nearest_decoy_damage = mean(row["mean_decoy_damage"] for row in nearest)
    max_decoy_damage = max(nearest_decoy_damage, 1e-8)

    rows: List[Dict[str, float | str]] = []
    for controller in CONTROLLERS:
        records = controller_records(grid, controller)
        mean_win = mean(row["mean_win"] for row in records)
        min_win = mean(row["min_win"] for row in records)
        mean_decoy_damage = mean(row["mean_decoy_damage"] for row in records)
        decoy_damage_reduction = 1.0 - min(mean_decoy_damage / max_decoy_damage, 1.0)
        win_gain_vs_nearest = mean_win - nearest_win
        local_trap_avoidance = decoy_damage_reduction * max(win_gain_vs_nearest, 0.0)
        external_score = mean_win * decoy_damage_reduction * max(win_gain_vs_nearest, 0.0)
        rows.append(
            {
                "controller": controller,
                "mean_win": mean_win,
                "min_win": min_win,
                "mean_decoy_damage": mean_decoy_damage,
                "target_decoy_rate_proxy": decoy_rates.get(controller, 0.0),
                "decoy_damage_reduction": decoy_damage_reduction,
                "win_gain_vs_nearest": win_gain_vs_nearest,
                "local_trap_avoidance": local_trap_avoidance,
                "external_decoy_ptc_score": external_score,
                "source": str(grid_path),
            }
        )
    return sorted(rows, key=lambda row: float(row["external_decoy_ptc_score"]), reverse=True)


def summarize(rows: Sequence[Mapping[str, float | str]]) -> Dict[str, Any]:
    top = rows[0] if rows else {}
    nearest = next((row for row in rows if row["controller"] == "nearest_only"), None)
    return {
        "evidence_level": "summary_level_external_swarm_decoy",
        "top_controller": top.get("controller"),
        "top_score": float(top.get("external_decoy_ptc_score", 0.0)) if top else 0.0,
        "nearest_score": (
            float(nearest.get("external_decoy_ptc_score", 0.0)) if nearest else 0.0
        ),
        "interpretation": {
            "local_trap": "nearest target selection follows immediate proximity and damages decoys",
            "preserved_structure": "role-aware target selection avoids decoys and preserves future win path",
            "score": "mean_win * decoy_damage_reduction * win_gain_vs_nearest",
            "limitation": "summary-level grid; not full trajectory distribution collapse",
        },
    }


def write_outputs(rows: Sequence[Mapping[str, float | str]], summary: Mapping[str, Any]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "external_decoy_ptc_summary.json").write_text(
        json.dumps({"summary": summary, "rows": list(rows)}, indent=2),
        encoding="utf-8",
    )
    columns = list(rows[0].keys()) if rows else []
    with (OUTPUTS / "external_decoy_ptc_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    rows = build_rows()
    summary = summarize(rows)
    write_outputs(rows, summary)
    print("controller,mean_win,decoy_damage,win_gain,decoy_reduction,score")
    for row in rows:
        print(
            f"{row['controller']},{float(row['mean_win']):.4f},"
            f"{float(row['mean_decoy_damage']):.4f},"
            f"{float(row['win_gain_vs_nearest']):.4f},"
            f"{float(row['decoy_damage_reduction']):.4f},"
            f"{float(row['external_decoy_ptc_score']):.6f}"
        )
    print(f"\nWrote {OUTPUTS / 'external_decoy_ptc_summary.json'}")
    print(f"Wrote {OUTPUTS / 'external_decoy_ptc_scores.csv'}")


if __name__ == "__main__":
    main()
