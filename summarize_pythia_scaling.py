"""Summarize held-out Pythia scaling results without changing verdict rules."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SIZES = ["160m", "410m", "1b", "1.4b", "2.8b"]


def tag(size: str) -> str:
    return "" if size == "160m" else f"_{size}"


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def run_summary(summary: Dict[str, Any] | None, run: str) -> Dict[str, Any]:
    if summary is None:
        return {"status": "missing"}
    item = summary.get("runs", {}).get(run)
    if item is None:
        return {"status": "missing_run"}
    stats = item["stats"]
    verdict = item["verdict"]
    return {
        "status": "ok",
        "emergent": int(verdict["emergent"]),
        "passes": verdict["passes"],
        "window_epoch": stats.get("window_epoch"),
        "usefulness_acc_gain": stats.get("usefulness_acc_gain"),
        "burstiness_ratio": stats.get("burstiness_ratio"),
        "final_acc": stats.get("final_test_acc"),
    }


def integrity_checks(rows: List[Dict[str, Any]], run: str) -> Dict[str, Any]:
    selected = [r for r in rows if r.get("run") == run]
    by_step = {int(float(r["epoch"])): r for r in selected}
    checks: Dict[str, Any] = {"n_steps": len(selected)}
    for a, b in ((0, 1), (0, 1000), (1000, 143000)):
        if a not in by_step or b not in by_step:
            continue
        diffs = {
            key: abs(fnum(by_step[a][key]) - fnum(by_step[b][key]))
            for key in ("test_acc", "test_entropy_bits", "collapse_bits")
        }
        checks[f"diff_step{a}_step{b}"] = diffs
        checks[f"identical_step{a}_step{b}"] = all(v == 0.0 for v in diffs.values())
    return checks


def summarize() -> Dict[str, Any]:
    scales: Dict[str, Any] = {}
    for size in SIZES:
        suffix = tag(size)
        collapse = load_json(OUTPUTS / f"pythia_collapse_summary{suffix}.json")
        tail = load_json(OUTPUTS / f"pythia_tail_summary{suffix}.json")
        collapse_rows = load_rows(OUTPUTS / f"pythia_collapse_timeseries{suffix}.csv")
        tail_rows = load_rows(OUTPUTS / f"pythia_tail_timeseries{suffix}.csv")
        scales[size] = {
            "collapse": {
                "agreement": run_summary(collapse, "pythia_agreement"),
                "random_target": run_summary(collapse, "pythia_random_target"),
                "shuffled_vocab": run_summary(collapse, "shuffled_vocab"),
                "integrity": integrity_checks(collapse_rows, "pythia_agreement"),
                "skipped_steps": [] if collapse is None else collapse.get("skipped_steps", []),
                "download_base": None if collapse is None else collapse.get("download_base"),
            },
            "tail": {
                "head_facts": run_summary(tail, "head_facts"),
                "tail_facts": run_summary(tail, "tail_facts"),
                "tail_words": run_summary(tail, "tail_words"),
                "integrity": integrity_checks(tail_rows, "head_facts"),
                "skipped_steps": [] if tail is None else tail.get("skipped_steps", []),
            },
        }

    valid_sizes = [s for s, v in scales.items()
                   if v["collapse"]["agreement"]["status"] == "ok"]
    agreement_passes = sum(
        scales[s]["collapse"]["agreement"].get("emergent", 0) for s in valid_sizes
    )
    controls_rejected = sum(
        1 - scales[s]["collapse"][run].get("emergent", 0)
        for s in valid_sizes
        for run in ("random_target", "shuffled_vocab")
        if scales[s]["collapse"][run]["status"] == "ok"
    )
    control_total = sum(
        1 for s in valid_sizes
        for run in ("random_target", "shuffled_vocab")
        if scales[s]["collapse"][run]["status"] == "ok"
    )
    tail_sizes = [s for s, v in scales.items()
                  if v["tail"]["head_facts"]["status"] == "ok"]
    head_passes = sum(scales[s]["tail"]["head_facts"].get("emergent", 0)
                      for s in tail_sizes)
    tail_rejections = {
        fam: sum(1 - scales[s]["tail"][fam].get("emergent", 0)
                 for s in tail_sizes
                 if scales[s]["tail"][fam]["status"] == "ok")
        for fam in ("tail_facts", "tail_words")
    }
    return {
        "scales": scales,
        "prediction_counts": {
            "agreement_passes": [agreement_passes, len(valid_sizes)],
            "controls_rejected": [controls_rejected, control_total],
            "head_fact_passes": [head_passes, len(tail_sizes)],
            "tail_rejections": tail_rejections,
        },
    }


def main() -> None:
    summary = summarize()
    out = OUTPUTS / "pythia_scaling_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["prediction_counts"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
