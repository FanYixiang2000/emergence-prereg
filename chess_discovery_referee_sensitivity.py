"""Referee-threshold sensitivity for the uncurated discovery run.

Pure re-analysis of stored per-position records: the referee label used a
frozen 150 cp deep-eval gap; here the label is recomputed at 100/125/150/
200/250 cp (floor unchanged) and AUROC/precision/lift are re-reported, plus a
decile calibration table for the do-gap score. No score is recomputed; the
predictor never sees the referee.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FLOOR_CP = -50


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum(float(np.sum(p > neg) + 0.5 * np.sum(p == neg)) for p in pos)
    return wins / (len(pos) * len(neg))


def main() -> None:
    data = json.loads((OUTPUTS / "chess_discovery_main.json").read_text())
    recs = data["records"]
    gap = np.array([r["referee_gap_cp"] for r in recs])
    best = np.array([r["referee_best_cp"] for r in recs])
    do_gap = np.array([r["do_gap"] for r in recs])
    shallow = np.array([r["shallow_gap_cp"] for r in recs])
    flags = np.array([r["flag"] for r in recs])

    sweep = {}
    for thr in (100, 125, 150, 200, 250):
        labels = ((gap >= thr) & (best >= FLOOR_CP)).astype(int)
        base = float(labels.mean())
        prec = float(labels[flags == 1].mean()) if flags.sum() else float("nan")
        sweep[str(thr)] = {
            "base_rate": base,
            "auroc_do_gap": auroc(do_gap, labels),
            "auroc_shallow_gap": auroc(shallow, labels),
            "flag_precision": prec,
            "flag_lift": prec / base if base > 0 else float("nan"),
        }

    order = np.argsort(-do_gap)
    deciles = []
    labels150 = ((gap >= 150) & (best >= FLOOR_CP)).astype(int)
    for d in range(10):
        idx = order[d * 40:(d + 1) * 40]
        deciles.append({
            "decile": d + 1,
            "mean_do_gap": float(do_gap[idx].mean()),
            "hit_rate_at_150cp": float(labels150[idx].mean()),
        })

    out = {
        "status": "post-hoc referee-threshold sensitivity (labels only; "
                  "scores and flags unchanged)",
        "frozen_threshold_cp": 150,
        "sweep": sweep,
        "do_gap_decile_calibration": deciles,
        "auroc_above_0.70_at_all_thresholds": all(
            v["auroc_do_gap"] >= 0.70 for v in sweep.values()),
        "lift_above_2x_at_all_thresholds": all(
            v["flag_lift"] >= 2.0 for v in sweep.values()),
    }
    path = OUTPUTS / "chess_discovery_referee_sensitivity.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for thr, v in sweep.items():
        print(f"{thr}cp: base {v['base_rate']:.3f}  AUROC {v['auroc_do_gap']:.3f}  "
              f"lift {v['flag_lift']:.2f}x")
    print("top-decile hit rate:", deciles[0]["hit_rate_at_150cp"])
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
