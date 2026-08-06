"""Different-lineage engine referee for the uncurated discovery runs.

The SF11-classical check removed the NNUE evaluator but kept the Stockfish
search framework. This script re-labels the stored discovery positions with
Toga II 3.0 (Fruit 2.1 lineage): an independently developed engine family
with its own search implementation and handcrafted evaluation, sharing no
code with Stockfish. Predictor scores are stored and untouched; this is
pure re-labelling under the frozen referee rule (depth 18, top-2 eval gap
>= 150 cp, best-line floor >= -50 cp).

Registered prediction (declared here, before running):
    TG-1  the stored collapse do-gap ranks Toga-referee value-critical
          positions at AUROC > 0.60 in BOTH months (the same transfer rule
          the SF11-classical check used).

Disclosed amendment after the first month completed: Toga's evaluation
scale is compressed relative to Stockfish, so the frozen absolute 150 cp
gap rule labels almost nothing (month-1 base rate 0.005) and TG-1 fails
through referee-scale mismatch. The failure is retained. This script
additionally stores per-position Toga gaps and reports a
quantile-matched referee (top-K Toga gaps, K = the NNUE referee's
positive count for that month), which tests transfer of the RANKING
rather than the absolute threshold; it is labelled a disclosed follow-up,
not a registered prediction.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Dict, Optional

import chess
import chess.engine
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
TOGA = (HERE / "external_chess" / "toga2_x" / "usr" / "games" / "toga2")

REFEREE_DEPTH = 18
REFEREE_GAP_CP = 150
REFEREE_FLOOR_CP = -50
MATE_CLIP = 1000

_ENGINE: Optional[chess.engine.SimpleEngine] = None


def get_engine() -> chess.engine.SimpleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = chess.engine.SimpleEngine.popen_uci(str(TOGA))
        _ENGINE.configure({"Hash": 128, "OwnBook": False})
    return _ENGINE


def worker_init() -> None:
    global _ENGINE
    _ENGINE = None
    get_engine()


def relabel(task: Dict) -> Optional[Dict]:
    engine = get_engine()
    board = chess.Board(task["fen"])
    side = board.turn
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=REFEREE_DEPTH),
                               multipv=4)
    except chess.engine.EngineError:
        return None
    if isinstance(infos, dict):
        infos = [infos]
    cps = [float(np.clip(i["score"].pov(side).score(mate_score=10000),
                         -MATE_CLIP, MATE_CLIP))
           for i in infos if i.get("pv")]
    if len(cps) < 2:
        return None
    gap = cps[0] - cps[1]
    return {
        "eligible_index": task["eligible_index"],
        "toga_gap_cp": gap,
        "toga_best_cp": cps[0],
        "toga_label": int(gap >= REFEREE_GAP_CP and cps[0] >= REFEREE_FLOOR_CP),
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum(float(np.sum(p > neg) + 0.5 * np.sum(p == neg)) for p in pos)
    return wins / (len(pos) * len(neg))


def run_month(tag: str, workers: int) -> Dict:
    data = json.loads((OUTPUTS / f"chess_discovery_{tag}.json").read_text())
    recs = data["records"]
    with Pool(workers, initializer=worker_init) as pool:
        labels = [r for r in pool.imap_unordered(relabel, recs)
                  if r is not None]
    by_idx = {l["eligible_index"]: l for l in labels}
    merged = [(r, by_idx[r["eligible_index"]]) for r in recs
              if r["eligible_index"] in by_idx]
    y = np.array([m["toga_label"] for _, m in merged])
    toga_gap = np.array([m["toga_gap_cp"] for _, m in merged])
    do_gap = np.array([r["do_gap"] for r, _ in merged])
    shallow = np.array([r["shallow_gap_cp"] for r, _ in merged])
    flags = np.array([r["flag"] for r, _ in merged])
    nnue = np.array([r["label"] for r, _ in merged])
    base = float(y.mean())
    prec = float(y[flags == 1].mean()) if flags.sum() else float("nan")

    # Disclosed follow-up: quantile-matched referee. K = the NNUE
    # referee's positive count, so both referees label the same fraction;
    # this tests ranking transfer without the absolute cp scale.
    k = int(nnue.sum())
    order = np.argsort(-toga_gap)
    y_q = np.zeros_like(y)
    y_q[order[:k]] = 1
    prec_q = float(y_q[flags == 1].mean()) if flags.sum() else float("nan")
    base_q = float(y_q.mean())
    return {
        "n": len(merged),
        "frozen_rule": {
            "toga_base_rate": base,
            "auroc_do_gap_vs_toga": auroc(do_gap, y),
            "auroc_shallow_gap_vs_toga": auroc(shallow, y),
            "flag_precision_vs_toga": prec,
            "flag_lift_vs_toga": prec / base if base > 0 else float("nan"),
            "label_agreement_nnue_vs_toga": float(np.mean(nnue == y)),
        },
        "quantile_matched": {
            "k_positives": k,
            "auroc_do_gap_vs_toga_topk": auroc(do_gap, y_q),
            "auroc_shallow_gap_vs_toga_topk": auroc(shallow, y_q),
            "flag_precision_vs_toga_topk": prec_q,
            "flag_lift_vs_toga_topk": (prec_q / base_q if base_q > 0
                                       else float("nan")),
            "label_agreement_nnue_vs_toga_topk": float(np.mean(nnue == y_q)),
        },
        "records": [
            {"eligible_index": r["eligible_index"],
             "toga_gap_cp": m["toga_gap_cp"],
             "toga_best_cp": m["toga_best_cp"],
             "toga_label_frozen": int(m["toga_label"]),
             "toga_label_topk": int(y_q[i])}
            for i, (r, m) in enumerate(merged)
        ],
    }


def main() -> None:
    set_start_method("spawn", force=True)
    out = {
        "status": ("different-lineage engine referee (Toga II 3.0 / Fruit "
                   "family; predictor scores untouched)"),
        "referee": {"engine": "Toga II 3.0 (Fruit 2.1 lineage)",
                    "depth": REFEREE_DEPTH, "gap_cp": REFEREE_GAP_CP,
                    "floor_cp": REFEREE_FLOOR_CP},
        "months": {},
    }
    for tag in ("main", "replication_2016_03"):
        out["months"][tag] = run_month(tag, workers=24)
        print(tag, json.dumps(
            {k: v for k, v in out["months"][tag].items() if k != "records"},
            indent=2), flush=True)
    both = [out["months"][t]["frozen_rule"]["auroc_do_gap_vs_toga"]
            for t in out["months"]]
    both_q = [out["months"][t]["quantile_matched"]
              ["auroc_do_gap_vs_toga_topk"] for t in out["months"]]
    out["registered_outcomes"] = {
        "TG1_auroc_gt_0.60_both_months": bool(all(a > 0.60 for a in both)),
        "TG1_note": ("frozen 150 cp rule fails through referee-scale "
                     "mismatch (Toga cp scale compressed; base rate ~0.005)"
                     " -- retained as a registered miss"),
        "followup_quantile_matched_auroc_gt_0.60_both_months": bool(
            all(a > 0.60 for a in both_q)),
    }
    path = OUTPUTS / "chess_discovery_toga_referee.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["registered_outcomes"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
