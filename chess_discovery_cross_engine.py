"""Cross-engine-family referee for the uncurated discovery runs.

Closes the same-family circularity objection: the discovery predictor and
the frozen referee both used Stockfish 14.1 NNUE at different depths. Here
the stored positions (scores untouched) are re-labelled by Stockfish 11
CLASSICAL evaluation (handcrafted evaluation function, pre-NNUE -- a
different evaluation family) at the same referee depth and gap rule, and
the stored do-gap scores are evaluated against the new labels.

Pure re-labelling: the predictor never sees any referee. No stored output
is modified.
"""

from __future__ import annotations

import json
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.engine
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SF11 = HERE / "external_chess" / "stockfish_11_x" / "usr" / "games" / "stockfish"

REFEREE_DEPTH = 18
REFEREE_GAP_CP = 150
REFEREE_FLOOR_CP = -50
MATE_CLIP = 1000

_ENGINE: Optional[chess.engine.SimpleEngine] = None


def get_engine() -> chess.engine.SimpleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = chess.engine.SimpleEngine.popen_uci(str(SF11))
        _ENGINE.configure({"Threads": 1, "Hash": 64})
    return _ENGINE


def worker_init() -> None:
    global _ENGINE
    _ENGINE = None
    get_engine()


def relabel(task: Dict) -> Optional[Dict]:
    engine = get_engine()
    board = chess.Board(task["fen"])
    side = board.turn
    infos = engine.analyse(board, chess.engine.Limit(depth=REFEREE_DEPTH),
                           multipv=4)
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
        "sf11_gap_cp": gap,
        "sf11_best_cp": cps[0],
        "sf11_label": int(gap >= REFEREE_GAP_CP and cps[0] >= REFEREE_FLOOR_CP),
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
    y = np.array([m["sf11_label"] for _, m in merged])
    do_gap = np.array([r["do_gap"] for r, _ in merged])
    shallow = np.array([r["shallow_gap_cp"] for r, _ in merged])
    flags = np.array([r["flag"] for r, _ in merged])
    nnue = np.array([r["label"] for r, _ in merged])
    base = float(y.mean())
    prec = float(y[flags == 1].mean()) if flags.sum() else float("nan")
    return {
        "n": len(merged),
        "sf11_base_rate": base,
        "auroc_do_gap_vs_sf11": auroc(do_gap, y),
        "auroc_shallow_gap_vs_sf11": auroc(shallow, y),
        "flag_precision_vs_sf11": prec,
        "flag_lift_vs_sf11": prec / base if base > 0 else float("nan"),
        "label_agreement_nnue_vs_sf11": float(np.mean(nnue == y)),
    }


def main() -> None:
    set_start_method("spawn", force=True)
    out = {
        "status": "cross-engine-family referee (SF11 classical eval; "
                  "predictor scores untouched)",
        "referee": {"engine": "Stockfish 11 classical", "depth": REFEREE_DEPTH,
                    "gap_cp": REFEREE_GAP_CP, "floor_cp": REFEREE_FLOOR_CP},
        "months": {},
    }
    for tag in ("main", "replication_2016_03"):
        out["months"][tag] = run_month(tag, workers=24)
        print(tag, json.dumps(out["months"][tag], indent=2), flush=True)
    both = [out["months"][t]["auroc_do_gap_vs_sf11"] for t in out["months"]]
    out["do_gap_transfers_across_engine_families"] = all(
        a > 0.60 for a in both)
    path = OUTPUTS / "chess_discovery_cross_engine.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
