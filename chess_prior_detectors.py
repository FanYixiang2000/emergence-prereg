"""Prior single-signal emergence detectors on the chess data (re-analysis).

Counterpart of prior_metrics_comparison.py, run on the external chess
positions instead of the internal battery. Exploratory question: given the measured
(position, move) pairs from the registered main run, can a single signal
identify the externally annotated key move among the counterfactual
moves, the way the useful-collapse composite does?

Task: for each sacrifice position, rank the four measured moves (key,
deep_alt, greedy, random) by a detector score; the detector succeeds on
a position if it ranks the annotated key move first. Detectors:

- collapse_only:    entropy drop of the future-basin distribution
                    (possibility collapse without the usefulness sign --
                    the "any collapse is emergence" reading)
- specificity_only: JS divergence of the move's future distribution
                    from the position's base distribution (how much the
                    move REDIRECTS the future, regardless of direction)
- local_value:      immediate material gain against best reply (the
                    greedy-account detector; also the reverse of the
                    "sacrifice = loss" signal)
- useful_shift:     P(win | do move) minus the same behaving-policy
                    baseline for every move in a position

Important limitation: subtracting a common within-position baseline does not
change move ranking. This diagnostic therefore does not test the manuscript's
martingale lesson or establish superiority of a composite criterion; it is
retained only as an exploratory ranking audit and excluded from main evidence.

Also reported: quiet-position separation. A detector that fires on quiet
best moves as strongly as on sacrifice key moves cannot distinguish
"a move exists" from "an emergence-grade trigger exists".

This is a re-analysis of stored measurements; no new engine calls, no
threshold fitting. JS-vs-base per move is recomputed from the stored
per-move distributions in the positions CSV where available; where only
summary columns exist, the detector uses the stored columns directly.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

MOVES = ("key", "deep_alt", "greedy", "random")


def load(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def rank_first(scores: Dict[str, float], target: str) -> bool:
    return all(scores[target] >= v for k, v in scores.items() if k != target)


def main() -> None:
    rows = load(OUTPUTS / "chess_collapse_main_positions.csv")
    sac = [r for r in rows if r["kind"] == "sacrifice"]
    qui = [r for r in rows if r["kind"] == "quiet"]

    # specificity_only proxy: unsigned displacement of the win-basin mass
    # (the stored per-move basin distributions were not kept in the CSV;
    # win-mass displacement is the dominant JS component on these
    # positions). It measures how much the move REDIRECTS the future,
    # without the usefulness sign.
    detectors = {
        "collapse_only": lambda r, m: float(r[f"collapse_{m}_bits"]),
        "specificity_only": lambda r, m: abs(float(r[f"useful_shift_{m}"])),
        "local_value": lambda r, m: float(r[f"local_cost_{m}"]),
        # Legacy output key retained for figure/bootstrap compatibility.
        "useful_collapse": lambda r, m: float(r[f"useful_shift_{m}"]),
    }

    results: Dict[str, Dict] = {}
    for name, fn in detectors.items():
        hits = 0
        usable = 0
        for r in sac:
            try:
                scores = {m: fn(r, m) for m in MOVES if r.get(f"{m}_uci")}
            except (KeyError, ValueError):
                continue
            if "key" not in scores or len(scores) < 3:
                continue
            usable += 1
            if rank_first(scores, "key"):
                hits += 1
        results[name] = {"key_top1_rate": hits / usable, "n": usable}

    # Quiet separation: does the detector value for the quiet best move
    # look like the sacrifice key move's value?
    def mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    quiet_sep = {}
    for name, fn in detectors.items():
        try:
            sac_vals = [fn(r, "key") for r in sac if r.get("key_uci")]
            qui_vals = [float(r[f"collapse_best_bits"]) if name == "collapse_only"
                        else abs(float(r["useful_shift_best"])) if name == "specificity_only"
                        else float(r["local_cost_best"]) if name == "local_value"
                        else float(r["useful_shift_best"])
                        for r in qui if r.get("best_uci")]
        except (KeyError, ValueError):
            continue
        quiet_sep[name] = {
            "sacrifice_key_mean": mean(sac_vals),
            "quiet_best_mean": mean(qui_vals),
            "separation": mean(sac_vals) - mean(qui_vals),
        }

    out = {"key_identification": results, "quiet_separation": quiet_sep}
    (OUTPUTS / "chess_prior_detectors.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
