"""Engine-independent realized-outcome referee for the chess discovery runs.

Reviewer objection addressed: both the collapse score (shallow Stockfish) and
the value-critical label (deep Stockfish) share one engine lineage, and the
SF11-classical relabel only removes the NNUE evaluator, not the search family.
This analysis uses a referee that involves NO engine at all: the realized
result of the human game, read from the same public PGN dumps the positions
were sampled from. Realized outcomes were never touched by any stage of the
discovery pipeline, so this is a fully independent behavioural check.

Design (internally frozen before running; both months analysed identically):

  For every scored position we recover from the PGN
    - the game result (mover's realized score: 1 win / 0.5 draw / 0 loss),
    - the move the human actually played at that ply,
    - the mover's rating (confounder audit only).
  played_m_star = (actual move == the shallow rank-1 move m*).

  Effect of interest: Delta = E[realized | played m*] - E[realized | other].
  Registered predictions:
    RO1  Delta(flagged) - Delta(unflagged) > 0 pooled over both months:
         at positions the collapse instrument flags as value-critical, the
         human's actual decision matters MORE for the realized game result.
    RO2  Delta(flagged) > 0 in each month separately.
  Inference: permutation test (10,000 shuffles of the flag column) for RO1;
  bootstrap CIs for the per-month Deltas. Mover-rating balance between
  flagged and unflagged positions is reported as a confounder audit.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.pgn
import numpy as np
import zstandard

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

RUNS = {
    "2015_08": {
        "records": OUTPUTS / "chess_discovery_main.json",
        "pgn": HERE / "external_chess" / "lichess_2015_08_prefix.pgn.zst",
    },
    "2016_03": {
        "records": OUTPUTS / "chess_discovery_replication_2016_03.json",
        "pgn": HERE / "external_chess" / "lichess_2016_03_prefix.pgn.zst",
    },
}
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
SEED = 20260717


def load_outcomes(pgn_path: Path, wanted: Dict[int, Dict]) -> Dict[int, Dict]:
    """Stream the PGN prefix and pull result/move/ratings for wanted games.

    wanted maps game_index -> record (with 'ply'). Returns game_index ->
    {result, played_uci, white_elo, black_elo}.
    """
    dctx = zstandard.ZstdDecompressor()
    found: Dict[int, Dict] = {}
    game_index = -1
    max_wanted = max(wanted)
    with pgn_path.open("rb") as raw:
        reader = io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8",
                                  errors="ignore")
        while game_index < max_wanted:
            game = chess.pgn.read_game(reader)
            if game is None:
                break
            game_index += 1
            if game_index not in wanted:
                continue
            ply = wanted[game_index]["ply"]
            moves = list(game.mainline_moves())
            if len(moves) <= ply:
                continue
            found[game_index] = {
                "result": game.headers.get("Result", "*"),
                "played_uci": moves[ply].uci(),
                "white_elo": int(game.headers.get("WhiteElo", "0") or 0),
                "black_elo": int(game.headers.get("BlackElo", "0") or 0),
            }
    return found


def realized_score(result: str, mover_is_white: bool) -> Optional[float]:
    if result == "1-0":
        return 1.0 if mover_is_white else 0.0
    if result == "0-1":
        return 0.0 if mover_is_white else 1.0
    if result == "1/2-1/2":
        return 0.5
    return None


def assemble_rows(month: str) -> List[Dict]:
    data = json.loads(RUNS[month]["records"].read_text(encoding="utf-8"))
    records = data["records"]
    wanted = {r["game_index"]: r for r in records}
    outcomes = load_outcomes(RUNS[month]["pgn"], wanted)
    rows: List[Dict] = []
    for rec in records:
        info = outcomes.get(rec["game_index"])
        if info is None:
            continue
        board = chess.Board(rec["fen"])
        mover_is_white = board.turn == chess.WHITE
        score = realized_score(info["result"], mover_is_white)
        if score is None:
            continue
        rows.append({
            "month": month,
            "flag": int(rec["flag"]),
            "label": int(rec["label"]),
            "do_gap": float(rec["do_gap"]),
            "played_m_star": int(info["played_uci"] == rec["m_star"]),
            "realized": score,
            "mover_elo": (info["white_elo"] if mover_is_white
                          else info["black_elo"]),
        })
    return rows


def delta(rows: List[Dict]) -> Optional[float]:
    """E[realized | played m*] - E[realized | other move] on these rows."""
    hit = [r["realized"] for r in rows if r["played_m_star"] == 1]
    miss = [r["realized"] for r in rows if r["played_m_star"] == 0]
    if not hit or not miss:
        return None
    return float(np.mean(hit) - np.mean(miss))


def interaction(rows: List[Dict]) -> Optional[float]:
    d_flag = delta([r for r in rows if r["flag"] == 1])
    d_unflag = delta([r for r in rows if r["flag"] == 0])
    if d_flag is None or d_unflag is None:
        return None
    return d_flag - d_unflag


def permutation_p(rows: List[Dict], observed: float,
                  rng: np.random.Generator) -> float:
    flags = np.array([r["flag"] for r in rows])
    count = 0
    valid = 0
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(flags)
        shuffled = [
            {**row, "flag": int(f)} for row, f in zip(rows, flags)
        ]
        stat = interaction(shuffled)
        if stat is None:
            continue
        valid += 1
        if stat >= observed:
            count += 1
    return (count + 1) / (valid + 1)


def bootstrap_ci(rows: List[Dict], rng: np.random.Generator):
    stats = []
    n = len(rows)
    for _ in range(N_BOOTSTRAP):
        sample = [rows[i] for i in rng.integers(0, n, size=n)]
        stat = delta([r for r in sample if r["flag"] == 1])
        if stat is not None:
            stats.append(stat)
    return [float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5))]


def main() -> None:
    rng = np.random.default_rng(SEED)
    all_rows: List[Dict] = []
    per_month: Dict[str, Dict] = {}
    for month in RUNS:
        rows = assemble_rows(month)
        all_rows.extend(rows)
        flagged = [r for r in rows if r["flag"] == 1]
        unflagged = [r for r in rows if r["flag"] == 0]
        per_month[month] = {
            "n_matched": len(rows),
            "n_flagged": len(flagged),
            "delta_flagged": delta(flagged),
            "delta_flagged_ci95": bootstrap_ci(rows, rng),
            "delta_unflagged": delta(unflagged),
            "played_rate_flagged": float(np.mean(
                [r["played_m_star"] for r in flagged])),
            "played_rate_unflagged": float(np.mean(
                [r["played_m_star"] for r in unflagged])),
            "mover_elo_flagged": float(np.mean(
                [r["mover_elo"] for r in flagged])),
            "mover_elo_unflagged": float(np.mean(
                [r["mover_elo"] for r in unflagged])),
        }
        print(f"{month}: matched {len(rows)}, "
              f"delta_flag {per_month[month]['delta_flagged']}, "
              f"delta_unflag {per_month[month]['delta_unflagged']}")

    observed = interaction(all_rows)
    p_value = permutation_p(all_rows, observed, rng)
    ro1 = observed is not None and observed > 0
    ro2 = all(
        per_month[m]["delta_flagged"] is not None
        and per_month[m]["delta_flagged"] > 0
        for m in per_month
    )
    summary = {
        "status": ("engine-independent realized-outcome referee; outcomes "
                   "never used in any scoring stage"),
        "predictions_frozen_in_docstring": True,
        "per_month": per_month,
        "pooled": {
            "n": len(all_rows),
            "interaction_delta_flag_minus_unflag": observed,
            "permutation_p_one_sided": p_value,
        },
        "registered_outcomes": {
            "RO1_interaction_positive_pooled": bool(ro1),
            "RO2_delta_flagged_positive_each_month": bool(ro2),
        },
        "confounder_note": (
            "Mover-rating means for flagged vs unflagged positions are "
            "reported above; the interaction contrast differences out the "
            "baseline association between playing the shallow-best move "
            "and winning."
        ),
    }
    out = OUTPUTS / "chess_realized_outcome.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["pooled"], indent=2))
    print(json.dumps(summary["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
