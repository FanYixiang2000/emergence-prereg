"""Prospective discovery on uncurated chess positions.

Protocol frozen in CHESS_DISCOVERY_PREREGISTRATION.md BEFORE any game data
was downloaded or scored. The shallow observer (multipv/temperature/playouts/
basins) is imported unchanged from chess_collapse_probe. The referee label
(deep eval gap) is computed after scoring and never fed back into scores.

Pipeline:
  1. stream a prefix of the public Lichess monthly dump (uncurated, no
     theme tags), apply the declared population filter, sample one position
     per eligible game with the frozen RNG;
  2. score each position with the shallow instrument only:
     potential H(P0), do_gap between the depth-4 rank-1 and rank-2 moves;
  3. referee: depth-18 multipv-4 eval gap labels;
  4. baselines, AUROC, registered predictions CD1-CD4.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.engine
import chess.pgn
import numpy as np
import zstandard

from chess_collapse_probe import (
    BASINS,
    MULTIPV,
    PLAYOUT_DEPTH,
    entropy_bits,
    future_distribution,
    get_engine,
    worker_init,
)

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
GAME_FILE = HERE / "external_chess" / "lichess_2015_08_prefix.pgn.zst"
GAME_URL = ("https://database.lichess.org/standard/"
            "lichess_db_standard_rated_2015-08.pgn.zst")
GLOBAL_SEED = 20260716

# Frozen replication month (see preregistration addendum): different year,
# same protocol, own sampling seed.
REPLICATION_GAME_FILE = (HERE / "external_chess"
                         / "lichess_2016_03_prefix.pgn.zst")
REPLICATION_SEED = 20260717

RATING_LO, RATING_HI = 1800, 2400
MIN_BASE_SECONDS = 300
MIN_PLIES = 30
PLY_LO, PLY_HI_CAP = 16, 60
MIN_LEGAL = 6

REFEREE_DEPTH = 18
REFEREE_GAP_CP = 150
REFEREE_FLOOR_CP = -50
MATE_CLIP = 1000

FLAG_POTENTIAL = 1.0   # frozen C4 cutoff, reused
FLAG_DO_GAP = 0.15     # frozen C3 cutoff, reused


# game sampling

def eligible(headers: chess.pgn.Headers) -> bool:
    try:
        white = int(headers.get("WhiteElo", "0"))
        black = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if not (RATING_LO <= white <= RATING_HI and RATING_LO <= black <= RATING_HI):
        return False
    tc = headers.get("TimeControl", "-")
    if "+" not in tc:
        return False
    try:
        base = int(tc.split("+")[0])
    except ValueError:
        return False
    if base < MIN_BASE_SECONDS:
        return False
    if headers.get("Variant", "Standard") != "Standard":
        return False
    return headers.get("Termination", "") != "Abandoned"


def sample_positions(n_wanted: int, skip: int = 0,
                     game_file: Path = GAME_FILE,
                     seed: int = GLOBAL_SEED) -> List[Dict]:
    """One frozen-RNG position per eligible game, streamed from the dump."""
    dctx = zstandard.ZstdDecompressor()
    tasks: List[Dict] = []
    game_index = -1
    eligible_index = -1
    with game_file.open("rb") as raw:
        reader = io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8",
                                  errors="ignore")
        while len(tasks) < n_wanted:
            game = chess.pgn.read_game(reader)
            if game is None:
                break
            game_index += 1
            if not eligible(game.headers):
                continue
            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLIES:
                continue
            eligible_index += 1
            if eligible_index < skip:
                continue
            rng = random.Random((seed << 24) | eligible_index)
            hi = min(PLY_HI_CAP, len(moves) - 8)
            if hi < PLY_LO:
                continue
            board = None
            for _attempt in range(5):
                ply = rng.randint(PLY_LO, hi)
                candidate = chess.Board()
                for move in moves[:ply]:
                    candidate.push(move)
                if (not candidate.is_check()
                        and candidate.legal_moves.count() >= MIN_LEGAL):
                    board = candidate
                    break
            if board is None:
                continue
            tasks.append({
                "game_index": game_index,
                "eligible_index": eligible_index,
                "site": game.headers.get("Site", ""),
                "ply": board.ply(),
                "fen": board.fen(),
                "seed": seed,
            })
    return tasks


# scoring

def clipped_cp(score: chess.engine.PovScore, side: chess.Color) -> float:
    return float(np.clip(score.pov(side).score(mate_score=10000),
                         -MATE_CLIP, MATE_CLIP))


def score_position(task: Dict) -> Optional[Dict]:
    engine = get_engine()
    rng = random.Random((task.get("seed", GLOBAL_SEED) << 32)
                        | task["eligible_index"])
    board = chess.Board(task["fen"])
    side = board.turn

    infos = engine.analyse(board, chess.engine.Limit(depth=PLAYOUT_DEPTH),
                           multipv=MULTIPV)
    if isinstance(infos, dict):
        infos = [infos]
    ranked = [(info["pv"][0], clipped_cp(info["score"], side))
              for info in infos if info.get("pv")]
    if len(ranked) < 2:
        return None
    m_star, shallow_best_cp = ranked[0]
    a_star, shallow_second_cp = ranked[1]

    base = future_distribution(engine, board, side, rng)

    def do_dist(move: chess.Move) -> Dict[str, float]:
        sim = board.copy(stack=False)
        sim.push(move)
        return future_distribution(engine, sim, side, rng)

    dist_m = do_dist(m_star)
    dist_a = do_dist(a_star)
    good = lambda d: d["win"] + d["adv"]

    legal = list(board.legal_moves)
    tactical = sum(1 for mv in legal
                   if board.is_capture(mv) or board.gives_check(mv))
    material = abs(sum(
        {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
         chess.QUEEN: 9, chess.KING: 0}[p.piece_type]
        * (1 if p.color == side else -1)
        for p in board.piece_map().values()))

    # referee (computed last; deep, independent depth)
    deep = engine.analyse(board, chess.engine.Limit(depth=REFEREE_DEPTH),
                          multipv=4)
    if isinstance(deep, dict):
        deep = [deep]
    deep_cps = [clipped_cp(info["score"], side)
                for info in deep if info.get("pv")]
    if len(deep_cps) < 2:
        return None
    referee_gap = deep_cps[0] - deep_cps[1]
    label = int(referee_gap >= REFEREE_GAP_CP
                and deep_cps[0] >= REFEREE_FLOOR_CP)

    return {
        **task,
        "potential_bits": entropy_bits(base),
        "do_gap": good(dist_m) - good(dist_a),
        "m_star": m_star.uci(),
        "a_star": a_star.uci(),
        "shallow_gap_cp": shallow_best_cp - shallow_second_cp,
        "shallow_eval_abs_cp": abs(shallow_best_cp),
        "tactical_density": tactical,
        "material_imbalance": material,
        "referee_gap_cp": referee_gap,
        "referee_best_cp": deep_cps[0],
        "label": label,
        "flag": int(entropy_bits(base) >= FLAG_POTENTIAL
                    and good(dist_m) - good(dist_a) >= FLAG_DO_GAP),
    }


# analysis

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def analyse(records: List[Dict]) -> Dict:
    labels = np.array([r["label"] for r in records])
    base_rate = float(labels.mean())
    scores = {
        "do_gap": np.array([r["do_gap"] for r in records]),
        "shallow_gap_cp": np.array([r["shallow_gap_cp"] for r in records]),
        "tactical_density": np.array(
            [r["tactical_density"] for r in records], dtype=float),
        "material_imbalance": np.array(
            [r["material_imbalance"] for r in records], dtype=float),
        "shallow_eval_abs_cp": np.array(
            [r["shallow_eval_abs_cp"] for r in records]),
    }
    aurocs = {name: auroc(vals, labels) for name, vals in scores.items()}
    flags = np.array([r["flag"] for r in records])
    n_flag = int(flags.sum())
    precision = float(labels[flags == 1].mean()) if n_flag else float("nan")
    recall = (float(flags[labels == 1].mean())
              if labels.sum() else float("nan"))
    flag_potentials = [r["potential_bits"] for r in records if r["flag"]]
    predictions = {
        "CD1_auroc_do_gap_ge_0.70": aurocs["do_gap"] >= 0.70,
        "CD2_precision_ge_2x_base_rate": (
            n_flag > 0 and precision >= 2 * base_rate),
        "CD3_beats_tactical_and_material": (
            aurocs["do_gap"] > aurocs["tactical_density"]
            and aurocs["do_gap"] > aurocs["material_imbalance"]),
        "CD4_flagged_median_potential_ge_1bit": (
            n_flag > 0 and float(np.median(flag_potentials)) >= 1.0),
    }
    return {
        "n_positions": len(records),
        "referee_base_rate": base_rate,
        "auroc": aurocs,
        "flag": {"n_flagged": n_flag, "precision": precision,
                 "recall": recall, "base_rate": base_rate},
        "flagged_median_potential": (
            float(np.median(flag_potentials)) if n_flag else None),
        "registered_predictions": predictions,
        "all_pass": all(predictions.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--replication", action="store_true",
                        help="Use the frozen 2016-03 replication month/seed.")
    args = parser.parse_args()

    game_file = REPLICATION_GAME_FILE if args.replication else GAME_FILE
    seed = REPLICATION_SEED if args.replication else GLOBAL_SEED
    tasks = sample_positions(args.n, skip=args.skip, game_file=game_file,
                             seed=seed)
    print(f"sampled {len(tasks)} positions "
          f"(eligible indices {tasks[0]['eligible_index']}-"
          f"{tasks[-1]['eligible_index']})", flush=True)

    set_start_method("spawn", force=True)
    records: List[Dict] = []
    with Pool(args.workers, initializer=worker_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(score_position, tasks)):
            if rec is not None:
                records.append(rec)
            if (i + 1) % 20 == 0:
                print(f"  scored {i + 1}/{len(tasks)}", flush=True)
    records.sort(key=lambda r: r["eligible_index"])

    result = {
        "status": ("prospectively frozen discovery run" if args.tag == "main"
                   else "feasibility pilot (worker count only)"),
        "protocol": "CHESS_DISCOVERY_PREREGISTRATION.md",
        "sample": {"n_scored": len(records), "skip": args.skip},
        "analysis": analyse(records) if records else None,
        "records": records,
    }
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / f"chess_discovery_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if records:
        print(json.dumps(result["analysis"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
