"""Within-state useful possibility collapse in chess (external real system).

Protocol, selection rules, candidate definitions, basins, thresholds and
predictions are frozen in CHESS_PREREGISTRATION.md BEFORE any measurement.
Positions and key-move labels come from the lichess puzzle database
(real rated human games, externally theme-tagged); the engine is the
Ubuntu-archive Stockfish 14.1 NNUE build. Zero authorial control over the
possibility space.

For each position s we estimate the future-outcome distribution P(B | s)
and P(B | s, a) by stochastic playouts under an imperfect-play observer
(softmax over engine multipv scores), then measure potential, collapse,
useful shift, specificity and local material cost for the externally
annotated key move versus deep-alternative / shallow-greedy / random
counterfactuals, plus a quiet-position control set.

Estimator parameters (playout depth, multipv, temperature, horizon,
rollouts, workers) are pilot-tunable per the registration; everything
else is frozen.

Pilot note (recorded per protocol; estimator parameters only, as the
registration allows): the registered initial estimator (multipv 4,
playout depth 6, temperature 150 cp) produced an observer too strong for
the potential component -- rollouts converged on the winning line before
the key move was played, so median potential on the sacrifice set was
0.27 bits (C4 fail; log kept in
outputs/chess_pilot_log_initial_params.txt). Weakening the observer to
multipv 6, playout depth 4, temperature 300 cp restored visible
aleatoric openness (median potential 1.46 bits) while keeping the
key/greedy/deep-alt contrasts sharp (C1-C4 pass on the pilot; log in
outputs/chess_pilot_log_depth4_temp300.txt). A softer observer still
(multipv 8, temperature 500 cp) degraded the contrasts themselves
(outputs/chess_pilot_log_depth4_temp500.txt) and was rejected. The main
run is frozen at multipv 6, depth 4, temperature 300, horizon 12, N=32,
classification depth 12. The pilot already showed the C5 effect-size
margin (>= 0.25) is at risk for a structural reason recorded before the
main run: P(B | s) is estimated under the observer's own policy, which
samples the key move endogenously at the first ply, so useful_shift(key)
is depressed relative to the counterfactual contrasts; thresholds are
frozen, so if C5 fails in the main run it is reported as a registered
failure with this route.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zlib
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine
import zstandard

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
PUZZLE_FILE = HERE / "external_chess" / "lichess_db_puzzle.csv.zst"
ENGINE_PATH = HERE / "external_chess" / "stockfish_14.1-1_x" / "usr" / "games" / "stockfish"

BASINS = ("win", "adv", "equal", "disadv", "loss")
GLOBAL_SEED = 20260706

# Estimator parameters (pilot-tunable per registration).
MULTIPV = 6
PLAYOUT_DEPTH = 4
TEMPERATURE_CP = 300.0
HORIZON_PLIES = 12
N_ROLLOUTS = 32
CLASSIFY_DEPTH = 12
DEEP_DEPTH = 16
GREEDY_DEPTH = 2
QUIET_EVAL_DEPTH = 12
COST_REPLY_DEPTH = 12
# Basin boundaries (frozen in the registration; sweepable for robustness).
WIN_CP = 300
ADV_CP = 100


def apply_overrides(overrides: Dict) -> None:
    """Set tunable module globals (used by the robustness grid).

    Must be called inside each spawned worker (worker_init), because
    spawn re-imports the module and discards parent-side mutations.
    """
    for key, value in overrides.items():
        globals()[key] = Path(value) if key == "ENGINE_PATH" else value

PIECE_VALUES = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}

_ENGINE: Optional[chess.engine.SimpleEngine] = None


def get_engine() -> chess.engine.SimpleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = chess.engine.SimpleEngine.popen_uci(str(ENGINE_PATH))
        _ENGINE.configure({"Threads": 1, "Hash": 64})
    return _ENGINE


def close_engine() -> None:
    global _ENGINE
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except chess.engine.EngineError:
            pass
        _ENGINE = None


def worker_init(overrides: Optional[Dict] = None) -> None:
    # Fresh engine per worker; never inherit the parent's handle.
    global _ENGINE
    _ENGINE = None
    if overrides:
        apply_overrides(overrides)
    get_engine()


def material_balance(board: chess.Board, side: chess.Color) -> float:
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, side))
        total -= value * len(board.pieces(piece_type, not side))
    return total


def entropy_bits(p: Dict[str, float]) -> float:
    return -sum(v * math.log(v, 2) for v in p.values() if v > 0)


def js_bits(p: Dict[str, float], q: Dict[str, float]) -> float:
    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        eps = 1e-12
        return sum(a[k] * math.log((a[k] + eps) / (b[k] + eps), 2)
                   for k in BASINS if a[k] > 0)
    m = {k: 0.5 * (p[k] + q[k]) for k in BASINS}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def classify_terminal(board: chess.Board, side: chess.Color) -> str:
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return "equal"
    return "win" if outcome.winner == side else "loss"


def classify_position(engine: chess.engine.SimpleEngine, board: chess.Board,
                      side: chess.Color) -> str:
    if board.is_game_over():
        return classify_terminal(board, side)
    info = engine.analyse(board, chess.engine.Limit(depth=CLASSIFY_DEPTH))
    cp = info["score"].pov(side).score(mate_score=100000)
    if cp >= WIN_CP:
        return "win"
    if cp >= ADV_CP:
        return "adv"
    if cp > -ADV_CP:
        return "equal"
    if cp > -WIN_CP:
        return "disadv"
    return "loss"


def sample_playout_move(engine: chess.engine.SimpleEngine, board: chess.Board,
                        rng: random.Random) -> Optional[chess.Move]:
    infos = engine.analyse(board, chess.engine.Limit(depth=PLAYOUT_DEPTH),
                           multipv=MULTIPV)
    if isinstance(infos, dict):
        infos = [infos]
    moves: List[chess.Move] = []
    scores: List[float] = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        moves.append(pv[0])
        scores.append(info["score"].pov(board.turn).score(mate_score=2000))
    if not moves:
        return None
    top = max(scores)
    weights = [math.exp((s - top) / TEMPERATURE_CP) for s in scores]
    return rng.choices(moves, weights=weights, k=1)[0]


def future_distribution(engine: chess.engine.SimpleEngine, board: chess.Board,
                        side: chess.Color, rng: random.Random) -> Dict[str, float]:
    counts = {basin: 0 for basin in BASINS}
    for _ in range(N_ROLLOUTS):
        sim = board.copy(stack=False)
        for _ply in range(HORIZON_PLIES):
            if sim.is_game_over():
                break
            move = sample_playout_move(engine, sim, rng)
            if move is None:
                break
            sim.push(move)
        counts[classify_position(engine, sim, side)] += 1
    return {basin: counts[basin] / N_ROLLOUTS for basin in BASINS}


def best_move_excluding(engine: chess.engine.SimpleEngine, board: chess.Board,
                        depth: int, exclude: Optional[chess.Move]) -> Optional[chess.Move]:
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
    if isinstance(infos, dict):
        infos = [infos]
    for info in infos:
        pv = info.get("pv")
        if pv and pv[0] != exclude:
            return pv[0]
    return None


def local_cost(engine: chess.engine.SimpleEngine, board: chess.Board,
               move: chess.Move, side: chess.Color) -> float:
    before = material_balance(board, side)
    sim = board.copy(stack=False)
    sim.push(move)
    if not sim.is_game_over():
        reply = engine.play(sim, chess.engine.Limit(depth=COST_REPLY_DEPTH)).move
        if reply is not None:
            sim.push(reply)
    return material_balance(sim, side) - before


def measure_position(task: Dict) -> Dict:
    engine = get_engine()
    stable = zlib.crc32(task["puzzle_id"].encode())
    rng = random.Random((GLOBAL_SEED << 32) | stable)
    board = chess.Board(task["fen"])
    for uci in task["setup_moves"]:
        board.push(chess.Move.from_uci(uci))
    side = board.turn

    key = chess.Move.from_uci(task["key_move"]) if task["key_move"] else None

    candidates: Dict[str, Optional[chess.Move]] = {}
    if task["kind"] == "sacrifice":
        candidates["key"] = key
        candidates["deep_alt"] = best_move_excluding(engine, board, DEEP_DEPTH, key)
        candidates["greedy"] = best_move_excluding(engine, board, GREEDY_DEPTH, key)
        legal = [m for m in board.legal_moves if m != key]
        candidates["random"] = rng.choice(legal) if legal else None
    else:
        candidates["best"] = best_move_excluding(engine, board, DEEP_DEPTH, None)

    base = future_distribution(engine, board, side, rng)
    record: Dict = {
        "puzzle_id": task["puzzle_id"],
        "kind": task["kind"],
        "rating": task["rating"],
        "themes": task["themes"],
        "potential_bits": entropy_bits(base),
        "p_win_base": base["win"],
        "base_dist": base,
    }
    dists: Dict[str, Dict[str, float]] = {}
    for name, move in candidates.items():
        if move is None:
            continue
        sim = board.copy(stack=False)
        sim.push(move)
        dist = future_distribution(engine, sim, side, rng)
        dists[name] = dist
        record[f"{name}_uci"] = move.uci()
        record[f"collapse_{name}_bits"] = entropy_bits(base) - entropy_bits(dist)
        record[f"useful_shift_{name}"] = dist["win"] - base["win"]
        record[f"p_win_{name}"] = dist["win"]
        record[f"local_cost_{name}"] = local_cost(engine, board, move, side)
    if "key" in dists and "deep_alt" in dists:
        record["specificity_js_bits"] = js_bits(dists["key"], dists["deep_alt"])
    return record


def select_positions(n_sacrifice: int, n_quiet: int) -> Tuple[List[Dict], List[Dict]]:
    engine = get_engine()
    sacrifice: List[Dict] = []
    quiet: List[Dict] = []
    with open(PUZZLE_FILE, "rb") as handle:
        reader = zstandard.ZstdDecompressor().stream_reader(handle)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        rows = csv.reader(text)
        next(rows)
        for row in rows:
            if len(sacrifice) >= n_sacrifice and len(quiet) >= n_quiet:
                break
            puzzle_id, fen, moves_str, rating_s, _, _, plays_s, themes = row[:8]
            themes_set = set(themes.split())
            moves = moves_str.split()
            try:
                rating = int(rating_s)
                plays = int(plays_s)
            except ValueError:
                continue
            if plays < 1000:
                continue
            if (len(sacrifice) < n_sacrifice and "sacrifice" in themes_set
                    and 1800 <= rating <= 2400 and len(moves) >= 4):
                try:
                    board = chess.Board(fen)
                    board.push(chess.Move.from_uci(moves[0]))
                    key = chess.Move.from_uci(moves[1])
                    if key not in board.legal_moves:
                        continue
                    probe = board.copy(stack=False)
                    probe.push(key)
                    if probe.is_checkmate():
                        continue
                except (ValueError, AssertionError):
                    continue
                sacrifice.append({
                    "puzzle_id": puzzle_id, "kind": "sacrifice", "fen": fen,
                    "setup_moves": [moves[0]], "key_move": moves[1],
                    "rating": rating, "themes": themes,
                })
            elif (len(quiet) < n_quiet and "middlegame" in themes_set
                  and "sacrifice" not in themes_set):
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue
                if board.is_check() or board.legal_moves.count() < 20:
                    continue
                info = engine.analyse(board, chess.engine.Limit(depth=QUIET_EVAL_DEPTH))
                cp = info["score"].pov(board.turn).score(mate_score=100000)
                if abs(cp) > 60:
                    continue
                quiet.append({
                    "puzzle_id": puzzle_id, "kind": "quiet", "fen": fen,
                    "setup_moves": [], "key_move": None,
                    "rating": rating, "themes": themes,
                })
    return sacrifice, quiet


def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact binomial sign test P(X >= wins | p=0.5, n=wins+losses)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def median(values: Sequence[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze(records: List[Dict]) -> Dict:
    sac = [r for r in records if r["kind"] == "sacrifice"
           and "useful_shift_key" in r and "useful_shift_deep_alt" in r
           and "useful_shift_greedy" in r]
    qui = [r for r in records if r["kind"] == "quiet" and "useful_shift_best" in r]

    cost_key = [r["local_cost_key"] for r in sac]
    cost_greedy = [r["local_cost_greedy"] for r in sac]
    c1 = {
        "median_local_cost_key": median(cost_key),
        "median_local_cost_greedy": median(cost_greedy),
        "pass": median(cost_key) < 0 and median(cost_greedy) > median(cost_key),
    }

    def sign_stats(a_field: str, b_field: str) -> Dict:
        wins = sum(1 for r in sac if r[a_field] > r[b_field])
        losses = sum(1 for r in sac if r[a_field] < r[b_field])
        return {"wins": wins, "losses": losses, "ties": len(sac) - wins - losses,
                "p_one_sided": sign_test_p(wins, losses)}

    vs_greedy = sign_stats("useful_shift_key", "useful_shift_greedy")
    vs_random = sign_stats("useful_shift_key", "useful_shift_random")
    c2 = {
        "mean_useful_shift_key": mean([r["useful_shift_key"] for r in sac]),
        "mean_useful_shift_greedy": mean([r["useful_shift_greedy"] for r in sac]),
        "mean_useful_shift_random": mean([r["useful_shift_random"] for r in sac
                                          if "useful_shift_random" in r]),
        "sign_vs_greedy": vs_greedy,
        "sign_vs_random": vs_random,
        "pass": vs_greedy["p_one_sided"] < 1e-3 and vs_random["p_one_sided"] < 1e-3,
    }

    p_win_key = mean([r["p_win_key"] for r in sac])
    p_win_alt = mean([r["p_win_deep_alt"] for r in sac])
    alt_sign = sign_stats("p_win_key", "p_win_deep_alt")
    c3 = {
        "mean_p_win_key": p_win_key,
        "mean_p_win_deep_alt": p_win_alt,
        "gap": p_win_key - p_win_alt,
        "sign_vs_deep_alt": alt_sign,
        "mean_specificity_js_bits": mean([r["specificity_js_bits"] for r in sac
                                          if "specificity_js_bits" in r]),
        "pass": (p_win_key - p_win_alt) >= 0.15 and alt_sign["p_one_sided"] < 1e-3,
    }

    pot = [r["potential_bits"] for r in sac]
    c4 = {"median_potential_bits": median(pot), "pass": median(pot) >= 1.0}

    quiet_shift = mean([r["useful_shift_best"] for r in qui])
    key_shift = mean([r["useful_shift_key"] for r in sac])
    c5 = {
        "mean_useful_shift_quiet_best": quiet_shift,
        "mean_useful_shift_sacrifice_key": key_shift,
        "gap": key_shift - quiet_shift,
        "mean_quiet_potential_bits": mean([r["potential_bits"] for r in qui]),
        "pass": quiet_shift < 0.10 and (key_shift - quiet_shift) >= 0.25,
    }

    return {
        "n_sacrifice_analyzed": len(sac),
        "n_quiet_analyzed": len(qui),
        "C1_local_cost": c1,
        "C2_useful_collapse": c2,
        "C3_selectivity": c3,
        "C4_potential": c4,
        "C5_quiet_control": c5,
        "all_pass": all(x["pass"] for x in (c1, c2, c3, c4, c5)),
        "estimator": {
            "multipv": MULTIPV, "playout_depth": PLAYOUT_DEPTH,
            "temperature_cp": TEMPERATURE_CP, "horizon_plies": HORIZON_PLIES,
            "n_rollouts": N_ROLLOUTS, "classify_depth": CLASSIFY_DEPTH,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--n_sacrifice", type=int, default=240)
    parser.add_argument("--n_quiet", type=int, default=120)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    n_sac = 12 if args.pilot else args.n_sacrifice
    n_qui = 6 if args.pilot else args.n_quiet

    print(f"selecting positions (sacrifice={n_sac}, quiet={n_qui}) ...", flush=True)
    sacrifice, quiet = select_positions(n_sac, n_qui)
    print(f"selected {len(sacrifice)} sacrifice + {len(quiet)} quiet", flush=True)
    close_engine()
    tasks = sacrifice + quiet

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=worker_init) as pool:
        records = []
        for i, rec in enumerate(pool.imap_unordered(measure_position, tasks)):
            records.append(rec)
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"measured {i + 1}/{len(tasks)}", flush=True)

    summary = analyze(records)
    tag = args.tag or ("pilot" if args.pilot else "main")
    OUTPUTS.mkdir(exist_ok=True)
    with open(OUTPUTS / f"chess_collapse_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    fields = sorted({k for r in records for k in r if k != "base_dist"})
    with open(OUTPUTS / f"chess_collapse_{tag}_positions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
