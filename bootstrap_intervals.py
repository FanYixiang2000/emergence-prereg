"""Bootstrap confidence intervals for the paper's headline effect sizes.

Statistical hardening pass: every headline number that is currently
reported as a point estimate (plus a sign test) gets a nonparametric
percentile bootstrap CI (B = 20,000 resamples, seed fixed). Resampling
units are the natural independent units of each analysis:

- chess: positions (240 sacrifice / 120 quiet);
- deep MARL (simple_spread) and LBF: evaluation episodes, pooled over
  seeds with per-seed medians also reported;
- prior detectors on chess: positions (binomial-style bootstrap of the
  rank-first rate);
- MultiBERTs burst-jump alignment keeps its existing permutation test
  (already distributional; not re-done here).

Writes outputs/bootstrap_intervals.json and prints a table.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

OUTPUTS = Path(__file__).resolve().parent / "outputs"
B = 20_000
SEED = 20260707
CI = (2.5, 97.5)


def percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * q / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def boot_ci(values: Sequence[float], stat, rng: random.Random,
            b: int = B) -> Dict[str, float]:
    values = list(values)
    n = len(values)
    stats = []
    for _ in range(b):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(stat(sample))
    stats.sort()
    return {
        "point": stat(values),
        "n": n,
        "ci_lo": percentile(stats, CI[0]),
        "ci_hi": percentile(stats, CI[1]),
    }


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fnum(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def chess_intervals(rng: random.Random) -> Dict[str, Dict]:
    rows = list(csv.DictReader(
        (OUTPUTS / "chess_collapse_main_positions.csv").open(encoding="utf-8")))
    sac = [r for r in rows if r["kind"] == "sacrifice"]
    quiet = [r for r in rows if r["kind"] == "quiet"]

    out: Dict[str, Dict] = {}
    # C3 headline: P(win | do key) - P(win | do best alternative).
    gaps = [fnum(r, "p_win_key") - fnum(r, "p_win_deep_alt") for r in sac
            if fnum(r, "p_win_key") is not None and fnum(r, "p_win_deep_alt") is not None]
    out["do_key_minus_do_alt_win_gap_mean"] = boot_ci(gaps, mean, rng)
    out["do_key_minus_do_alt_win_gap_median"] = boot_ci(gaps, median, rng)

    # C2 headline: useful shift of key vs greedy / random.
    kg = [fnum(r, "useful_shift_key") - fnum(r, "useful_shift_greedy") for r in sac
          if fnum(r, "useful_shift_key") is not None and fnum(r, "useful_shift_greedy") is not None]
    out["useful_shift_key_minus_greedy_mean"] = boot_ci(kg, mean, rng)
    kr = [fnum(r, "useful_shift_key") - fnum(r, "useful_shift_random") for r in sac
          if fnum(r, "useful_shift_key") is not None and fnum(r, "useful_shift_random") is not None]
    out["useful_shift_key_minus_random_mean"] = boot_ci(kr, mean, rng)

    # C1 headline: local cost of the key move (material pawns).
    costs = [fnum(r, "local_cost_key") for r in sac if fnum(r, "local_cost_key") is not None]
    out["key_local_cost_median"] = boot_ci(costs, median, rng)
    out["key_strict_loss_rate"] = boot_ci(
        [1.0 if c < 0 else 0.0 for c in costs], mean, rng)

    # C4/C5: potential in sacrifice vs quiet positions (two-sample:
    # bootstrap each group independently for the difference of medians).
    pot_s = [fnum(r, "potential_bits") for r in sac if fnum(r, "potential_bits") is not None]
    pot_q = [fnum(r, "potential_bits") for r in quiet if fnum(r, "potential_bits") is not None]
    diffs = []
    for _ in range(B):
        s = [pot_s[rng.randrange(len(pot_s))] for _ in range(len(pot_s))]
        q = [pot_q[rng.randrange(len(pot_q))] for _ in range(len(pot_q))]
        diffs.append(median(q) - median(s))
    diffs.sort()
    out["quiet_minus_sacrifice_potential_median"] = {
        "point": median(pot_q) - median(pot_s),
        "n": f"{len(pot_q)}q/{len(pot_s)}s",
        "ci_lo": percentile(diffs, CI[0]),
        "ci_hi": percentile(diffs, CI[1]),
    }
    return out


CHESS_MOVES = ("key", "deep_alt", "greedy", "random")


def chess_detector_intervals(rng: random.Random) -> Dict[str, Dict]:
    """Recompute per-position rank-first flags with the exact detector
    definitions of chess_prior_detectors.py, then bootstrap the rates."""
    rows = list(csv.DictReader(
        (OUTPUTS / "chess_collapse_main_positions.csv").open(encoding="utf-8")))
    sac = [r for r in rows if r["kind"] == "sacrifice"]
    detectors = {
        "collapse_only": lambda r, m: float(r[f"collapse_{m}_bits"]),
        "specificity_only": lambda r, m: abs(float(r[f"useful_shift_{m}"])),
        "local_value": lambda r, m: float(r[f"local_cost_{m}"]),
        "useful_collapse": lambda r, m: float(r[f"useful_shift_{m}"]),
    }
    out: Dict[str, Dict] = {}
    for name, fn in detectors.items():
        flags: List[float] = []
        for r in sac:
            try:
                scores = {m: fn(r, m) for m in CHESS_MOVES if r.get(f"{m}_uci")}
            except (KeyError, ValueError):
                continue
            if "key" not in scores or len(scores) < 3:
                continue
            first = all(scores["key"] >= v for k, v in scores.items() if k != "key")
            flags.append(1.0 if first else 0.0)
        out[f"rank_first_{name}"] = boot_ci(flags, mean, rng)
    return out


def marl_intervals(rng: random.Random, path: Path, label: str,
                   prefix: str = "trained_") -> Dict[str, Dict]:
    data = json.loads(path.read_text())
    conditions = data["conditions"]
    gaps: List[float] = []
    per_seed: Dict[str, float] = {}
    for name, cond in conditions.items():
        if not name.startswith(prefix):
            continue
        seed_gaps = [e["p_win_do_commit"] - e["p_win_do_block"]
                     for e in cond["episodes"] if "p_win_do_commit" in e]
        gaps.extend(seed_gaps)
        if seed_gaps:
            per_seed[name] = median(seed_gaps)
    out = {
        f"{label}_do_gap_median_pooled": boot_ci(gaps, median, rng),
        f"{label}_do_gap_mean_pooled": boot_ci(gaps, mean, rng),
    }
    out[f"{label}_do_gap_median_per_seed"] = {"point": per_seed}
    return out


def main() -> None:
    rng = random.Random(SEED)
    results: Dict[str, Dict] = {}
    results["chess"] = chess_intervals(rng)

    det_path = OUTPUTS / "chess_prior_detectors.json"
    if det_path.exists():
        det = chess_detector_intervals(rng)
        if det:
            results["chess_prior_detectors"] = det

    marl_path = OUTPUTS / "deep_marl_collapse_mappo_seed11.json"
    agg: Dict[str, Dict] = {}
    pooled_gaps: List[float] = []
    per_seed: Dict[str, float] = {}
    for seed in (11, 22, 33):
        p = OUTPUTS / f"deep_marl_collapse_mappo_seed{seed}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for name, cond in data["conditions"].items():
            if not name.startswith("trained_"):
                continue
            seed_gaps = [e["p_win_do_commit"] - e["p_win_do_block"]
                         for e in cond["episodes"] if "p_win_do_commit" in e]
            pooled_gaps.extend(seed_gaps)
            if seed_gaps:
                per_seed[name] = median(seed_gaps)
    if pooled_gaps:
        agg["do_gap_median_pooled"] = boot_ci(pooled_gaps, median, rng)
        agg["do_gap_mean_pooled"] = boot_ci(pooled_gaps, mean, rng)
        agg["do_gap_median_per_seed"] = {"point": per_seed}
        results["deep_marl_simple_spread"] = agg

    lbf_path = OUTPUTS / "lbf_collapse_main.json"
    if lbf_path.exists():
        results["deep_marl_lbf"] = marl_intervals(rng, lbf_path, "lbf")

    (OUTPUTS / "bootstrap_intervals.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    for domain, entries in results.items():
        print(f"== {domain}")
        for key, ci in entries.items():
            if "ci_lo" in ci:
                print(f"  {key:45s} {ci['point']:+.4f}  "
                      f"[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]  n={ci['n']}")
            else:
                print(f"  {key:45s} {ci['point']}")
    print(f"Wrote {OUTPUTS / 'bootstrap_intervals.json'}")


if __name__ == "__main__":
    main()
