"""Independent cross-implementation check of the trajectory-KL machinery.

The chain-rule computation in trajectory_basin_coupling.py is validated
against brute-force enumeration of every finite path on small synthetic
absorbing chains (same duck-typed interface: clocked nonterminal states, so
each path visits a state at most once). Both implementations must agree to
machine precision over many random chain pairs, including cases with
disjoint-support rows (infinite KL) and identical rows (zero KL).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from trajectory_basin_coupling import trajectory_kl_bits

OUTPUTS = Path(__file__).resolve().parent / "outputs"

N_STATES = 4        # per clock tick
N_STEPS = 5
N_TRIALS = 2000


class TinyChain:
    """Minimal object satisfying the trajectory_kl_bits interface."""

    def __init__(self, rows, init, n_nonterminal, states, index):
        self.rows = rows
        self.init = init
        self.n_nonterminal = n_nonterminal
        self.states = states
        self.index = index


def build_pair(rng: random.Random, force_zero: bool = False,
               force_singular: bool = False) -> Tuple[TinyChain, TinyChain]:
    """Two chains sharing a state space: clocked layers then one absorber."""
    states = []
    for t in range(N_STEPS):
        for k in range(N_STATES):
            states.append(("s", 0, 0, 0, 0, t, k))  # clock at position 5
    n_nonterminal = len(states)
    states.append(("basin", "end"))
    index = {s: i for i, s in enumerate(states)}

    def random_rows():
        rows = []
        for i in range(n_nonterminal):
            t = states[i][5]
            if t + 1 < N_STEPS:
                succ = [index[("s", 0, 0, 0, 0, t + 1, k)]
                        for k in range(N_STATES)]
            else:
                succ = [n_nonterminal]
            weights = [rng.random() + 0.05 for _ in succ]
            total = sum(weights)
            rows.append([(j, w / total) for j, w in zip(succ, weights)])
        rows.append([(n_nonterminal, 1.0)])
        return rows

    rows_p = random_rows()
    if force_zero:
        rows_q = [list(r) for r in rows_p]
    elif force_singular:
        rows_q = random_rows()
        # remove one successor from Q's support that P uses
        i = rng.randrange(N_STATES)  # a t=0 state, always visited region
        full = rows_q[i]
        if len(full) > 1:
            kept = full[1:]
            total = sum(w for _, w in kept)
            rows_q[i] = [(j, w / total) for j, w in kept]
    else:
        rows_q = random_rows()

    init = {index[("s", 0, 0, 0, 0, 0, k)]: 1.0 / N_STATES
            for k in range(N_STATES)}
    make = lambda rows: TinyChain(rows, init, n_nonterminal, states, index)
    return make(rows_p), make(rows_q)


def enumerate_paths_kl(chain_p: TinyChain, chain_q: TinyChain) -> float:
    """Brute-force sum over every finite path from init to absorption."""
    total = 0.0

    def q_prob(path: List[int]) -> float:
        prob = chain_q.init.get(path[0], 0.0)
        for a, b in zip(path, path[1:]):
            step = dict(chain_q.rows[a]).get(b, 0.0)
            prob *= step
        return prob

    stack: List[Tuple[List[int], float]] = [
        ([i], w) for i, w in chain_p.init.items()]
    while stack:
        path, p_prob = stack.pop()
        last = path[-1]
        if last >= chain_p.n_nonterminal:
            if p_prob > 0:
                q = q_prob(path)
                if q <= 0:
                    return float("inf")
                total += p_prob * math.log2(p_prob / q)
            continue
        for j, w in chain_p.rows[last]:
            stack.append((path + [j], p_prob * w))
    return total


def main() -> None:
    rng = random.Random(20260716)
    max_gap = 0.0
    singular_ok = zero_ok = 0
    for trial in range(N_TRIALS):
        kind = trial % 10
        p, q = build_pair(rng, force_zero=(kind == 8),
                          force_singular=(kind == 9))
        chain_rule = trajectory_kl_bits(p, q)
        brute = enumerate_paths_kl(p, q)
        if math.isinf(brute) or math.isinf(chain_rule):
            assert math.isinf(brute) == math.isinf(chain_rule), \
                f"singularity mismatch at trial {trial}"
            singular_ok += 1
            continue
        gap = abs(chain_rule - brute)
        max_gap = max(max_gap, gap)
        if kind == 8:
            assert brute < 1e-12
            zero_ok += 1
    report = {
        "status": "cross-implementation validation "
                  "(chain rule vs full path enumeration)",
        "n_trials": N_TRIALS,
        "n_states_per_layer": N_STATES,
        "n_steps": N_STEPS,
        "max_abs_gap_bits": max_gap,
        "singular_cases_agreeing": singular_ok,
        "zero_kl_cases_verified": zero_ok,
        "pass": max_gap < 1e-9,
    }
    out = OUTPUTS / "trajectory_kl_implementation_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"max |chain rule - enumeration| = {max_gap:.3e} bits over "
          f"{N_TRIALS} trials ({singular_ok} singular, {zero_ok} zero-KL)")
    print(f"Wrote {out}")
    assert report["pass"]


if __name__ == "__main__":
    main()
