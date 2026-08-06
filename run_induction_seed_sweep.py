"""Multi-seed replication of the induction-head possibility-collapse result.

Repeats all four registered conditions of induction_head_emergence.py on
fresh seeds (the registered run used seed 7; the sweep uses 101, 202, 303).
Predictions are unchanged and frozen: induction_2layer passes the criterion
on every seed; induction_1layer, no_structure, and memorizer fail on every
seed. Any deviation is reported, not patched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from grokking_collapse_bridge import THRESHOLDS, verdict
from induction_head_emergence import train_run

OUTPUTS = Path(__file__).resolve().parent / "outputs"

EXPECTED = {
    "induction_2layer": 1,
    "induction_1layer": 0,
    "no_structure": 0,
    "memorizer": 0,
}

CONDITIONS = {
    "induction_2layer": dict(n_layers=2, structured=True, memorize=False),
    "induction_1layer": dict(n_layers=1, structured=True, memorize=False),
    "no_structure": dict(n_layers=2, structured=False, memorize=False),
    "memorizer": dict(n_layers=2, structured=True, memorize=True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Induction-head seed sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(4)

    summary: Dict[str, Dict] = {"thresholds": THRESHOLDS, "seeds": {}}
    n_checks = 0
    n_correct = 0
    for seed in args.seeds:
        summary["seeds"][str(seed)] = {}
        for name, cfg in CONDITIONS.items():
            rows, stats, n_params = train_run(
                name, cfg["n_layers"], cfg["structured"], cfg["memorize"],
                vocab=64, seq_len=64, k_min=8, k_max=32, d_model=64,
                n_heads=4, steps=args.steps, batch=64, lr=1e-3, seed=seed,
                eval_every=60,
            )
            v = verdict(stats, prespecified=False)
            match = int(v["emergent"] == EXPECTED[name])
            n_checks += 1
            n_correct += match
            summary["seeds"][str(seed)][name] = {
                "stats": stats,
                "verdict": v,
                "expected": EXPECTED[name],
                "match": match,
            }
            failed = ";".join(k for k, ok in v["passes"].items() if not ok) or "-"
            print(f"seed {seed} {name:18s} emergent={v['emergent']} "
                  f"expected={EXPECTED[name]} match={match} failed={failed}",
                  flush=True)
    summary["n_checks"] = n_checks
    summary["n_correct"] = n_correct
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "induction_seed_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"TOTAL {n_correct}/{n_checks} registered predictions correct")
    print(f"Wrote {args.output_dir / 'induction_seed_sweep_summary.json'}")


if __name__ == "__main__":
    main()
