"""Command-line runner for the learned sacrifice PTC experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from learned_sacrifice_gridworld import REGIMES, run_all_learned


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train tabular policies and measure Potential-Trigger-Collapse."
    )
    parser.add_argument("--train_episodes", type=int, default=8000)
    parser.add_argument("--eval_episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eval_temperature", type=float, default=0.35)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=list(REGIMES.keys()),
        choices=list(REGIMES.keys()),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all_learned(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        eval_temperature=args.eval_temperature,
        output_dir=args.output_dir,
        regimes=args.regimes,
    )
    print(f"\nWrote {args.output_dir / 'learned_ptc_results.json'}")
    print(f"Wrote {args.output_dir / 'learned_ptc_summary.csv'}")


if __name__ == "__main__":
    main()
