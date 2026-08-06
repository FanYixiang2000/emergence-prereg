"""Generality of the grokking-collapse result: second task, multiple seeds.

fig23 established, on one task (modular addition) and one seed, that
grokking is a useful, sudden, endogenous possibility collapse, and that the
registered controls fail the registered components. A single task/seed is
anecdote. This sweep repeats the grokking and memorizer conditions on:

    tasks:  modular addition  (a + b) mod p          [original]
            modular multiplication (a * b) mod p, a,b in [1, p-1]  [new]
    seeds:  {1234, 2345, 3456}

Registered expectations (thresholds unchanged from grokking_collapse_bridge):

    G1: the grokking condition is classified emergent (all four components)
        on every task x seed cell.
    G2: the memorizer condition fails usefulness on every cell.
    G3: in every grokking cell the train-accuracy epoch (>= 0.99) precedes
        the test-accuracy epoch (>= 0.90) by at least 3x (delayed
        generalization, the grokking signature).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from grokking_collapse_bridge import (
    THRESHOLDS,
    GrokNet,
    analyze_run,
    entropy_bits,
    kl_bits,
    verdict,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def make_dataset_op(p: int, op: str, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    if op == "add":
        pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
        labels = (pairs[:, 0] + pairs[:, 1]) % p
    elif op == "mul":
        pairs = np.array(
            [(a, b) for a in range(1, p) for b in range(1, p)], dtype=np.int64
        )
        labels = (pairs[:, 0] * pairs[:, 1]) % p
    else:
        raise ValueError(op)
    order = rng.permutation(len(pairs))
    n_train = int(train_frac * len(pairs))
    train_idx, test_idx = order[:n_train], order[n_train:]
    return (
        torch.from_numpy(pairs[train_idx]),
        torch.from_numpy(labels[train_idx]),
        torch.from_numpy(pairs[test_idx]),
        torch.from_numpy(labels[test_idx]),
    )


def run_cell(op: str, condition: str, p: int, train_frac: float, epochs: int,
             lr: float, weight_decay: float, seed: int, eval_every: int):
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = make_dataset_op(p, op, train_frac, seed)
    model = GrokNet(p)
    wd = weight_decay if condition == "grokking" else 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    probe = x_test[: min(256, len(x_test))]
    p0 = None
    prev_emb = None
    rows: List[Dict[str, float]] = []
    for epoch in range(epochs + 1):
        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x_test)
                probs = torch.softmax(logits, dim=1)
                if p0 is None:
                    p0 = probs.clone()
                emb = model.penultimate(probe)
                emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
                jump = float((emb - prev_emb).norm(dim=1).mean()) if prev_emb is not None else 0.0
                prev_emb = emb.clone()
                rows.append({
                    "run": f"{op}_{condition}_{seed}",
                    "epoch": epoch,
                    "train_acc": float((model(x_train).argmax(1) == y_train).float().mean()),
                    "test_acc": float((logits.argmax(1) == y_test).float().mean()),
                    "test_entropy_bits": entropy_bits(probs),
                    "collapse_bits": kl_bits(probs, p0),
                    "embedding_jump": jump,
                })
            model.train()
        if epoch == epochs:
            break
        opt.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        opt.step()
    return rows, analyze_run(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grokking generality sweep.")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=150)
    parser.add_argument("--seeds", type=str, default="1234,2345,3456")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    torch.set_num_threads(2)
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = []
    all_rows: List[Dict[str, float]] = []
    for op in ("add", "mul"):
        for seed in seeds:
            for condition in ("grokking", "memorizer"):
                print(f"=== {op} / {condition} / seed {seed} ===")
                rows, stats = run_cell(
                    op, condition, args.p, args.train_frac, args.epochs,
                    args.lr, args.weight_decay, seed, args.eval_every,
                )
                all_rows.extend(rows)
                v = verdict(stats, prespecified=False)
                cells.append({
                    "op": op, "condition": condition, "seed": seed,
                    "stats": stats, "verdict": v,
                })
                print(
                    f"  test {stats['final_test_acc']:.3f} burstiness "
                    f"{stats['burstiness_ratio']:.1f} gain {stats['usefulness_acc_gain']:+.3f} "
                    f"emergent={v['emergent']}"
                )

    grok_cells = [c for c in cells if c["condition"] == "grokking"]
    mem_cells = [c for c in cells if c["condition"] == "memorizer"]
    checks = {
        "g1_grokking_emergent_all_cells": all(c["verdict"]["emergent"] == 1 for c in grok_cells),
        "g2_memorizer_fails_usefulness_all_cells": all(
            not c["verdict"]["passes"]["usefulness"] for c in mem_cells
        ),
        "g3_delayed_generalization_all_cells": all(
            (c["stats"]["test_acc_90_epoch"] or 0)
            >= 3 * max(c["stats"]["train_acc_99_epoch"], 1)
            for c in grok_cells
            if not np.isnan(c["stats"]["test_acc_90_epoch"])
        ) and all(
            not np.isnan(c["stats"]["test_acc_90_epoch"]) for c in grok_cells
        ),
    }
    summary = {
        "thresholds": THRESHOLDS,
        "checks": checks,
        "cells": [
            {
                "op": c["op"], "condition": c["condition"], "seed": c["seed"],
                "emergent": c["verdict"]["emergent"],
                "failed": [k for k, ok in c["verdict"]["passes"].items() if not ok],
                "final_test_acc": c["stats"]["final_test_acc"],
                "burstiness_ratio": c["stats"]["burstiness_ratio"],
                "usefulness_acc_gain": c["stats"]["usefulness_acc_gain"],
                "train_acc_99_epoch": c["stats"]["train_acc_99_epoch"],
                "test_acc_90_epoch": c["stats"]["test_acc_90_epoch"],
            }
            for c in cells
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "grokking_generality_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / "grokking_generality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nchecks:")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"Wrote {args.output_dir / 'grokking_generality_summary.json'}")


if __name__ == "__main__":
    main()
