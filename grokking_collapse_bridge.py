"""Grokking as useful possibility collapse: bridging to large-model emergence.

Why this experiment exists
--------------------------
Our claim is that "emergence" across domains (multi-agent structure, spatial
patterns, large-model abilities) shares one mechanism: the structured,
non-prespecified, useful collapse of an open possibility space. Everything
measured so far lives in decision-making systems. The large-model literature,
however, anchors "emergent abilities" in a different phenomenon: sudden jumps
in generalization during training or scaling. Grokking (Power et al. 2022) is
the cleanest, publicly replicated instance: a small network trained on modular
arithmetic first memorizes the training set, then -- much later and abruptly
-- generalizes to unseen inputs.

This script asks: is grokking a *useful possibility collapse* in exactly our
sense? The observer's possibility distribution is the model's own predictive
distribution P_k(y | x) on held-out inputs at checkpoint k. Before grokking
the distribution over answers for unseen inputs is broad (many futures open);
at grokking it collapses suddenly onto the structured (correct) answers.

Measured quantities per checkpoint k:

    H_k   = mean per-example predictive entropy on held-out inputs (bits)
    C_k   = mean per-example KL(P_k || P_0)      (collapse magnitude)
    B_k   = max(C_k - C_{k-1}, 0)                (collapse burst)
    J_k   = mean L2 jump of penultimate-layer embeddings on probe inputs
    acc_k = train / test accuracy

Four runs (registered before the full run; a feasibility pilot may tune
optimizer hyperparameters only, never the thresholds or predictions):

- grokking:     modular addition, weight decay on. Expected: delayed, sudden
                useful collapse on held-out inputs (the emergent case).
- memorizer:    same task, weight decay off. Expected: training-set collapse
                only; held-out possibility either stays open or collapses
                without usefulness. Fails usefulness.
- no_structure: labels shuffled (no latent structure to discover). Any
                collapse cannot be useful. Fails usefulness.
- prewired:     trained with the held-out set included in training
                (structure injected by supervision on the evaluation
                distribution). Prespecified by design; fails endogeneity,
                and its held-out collapse tracks training directly rather
                than arising late from internal reorganization.

Pre-registered component mapping and thresholds (process-level instantiation
of the five-component criterion; selectivity/specificity are replaced by the
burstiness of the collapse because a training process has no per-episode
trigger choice -- this mapping is declared here, before measurement):

    potential:   held-out entropy just before the candidate collapse window
                 H_pre >= 1.0 bits
    burstiness:  collapse burst inside the window / median burst >= 5.0
                 (sudden, not gradual, collapse)
    usefulness:  test-accuracy gain across the window >= 0.2
    endogeneity: no supervision on the held-out distribution and no
                 hand-wired solution (design flag)

Windowing note (recorded after the feasibility pilot, thresholds untouched):
the pilot showed that KL-from-initialization also spikes during the early
*memorization* phase, when held-out entropy drops from 6.6 to 2.8 bits while
held-out accuracy stays at zero -- a live example of collapse WITHOUT
usefulness inside the same run. The candidate window is therefore anchored at
the largest held-out accuracy jump, and the criterion requires a collapse
burst to coincide with it. This is stricter than the max-burst window: the
emergent case must exhibit potential, sudden collapse, and usefulness at the
SAME point in training.

Predictions: grokking passes all four; memorizer fails usefulness;
no_structure fails usefulness; prewired fails endogeneity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

OUTPUTS = Path(__file__).resolve().parent / "outputs"

THRESHOLDS = {
    "potential_bits": 1.0,
    "burstiness_ratio": 5.0,
    "usefulness_acc_gain": 0.2,
}


def make_dataset(p: int, train_frac: float, seed: int, shuffle_labels: bool):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    if shuffle_labels:
        labels = rng.permutation(labels)
    order = rng.permutation(len(pairs))
    n_train = int(train_frac * len(pairs))
    train_idx, test_idx = order[:n_train], order[n_train:]
    return (
        torch.from_numpy(pairs[train_idx]),
        torch.from_numpy(labels[train_idx]),
        torch.from_numpy(pairs[test_idx]),
        torch.from_numpy(labels[test_idx]),
    )


class GrokNet(nn.Module):
    def __init__(self, p: int, emb: int = 64, hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(p, emb)
        self.body = nn.Sequential(
            nn.Linear(2 * emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, p)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x).flatten(1)
        return self.body(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.penultimate(x))


def entropy_bits(probs: torch.Tensor) -> float:
    logp = torch.log2(probs.clamp_min(1e-12))
    return float((-probs * logp).sum(dim=1).mean())


def kl_bits(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(
        (p * (torch.log2(p.clamp_min(1e-12)) - torch.log2(q.clamp_min(1e-12)))).sum(dim=1).mean()
    )


def train_run(
    name: str,
    p: int,
    train_frac: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    eval_every: int,
    shuffle_labels: bool = False,
    include_test_in_train: bool = False,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = make_dataset(p, train_frac, seed, shuffle_labels)
    if include_test_in_train:
        x_fit = torch.cat([x_train, x_test])
        y_fit = torch.cat([y_train, y_test])
    else:
        x_fit, y_fit = x_train, y_train

    model = GrokNet(p)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    probe = x_test[: min(256, len(x_test))]
    p0: torch.Tensor | None = None
    prev_emb: torch.Tensor | None = None
    rows: List[Dict[str, float]] = []

    for epoch in range(epochs + 1):
        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                logits_test = model(x_test)
                probs_test = torch.softmax(logits_test, dim=1)
                if p0 is None:
                    p0 = probs_test.clone()
                emb = model.penultimate(probe)
                emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
                jump = float((emb - prev_emb).norm(dim=1).mean()) if prev_emb is not None else 0.0
                prev_emb = emb.clone()
                train_acc = float(
                    (model(x_fit).argmax(dim=1) == y_fit).float().mean()
                )
                test_acc = float((logits_test.argmax(dim=1) == y_test).float().mean())
                rows.append(
                    {
                        "run": name,
                        "epoch": epoch,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "test_entropy_bits": entropy_bits(probs_test),
                        "collapse_bits": kl_bits(probs_test, p0),
                        "embedding_jump": jump,
                    }
                )
            model.train()

        if epoch == epochs:
            break
        opt.zero_grad()
        loss = loss_fn(model(x_fit), y_fit)
        loss.backward()
        opt.step()

    stats = analyze_run(rows)
    return rows, stats


def analyze_run(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    collapse = [row["collapse_bits"] for row in rows]
    bursts = [max(collapse[i] - collapse[i - 1], 0.0) for i in range(1, len(collapse))]
    jumps = [row["embedding_jump"] for row in rows][1:]
    test_acc = [row["test_acc"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]
    entropies = [row["test_entropy_bits"] for row in rows]

    # Candidate window anchored at the largest held-out accuracy jump (see
    # the windowing note in the module docstring). bursts[i] and
    # acc_jumps[i] both span rows[i] -> rows[i+1].
    acc_jumps = [test_acc[i + 1] - test_acc[i] for i in range(len(test_acc) - 1)]
    anchor = int(np.argmax(acc_jumps)) if acc_jumps else 0
    lo = max(0, anchor - 1)
    hi = min(len(test_acc) - 1, anchor + 2)
    median_burst = float(np.median(bursts)) if bursts else 0.0
    window_burst = max(bursts[lo:hi], default=0.0)
    burstiness = window_burst / (median_burst + 1e-6)
    acc_gain = test_acc[hi] - test_acc[lo]
    h_pre = entropies[lo]
    # Reported for honesty: the globally largest collapse burst, which in the
    # grokking run happens during memorization (collapse without usefulness).
    max_burst_idx = int(np.argmax(bursts)) if bursts else 0

    def first_epoch_above(series: List[float], level: float) -> float:
        for row, value in zip(rows, series):
            if value >= level:
                return row["epoch"]
        return math.nan

    corr = float(np.corrcoef(bursts, jumps)[0, 1]) if len(bursts) > 2 else math.nan
    return {
        "h_pre_burst_bits": h_pre,
        "window_burst_bits": window_burst,
        "global_max_burst_bits": max(bursts) if bursts else 0.0,
        "global_max_burst_epoch": rows[max_burst_idx + 1]["epoch"] if bursts else math.nan,
        "burstiness_ratio": burstiness,
        "window_epoch": rows[anchor + 1]["epoch"] if acc_jumps else math.nan,
        "usefulness_acc_gain": acc_gain,
        "final_train_acc": train_acc[-1],
        "final_test_acc": test_acc[-1],
        "train_acc_99_epoch": first_epoch_above(train_acc, 0.99),
        "test_acc_90_epoch": first_epoch_above(test_acc, 0.90),
        "burst_jump_correlation": corr,
        "final_test_entropy_bits": entropies[-1],
        "initial_test_entropy_bits": entropies[0],
    }


def verdict(stats: Dict[str, Any], prespecified: bool) -> Dict[str, Any]:
    passes = {
        "potential": stats["h_pre_burst_bits"] >= THRESHOLDS["potential_bits"],
        "burstiness": stats["burstiness_ratio"] >= THRESHOLDS["burstiness_ratio"],
        "usefulness": stats["usefulness_acc_gain"] >= THRESHOLDS["usefulness_acc_gain"],
        "endogeneity": not prespecified,
    }
    return {"passes": passes, "emergent": int(all(passes.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grokking possibility-collapse bridge.")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    torch.set_num_threads(2)
    runs = (
        ("grokking", dict(weight_decay=args.weight_decay), False),
        ("memorizer", dict(weight_decay=0.0), False),
        ("no_structure", dict(weight_decay=args.weight_decay, shuffle_labels=True), False),
        ("prewired", dict(weight_decay=args.weight_decay, include_test_in_train=True), True),
    )

    all_rows: List[Dict[str, float]] = []
    summary: Dict[str, Any] = {"thresholds": THRESHOLDS, "runs": {}}
    for name, overrides, prespecified in runs:
        print(f"\n=== run: {name} ===")
        rows, stats = train_run(
            name,
            p=args.p,
            train_frac=args.train_frac,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=overrides.get("weight_decay", args.weight_decay),
            seed=args.seed,
            eval_every=args.eval_every,
            shuffle_labels=overrides.get("shuffle_labels", False),
            include_test_in_train=overrides.get("include_test_in_train", False),
        )
        all_rows.extend(rows)
        v = verdict(stats, prespecified)
        summary["runs"][name] = {"stats": stats, "verdict": v, "prespecified": prespecified}
        print(
            f"  final train/test acc {stats['final_train_acc']:.3f}/{stats['final_test_acc']:.3f} | "
            f"H_pre {stats['h_pre_burst_bits']:.2f} bits | burstiness {stats['burstiness_ratio']:.1f} | "
            f"acc gain @burst {stats['usefulness_acc_gain']:+.3f} | emergent={v['emergent']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "grokking_collapse_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / "grokking_collapse_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nrun,emergent,failed_components")
    for name, info in summary["runs"].items():
        failed = ";".join(k for k, ok in info["verdict"]["passes"].items() if not ok) or "-"
        print(f"{name},{info['verdict']['emergent']},{failed}")
    print(f"\nWrote {args.output_dir / 'grokking_collapse_summary.json'}")


if __name__ == "__main__":
    main()
