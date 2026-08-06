"""Scale emergence decomposed: is the ability jump also a collapse jump?

The debate this addresses
-------------------------
Emergent abilities of large models are reported as sharp accuracy jumps at
some scale (Wei et al. 2022). Schaeffer et al. (2023) countered that the
jumps are artifacts of discontinuous metrics (exact match): under continuous
metrics the underlying competence grows smoothly, so "emergence is a mirage".

Our framework makes the disagreement measurable instead of rhetorical. The
model's predictive distribution on held-out inputs IS a possibility
distribution; its collapse C (KL from initialization, in bits) is a
continuous, mechanism-level quantity. At every scale we measure both:

    discontinuous observable: exact-match accuracy on held-out inputs
    continuous mechanism:     held-out collapse C and entropy drop

Two registered, distinguishable hypotheses (decided by data, not by us):

    H_smooth: collapse grows smoothly with scale while accuracy jumps ->
              the ability jump is a threshold effect on top of a continuous
              mechanism (Schaeffer-consistent, and our framework supplies
              the continuous latent quantity that the metric hides).
    H_jump:   collapse itself jumps at the same scale as accuracy ->
              genuine mechanism-level emergence at a critical scale
              (Wei-consistent).

Either outcome is informative; the framework is what lets the question be
posed quantitatively. Additionally registered:

    R1: accuracy-vs-scale has a sharper transition than collapse-vs-scale,
        quantified by the max single-step jump of each curve after min-max
        normalization over the scale grid.
    R2: at every scale, held-out collapse is necessary for held-out
        accuracy (no scale shows high accuracy with low collapse);
        concretely acc <= collapse_normalized + 0.15 at every scale.

Protocol: modular addition (p = 97, train fraction 0.5, AdamW wd = 1.0,
8000 epochs -- the fig23 grokking recipe), width scale factor
s in {1/16, 1/8, 1/4, 1/2, 1, 2} applied to both embedding (64 s) and
hidden (256 s), 2 seeds per scale. Measured at the final checkpoint.
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

from grokking_collapse_bridge import entropy_bits, kl_bits, make_dataset

OUTPUTS = Path(__file__).resolve().parent / "outputs"


class ScaledNet(nn.Module):
    def __init__(self, p: int, emb: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(p, emb)
        self.body = nn.Sequential(
            nn.Linear(2 * emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.embed(x).flatten(1)))


def train_at_scale(p: int, scale: float, seed: int, epochs: int,
                   lr: float, weight_decay: float, train_frac: float) -> Dict[str, Any]:
    torch.manual_seed(seed)
    emb = max(2, round(64 * scale))
    hidden = max(4, round(256 * scale))
    x_train, y_train, x_test, y_test = make_dataset(p, train_frac, seed, False)
    model = ScaledNet(p, emb, hidden)
    n_params = sum(param.numel() for param in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        p0 = torch.softmax(model(x_test), dim=1)
    h0 = entropy_bits(p0)

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        probs = torch.softmax(logits, dim=1)
        test_acc = float((logits.argmax(dim=1) == y_test).float().mean())
        train_acc = float((model(x_train).argmax(dim=1) == y_train).float().mean())
        collapse = kl_bits(probs, p0)
        h_final = entropy_bits(probs)
        # Continuous competence metric (Schaeffer-style): mean log-prob of
        # the correct answer, in bits.
        correct_logp = float(
            torch.log2(
                probs.gather(1, y_test.unsqueeze(1)).clamp_min(1e-12)
            ).mean()
        )
    return {
        "scale": scale,
        "seed": seed,
        "n_params": n_params,
        "emb": emb,
        "hidden": hidden,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "collapse_bits": collapse,
        "initial_entropy_bits": h0,
        "final_entropy_bits": h_final,
        "correct_logprob_bits": correct_logp,
    }


def max_normalized_jump(values: List[float]) -> float:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return 0.0
    norm = [(v - lo) / (hi - lo) for v in values]
    return max(abs(norm[i + 1] - norm[i]) for i in range(len(norm) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale-emergence decomposition.")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--scales", type=str, default="0.0625,0.125,0.25,0.5,1.0,2.0")
    parser.add_argument("--seeds", type=str, default="1234,2345")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    torch.set_num_threads(2)
    scales = [float(s) for s in args.scales.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    rows: List[Dict[str, Any]] = []
    for scale in scales:
        for seed in seeds:
            row = train_at_scale(
                args.p, scale, seed, args.epochs, args.lr,
                args.weight_decay, args.train_frac,
            )
            rows.append(row)
            print(
                f"scale {scale:7.4f} seed {seed} params {row['n_params']:8d} "
                f"train {row['train_acc']:.3f} test {row['test_acc']:.3f} "
                f"C {row['collapse_bits']:.2f} logp {row['correct_logprob_bits']:.2f}"
            )

    # Seed-averaged curves over the scale grid.
    acc_curve = [
        float(np.mean([r["test_acc"] for r in rows if r["scale"] == s])) for s in scales
    ]
    collapse_curve = [
        float(np.mean([r["collapse_bits"] for r in rows if r["scale"] == s])) for s in scales
    ]
    logp_curve = [
        float(np.mean([r["correct_logprob_bits"] for r in rows if r["scale"] == s]))
        for s in scales
    ]
    acc_jump = max_normalized_jump(acc_curve)
    collapse_jump = max_normalized_jump(collapse_curve)
    lo, hi = min(collapse_curve), max(collapse_curve)
    collapse_norm = [(c - lo) / (hi - lo + 1e-9) for c in collapse_curve]
    r2_ok = all(a <= cn + 0.15 for a, cn in zip(acc_curve, collapse_norm))

    verdict = "H_smooth" if acc_jump > collapse_jump + 0.15 else (
        "H_jump" if collapse_jump > acc_jump + 0.15 else "comparable"
    )
    summary = {
        "registered_hypotheses": ["H_smooth", "H_jump"],
        "scales": scales,
        "acc_curve": acc_curve,
        "collapse_curve": collapse_curve,
        "logp_curve": logp_curve,
        "max_normalized_jump_accuracy": acc_jump,
        "max_normalized_jump_collapse": collapse_jump,
        "R1_accuracy_sharper_than_collapse": acc_jump > collapse_jump,
        "R2_collapse_necessary_for_accuracy": r2_ok,
        "data_verdict": verdict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scale_emergence_grid.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (args.output_dir / "scale_emergence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nmax normalized jump: accuracy {acc_jump:.3f} vs collapse {collapse_jump:.3f}")
    print(f"R1 accuracy sharper: {summary['R1_accuracy_sharper_than_collapse']}")
    print(f"R2 collapse necessary: {r2_ok}")
    print(f"data verdict: {verdict}")
    print(f"Wrote {args.output_dir / 'scale_emergence_summary.json'}")


if __name__ == "__main__":
    main()
