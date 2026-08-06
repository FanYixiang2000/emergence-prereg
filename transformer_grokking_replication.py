"""Transformer replication of the grokking-collapse result.

The MLP grokking bridge (grokking_collapse_bridge.py, fig23/fig27) could be
an architecture artifact. This script repeats the grokking and memorizer
conditions with a small causal transformer (the architecture family of the
original grokking report, Power et al. 2022) on modular addition, sequences
"a b -> c". Registered expectations are unchanged from the bridge module:
grokking passes all four process-level components; memorizer fails
usefulness. Thresholds are imported, not redefined.

Pilot note (optimizer only; thresholds and predictions untouched): a first
pilot used weight_decay = 1.0 and lr = 1e-3, the MLP-bridge setting. The
transformer grokked cleanly (held-out accuracy 0 -> 1.0 by epoch 4500) but
then exhibited repeated "slingshot" destabilizations (accuracy collapsing
back to chance and re-grokking), a known optimization artifact of aggressive
weight decay on transformers. Those crashes corrupted the accuracy-jump
window used by analyze_run. Per the registration, optimizer hyperparameters
(only) were retuned to weight_decay = 0.5, lr = 5e-4 for a stable grok; the
raw unstable pilot log is kept at
outputs/transformer_grokking_log_wd1_pilot.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from grokking_collapse_bridge import (
    THRESHOLDS,
    analyze_run,
    entropy_bits,
    kl_bits,
    make_dataset,
    verdict,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, seq_len: int = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(x) + self.pos_emb[: x.shape[1]]
        h = self.encoder(h)
        return self.norm(h[:, -1])  # representation at the answer position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.penultimate(x))


def train_run(name: str, p: int, train_frac: float, epochs: int, lr: float,
              weight_decay: float, seed: int, eval_every: int):
    torch.manual_seed(seed)
    x_train, y_train, x_test, y_test = make_dataset(p, train_frac, seed, False)
    model = TinyTransformer(vocab=p)
    n_params = sum(t.numel() for t in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                    "run": name,
                    "epoch": epoch,
                    "train_acc": float((model(x_train).argmax(1) == y_train).float().mean()),
                    "test_acc": float((logits.argmax(1) == y_test).float().mean()),
                    "test_entropy_bits": entropy_bits(probs),
                    "collapse_bits": kl_bits(probs, p0),
                    "embedding_jump": jump,
                })
                r = rows[-1]
                if epoch % (eval_every * 10) == 0:
                    print(f"  epoch {epoch:6d} train {r['train_acc']:.3f} "
                          f"test {r['test_acc']:.3f} H {r['test_entropy_bits']:.2f}")
            model.train()
        if epoch == epochs:
            break
        opt.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        opt.step()
    return rows, analyze_run(rows), n_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer grokking replication.")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    torch.set_num_threads(4)
    all_rows: List[Dict[str, float]] = []
    summary: Dict[str, Dict] = {"thresholds": THRESHOLDS, "runs": {}}
    for name, wd, prespecified in (
        ("transformer_grokking", args.weight_decay, False),
        ("transformer_memorizer", 0.0, False),
    ):
        print(f"=== {name} ===")
        rows, stats, n_params = train_run(
            name, args.p, args.train_frac, args.epochs, args.lr, wd,
            args.seed, args.eval_every,
        )
        all_rows.extend(rows)
        v = verdict(stats, prespecified)
        summary["runs"][name] = {
            "stats": stats, "verdict": v, "n_params": n_params,
        }
        print(f"  -> emergent={v['emergent']} "
              f"failed={';'.join(k for k, ok in v['passes'].items() if not ok) or '-'} "
              f"({n_params} params)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "transformer_grokking_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / "transformer_grokking_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'transformer_grokking_summary.json'}")


if __name__ == "__main__":
    main()
