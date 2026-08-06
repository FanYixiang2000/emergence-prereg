"""Induction-head formation as useful possibility collapse.

Why this experiment exists
--------------------------
Every phenomenon measured so far was either designed by us (gridworlds,
swarm) or is a training-dynamics effect we chose to study (grokking). The
strongest remaining objection is that the criterion has never been pointed
at an emergent ability that the broader community discovered and documented
independently, with its own mechanistic account. Induction heads are exactly
that: Olsson et al. (2022) documented an abrupt phase change during
transformer training in which a two-layer circuit (previous-token head
composed with an induction head) forms and suddenly enables in-context
copying, and Elhage et al. (2021) proved that one-layer attention-only
transformers CANNOT implement this circuit. The phenomenon, its timing, and
its architectural prerequisite are all external facts, not our design.

Setup
-----
Each sequence is a prefix of DISTINCT random tokens, cyclically repeated to
a fixed length. The prefix length k is drawn per sequence from a wide range,
so the repeat offset varies from sequence to sequence: a fixed positional
lookup cannot predict the repeated region, only content-based induction can
(find the earlier occurrence of the current token, copy its successor).
Evaluation positions are the final half of the sequence, which lie inside
the repeated region for every k in the range. On fresh sequences nothing can
be memorized. The observer's possibility space is the model's own predictive
distribution over the vocabulary at those positions on a FIXED probe set of
held-out sequences.

Pilot note (recorded before the registered run, criterion untouched): a
first pilot used a FIXED prefix length. The one-layer control reached
perfect copy accuracy in that pilot -- with a constant repeat offset, a
single positional attention head ("attend to position t - k") solves the
task without any induction circuit, exactly the positional shortcut the
mechanistic literature warns about. The task, not the criterion, was
revised: prefix lengths now vary per sequence (k uniform in [8, 32] with
sequence length 64), which removes the shortcut and restores the
architectural impossibility result for one layer. The pilot log is kept at
outputs/induction_head_log_fixed_offset_pilot.txt.

Per checkpoint k (identical quantities to the grokking bridge):

    H_k   = mean predictive entropy at second-half probe positions (bits)
    C_k   = mean KL(P_k || P_0) at those positions (collapse magnitude)
    acc_k = induction accuracy on fresh sequences (usefulness)
    J_k   = mean L2 jump of final-layer residual embeddings on the probe

Runs (registered before the full run; the criterion, its component mapping,
and all numeric thresholds are IMPORTED FROZEN from grokking_collapse_bridge
-- nothing is retuned for this domain):

- induction_2layer: two-layer attention-only transformer on repeat data.
  Expected: abrupt useful collapse when the induction circuit forms
  (passes all four components).
- induction_1layer: one-layer attention-only transformer, same data, same
  training. Architecturally cannot form the circuit (Elhage et al. 2021).
  Expected: fails usefulness (partial entropy reduction from direct
  copying statistics is possible, exact continuation prediction is not).
- no_structure: two-layer model, but sequences are i.i.d. random tokens
  (no repeat structure). Nothing useful to collapse onto; fails usefulness.
- memorizer: two-layer model trained on 32 FIXED sequences only, evaluated
  on fresh sequences. Collapses on its training set but cannot transfer;
  fails usefulness on the held-out possibility space.

Predictions: only induction_2layer is emergent under the frozen criterion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from grokking_collapse_bridge import THRESHOLDS, analyze_run, entropy_bits, kl_bits, verdict

OUTPUTS = Path(__file__).resolve().parent / "outputs"


class AttentionOnlyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        return x + out


class AttentionOnlyTransformer(nn.Module):
    """Attention-only (no MLP) causal transformer, per Elhage et al. 2021."""

    def __init__(self, vocab: int, seq_len: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(0.02 * torch.randn(seq_len, d_model))
        self.blocks = nn.ModuleList(
            AttentionOnlyBlock(d_model, n_heads) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.token_emb(x) + self.pos_emb[:t]
        for block in self.blocks:
            h = block(h, self.mask[:t, :t])
        return self.ln_f(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.residual(x))


def make_batch(rng: np.random.Generator, batch: int, vocab: int, seq_len: int,
               k_min: int, k_max: int, structured: bool) -> torch.Tensor:
    """Distinct-token prefix of random length k, cyclically repeated.

    Variable k removes the constant repeat offset, so a purely positional
    head cannot solve the task; distinct prefix tokens make the induction
    rule (previous occurrence -> successor) unambiguous.
    """
    if not structured:
        return torch.from_numpy(rng.integers(0, vocab, size=(batch, seq_len)))
    rows = np.empty((batch, seq_len), dtype=np.int64)
    for i in range(batch):
        k = int(rng.integers(k_min, k_max + 1))
        prefix = rng.choice(vocab, size=k, replace=False)
        rows[i] = np.resize(prefix, seq_len)
    return torch.from_numpy(rows)


def eval_model(model: AttentionOnlyTransformer, probe: torch.Tensor, half: int,
               p0: torch.Tensor | None, prev_emb: torch.Tensor | None):
    """Metrics on final-half positions of the fixed probe set.

    Positions half .. seq_len-2 predict tokens half+1 .. seq_len-1; for every
    prefix length k <= half these positions lie inside the repeated region
    and their current token has an earlier occurrence at distance k.
    """
    model.eval()
    with torch.no_grad():
        logits = model(probe[:, :-1])
        sel = logits[:, half:, :].reshape(-1, logits.shape[-1])
        probs = torch.softmax(sel, dim=1)
        targets = probe[:, half + 1 :].reshape(-1)
        acc = float((sel.argmax(dim=1) == targets).float().mean())
        emb = model.residual(probe[:, :-1])[:, half:, :].reshape(-1, logits.shape[-1])
        emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
    model.train()
    return probs, acc, emb


def train_run(name: str, n_layers: int, structured: bool, memorize: bool,
              vocab: int, seq_len: int, k_min: int, k_max: int, d_model: int,
              n_heads: int, steps: int, batch: int, lr: float, seed: int,
              eval_every: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    half = seq_len // 2
    model = AttentionOnlyTransformer(vocab, seq_len, d_model, n_heads, n_layers)
    n_params = sum(t.numel() for t in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    probe_rng = np.random.default_rng(99991)  # fixed held-out probe set
    probe = make_batch(probe_rng, 256, vocab, seq_len, k_min, k_max, structured=True)
    fixed_train = (
        make_batch(rng, 32, vocab, seq_len, k_min, k_max, structured)
        if memorize else None
    )

    p0: torch.Tensor | None = None
    prev_emb: torch.Tensor | None = None
    rows: List[Dict[str, float]] = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            probs, acc, emb = eval_model(model, probe, half, p0, prev_emb)
            if p0 is None:
                p0 = probs.clone()
            jump = float((emb - prev_emb).norm(dim=1).mean()) if prev_emb is not None else 0.0
            prev_emb = emb.clone()
            rows.append({
                "run": name,
                "epoch": step,
                "train_acc": acc,  # kept for schema parity with the bridge
                "test_acc": acc,
                "test_entropy_bits": entropy_bits(probs),
                "collapse_bits": kl_bits(probs, p0),
                "embedding_jump": jump,
            })
            if step % (eval_every * 10) == 0:
                r = rows[-1]
                print(f"  step {step:5d} induction_acc {acc:.3f} "
                      f"H {r['test_entropy_bits']:.2f} C {r['collapse_bits']:.2f}")
        if step == steps:
            break
        if memorize:
            idx = rng.integers(0, len(fixed_train), size=batch)
            x = fixed_train[idx]
        else:
            x = make_batch(rng, batch, vocab, seq_len, k_min, k_max, structured)
        opt.zero_grad()
        logits = model(x[:, :-1])
        loss_fn(logits.reshape(-1, vocab), x[:, 1:].reshape(-1)).backward()
        opt.step()
    return rows, analyze_run(rows), n_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Induction-head possibility collapse.")
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--k_min", type=int, default=8)
    parser.add_argument("--k_max", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval_every", type=int, default=60)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    torch.set_num_threads(4)
    runs = (
        ("induction_2layer", dict(n_layers=2, structured=True, memorize=False), False),
        ("induction_1layer", dict(n_layers=1, structured=True, memorize=False), False),
        ("no_structure", dict(n_layers=2, structured=False, memorize=False), False),
        ("memorizer", dict(n_layers=2, structured=True, memorize=True), False),
    )
    all_rows: List[Dict[str, float]] = []
    summary: Dict[str, Dict] = {"thresholds": THRESHOLDS, "runs": {}}
    for name, cfg, prespecified in runs:
        print(f"=== {name} ===")
        rows, stats, n_params = train_run(
            name, cfg["n_layers"], cfg["structured"], cfg["memorize"],
            args.vocab, args.seq_len, args.k_min, args.k_max, args.d_model,
            args.n_heads, args.steps, args.batch, args.lr, args.seed,
            args.eval_every,
        )
        all_rows.extend(rows)
        v = verdict(stats, prespecified)
        summary["runs"][name] = {"stats": stats, "verdict": v, "n_params": n_params}
        failed = ";".join(k for k, ok in v["passes"].items() if not ok) or "-"
        print(f"  -> emergent={v['emergent']} failed={failed} ({n_params} params)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "induction_head_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / "induction_head_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'induction_head_summary.json'}")


if __name__ == "__main__":
    main()
