"""Boundary probe: what does the four-component process proxy say about a
strong, smooth, ORDINARY supervised learner? (Exploratory; not a
preregistered confirmation.)

Design history, disclosed in full:

- Attempt 1 (quarantined as
  ordinary_learner_control_attempt1_failed_design.json): task y=(a+b)//2
  with 97 ordinal classes was too hard for the frozen architecture
  (final accuracy 0.07); rejected via usefulness. A failed DESIGN, not an
  informative control.
- Feasibility pilots (logged): y=(a+b)//20 plateaus at 0.868;
  y=(a+b)//40 reaches 0.93 quickly and smoothly.
- This probe: y=(a+b)//40 (5 coarse ordinal classes), same architecture,
  optimizer and frozen thresholds as the grokking bridge; three seeds;
  both the standard (40-epoch) and a dense (5-epoch) evaluation grid.

FINDING ESTABLISHED BY THE PILOT AND CONFIRMED HERE (recorded before this
final run only in the sense of the pilot; the probe is exploratory): the
proxy ACCEPTS this ordinary learner. Fast learning of easy structure is
temporally concentrated collapse with a large anchored-window gain, so
potential/burstiness/usefulness/endogeneity all pass. This is a measured
scope boundary of the process proxy: it detects burst-concentrated useful
acquisition and cannot, by itself, separate emergent abilities from easy
abilities acquired quickly. That separation is exactly the job of the
episode-level components (conditional selectivity, specificity,
do-contrasts) in the full six-component criterion -- which is why the
instrument map never treats a lone process-proxy pass as an emergence
verdict, and why paired controls carry the discriminative weight in the
public-checkpoint studies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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

P = 97
DIVISOR = 40           # 5 coarse ordinal classes
TRAIN_FRAC = 0.5
EPOCHS = 3000
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEEDS = (111, 222, 333)


def make_dataset(seed: int):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(P) for b in range(P)],
                     dtype=np.int64)
    labels = (pairs[:, 0] + pairs[:, 1]) // DIVISOR
    order = rng.permutation(len(pairs))
    n_train = int(TRAIN_FRAC * len(pairs))
    tr, te = order[:n_train], order[n_train:]
    return (torch.from_numpy(pairs[tr]), torch.from_numpy(labels[tr]),
            torch.from_numpy(pairs[te]), torch.from_numpy(labels[te]))


def train_run(seed: int, eval_every: int) -> List[Dict[str, float]]:
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = make_dataset(seed)
    model = GrokNet(P)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    rows: List[Dict[str, float]] = []
    p0 = None
    for epoch in range(EPOCHS + 1):
        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(x_te), dim=1)
                if p0 is None:
                    p0 = probs.clone()
                rows.append({
                    "run": f"ordinary_{seed}_e{eval_every}", "epoch": epoch,
                    "train_acc": float((model(x_tr).argmax(1) == y_tr)
                                       .float().mean()),
                    "test_acc": float((probs.argmax(1) == y_te)
                                      .float().mean()),
                    "test_entropy_bits": entropy_bits(probs),
                    "collapse_bits": kl_bits(probs, p0),
                    "embedding_jump": 0.0,
                })
            model.train()
        if epoch == EPOCHS:
            break
        opt.zero_grad()
        loss_fn(model(x_tr), y_tr).backward()
        opt.step()
    return rows


def main() -> None:
    torch.set_num_threads(24)
    summary = {
        "status": "exploratory boundary probe (not a preregistered "
                  "confirmation); attempt-1 design failure quarantined",
        "task": f"y=(a+b)//{DIVISOR} (5 ordinal classes)",
        "thresholds": THRESHOLDS,
        "runs": {},
    }
    accepted = 0
    strong = 0
    for seed in SEEDS:
        for eval_every in (40, 5):
            rows = train_run(seed, eval_every)
            stats = analyze_run(rows)
            v = verdict(stats, prespecified=False)
            key = f"seed{seed}_grid{eval_every}"
            summary["runs"][key] = {"stats": stats, "verdict": v}
            accepted += v["emergent"]
            strong += stats["final_test_acc"] >= 0.9
            print(f"{key}: final {stats['final_test_acc']:.3f} "
                  f"emergent={v['emergent']} "
                  f"burst {stats['burstiness_ratio']:.1f} "
                  f"gain {stats['usefulness_acc_gain']:.3f}", flush=True)
    n = len(summary["runs"])
    summary["finding"] = {
        "runs_accepted_by_proxy": f"{accepted}/{n}",
        "runs_with_final_acc_ge_0.9": f"{strong}/{n}",
        "scope_statement": (
            "The four-component process proxy accepts a strong ordinary "
            "learner: fast smooth acquisition of easy structure is "
            "burst-concentrated useful collapse at these grid resolutions. "
            "The proxy is an acquisition-shape instrument, not an emergence "
            "verdict; separating emergent from merely-fast abilities "
            "requires the episode-level selectivity/specificity components "
            "and paired controls."
        ),
    }
    out = OUTPUTS / "ordinary_learner_control.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["finding"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
