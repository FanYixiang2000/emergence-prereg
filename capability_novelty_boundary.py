"""Capability emergence versus ordinary learning: lower-order novelty test.

The measured boundary problem is real: the frozen four-component process
proxy accepts 6/6 runs of a fast ordinary supervised learner. Therefore
gradual-vs-abrupt acquisition cannot define emergence, and useful output
collapse alone is insufficient.

This experiment adds a capability-specific product test:

    N_cap = Acc(full learned system) - Acc(frozen lower-order hypothesis)

The lower-order model is an additive classifier

    logits(a,b) = U[a] + V[b] + bias,

which sees the same two inputs but has no interaction/composition term.
It should solve the ordinary coarse-sum task, whose classes are ordered
stripes in a+b, but fail modular addition, whose wrap-around interaction
cannot be represented additively. The induction-head 1-layer vs 2-layer
comparison supplies an independent architectural-composition instance.

This is not claimed as the final universal novelty measure. It is a direct
falsification repair: ordinary learning must not be rescued merely because
it is fast, useful and entropy-reducing.

REGISTERED PREDICTIONS (frozen before running):
  CN-1  Ordinary coarse-sum: additive baseline test accuracy >= 0.85 and
        mean novelty gap versus the stored full learner <= 0.10.
  CN-2  Modular addition: additive baseline test accuracy <= 0.10 while
        stored grokking full accuracy >= 0.90, novelty gap >= 0.80.
  CN-3  Induction: stored 2-layer minus 1-layer test-accuracy gap >= 0.70.
  CN-4  The old process proxy accepts ordinary learning (6/6), but the
        novelty-qualified capability rule rejects ordinary and accepts
        modular grokking + induction (3/3 classifications correct).
Misses are retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

OUTPUTS = Path(__file__).resolve().parent / "outputs"
P = 97
SEEDS = (111, 222, 333)


class AdditiveClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.ua = nn.Embedding(P, n_classes)
        self.ub = nn.Embedding(P, n_classes)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x):
        return self.ua(x[:, 0]) + self.ub(x[:, 1]) + self.bias


def dataset(kind: str, seed: int) -> Tuple[torch.Tensor, ...]:
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(P) for b in range(P)],
                     dtype=np.int64)
    if kind == "ordinary":
        labels = (pairs[:, 0] + pairs[:, 1]) // 40
        n_classes = 5
    elif kind == "mod_add":
        labels = (pairs[:, 0] + pairs[:, 1]) % P
        n_classes = P
    else:
        raise ValueError(kind)
    order = rng.permutation(len(pairs))
    cut = len(pairs) // 2
    tr, te = order[:cut], order[cut:]
    return (torch.from_numpy(pairs[tr]), torch.from_numpy(labels[tr]),
            torch.from_numpy(pairs[te]), torch.from_numpy(labels[te]),
            n_classes)


def train_additive(kind: str, seed: int) -> float:
    torch.manual_seed(seed)
    xtr, ytr, xte, yte, n_classes = dataset(kind, seed)
    model = AdditiveClassifier(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    # Full-batch optimization is deterministic and cheap (<= 19k params).
    for _ in range(1200):
        loss = nn.functional.cross_entropy(model(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((model(xte).argmax(1) == yte).float().mean())


def main() -> None:
    torch.set_num_threads(8)
    ordinary = json.loads(
        (OUTPUTS / "ordinary_learner_control.json").read_text())
    grok = json.loads(
        (OUTPUTS / "grokking_generality_summary.json").read_text())
    induction = json.loads(
        (OUTPUTS / "induction_head_summary.json").read_text())

    ordinary_full = [
        ordinary["runs"][f"seed{s}_grid40"]["stats"]["final_test_acc"]
        for s in SEEDS
    ]
    ordinary_low = [train_additive("ordinary", s) for s in SEEDS]
    mod_low = [train_additive("mod_add", s) for s in SEEDS]
    mod_full = [
        c["final_test_acc"] for c in grok["cells"]
        if c["op"] == "add" and c["condition"] == "grokking"
    ]
    ih2 = induction["runs"]["induction_2layer"]["stats"]["final_test_acc"]
    ih1 = induction["runs"]["induction_1layer"]["stats"]["final_test_acc"]

    values = {
        "ordinary": {
            "full_acc": ordinary_full,
            "additive_acc": ordinary_low,
            "mean_full": float(np.mean(ordinary_full)),
            "mean_additive": float(np.mean(ordinary_low)),
            "novelty_gap": float(np.mean(ordinary_full)
                                 - np.mean(ordinary_low)),
            "old_proxy_accepted": ordinary["finding"]
            ["runs_accepted_by_proxy"],
        },
        "modular_addition": {
            "full_acc": mod_full,
            "additive_acc": mod_low,
            "mean_full": float(np.mean(mod_full)),
            "mean_additive": float(np.mean(mod_low)),
            "novelty_gap": float(np.mean(mod_full) - np.mean(mod_low)),
        },
        "induction": {
            "two_layer_acc": ih2,
            "one_layer_acc": ih1,
            "novelty_gap": ih2 - ih1,
        },
    }
    cn1 = (values["ordinary"]["mean_additive"] >= 0.85
           and values["ordinary"]["novelty_gap"] <= 0.10)
    cn2 = (values["modular_addition"]["mean_additive"] <= 0.10
           and values["modular_addition"]["mean_full"] >= 0.90
           and values["modular_addition"]["novelty_gap"] >= 0.80)
    cn3 = values["induction"]["novelty_gap"] >= 0.70
    novelty_verdict = {
        "ordinary": values["ordinary"]["novelty_gap"] >= 0.30,
        "modular_grokking": values["modular_addition"]["novelty_gap"] >= 0.30,
        "induction": values["induction"]["novelty_gap"] >= 0.30,
    }
    cn4 = (values["ordinary"]["old_proxy_accepted"] == "6/6"
           and novelty_verdict == {
               "ordinary": False,
               "modular_grokking": True,
               "induction": True,
           })
    report = {
        "status": ("capability novelty boundary; lower-order additive / "
                   "architectural controls distinguish ordinary learning "
                   "from capability-emergence candidates"),
        "novelty_threshold": 0.30,
        "values": values,
        "novelty_qualified_verdict": novelty_verdict,
        "registered_outcomes": {
            "CN1_ordinary_low_order_suffices": bool(cn1),
            "CN2_modular_rule_requires_interaction": bool(cn2),
            "CN3_induction_requires_composition": bool(cn3),
            "CN4_old_proxy_false_positive_repaired": bool(cn4),
        },
    }
    out = OUTPUTS / "capability_novelty_boundary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2))
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
