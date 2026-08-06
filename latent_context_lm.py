"""Latent-context sequence domain: full six-component criterion on a
language model (third full-criterion domain, sequence modality).

Task (no context label is ever given to the model):
  prefix (12 tokens) + SEP + continuation (4 tokens).
  context 1: the prefix's first token recurs once mid-prefix (long-range
             statistical marker); the valued continuation is the LONG-RANGE
             rule R1 = repeat prefix[0] four times.
  context 0: no such recurrence; the valued continuation is the LOCAL rule
             R0 = repeat the last prefix token four times.
  Training corpus: sequences with the context-appropriate continuation
  (the environment's own data distribution; next-token cross-entropy; no
  context token, no rule label -- endogeneity by construction).

Observer (mirrors the Contextual LBF instrument):
  trigger   = first generated continuation token equals prefix[0]
              (the long-range commitment; appropriate only in context 1);
  basins    = {R1, R0, other} x {context-correct or not} collapsed to
              {win_R1, loss_R1, win_R0, loss_R0, none} by generated-token
              majority (>=3 of 4);
  do_trigger     = force the first continuation token to prefix[0];
  do_non_trigger = renormalize the first token away from prefix[0];
  value    = P(context-correct continuation);
  potential = H(basin distribution) under natural sampling;
  conditional selectivity = |P(trigger|ctx1) - P(trigger|ctx0)|;
  specificity = JS(P(B|do_trigger), P(B|do_non_trigger));
  usefulness = value(natural) - value(do_non_trigger);
  acquisition = selectivity(learned) - selectivity(same-seed init twin).

Thresholds are copied UNCHANGED from the frozen Contextual LBF criterion.
Systems: learned, initial twin, oracle context router (scripted; fails
endogeneity/acquisition), fixed_R0, fixed_R1.

Runs tagged ``pilot`` are design-feasibility analyses (training and
estimator parameters only). A confirmatory run must use fresh seeds after
LATENT_CONTEXT_PREREGISTRATION.md is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

V = 12            # content tokens 0..9, SEP=10, BOS=11
SEP, BOS = 10, 11
PREFIX_LEN = 12
CONT_LEN = 4
SEQ_LEN = 1 + PREFIX_LEN + 1 + CONT_LEN   # BOS prefix SEP continuation

THRESHOLDS = {                 # copied unchanged from contextual_lbf_transfer
    "potential_bits": 0.5,
    "conditional_selectivity": 0.5,
    "specificity_js_bits": 0.2,
    "usefulness_gap": 0.0,
    "acquisition": 0.3,
}
BASINS = ("win_R1", "loss_R1", "win_R0", "loss_R0", "none")
CONTEXTS = (0, 1)

TRAIN_STEPS = 3000
BATCH = 128
LR = 3e-4
EVAL_TEMP = 1.0                # estimator parameter (pilot-tunable)


# ------------------------------------------------------------------ data

def sample_prefix(rng: np.random.Generator, context: int) -> List[int]:
    while True:
        toks = rng.integers(0, 10, size=PREFIX_LEN).tolist()
        if context == 1:
            j = int(rng.integers(5, 10))
            toks[j] = toks[0]
            # no other recurrence of toks[0]; last token differs from first
            for k in range(1, PREFIX_LEN):
                if k != j and toks[k] == toks[0]:
                    toks[k] = int((toks[k] + 1 + rng.integers(0, 8)) % 10)
                    if toks[k] == toks[0]:
                        toks[k] = (toks[k] + 1) % 10
        else:
            for k in range(1, PREFIX_LEN):
                if toks[k] == toks[0]:
                    toks[k] = int((toks[k] + 1 + rng.integers(0, 8)) % 10)
                    if toks[k] == toks[0]:
                        toks[k] = (toks[k] + 1) % 10
        if toks[-1] != toks[0]:
            return toks


def correct_continuation(prefix: List[int], context: int) -> List[int]:
    target = prefix[0] if context == 1 else prefix[-1]
    return [target] * CONT_LEN


def build_batch(rng: np.random.Generator, batch: int) -> torch.Tensor:
    rows = []
    for _ in range(batch):
        ctx = int(rng.integers(0, 2))
        prefix = sample_prefix(rng, ctx)
        seq = [BOS] + prefix + [SEP] + correct_continuation(prefix, ctx)
        rows.append(seq)
    return torch.tensor(rows, dtype=torch.long)


# ------------------------------------------------------------------ model

class TinyLM(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            batch_first=True, dropout=0.0, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(d, V)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(n, device=x.device))
        h = self.blocks(h, mask=self.mask[:n, :n])
        return self.head(h)


def train_lm(seed: int, steps: int, device: str) -> TinyLM:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyLM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(steps):
        batch = build_batch(rng, BATCH).to(device)
        logits = model(batch[:, :-1])
        loss = loss_fn(logits.reshape(-1, V), batch[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  seed {seed} step {step + 1}: loss {loss.item():.4f}",
                  flush=True)
    model.eval()
    return model


def initial_twin(seed: int, device: str) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM().to(device).eval()


# ------------------------------------------------------------------ systems

class System:
    """learned/twin (neural) or scripted (router / fixed rules)."""

    def __init__(self, kind: str, model: Optional[TinyLM] = None,
                 device: str = "cpu"):
        self.kind = kind
        self.model = model
        self.device = device

    def generate(self, prefix: List[int], context: int,
                 rng: np.random.Generator,
                 intervention: Optional[str]) -> List[int]:
        first_forced: Optional[int] = None
        banned: Optional[int] = None
        if intervention == "do_trigger":
            first_forced = prefix[0]
        elif intervention == "do_non_trigger":
            banned = prefix[0]

        if self.kind in ("router", "fixed_R0", "fixed_R1"):
            if self.kind == "router":
                target = prefix[0] if context == 1 else prefix[-1]
            elif self.kind == "fixed_R1":
                target = prefix[0]
            else:
                target = prefix[-1]
            out = [target] * CONT_LEN
            if first_forced is not None:
                out[0] = first_forced
            elif banned is not None and out[0] == banned:
                out[0] = prefix[-1] if banned == prefix[0] else prefix[0]
            return out

        seq = [BOS] + prefix + [SEP]
        out: List[int] = []
        with torch.no_grad():
            for pos in range(CONT_LEN):
                x = torch.tensor([seq + out], dtype=torch.long,
                                 device=self.device)
                logits = self.model(x)[0, -1, :10] / EVAL_TEMP
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                if pos == 0 and first_forced is not None:
                    tok = first_forced
                elif pos == 0 and banned is not None:
                    probs[banned] = 0.0
                    probs = probs / probs.sum()
                    tok = int(rng.choice(10, p=probs))
                else:
                    tok = int(rng.choice(10, p=probs))
                out.append(tok)
        return out


def classify(prefix: List[int], context: int, cont: List[int]) -> Dict:
    r1 = sum(t == prefix[0] for t in cont)
    r0 = sum(t == prefix[-1] for t in cont)
    if r1 >= 3:
        rule = "R1"
    elif r0 >= 3:
        rule = "R0"
    else:
        rule = "none"
    correct_rule = "R1" if context == 1 else "R0"
    win = rule == correct_rule
    basin = "none" if rule == "none" else f"{'win' if win else 'loss'}_{rule}"
    return {"basin": basin, "trigger": int(cont[0] == prefix[0]),
            "value": float(win)}


# ------------------------------------------------------------------ metrics

def entropy(d: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in d.values() if p > 0)


def js(p: Dict[str, float], q: Dict[str, float]) -> float:
    out = 0.0
    for k in set(p) | set(q):
        a, b = p.get(k, 0.0), q.get(k, 0.0)
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def norm_counts(rows: List[Dict]) -> Dict[str, float]:
    total = len(rows) or 1
    return {b: sum(r["basin"] == b for r in rows) / total for b in BASINS}


def evaluate(system: System, n_eval: int, seed_offset: int) -> Dict[str, Any]:
    rows: List[Dict] = []
    for context in CONTEXTS:
        data_rng = np.random.default_rng(seed_offset + context)
        prefixes = [sample_prefix(data_rng, context) for _ in range(n_eval)]
        for ep, prefix in enumerate(prefixes):
            for mode in (None, "do_trigger", "do_non_trigger"):
                gen_rng = np.random.default_rng(
                    seed_offset + 7919 * context + 104729 * ep
                    + {"do_trigger": 1, "do_non_trigger": 2}.get(mode, 0))
                cont = system.generate(prefix, context, gen_rng, mode)
                row = classify(prefix, context, cont)
                row["context"] = context
                row["mode"] = mode or "natural"
                rows.append(row)

    nat = [r for r in rows if r["mode"] == "natural"]
    do_t = [r for r in rows if r["mode"] == "do_trigger"]
    do_n = [r for r in rows if r["mode"] == "do_non_trigger"]
    rate = {c: float(np.mean([r["trigger"] for r in nat
                              if r["context"] == c])) for c in CONTEXTS}
    return {
        "trigger_rates": {str(c): rate[c] for c in CONTEXTS},
        "potential_bits": entropy(norm_counts(nat)),
        "conditional_selectivity": abs(rate[1] - rate[0]),
        "specificity_js_bits": js(norm_counts(do_t), norm_counts(do_n)),
        "natural_value": float(np.mean([r["value"] for r in nat])),
        "do_non_trigger_value": float(np.mean([r["value"] for r in do_n])),
        "usefulness_gap": float(np.mean([r["value"] for r in nat])
                                - np.mean([r["value"] for r in do_n])),
    }


def component_verdict(metrics: Dict[str, Any], endogenous: bool,
                      acquisition: float) -> Dict[str, Any]:
    passes = {
        "potential": metrics["potential_bits"] >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": (
            metrics["conditional_selectivity"]
            >= THRESHOLDS["conditional_selectivity"]),
        "specificity": (metrics["specificity_js_bits"]
                        >= THRESHOLDS["specificity_js_bits"]),
        "usefulness": metrics["usefulness_gap"] > THRESHOLDS["usefulness_gap"],
        "endogeneity": endogenous,
        "acquisition": acquisition >= THRESHOLDS["acquisition"],
    }
    return {"passes": passes, "emergent": int(all(passes.values()))}


def run_seed(seed: int, train_steps: int, n_eval: int,
             device: str) -> Dict[str, Any]:
    model = train_lm(seed, train_steps, device)
    torch.save(model.state_dict(), OUTPUTS / f"latent_context_lm_seed{seed}.pt")
    learned = evaluate(System("learned", model, device), n_eval,
                       30_000_000 + seed * 100_000)
    init = evaluate(System("twin", initial_twin(seed, device), device),
                    n_eval, 30_000_000 + seed * 100_000)
    acquisition = (learned["conditional_selectivity"]
                   - init["conditional_selectivity"])
    systems: Dict[str, Any] = {
        "learned": {"metrics": learned, "endogenous": True,
                    "acquisition": acquisition},
        "initial_twin": {"metrics": init, "endogenous": True,
                         "acquisition": 0.0},
    }
    for kind in ("router", "fixed_R0", "fixed_R1"):
        metrics = evaluate(System(kind), n_eval,
                           30_000_000 + seed * 100_000)
        systems[kind] = {"metrics": metrics, "endogenous": False,
                         "acquisition": 0.0}
    for name, item in systems.items():
        item["verdict"] = component_verdict(
            item["metrics"], item["endogenous"], item["acquisition"])
    expected = {"learned": 1, "initial_twin": 0, "router": 0,
                "fixed_R0": 0, "fixed_R1": 0}
    return {
        "systems": systems,
        "expected": expected,
        "all_expected": all(systems[k]["verdict"]["emergent"] == v
                            for k, v in expected.items()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[2001])
    parser.add_argument("--train_steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--n_eval", type=int, default=80)
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--device", default="cuda:0"
                        if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    result: Dict[str, Any] = {
        "status": ("exploratory design pilot" if "pilot" in args.tag
                   else "prospectively frozen fresh-seed run"),
        "thresholds": THRESHOLDS,
        "train_steps": args.train_steps,
        "n_eval_per_context": args.n_eval,
        "eval_temperature": EVAL_TEMP,
        "seeds": {},
    }
    for seed in args.seeds:
        print(f"latent-context LM seed {seed}", flush=True)
        result["seeds"][str(seed)] = run_seed(
            seed, args.train_steps, args.n_eval, args.device)
        learned = result["seeds"][str(seed)]["systems"]["learned"]
        print(json.dumps({"metrics": learned["metrics"],
                          "acquisition": learned["acquisition"],
                          "verdict": learned["verdict"]}, indent=2),
              flush=True)
    result["summary"] = {
        "learned_passes": sum(
            s["systems"]["learned"]["verdict"]["emergent"]
            for s in result["seeds"].values()),
        "all_expected_by_seed": {
            k: s["all_expected"] for k, s in result["seeds"].items()},
    }
    out = OUTPUTS / f"latent_context_lm_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
