"""Generalization audit for the latent-context sequence domain.

Frozen in this docstring BEFORE any perturbed evaluation (saved confirmation
models only; no retraining). Training sampled the long-range marker at
positions j in [5,10). Perturbations:

- G0 baseline: training distribution (marker j in [5,10)).
- G1 unseen-early markers: j in [2,5) -- positions never seen in training.
- G2 unseen-late markers: j in {10, 11} -- adjacent to SEP, never seen.
- G3 distractor: context-0 prefixes additionally contain a repeated
  non-first token pair (a long-range recurrence of the WRONG token);
  context-1 prefixes unchanged. Tests false triggering.
- G4 double-marker: context-1 prefixes contain the first-token recurrence
  twice (two markers). Tests robustness of true triggering.

Registered predictions (registered failures kept if wrong):

- LG1 (no false triggers): under G3, the context-0 natural trigger rate
  stays <= 0.2 for at least 8/10 models.
- LG2 (positional generality, genuine risk): under G1 and G2, at least
  8/10 models retain >= 50% of their own G0 selectivity.
- LG3 (causal retention where activation persists): in every perturbation
  where a model retains >= 50% selectivity, its usefulness gap stays
  positive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import latent_context_lm as lm

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = [2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110]
N_EVAL = 80
OFFSET = 40_000_000


def sample_prefix_perturbed(rng: np.random.Generator, context: int,
                            mode: str) -> List[int]:
    if mode in ("G0", "G3", "G4"):
        toks = lm.sample_prefix(rng, context)
    else:
        lo, hi = ((2, 5) if mode == "G1" else (10, 12))
        while True:
            toks = rng.integers(0, 10, size=lm.PREFIX_LEN).tolist()
            if context == 1:
                j = int(rng.integers(lo, hi))
                toks[j] = toks[0]
                for k in range(1, lm.PREFIX_LEN):
                    if k != j and toks[k] == toks[0]:
                        toks[k] = (toks[k] + 1) % 10
                        if toks[k] == toks[0]:
                            toks[k] = (toks[k] + 1) % 10
            else:
                for k in range(1, lm.PREFIX_LEN):
                    if toks[k] == toks[0]:
                        toks[k] = (toks[k] + 1) % 10
                        if toks[k] == toks[0]:
                            toks[k] = (toks[k] + 1) % 10
            if toks[-1] != toks[0]:
                break
    if mode == "G3" and context == 0:
        # insert a repeated non-first, non-last token pair (wrong-token
        # long-range recurrence) at fixed slots 3 and 7
        candidates = [t for t in range(10)
                      if t not in (toks[0], toks[-1])]
        d = int(candidates[int(rng.integers(0, len(candidates)))])
        toks[3] = d
        toks[7] = d
        if toks[-1] == toks[0]:
            toks[-1] = (toks[-1] + 1) % 10
    if mode == "G4" and context == 1:
        positions = [k for k in range(2, lm.PREFIX_LEN - 1)
                     if toks[k] != toks[0]]
        j2 = positions[int(rng.integers(0, len(positions)))]
        toks[j2] = toks[0]
    return toks


def evaluate(system: lm.System, gmode: str, seed_offset: int) -> Dict:
    rows = []
    for context in lm.CONTEXTS:
        data_rng = np.random.default_rng(seed_offset + context)
        prefixes = [sample_prefix_perturbed(data_rng, context, gmode)
                    for _ in range(N_EVAL)]
        for ep, prefix in enumerate(prefixes):
            for mode in (None, "do_non_trigger"):
                gen_rng = np.random.default_rng(
                    seed_offset + 7919 * context + 104729 * ep
                    + (2 if mode else 0))
                cont = system.generate(prefix, context, gen_rng, mode)
                row = lm.classify(prefix, context, cont)
                row["context"] = context
                row["mode"] = mode or "natural"
                rows.append(row)
    nat = [r for r in rows if r["mode"] == "natural"]
    non = [r for r in rows if r["mode"] == "do_non_trigger"]
    rate = {c: float(np.mean([r["trigger"] for r in nat
                              if r["context"] == c])) for c in lm.CONTEXTS}
    return {
        "trigger_rate_ctx0": rate[0],
        "trigger_rate_ctx1": rate[1],
        "conditional_selectivity": abs(rate[1] - rate[0]),
        "usefulness_gap": float(np.mean([r["value"] for r in nat])
                                - np.mean([r["value"] for r in non])),
    }


def main() -> None:
    torch.set_num_threads(16)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results: Dict[str, Dict[str, Dict]] = {}
    for seed in SEEDS:
        model = lm.TinyLM().to(device)
        model.load_state_dict(torch.load(
            OUTPUTS / f"latent_context_lm_seed{seed}.pt",
            weights_only=True, map_location=device))
        model.eval()
        system = lm.System("learned", model, device)
        results[str(seed)] = {}
        for gmode in ("G0", "G1", "G2", "G3", "G4"):
            results[str(seed)][gmode] = evaluate(
                system, gmode, OFFSET + seed * 100_000)
        print(f"seed {seed}: " + " ".join(
            f"{g}:sel{results[str(seed)][g]['conditional_selectivity']:.2f}"
            for g in ("G0", "G1", "G2", "G3", "G4")), flush=True)

    lg1 = sum(results[str(s)]["G3"]["trigger_rate_ctx0"] <= 0.2
              for s in SEEDS) >= 8
    lg2_counts = {}
    for g in ("G1", "G2"):
        lg2_counts[g] = sum(
            results[str(s)][g]["conditional_selectivity"]
            >= 0.5 * results[str(s)]["G0"]["conditional_selectivity"]
            for s in SEEDS)
    lg2 = all(v >= 8 for v in lg2_counts.values())
    lg3_violations = []
    for s in SEEDS:
        for g in ("G1", "G2", "G3", "G4"):
            r = results[str(s)][g]
            if (r["conditional_selectivity"]
                    >= 0.5 * results[str(s)]["G0"]["conditional_selectivity"]
                    and r["usefulness_gap"] <= 0):
                lg3_violations.append(f"{s}:{g}")
    lg3 = not lg3_violations

    summary = {
        "status": "prospectively frozen generalization audit "
                  "(saved models; no retraining)",
        "results": results,
        "predictions": {
            "LG1_no_false_triggers": {"pass": lg1},
            "LG2_positional_generality": {"pass": lg2, "counts": lg2_counts},
            "LG3_causal_retention_where_activated": {
                "pass": lg3, "violations": lg3_violations},
        },
        "all_pass": all([lg1, lg2, lg3]),
    }
    out = OUTPUTS / "latent_context_generalization.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["predictions"], indent=2))
    print("all_pass:", summary["all_pass"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
