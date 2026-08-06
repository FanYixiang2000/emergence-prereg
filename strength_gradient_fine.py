"""Fine-grid re-measurement of the strength-gradient acquisition shape.

The frozen ST-3 prediction (suddenness ordering) FAILED on the 20-point
grid over 60k episodes: both provenances complete acquisition inside the
first checkpoint interval, so the grid cannot resolve their difference --
the same grid-resolution boundary the Pythia 2.8B thinning audit measured.
The ST-3 miss is retained in strength_gradient_battery.json; this script
is the DISCLOSED follow-up at adequate resolution, not a replacement.

Design: same environment, rewards and seeds; 40 checkpoints over the
first 10,000 episodes (250-episode spacing), 200 evaluation episodes per
checkpoint. Declared quantities:

    discovery time      first checkpoint with pattern probability >= 0.5;
    pre-discovery rarity  mean C_t over checkpoints before discovery
                        (the provenance search prior actually experienced);
    fine suddenness     largest single-step drop in C_t divided by the
                        mean absolute drop, on the fine grid.

Declared expectations (follow-up, stated before running): outcome-only
discovers later, spends more checkpoints at high rarity, and shows equal
or more concentrated collapse than shaped; shaped discovers almost
immediately because the process reward names the trigger.

Failure counts as failure and is reported unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import strength_gradient_battery as base

OUTPUTS = Path(__file__).resolve().parent / "outputs"

FINE_EPISODES = 10_000
FINE_CHECKPOINTS = 40
DISCOVERY_P = 0.5


def fine_stats(trace: List[float]) -> Dict[str, Any]:
    c = [-math.log2(p) for p in trace]
    discovery = next(
        (i for i, p in enumerate(trace) if p >= DISCOVERY_P), None)
    if discovery is None or discovery == 0:
        pre_rarity = c[0] if discovery == 0 else float(sum(c) / len(c))
    else:
        pre_rarity = sum(c[:discovery]) / discovery
    drops = [max(0.0, c[i - 1] - c[i]) for i in range(1, len(c))]
    mean_drop = sum(drops) / len(drops)
    return {
        "discovery_checkpoint": discovery,
        "discovery_episode": (None if discovery is None
                              else discovery * FINE_EPISODES
                              // FINE_CHECKPOINTS),
        "pre_discovery_rarity_bits": pre_rarity,
        "fine_suddenness_ratio": (max(drops) / mean_drop
                                  if mean_drop > 1e-9 else 0.0),
    }


def main() -> None:
    base.N_CHECKPOINTS = FINE_CHECKPOINTS
    report: Dict[str, Any] = {
        "status": ("disclosed fine-grid follow-up to the retained ST-3 "
                   "grid-resolution miss; expectations declared in the "
                   "module docstring before running"),
        "grid": {"episodes": FINE_EPISODES,
                 "checkpoints": FINE_CHECKPOINTS,
                 "eval_episodes": base.EVAL_EPISODES},
        "systems": {},
    }
    for provenance in ("shaped", "outcome_only"):
        seeds_out = {}
        for seed in base.SEEDS:
            print(f"fine grid: {provenance} seed {seed}", flush=True)
            trace = base.train_with_pattern_trace(
                provenance, FINE_EPISODES, seed)
            stats = fine_stats(trace)
            seeds_out[str(seed)] = {"trace": trace, **stats}
            print(f"  discovery ckpt {stats['discovery_checkpoint']}, "
                  f"pre-rarity {stats['pre_discovery_rarity_bits']:.2f} "
                  f"bits, suddenness "
                  f"{stats['fine_suddenness_ratio']:.2f}", flush=True)
        discoveries = [seeds_out[str(s)]["discovery_checkpoint"]
                       for s in base.SEEDS]
        rarities = [seeds_out[str(s)]["pre_discovery_rarity_bits"]
                    for s in base.SEEDS]
        suddens = [seeds_out[str(s)]["fine_suddenness_ratio"]
                   for s in base.SEEDS]
        report["systems"][provenance] = {
            "seeds": seeds_out,
            "mean_discovery_checkpoint": sum(discoveries) / len(discoveries),
            "mean_pre_discovery_rarity_bits": sum(rarities) / len(rarities),
            "mean_fine_suddenness": sum(suddens) / len(suddens),
        }
    shaped = report["systems"]["shaped"]
    outcome = report["systems"]["outcome_only"]
    report["follow_up_outcomes"] = {
        "later_discovery_outcome_only": (
            outcome["mean_discovery_checkpoint"]
            > shaped["mean_discovery_checkpoint"]),
        "higher_pre_discovery_rarity_outcome_only": (
            outcome["mean_pre_discovery_rarity_bits"]
            > shaped["mean_pre_discovery_rarity_bits"]),
        "suddenness_geq_outcome_only": (
            outcome["mean_fine_suddenness"]
            >= shaped["mean_fine_suddenness"]),
    }
    out = OUTPUTS / "strength_gradient_fine.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["follow_up_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
