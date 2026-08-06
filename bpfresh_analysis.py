"""BP-FRESH analysis: frozen BPF-1 / BPF-2 checks.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (BP-FRESH execution
contract). Detector identical to the frozen BP method: one-segment
vs continuous two-segment hinge on log10(steps), Delta-BIC >= 2.
BPF-1: C_env hinge inside [480k, 1.25M] in >= 2/3 fresh seeds.
BPF-2: every positive seed survives 2x thinning (both parities,
verdict kept, hinge within +/- one grid step).
"""

from __future__ import annotations

import json
from pathlib import Path

from breakpoint_model_comparison import breakpoint_test

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = (93004, 93005, 93006)
WINDOW = (480_000, 1_250_000)


def main() -> None:
    per_seed = {}
    for seed in SEEDS:
        r = json.loads(
            (OUTPUTS / f"overcooked_joint_collapse_bpfresh_s{seed}.json")
            .read_text(encoding="utf-8"))
        grid = r["checkpoint_grid"]
        detail = {}
        for series in ("C_env", "collapse_norm", "C_individual",
                       "C_relational"):
            y = [r["curve"][str(c)][series] for c in grid]
            detail[series] = breakpoint_test(grid, y)
        t = detail["C_env"]
        pos = bool(t["verdict"]
                   and WINDOW[0] <= t["breakpoint_step"] <= WINDOW[1])
        thin = {}
        if pos:
            y = [r["curve"][str(c)]["C_env"] for c in grid]
            bi = grid.index(t["breakpoint_step"])
            lo = grid[max(bi - 1, 0)]
            hi = grid[min(bi + 1, len(grid) - 1)]
            for parity in (0, 1):
                t2 = breakpoint_test(grid[parity::2], y[parity::2])
                t2["hinge_ok"] = bool(lo <= t2["breakpoint_step"] <= hi)
                thin[f"parity{parity}"] = t2
        per_seed[str(seed)] = {"detail": detail, "env_positive": pos,
                               "thinning": thin,
                               "thinning_ok": bool(thin) and all(
                                   v["verdict"] and v["hinge_ok"]
                                   for v in thin.values())}

    n_pos = sum(per_seed[s]["env_positive"] for s in per_seed)
    bpf1 = n_pos >= 2
    positives = [s for s in per_seed if per_seed[s]["env_positive"]]
    bpf2 = bool(positives) and all(per_seed[s]["thinning_ok"]
                                   for s in positives)
    outcomes = {"BPF1_env_breakpoint_majority": bool(bpf1),
                "BPF2_thinning_persistence": bool(bpf2),
                "n_positive_seeds": int(n_pos)}
    report = {
        "status": ("BP-FRESH frozen analysis; fresh seeds, ladder-only "
                   "curves; detector identical to BP"),
        "window": list(WINDOW),
        "per_seed": per_seed,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "bpfresh_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    for s in per_seed:
        d = per_seed[s]["detail"]["C_env"]
        print(s, "C_env dBIC", d["delta_bic"], "hinge",
              d["breakpoint_step"], "| positive:",
              per_seed[s]["env_positive"])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
