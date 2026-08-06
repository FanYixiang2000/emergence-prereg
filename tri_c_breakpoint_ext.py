"""TRI-C-BP-EXT: five fresh seeds for the high-order breakpoint.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Reuses
tri_c_breakpoint's run_seed and adjudicate unchanged.
"""
import json
from pathlib import Path

import torch

from tri_c_breakpoint import adjudicate, run_seed

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEEDS = (96_401, 96_402, 96_403, 96_404, 96_405)


def main() -> None:
    torch.set_num_threads(4)
    seeds_out, passing = {}, []
    for seed in SEEDS:
        curve = run_seed(seed)
        v = adjudicate(curve)
        seeds_out[str(seed)] = {"curve": curve, "verdict": v}
        if v["tricbp1_seed"]:
            passing.append(seed)
        print(f"seed {seed}: dBIC={v['full']['delta_bic']} "
              f"t*={v['full']['t_star']:.0f} onset={v['full']['onset_type']} "
              f"r3_cross={v['r3_090_cross']} lead={v['tricbp3_seed']}",
              flush=True)
    tce1 = len(passing) >= 4
    tce2 = all(seeds_out[str(s)]["verdict"]["tricbp3_seed"] for s in passing)
    outcomes = {"TCE1_onset_4of5": bool(tce1),
                "TCE2_collapse_leads": bool(tce2 and tce1),
                "passing_seeds": passing}
    report = {"status": "TRI-C-BP-EXT five fresh seeds; registered before run",
              "seeds": seeds_out, "registered_outcomes": outcomes}
    out = OUTPUTS / "tri_c_breakpoint_ext.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
