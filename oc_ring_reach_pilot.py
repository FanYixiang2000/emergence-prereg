"""REACH timing/sanity pilot: two continuations of seed 95101 from
checkpoint 500k to the full 2M-step horizon.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Pilot runs are excluded from the
confirmatory cells; the only parameter the pilot may adjust is worker
parallelism.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEED = 95_101
CKPT = 500_000
HORIZON = 2_000_000
PILOT_JS = (900, 901)


def run_one(j):
    import torch
    torch.set_num_threads(4)
    import oc_ring_intervention as oci
    from overcooked_pilot import PolicyNet
    from overcooked_ring_convention import eval_checkpoint

    t0 = time.time()
    path = OUTPUTS / f"overcooked_genesis_{oci.TAGS[SEED]}_s{SEED}_{CKPT}.pt"
    net = PolicyNet()
    net.load_state_dict(torch.load(path, weights_only=True,
                                   map_location="cpu"))
    rng_seed = 1_000_003 * SEED + 31 * (CKPT // 1000) + j
    oci.resume_training(net, "coordination_ring", rng_seed, CKPT,
                        HORIZON - CKPT)
    tmp = OUTPUTS / f"oc_reach_pilot_{j}.pt"
    torch.save(net.state_dict(), tmp)
    ev = eval_checkpoint(tmp, "coordination_ring", SEED)
    tmp.unlink()
    return {"j": j, "rng_seed": rng_seed, "final_p_ccw": ev["p_ccw"],
            "final_soups": ev["mean_soups"],
            "committed": abs(ev["p_ccw"] - 0.5) >= 0.3,
            "wall_seconds": round(time.time() - t0, 1)}


def main() -> None:
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for r in ex.map(run_one, PILOT_JS):
            rows.append(r)
            print(r, flush=True)
    (OUTPUTS / "oc_ring_reach_pilot.json").write_text(json.dumps({
        "status": ("REACH timing/sanity pilot; runs excluded from "
                   "confirmatory cells"),
        "config": {"seed": SEED, "ckpt": CKPT, "horizon": HORIZON},
        "runs": rows}, indent=2), encoding="utf-8")
    print("Wrote oc_ring_reach_pilot.json")


if __name__ == "__main__":
    main()
