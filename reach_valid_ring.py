"""REACH-VALID gate VH: continuation-horizon convergence in the ring.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Gate runs use seed 95101 with
continuation indices j = 902..909 and are excluded from the
confirmatory REACH cells. Each continuation is evaluated at horizons
1.2M/1.6M/2.0M by re-running resume_training from the checkpoint with
the same rng seed, so shorter horizons are prefixes of the longer run
when the pipeline is deterministic; a determinism check is run first.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEED = 95_101
CKPTS = (500_000, 960_000)
JS = tuple(range(902, 910))
HORIZONS = (1_200_000, 1_600_000, 2_000_000)


def run_one(args):
    ckpt, j, horizon, det_tag = args
    import torch

    torch.set_num_threads(4)
    import oc_ring_intervention as oci
    from overcooked_pilot import PolicyNet
    from overcooked_ring_convention import eval_checkpoint

    path = OUTPUTS / f"overcooked_genesis_{oci.TAGS[SEED]}_s{SEED}_{ckpt}.pt"
    net = PolicyNet()
    net.load_state_dict(torch.load(path, weights_only=True,
                                   map_location="cpu"))
    rng_seed = 1_000_003 * SEED + 31 * (ckpt // 1000) + j
    oci.resume_training(net, "coordination_ring", rng_seed, ckpt,
                        horizon - ckpt)
    tmp = OUTPUTS / f"reach_valid_{ckpt}_{j}_{horizon}{det_tag}.pt"
    torch.save(net.state_dict(), tmp)
    ev = eval_checkpoint(tmp, "coordination_ring", SEED)
    tmp.unlink()
    return {"ckpt": ckpt, "j": j, "horizon": horizon, "det": det_tag,
            "p_ccw": ev["p_ccw"], "soups": ev["mean_soups"],
            "label_ccw": bool(ev["p_ccw"] > 0.5)}


def main() -> None:
    jobs = [(960_000, 902, 1_600_000, "_det")]  # determinism duplicate
    jobs += [(c, j, h, "") for c in CKPTS for j in JS for h in HORIZONS]
    rows = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)
            print(r, flush=True)

    def cell(ckpt, j, h):
        return next(r for r in rows
                    if (r["ckpt"], r["j"], r["horizon"], r["det"])
                    == (ckpt, j, h, ""))

    det_a = next(r for r in rows if r["det"] == "_det")
    det_b = cell(960_000, 902, 1_600_000)
    det_ok = det_a["p_ccw"] == det_b["p_ccw"]

    vh = {}
    for c in CKPTS:
        agree = sum(cell(c, j, 1_600_000)["label_ccw"]
                    == cell(c, j, 2_000_000)["label_ccw"] for j in JS)
        k16 = sum(cell(c, j, 1_600_000)["label_ccw"] for j in JS)
        k20 = sum(cell(c, j, 2_000_000)["label_ccw"] for j in JS)
        vh[str(c)] = {"agree_16_20": f"{agree}/8", "k_ccw_16": k16,
                      "k_ccw_20": k20,
                      "primary_pass": bool(agree >= 7),
                      "fallback_pass": bool(abs(k16 - k20) <= 1)}
    key = "primary_pass" if det_ok else "fallback_pass"
    outcomes = {"determinism_check_pass": bool(det_ok),
                "reading_used": key,
                "per_checkpoint": vh,
                "VH1_pass": bool(all(v[key] for v in vh.values()))}
    (OUTPUTS / "reach_valid_ring.json").write_text(json.dumps({
        "status": ("REACH-VALID gate VH; gate runs excluded from "
                   "confirmatory REACH cells"),
        "config": {"seed": SEED, "ckpts": CKPTS, "js": JS,
                   "horizons": HORIZONS},
        "runs": rows,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print("Wrote reach_valid_ring.json")


if __name__ == "__main__":
    main()
