"""OC-RING-FIXT: fixed-time causal commitment test in coordination_ring.

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Every seed is perturbed at the same
training step T_FIX = 960,000 (rule: common-grid checkpoint maximizing
cross-seed openness variance, all seeds >= 20 committed episodes),
so training time is identical by construction and only openness
varies across seeds. Mechanics are byte-identical to OC-RING-INT.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"

T_FIX = 960_000
OPEN_THRESHOLD = 0.5
RING_SEEDS = (95_101, 95_202, 95_303, 95_606, 95_707, 95_808, 95_909, 96_010)
NOISE_SCALES = (0.25, 0.5)


def run_one(args):
    seed, s = args
    import torch
    torch.set_num_threads(4)
    import oc_ring_intervention as oci
    from overcooked_pilot import PolicyNet
    from overcooked_ring_convention import eval_checkpoint

    rec = oci.formation_record(seed)
    orig_dir = 1 if rec["curves"][-1]["p_ccw"] > 0.5 else -1

    path = OUTPUTS / f"overcooked_genesis_{oci.TAGS[seed]}_s{seed}_{T_FIX}.pt"
    net = PolicyNet()
    net.load_state_dict(torch.load(path, weights_only=True,
                                   map_location="cpu"))
    rng_seed = 7 * seed + T_FIX + round(100 * s)
    gen = torch.Generator().manual_seed(rng_seed)
    oci.perturb(net, s, gen)
    oci.resume_training(net, "coordination_ring", rng_seed, T_FIX,
                        oci.RESUME_STEPS)
    tmp = OUTPUTS / f"oc_fixt_{seed}_{s}.pt"
    torch.save(net.state_dict(), tmp)
    ev = eval_checkpoint(tmp, "coordination_ring", seed)
    p = ev["p_ccw"]
    new_dir = 1 if p > 0.5 else -1
    committed = abs(p - 0.5) >= oci.COMMIT_MARGIN
    outcome = ("flip" if committed and new_dir != orig_dir
               else "held" if committed else "uncommitted")
    return {"seed": seed, "ckpt": T_FIX, "scale": s,
            "openness_at_perturbation": oci.openness_at(seed, T_FIX),
            "orig_dir": orig_dir,
            "unperturbed_final_soups": rec["final_soups"],
            "final_p_ccw": p, "final_soups": ev["mean_soups"],
            "outcome": outcome}


def eval_baseline(seed):
    """OCF5 maturity control: unperturbed checkpoint at T_FIX."""
    import torch
    torch.set_num_threads(4)
    import oc_ring_intervention as oci
    from overcooked_ring_convention import eval_checkpoint
    path = OUTPUTS / f"overcooked_genesis_{oci.TAGS[seed]}_s{seed}_{T_FIX}.pt"
    ev = eval_checkpoint(path, "coordination_ring", seed)
    return seed, {"mean_soups_at_tfix": ev["mean_soups"],
                  "p_ccw_at_tfix": ev["p_ccw"],
                  "openness_at_tfix": oci.openness_at(seed, T_FIX)}


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or not labels.any():
        return None
    pos, neg = scores[labels], scores[~labels]
    return float(np.mean([(pi > ni) + 0.5 * (pi == ni)
                          for pi in pos for ni in neg]))


def main() -> None:
    jobs = [(seed, s) for seed in RING_SEEDS for s in NOISE_SCALES]
    runs = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run_one, jobs):
            runs.append(r)
            print(f"seed {r['seed']} s={r['scale']}: {r['outcome']} "
                  f"p_ccw={r['final_p_ccw']} soups={r['final_soups']}",
                  flush=True)
            (OUTPUTS / "oc_fixt_partial.json").write_text(
                json.dumps(runs, indent=1))

    baselines = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for seed, row in ex.map(eval_baseline, RING_SEEDS):
            baselines[str(seed)] = row
            print(f"baseline seed {seed}: {row}", flush=True)

    runs.sort(key=lambda r: (r["seed"], r["scale"]))
    open_seeds = [s for s in RING_SEEDS
                  if baselines[str(s)]["openness_at_tfix"] >= OPEN_THRESHOLD]
    committed_seeds = [s for s in RING_SEEDS if s not in open_seeds]

    def moved(r):
        return r["outcome"] in ("flip", "uncommitted")

    movable = {s: any(moved(r) for r in runs if r["seed"] == s)
               for s in RING_SEEDS}
    a = sum(movable[s] for s in open_seeds)
    b = sum(movable[s] for s in committed_seeds)
    from scipy.stats import fisher_exact
    _, p_seed = fisher_exact(
        [[a, len(open_seeds) - a], [b, len(committed_seeds) - b]],
        alternative="greater")

    open_runs = [r for r in runs if r["seed"] in open_seeds]
    com_runs = [r for r in runs if r["seed"] in committed_seeds]
    om, cm = sum(map(moved, open_runs)), sum(map(moved, com_runs))
    _, p_run = fisher_exact(
        [[om, len(open_runs) - om], [cm, len(com_runs) - cm]],
        alternative="greater")

    auc_open = auc([r["openness_at_perturbation"] for r in runs],
                   [moved(r) for r in runs])
    auc_soups = auc([-baselines[str(r["seed"])]["mean_soups_at_tfix"]
                     for r in runs],
                    [moved(r) for r in runs])

    strict_flips_open = sum(r["outcome"] == "flip" for r in open_runs)
    strict_flips_com = sum(r["outcome"] == "flip" for r in com_runs)

    outcomes = {
        "T_FIX": T_FIX,
        "open_seeds": open_seeds,
        "committed_seeds": committed_seeds,
        "OCF1_movable_open": f"{a}/{len(open_seeds)}",
        "OCF1_movable_committed": f"{b}/{len(committed_seeds)}",
        "OCF1_seed_fisher_p": float(p_seed),
        "OCF1_pass": bool(p_seed < 0.05),
        "OCF2_moved_open_runs": f"{om}/{len(open_runs)}",
        "OCF2_moved_committed_runs": f"{cm}/{len(com_runs)}",
        "OCF2_run_fisher_p": float(p_run),
        "OCF3_auc_openness": auc_open,
        "OCF3_pass": bool(auc_open is not None and auc_open >= 0.70),
        "OCF4_strict_flips_open": strict_flips_open,
        "OCF4_strict_flips_committed": strict_flips_com,
        "OCF4_flips_only_in_open": bool(strict_flips_com == 0),
        "OCF5_auc_soups_deficit": auc_soups,
    }
    (OUTPUTS / "oc_ring_fixed_time.json").write_text(json.dumps({
        "status": ("OC-RING-FIXT fixed-time causal commitment test; all "
                   "seeds perturbed at the same training step; registered "
                   "in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md "
                   "before run"),
        "config": {"t_fix": T_FIX, "open_threshold": OPEN_THRESHOLD,
                   "resume_steps": 400_000, "noise_scales": NOISE_SCALES,
                   "commit_margin": 0.3},
        "baselines_at_tfix": baselines,
        "runs": runs, "registered_outcomes": outcomes}, indent=1))
    print(json.dumps(outcomes, indent=1))


if __name__ == "__main__":
    main()
