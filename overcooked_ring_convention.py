"""OC-RING: mechanism recovery for the Overcooked negative.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Standard
overcooked_ai layouts only; training mechanics imported unchanged from
overcooked_genesis_curve.train_with_checkpoints. The official
coordination_ring layout supplies the one ingredient the theory says
cramped_room lacks: two mirror-equivalent joint conventions (CW/CCW
circulation) whose value exists only when shared.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from ant_fine_onset import adjudicate
from overcooked_genesis_curve import train_with_checkpoints
from overcooked_pilot import PolicyNet

OUTPUTS = Path(__file__).resolve().parent / "outputs"
TOTAL_STEPS = 2_000_000
CKPT_EVERY = 20_000
CHECKPOINTS = tuple(range(CKPT_EVERY, TOTAL_STEPS + 1, CKPT_EVERY))
RING_SEEDS = (95_101, 95_202, 95_303)
CRAMPED_SEEDS = (95_404, 95_505)
EVAL_EPISODES = 30
HORIZON = 200
LAPS_MIN = 0.5
LOG2_3 = math.log2(3)
RING_CENTER = (2.0, 2.0)  # central counter block of the 5x5 ring


def winding_laps(positions) -> float:
    """Net winding (in laps) of one agent's trajectory around the ring
    centre; positive = counterclockwise in grid coordinates."""
    total = 0.0
    prev = None
    for (x, y) in positions:
        ang = math.atan2(y - RING_CENTER[1], x - RING_CENTER[0])
        if prev is not None:
            d = ang - prev
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            total += d
        prev = ang
    return total / (2 * math.pi)


def eval_checkpoint(ckpt_path: Path, layout: str, seed: int):
    net = PolicyNet()
    net.load_state_dict(torch.load(ckpt_path, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    env = oc.make_env(layout)
    dir_signs, soups, ent_sum, ent_n = [], [], 0.0, 0
    for ep in range(EVAL_EPISODES):
        torch.manual_seed(seed * 10_000 + ep)
        env.reset()
        pos = [[], []]
        sparse_total = 0.0
        for _t in range(HORIZON):
            obs = oc.featurize(env)
            with torch.no_grad():
                logits, _ = net(torch.tensor(np.stack(obs)))
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log2(probs.clamp(min=1e-12))
                        ).sum(dim=-1)
                ent_sum += float(ent.mean().item())
                ent_n += 1
                acts = torch.distributions.Categorical(probs=probs).sample()
            actions = [Action.ALL_ACTIONS[a] for a in acts.tolist()]
            for i, p in enumerate(env.state.players):
                pos[i].append(p.position)
            _s, sparse_r, done, _info = env.step(actions)
            sparse_total += sparse_r
            if done:
                break
        soups.append(sparse_total / 20.0)  # soups delivered (reward 20 each)
        w = winding_laps(pos[0]) + winding_laps(pos[1])
        if abs(w) >= LAPS_MIN:
            dir_signs.append(1 if w > 0 else -1)
    n_ccw = sum(1 for s in dir_signs if s > 0)
    n_com = len(dir_signs)
    p_ccw = (n_ccw + 1) / (n_com + 2)  # Laplace smoothing
    h2 = -(p_ccw * math.log2(p_ccw) + (1 - p_ccw) * math.log2(1 - p_ccw))
    return {
        "circulation_openness": round(h2, 5),
        "p_ccw": round(p_ccw, 4),
        "n_committed_episodes": n_com,
        "mean_soups": round(float(np.mean(soups)), 4),
        "mean_policy_entropy_norm": round(ent_sum / max(ent_n, 1)
                                          / math.log2(6), 5),
    }


def run_system(layout: str, seed: int, tag: str):
    print(f"=== training {layout} seed {seed}", flush=True)
    saved = train_with_checkpoints((layout, layout), seed, TOTAL_STEPS,
                                   CHECKPOINTS, tag)
    grid, rows = [], []
    for ck in sorted(saved):
        r = eval_checkpoint(saved[ck], layout, seed)
        grid.append(ck)
        rows.append(r)
        if ck % 200_000 == 0:
            print(f"  {layout} s{seed} ck={ck}: open={r['circulation_openness']} "
                  f"p_ccw={r['p_ccw']} n_com={r['n_committed_episodes']} "
                  f"soups={r['mean_soups']} Hpol={r['mean_policy_entropy_norm']}",
                  flush=True)
    circ = np.array([r["circulation_openness"] for r in rows])
    pol = np.array([r["mean_policy_entropy_norm"] for r in rows])
    soups = np.array([r["mean_soups"] for r in rows])
    adj_circ = adjudicate(grid, circ * LOG2_3)
    adj_pol = adjudicate(grid, pol * LOG2_3)
    final_rate = soups[-1]
    cross = next((grid[i] for i in range(len(soups))
                  if soups[i] >= 0.5 * final_rate and final_rate > 0), None)
    h = adj_circ.get("hinge", {})
    return {
        "grid": grid,
        "curves": rows,
        "final_soups": float(final_rate),
        "capability_crossing": cross,
        "final_p_ccw": rows[-1]["p_ccw"],
        "circ_adj": {"b5_onset": adj_circ["b5_onset"],
                     "t_star": h.get("t_star"),
                     "delta_bic": h.get("delta_bic")},
        "pol_adj": {"b5_onset": adj_pol["b5_onset"],
                    "t_star": adj_pol.get("hinge", {}).get("t_star"),
                    "delta_bic": adj_pol.get("hinge", {}).get("delta_bic")},
    }


def main() -> None:
    torch.set_num_threads(4)
    report = {"ring": {}, "cramped": {}}
    for seed in RING_SEEDS:
        report["ring"][str(seed)] = run_system("coordination_ring", seed,
                                               f"ring{seed}")
    for seed in CRAMPED_SEEDS:
        report["cramped"][str(seed)] = run_system("cramped_room", seed,
                                                  f"crmp{seed}")

    ring = report["ring"].values()
    onset_seeds = [r for r in ring if r["circ_adj"]["b5_onset"]]
    dirs = [(">0.5" if r["final_p_ccw"] > 0.5 else "<0.5")
            for r in ring if r["circ_adj"]["b5_onset"]]
    ocr3 = all(r["circ_adj"]["t_star"] <= r["capability_crossing"]
               for r in onset_seeds if r["capability_crossing"] is not None)
    outcomes = {
        "OCR1_ring_onset_ge_2of3": bool(len(onset_seeds) >= 2),
        "OCR2_directions": dict(Counter(dirs)),
        "OCR3_collapse_leads_capability": bool(ocr3 and onset_seeds),
        "OCR4_policy_entropy_no_onset_ring": bool(
            all(not r["pol_adj"]["b5_onset"] for r in ring)),
        "OCR5_cramped_no_onset_matched_grid": bool(
            all(not r["pol_adj"]["b5_onset"]
                for r in report["cramped"].values())),
        "n_ring_onset": len(onset_seeds),
    }
    out = OUTPUTS / "overcooked_ring_convention.json"
    out.write_text(json.dumps({
        "status": ("OC-RING mechanism recovery; standard overcooked_ai "
                   "layouts, training mechanics imported unchanged; "
                   "registered before run"),
        "config": {"total_steps": TOTAL_STEPS, "ckpt_every": CKPT_EVERY,
                   "eval_episodes": EVAL_EPISODES, "horizon": HORIZON,
                   "laps_min": LAPS_MIN, "ring_seeds": RING_SEEDS,
                   "cramped_seeds": CRAMPED_SEEDS},
        "systems": report,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
