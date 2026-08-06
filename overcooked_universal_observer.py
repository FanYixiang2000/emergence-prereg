"""Universal-observer recipe on the public Overcooked-AI domain (U-4).

Completes the cross-domain universality claim: the same semantics-free
possibility-space recipe validated on the gridworld battery, the crowd
domain and CLBF (universal_observer.py) is applied, with the same
frozen function objects and hyperparameters, to the PUBLIC third-party
benchmark using the stored round-1 confirmation policies.

Raw encoding (zero benchmark semantics): per-episode joint-action
histogram (36 opaque action-pair counts, normalized by horizon) plus
the raw sparse score scaled by 1/100. No pot, role, layout or recipe
knowledge enters the features.

Declared systems: seeds 77002, 77007, 77010 (accepted in round 1) and
77005 (selectivity-rejected in round 1), five systems each (learned,
initialization twin, scripted roles, BC clone, untrained-other),
re-evaluated at 20 episodes per context per condition (reduced from
the confirmatory 40 for compute; disclosed). Hand-basin verdicts are
recomputed side by side on the same rollout budget, so the comparison
is like-for-like (the lesson of the disclosed run-1 correction in
universal_observer.py).

Registered predictions (frozen before running):

    U4a  Verdict agreement (universal vs hand, same budget) on
         >= 18/20 system-verdicts.
    U4b  All 16 control verdicts under the universal recipe are
         rejections.
    U4c  Seed 77005's learned policy remains rejected under the
         universal recipe, and every accepted-seed learned policy
         remains accepted.

Misses are retained.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

np.Inf = np.inf
import torch

from overcooked_ai_py.mdp.actions import Action

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from universal_observer import (
    fit_universal_basins,
    assign,
    cluster_dist,
    basin_entropy,
    basin_js,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

SEEDS = (77002, 77007, 77010, 77005)
LAYOUTS = ("cramped_room", "asymmetric_advantages")
N_EVAL = 20
N_ACTIONS = len(Action.ALL_ACTIONS)


def rollout(policy: oc.TeamPolicy, layout: str, seed: int,
            intervention: Optional[str]) -> Dict:
    """oc.run_episode with a raw action-pair histogram added."""
    env = oc.make_env(layout)
    env.reset()
    rng = random.Random(seed)
    first_potter = None
    agent1_potted = False
    sparse_total = 0.0
    committed_done = False
    hist = np.zeros(N_ACTIONS * N_ACTIONS)
    steps = 0
    while True:
        obs = oc.featurize(env)
        actions = policy.actions(env, obs, rng)
        if intervention == "do_commit" and first_potter is None \
                and not committed_done:
            actions[0] = oc.onion_to_pot_action(env, 0, rng)
        if intervention == "do_block" and not agent1_potted:
            if oc.is_potting_interact(env, 0, actions[0]):
                actions[0] = Action.STAY
        a0 = Action.ALL_ACTIONS.index(actions[0]) \
            if actions[0] in Action.ALL_ACTIONS else N_ACTIONS - 1
        a1 = Action.ALL_ACTIONS.index(actions[1]) \
            if actions[1] in Action.ALL_ACTIONS else N_ACTIONS - 1
        hist[a0 * N_ACTIONS + a1] += 1
        steps += 1
        _s, sparse_r, done, _info = env.step(actions)
        sparse_total += sparse_r
        gs = env.game_stats
        pots0 = list(gs.get("potting_onion", [[], []])[0]) + \
            list(gs.get("potting_tomato", [[], []])[0])
        pots1 = list(gs.get("potting_onion", [[], []])[1]) + \
            list(gs.get("potting_tomato", [[], []])[1])
        if first_potter is None:
            t0 = min(pots0) if len(pots0) else None
            t1 = min(pots1) if len(pots1) else None
            if t0 is not None and (t1 is None or t0 <= t1):
                first_potter = 0
            elif t1 is not None:
                first_potter = 1
        if len(pots1):
            agent1_potted = True
        if first_potter == 0:
            committed_done = True
        if done:
            break
    potter = {0: "pot0", 1: "pot1", None: "potnone"}[first_potter]
    deliver = "deliver" if sparse_total > 0 else "nodeliver"
    return {
        "tokens": [],
        "numeric": (hist / max(1, steps)).tolist()
        + [sparse_total / 100.0],
        "score": sparse_total,
        "trigger": int(first_potter == 0),
        "hand_basin": f"{potter}_{deliver}",
    }


def collect(policy: oc.TeamPolicy, offset: int) -> Dict[str, List[Dict]]:
    rolls = {m: [] for m in ("natural", "do_commit", "do_block")}
    for mode in rolls:
        iv = None if mode == "natural" else mode
        for ci, layout in enumerate(LAYOUTS):
            for j in range(N_EVAL):
                r = rollout(policy, layout, offset + 10_000 * ci + j, iv)
                r["context"] = ci
                rolls[mode].append(r)
    return rolls


def components(rolls: Dict[str, List[Dict]], basin_of) -> Dict:
    nat = rolls["natural"]
    trig = {c: float(np.mean([r["trigger"] for r in nat
                              if r["context"] == c])) for c in (0, 1)}

    def dist(rows):
        labels = [basin_of(r) for r in rows]
        keys = sorted(set(labels))
        return {k: labels.count(k) / len(labels) for k in keys}

    mean_s = lambda rows: float(np.mean([r["score"] for r in rows]))
    return {
        "potential_bits": basin_entropy(dist(nat)),
        "conditional_selectivity": abs(trig[0] - trig[1]),
        "specificity_js_bits": basin_js(dist(rolls["do_commit"]),
                                        dist(rolls["do_block"])),
        "usefulness_gap": mean_s(nat) - mean_s(rolls["do_block"]),
    }


def main() -> None:
    torch.set_num_threads(16)
    report = {"status": ("universal recipe on public Overcooked "
                         "(U-4a..c frozen in docstring); reduced "
                         "20-episode budget disclosed"), "seeds": {}}
    agree = 0
    controls_rejected = 0
    learned_ok = True
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        net = PolicyNet()
        net.load_state_dict(torch.load(
            OUTPUTS / f"overcooked_confirm_s{seed}.pt",
            weights_only=True))
        net.eval()
        torch.manual_seed(seed)
        twin = PolicyNet()
        twin.eval()
        torch.manual_seed(seed + 999)
        untrained = PolicyNet()
        untrained.eval()
        clone = oc.train_bc_clone(LAYOUTS, seed + 31)
        systems = {
            "learned": (oc.TeamPolicy("net", net=net), True),
            "initial_twin": (oc.TeamPolicy("net", net=twin), True),
            "scripted_roles": (oc.TeamPolicy("scripted_roles",
                                             cook_agent=0), False),
            "bc_clone": (oc.TeamPolicy("clone", net=clone), False),
            "untrained_other": (oc.TeamPolicy("net", net=untrained),
                                True),
        }
        offset = 80_000_000 + seed * 100_000
        rolls = {n: collect(pol, offset)
                 for n, (pol, _e) in systems.items()}
        pooled = [r for n in rolls for r in rolls[n]["natural"]]
        vocab, scaler, km, k = fit_universal_basins(pooled)
        print(f"  universal recipe chose k={k}", flush=True)

        uni_label = {}
        for n in rolls:
            for mode in rolls[n]:
                labels = assign(rolls[n][mode], vocab, scaler, km)
                for r, lab in zip(rolls[n][mode], labels):
                    r["uni_basin"] = str(lab)

        seed_rows = {}
        m_uni_all = {n: components(rolls[n], lambda r: r["uni_basin"])
                     for n in systems}
        m_hand_all = {n: components(rolls[n], lambda r: r["hand_basin"])
                      for n in systems}
        for n, (pol, endo) in systems.items():
            acq_u = acq_h = 0.0
            if n == "learned":
                acq_u = (m_uni_all["learned"]["conditional_selectivity"]
                         - m_uni_all["initial_twin"][
                             "conditional_selectivity"])
                acq_h = (m_hand_all["learned"]["conditional_selectivity"]
                         - m_hand_all["initial_twin"][
                             "conditional_selectivity"])
            v_uni = oc.verdict(m_uni_all[n], endo, acq_u)
            v_hand = oc.verdict(m_hand_all[n], endo, acq_h)
            agree += int(v_uni["emergent"] == v_hand["emergent"])
            if n != "learned":
                controls_rejected += int(v_uni["emergent"] == 0)
            seed_rows[n] = {"universal": v_uni["emergent"],
                            "hand": v_hand["emergent"],
                            "universal_failed": v_uni["failed"],
                            "k": k}
            print(f"  {n}: universal {v_uni['emergent']} vs hand "
                  f"{v_hand['emergent']}", flush=True)
        expected = 0 if seed == 77005 else 1
        if seed_rows["learned"]["universal"] != expected:
            learned_ok = False
        report["seeds"][str(seed)] = seed_rows
        out = OUTPUTS / "overcooked_universal_observer.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["registered_outcomes"] = {
        "U4a_agreement": f"{agree}/20 -> {agree >= 18}",
        "U4b_controls_rejected":
            f"{controls_rejected}/16 -> {controls_rejected == 16}",
        "U4c_learned_pattern_preserved": learned_ok,
    }
    out = OUTPUTS / "overcooked_universal_observer.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
