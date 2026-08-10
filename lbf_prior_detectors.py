"""Prior emergence detectors on the deep-MARL domain (LBF).

Closes the last domain-coverage gap in the detector comparison: single-
signal literature detectors have been run on the gridworld battery
(prior_metrics_comparison.py, plus exact rival formalisms in
exact_prior_formalisms.py) and on external chess (chess_prior_detectors
.py), but not on either deep-MARL task. This script runs them on
Level-Based Foraging using the SAVED main-run networks (no retraining).

Systems (7) and ground-truth labels:

- trained_seed{11,22,33} (label 1): the main-run policies; the frozen
  registered criterion already accepted them (4/4 predictions,
  lbf_collapse_main.json).
- untrained (0): fresh random network.
- greedy_nearest (0): scripted controller -- structure prespecified.
- noise (0): uniform random actions.
- forced_commit (0): the trained seed-11 network with agent 0 held
  under a PERMANENT measurement-time do_commit toward the nearest
  remaining food. The deep-MARL analogue of the battery's
  blind_trigger: identical underlying policy, but the trigger is
  imposed by the observer, not chosen by the system, so endogeneity
  fails BY CONSTRUCTION (the composite criterion rejects it on that
  component definitionally, exactly as blind_trigger was labeled).

Detectors (each scored per system, then given its hindsight-OPTIMAL
threshold, either direction -- the rival's best case):

1. performance_level: final win rate ("the ability is present, hence
   emergence" -- the colloquial reading of ability curves).
2. specificity_js: JS(P(B | do_commit), P(B | do_block)) at the episode
   start state, mean over episodes (do-operator macro effectiveness
   without a value sign).
3. synergy: plug-in I((X1,X2); B) - I(X1; B) - I(X2; B), agent
   quadrants at step 2 vs final basin (PID-flavored).
4. psi_rosas: Rosas' practical criterion on the pooled step-level
   behavioral series, Psi = I(V_t; V_{t+1}) - sum_j I(X^j_t; V_{t+1}),
   V = remaining-food set, X^j = agent-j quadrant (sampled analogue of
   the exact computation in exact_prior_formalisms.py).
5. ei_do_gap: Hoel-flavored proxy, I(macro; B) - I(micro; B), macro =
   uniform {do_commit, do_block} at t=0, micro = agent-0 first action
   forced uniform then on-policy.

REGISTERED PREDICTION (before running, same practice as
prior_metrics_comparison.py): forced_commit inherits the trained
policy's coordination machinery, so every detector that reads structure
or do-response (2-5) and the ability reading (1) should score it in the
trained range -- i.e. each single detector misclassifies forced_commit
(or pays for excluding it by losing a trained system). None of the five
signals can see WHO chose the trigger. If some single detector
separates all 7 correctly, that is a real finding against the
comparison claim in this domain and gets reported as such.

ROUND-1 OUTCOME (recorded, archived in lbf_prior_detectors_round1.json
before any change): the prediction FAILED for detector 1 --
performance_level separated all 7 systems (best acc 1.000), because
forced_commit's permanent commitment halves its win rate (0.48 vs
0.85+ trained), so the initial system set contained NO competent
imitation: every label-0 system also had low win rate. Detectors 2-5
behaved as registered (0.714-0.857, specificity missing forced_commit
exactly as predicted). This is a set-composition gap, not evidence
that win rate reads emergence: the gridworld battery already witnesses
performance accepting reward-forced systems (shaped_process,
useful_habit). Fix: add the missing control-family member.

ROUND 2 (this file): + scripted_coop (label 0), a deterministic
hand-coded coordinator -- both agents path (with collision avoidance)
to the SAME food and LOAD together; measured win rate 1.0 in the
smoke test; the structure is 100% prespecified by the designer, so
endogeneity and acquisition fail by construction (the composite
criterion rejects it definitionally, as with the battery's
shaped_process). REGISTERED ROUND-2 PREDICTION (frozen before the
round-2 run): performance_level now misclassifies scripted_coop (it
must accept the highest-win-rate system or lose the trained ones);
the structure detectors (specificity, synergy, psi) score
scripted_coop in or above the trained range (its coordination is
tighter than the learned one); hence no single detector reaches 1.0
on the 8-system set.

Output: outputs/lbf_prior_detectors.json (round 2, 8 systems)
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import lbf_collapse_probe as lbf

OUTPUTS = Path(__file__).resolve().parent / "outputs"

EPISODES = 60
PROBE_ROLLOUTS = 32
PROBE_TEMPERATURE = 6.0  # frozen main-run value

TRUTH = {
    "trained_seed11": 1, "trained_seed22": 1, "trained_seed33": 1,
    "untrained": 0, "greedy_nearest": 0, "noise": 0, "forced_commit": 0,
    "scripted_coop": 0,
}

MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


class BlockAwareController(lbf.Controller):
    """Adds do_block support for non-policy kinds (uniform over the
    allowed mask) so specificity/EI detectors run on every system."""

    def act(self, env, obs, rng, interventions=None):
        if self.kind == "policy" or (not interventions
                                     and self.kind != "scripted_coop"):
            return super().act(env, obs, rng, interventions)
        acts: List[int] = []
        for i in range(lbf.N_AGENTS):
            iv = (interventions or {}).get(i)
            if iv and iv["type"] == "do_commit":
                acts.append(lbf.greedy_action_toward(env, i, iv["target"]))
                continue
            if iv and iv["type"] == "do_block":
                pos = env.players[i].position
                target = iv["target"]
                d0 = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
                allowed = [1.0] * lbf.N_ACTIONS
                for a, (dr, dc) in moves.items():
                    nd = abs(pos[0] + dr - target[0]) + abs(pos[1] + dc - target[1])
                    if nd < d0:
                        allowed[a] = 0.0
                if lbf.adjacent(pos, target):
                    allowed[5] = 0.0
                choices = [a for a in range(lbf.N_ACTIONS) if allowed[a] > 0]
                acts.append(rng.choice(choices or list(range(lbf.N_ACTIONS))))
                continue
            if self.kind == "scripted_coop":
                acts.append(self.coop_action(env, i, rng))
                continue
            single = super().act(env, obs, rng, None)
            acts.append(single[i])
        return tuple(acts)

    def coop_action(self, env, i: int, rng: random.Random) -> int:
        """Prespecified coordination: both agents path (collision-aware)
        to the SAME food -- nearest to agent 0 -- and LOAD together."""
        foods = lbf.food_positions(env)
        if not foods:
            return 0
        pos0 = env.players[0].position
        target = min(foods,
                     key=lambda f: abs(pos0[0] - f[0]) + abs(pos0[1] - f[1]))
        player = env.players[i]
        pos = player.position
        if lbf.adjacent(pos, target):
            return 5
        occupied = {p.position for p in env.players if p is not player}
        rows, cols = env.field.shape
        best: List[int] = []
        bestd: Optional[int] = None
        for a, (dr, dc) in MOVES.items():
            cand = (pos[0] + dr, pos[1] + dc)
            if not (0 <= cand[0] < rows and 0 <= cand[1] < cols):
                continue
            if cand in occupied or env.field[cand] > 0:
                continue
            d = abs(cand[0] - target[0]) + abs(cand[1] - target[1])
            if bestd is None or d < bestd:
                bestd, best = d, [a]
            elif d == bestd:
                best.append(a)
        return rng.choice(best) if best else 0


class ForcedCommitController(BlockAwareController):
    """Trained policy with agent 0 permanently committed (dynamic target:
    nearest remaining food). Observer-imposed trigger -> endogeneity
    fails by construction."""

    def act(self, env, obs, rng, interventions=None):
        foods = lbf.food_positions(env)
        merged = dict(interventions or {})
        if foods and 0 not in merged:
            pos = env.players[0].position
            target = min(foods,
                         key=lambda f: abs(pos[0] - f[0]) + abs(pos[1] - f[1]))
            merged[0] = {"type": "do_commit", "target": target}
        return super().act(env, obs, rng, merged or None)


def quadrant(pos: Tuple[int, int]) -> int:
    return (2 if pos[0] >= 3 else 0) + (1 if pos[1] >= 3 else 0)


def entropy_bits_counter(c: Counter) -> float:
    total = sum(c.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in c.values() if v > 0)


def mi_bits(pairs: List[Tuple]) -> float:
    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    return (entropy_bits_counter(left) + entropy_bits_counter(right)
            - entropy_bits_counter(joint))


def js_bits(p: Dict, q: Dict) -> float:
    keys = set(p) | set(q)
    tp = sum(p.values()) or 1
    tq = sum(q.values()) or 1
    m = {k: 0.5 * (p.get(k, 0) / tp + q.get(k, 0) / tq) for k in keys}
    def kl(a, ta):
        out = 0.0
        for k in keys:
            ak = a.get(k, 0) / ta
            if ak > 0:
                out += ak * math.log2(ak / m[k])
        return out
    return 0.5 * kl(p, tp) + 0.5 * kl(q, tq)


def probe_system(name: str, controller: lbf.Controller, seed: int) -> Dict:
    """Behaving episodes + t=0 probe measurements for all detectors."""
    rng = random.Random(seed * 6151 + 7)
    env = lbf.make_raw_env(seed)
    sim = lbf.make_raw_env(seed + 10 ** 6)
    if controller.kind == "policy":
        probe = type(controller)("policy", controller.net, PROBE_TEMPERATURE)
    else:
        probe = controller

    wins: List[float] = []
    spec_js: List[float] = []
    syn_samples: List[Tuple[Tuple, int, Tuple]] = []  # (x1, x2 packed later)
    psi_pairs: List[Tuple[Tuple, Tuple]] = []  # ((V,X1,X2)_t, V_{t+1}) raw
    macro_pairs: List[Tuple[str, Tuple]] = []
    micro_pairs: List[Tuple[int, Tuple]] = []

    for ep in range(EPISODES):
        env.reset(seed=2_000_000 + 997 * seed + ep)
        findex = lbf.FoodIndex(env)
        snap0 = lbf.world_snapshot(env, ())

        # t=0 probes
        foods = lbf.food_positions(env)
        pos0 = env.players[0].position
        target = min(foods, key=lambda f: abs(pos0[0] - f[0]) + abs(pos0[1] - f[1]))
        d_commit = lbf.future_distribution(
            sim, snap0, findex, probe, rng,
            {0: {"type": "do_commit", "target": target}}, rollouts=PROBE_ROLLOUTS)
        d_block = lbf.future_distribution(
            sim, snap0, findex, probe, rng,
            {0: {"type": "do_block", "target": target}}, rollouts=PROBE_ROLLOUTS)
        spec_js.append(js_bits(d_commit, d_block))

        macro = "do_commit" if ep % 2 == 0 else "do_block"
        iv = {0: {"type": "do_" + ("commit" if macro == "do_commit" else "block"),
                  "target": target}}
        macro_pairs.append(
            (macro, lbf.rollout_basin(sim, snap0, findex, probe, rng, iv)))
        forced_a0 = rng.randrange(lbf.N_ACTIONS)
        lbf.world_restore(sim, snap0)
        obs = lbf.obs_all(sim)
        acts = list(probe.act(sim, obs, rng))
        acts[0] = forced_a0
        sim.step(tuple(acts))
        snap1 = lbf.world_snapshot(sim, ())
        # basin ids from snap1 must include already-consumed foods
        before0 = set(findex.positions)
        after1 = set(lbf.food_positions(sim))
        pre_order = findex.consumed_now(before0, after1) if after1 != before0 else []
        snap1 = lbf.world_snapshot(sim, tuple(pre_order))
        micro_pairs.append(
            (forced_a0, lbf.rollout_basin(sim, snap1, findex, probe, rng)))

        # behaving episode with step-level series
        order: List[int] = []
        before = set(lbf.food_positions(env))
        series: List[Tuple[Tuple, int, int]] = []  # (V, x1, x2)
        x_step2 = None
        for t in range(lbf.MAX_STEPS):
            if env.field.sum() == 0:
                break
            remaining = frozenset(
                findex.positions.index(p) for p in lbf.food_positions(env))
            q1 = quadrant(env.players[0].position)
            q2 = quadrant(env.players[1].position)
            series.append((tuple(sorted(remaining)), q1, q2))
            if t == 2:
                x_step2 = (q1, q2)
            obs = lbf.obs_all(env)
            env.step(controller.act(env, obs, rng))
            after = set(lbf.food_positions(env))
            if after != before:
                order.extend(findex.consumed_now(before, after))
                before = after
        basin = tuple(order)
        wins.append(1.0 if lbf.is_win(basin) else 0.0)
        if x_step2 is not None:
            syn_samples.append((x_step2[0], x_step2[1], basin))
        for a, b in zip(series, series[1:]):
            psi_pairs.append((a, b[0]))  # ((V,x1,x2)_t, V_{t+1})

    synergy = (mi_bits([((x1, x2), b) for x1, x2, b in syn_samples])
               - mi_bits([(x1, b) for x1, _, b in syn_samples])
               - mi_bits([(x2, b) for _, x2, b in syn_samples]))
    psi = (mi_bits([(v, v2) for (v, _, _), v2 in psi_pairs])
           - mi_bits([(x1, v2) for (_, x1, _), v2 in psi_pairs])
           - mi_bits([(x2, v2) for (_, _, x2), v2 in psi_pairs]))
    ei = mi_bits(macro_pairs) - mi_bits(micro_pairs)

    return {
        "performance_level": float(np.mean(wins)),
        "specificity_js": float(np.mean(spec_js)),
        "synergy": synergy,
        "psi_rosas": psi,
        "ei_do_gap": ei,
    }


def hindsight_best(scores: Dict[str, float]) -> Dict:
    best: Dict = {"accuracy": -1.0}
    values = sorted(set(scores.values()))
    cuts = ([values[0] - 1.0]
            + [0.5 * (a + b) for a, b in zip(values, values[1:])]
            + [values[-1] + 1.0])
    for direction in (1, -1):
        for cut in cuts:
            preds = {s: int(direction * v > direction * cut)
                     for s, v in scores.items()}
            acc = sum(int(preds[s] == TRUTH[s]) for s in scores) / len(scores)
            if acc > best["accuracy"]:
                best = {"accuracy": acc, "threshold": cut,
                        "direction": direction,
                        "misclassified": sorted(
                            s for s in scores if preds[s] != TRUTH[s])}
    return best


def main() -> None:
    torch.set_num_threads(16)
    systems: Dict[str, Tuple[lbf.Controller, int]] = {}
    for seed in (11, 22, 33):
        net = lbf.PolicyNet()
        net.load_state_dict(torch.load(OUTPUTS / f"lbf_net_seed{seed}.pt",
                                       weights_only=True))
        net.eval()
        systems[f"trained_seed{seed}"] = (
            BlockAwareController("policy", net), seed)
    torch.manual_seed(999)
    systems["untrained"] = (BlockAwareController("policy", lbf.PolicyNet()), 44)
    systems["greedy_nearest"] = (BlockAwareController("greedy_nearest"), 55)
    systems["noise"] = (BlockAwareController("noise"), 66)
    net11 = systems["trained_seed11"][0].net
    systems["forced_commit"] = (ForcedCommitController("policy", net11), 77)
    systems["scripted_coop"] = (BlockAwareController("scripted_coop"), 88)

    per_system: Dict[str, Dict] = {}
    for name, (controller, seed) in systems.items():
        per_system[name] = probe_system(name, controller, seed)
        print(name, json.dumps(
            {k: round(v, 4) for k, v in per_system[name].items()}), flush=True)

    detectors = ("performance_level", "specificity_js", "synergy",
                 "psi_rosas", "ei_do_gap")
    results: Dict[str, Dict] = {}
    print(f"\n{'detector':20s} {'best_acc':>8s}  misclassified")
    for det in detectors:
        scores = {s: per_system[s][det] for s in per_system}
        hb = hindsight_best(scores)
        results[det] = {"scores": scores, **hb}
        print(f"{det:20s} {hb['accuracy']:8.3f}  "
              f"{';'.join(hb['misclassified']) or '-'}")

    out = {
        "note": ("Prior single-signal detectors on the LBF deep-MARL domain, "
                 "saved main-run nets, hindsight-optimal thresholds. "
                 "forced_commit = trained policy + permanent observer-imposed "
                 "do_commit (blind_trigger analogue; endogeneity fails by "
                 "construction). Registered prediction in module docstring: "
                 "every single detector misclassifies forced_commit or pays "
                 "elsewhere for excluding it."),
        "truth": TRUTH,
        "per_system": per_system,
        "detectors": results,
    }
    (OUTPUTS / "lbf_prior_detectors.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUTS / 'lbf_prior_detectors.json'}")


if __name__ == "__main__":
    main()
