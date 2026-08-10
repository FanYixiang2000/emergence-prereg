"""EXACT prior-formalism detectors on the battery (no proxies, no sampling).

Closes the "strawman" objection to the prior-detector comparison
(prior_metrics_comparison.py): the flavored detectors there are replaced
here by the rival theories' own formal quantities, computed EXACTLY on
the policy-closed Markov chain of each battery system. The gridworld is
small enough to enumerate the full closed state space

    (mode, preference_context, a0_pos, a1_pos, switch_flag, t)

(<= 75,000 non-terminal states + 4 absorbing basin states); transitions
are deterministic given the joint action and the softmax policy is an
explicit formula, so the transition probability matrix is exact and
every information quantity below carries ZERO Monte-Carlo error.

1. Causal emergence, CE 1.0 (Hoel, Hoel-Albantakis-Tononi PNAS 2013):
       EI(TPM) = I(X_{t+1}; X_t),  X_t ~ maximum-entropy (uniform)
               = (1/N) sum_s KL( T(s, .) || mean-row )
   on the exact micro TPM, and on exact macro TPMs (macro row = uniform
   average of member micro rows, Hoel's macro mechanism) for a family
   of candidate coarse-grainings. Hoel's framework asks whether THERE
   EXISTS a macro beating the micro; exhaustive partition search is
   super-exponential and is not performed in the original papers either
   -- we search a structured family that includes the mechanistically
   privileged partitions (switch flag, mode x switch, basin forecast,
   forecast x switch, win-probability quartiles) and give the rival its
   best member:
       CE = max_partition EI(macro) - EI(micro).

2. Information-decomposition causal emergence (Rosas et al., PLoS
   Comput Biol 2020), practical criterion Psi (their first-order form,
   which they state as the sufficient condition Psi > 0):
       Psi(V) = I(V_t; V_{t+1}) - sum_j I(X^j_t; V_{t+1})
   with V a supervenient macro feature and micro components taken from
   TWO decompositions, the rival keeping the better one: agents-only
   (X^1 = a0 position, X^2 = a1 position -- generous: fewer subtracted
   terms) and full-state (adding X^3 = (switch flag, t) and
   X^4 = (mode, context)).
   Computed exactly from the behavioral occupancy measure of the chain
   (exact forward propagation from the episode initial distribution,
   pooled one-step joints over the horizon, terminal self-loops
   excluded). We evaluate a candidate family of V's and give the rival
   its best member. Because the final verdict uses the hindsight-
   optimal threshold, constant level shifts (e.g. Psi's redundancy
   bias) cannot hurt the rival -- only its ORDERING of systems matters.

Scoring mirrors prior_metrics_comparison.py: each detector gets its
hindsight-OPTIMAL threshold (either direction) on the audited battery
labels; we also report the rivals' own natural thresholds (score > 0).
Ground truth = the battery's audited labels; the full criterion scores
10/10 on the same battery with frozen thresholds.

Output: outputs/exact_prior_formalisms.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from contextual_sacrifice_gridworld import (
    A0_START,
    A1_START,
    GRID_SIZE,
    HIGH_GOAL,
    JOINT_ACTIONS,
    MAX_STEPS,
    MODES,
    SAFE_EXIT,
    SWITCH,
    TEAM_A0,
    TEAM_A1,
    manhattan,
    move_position,
    q_values,
    train_policy,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

TEMPERATURE = 0.25  # the battery's behaving temperature
BASINS = ("sacrifice_rescue", "team_direct", "selfish_escape", "failed_noise")

# system -> (base regime or None for untrained, behavior intervention, modes)
SYSTEMS: Dict[str, Tuple[Optional[str], object, List[str]]] = {
    "latent_conditional": ("uncertain_preference", None, list(MODES)),
    "converged_team": ("pure_team", None, list(MODES)),
    "shaped_process": ("dense_shaping", None, list(MODES)),
    "noise_policy": ("random_noise", None, list(MODES)),
    "untrained_uniform": (None, None, list(MODES)),
    "blind_trigger": ("uncertain_preference", "do_trigger", list(MODES)),
    "harmful_decoy": ("uncertain_preference", "do_trigger", ["bridge"]),
    "useful_habit": ("uncertain_preference", "do_trigger", ["rescue"]),
    "wrong_selector": ("uncertain_preference",
                       {"rescue": None, "bridge": "do_trigger"}, list(MODES)),
    "anti_selector": ("uncertain_preference",
                      {"rescue": "do_non_trigger", "bridge": "do_trigger"},
                      list(MODES)),
}

TRUTH = {
    "latent_conditional": 1, "converged_team": 0, "shaped_process": 0,
    "noise_policy": 1, "untrained_uniform": 0, "blind_trigger": 0,
    "harmful_decoy": 0, "useful_habit": 0, "wrong_selector": 0,
    "anti_selector": 0,
}


def contexts_for(regime: Optional[str]) -> Tuple[str, ...]:
    if regime == "uncertain_preference":
        return ("self_preservation", "visible_teamwork", "latent_sacrifice")
    return ("fixed",)


# Exact policy-closed Markov chain (integer-indexed)

Position = Tuple[int, int]
# Non-terminal chain state: (mode, context, a0, a1, switch_used, t)
# Absorbing: ("basin", name)


def positions() -> List[Position]:
    return [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]


def softmax_probs(q_table, state, intervention: Optional[str]
                  ) -> Dict[Tuple[str, str], float]:
    """Exact softmax action distribution with the battery's do-operator
    action restrictions (mirrors choose_with_intervention exactly)."""
    mode, context, a0_pos, a1_pos, switch_used, t = state
    env_state = (mode, a0_pos, a1_pos, switch_used, switch_used, t)
    values = q_values(q_table, env_state, context)
    actions = list(JOINT_ACTIONS)
    if intervention == "do_trigger" and not switch_used:
        toward = [a for a in actions
                  if manhattan(move_position(a0_pos, a[0]), SWITCH)
                  < manhattan(a0_pos, SWITCH)]
        if toward:
            actions = toward
    elif intervention == "do_non_trigger" and not switch_used:
        actions = [a for a in actions
                   if move_position(a0_pos, a[0]) != SWITCH] or list(JOINT_ACTIONS)
    max_v = max(values[a] for a in actions)
    weights = {a: math.exp((values[a] - max_v) / TEMPERATURE) for a in actions}
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


def transition(state, action: Tuple[str, str]):
    """Deterministic env dynamics -> next chain state or absorbing basin.
    Mirrors ContextualSacrificeEnv.step + classify_basin exactly
    (gate_open == switch_used throughout)."""
    mode, context, a0_pos, a1_pos, switch_used, t = state
    next_a0 = move_position(a0_pos, action[0])
    next_a1 = move_position(a1_pos, action[1])
    switch = switch_used or (next_a0 == SWITCH)
    if next_a1 == HIGH_GOAL and switch:
        return ("basin",
                "sacrifice_rescue" if mode == "rescue" else "failed_noise")
    if next_a0 == SAFE_EXIT and not switch:
        return ("basin", "selfish_escape")
    if next_a0 == TEAM_A0 and next_a1 == TEAM_A1:
        return ("basin", "team_direct")
    if t + 1 >= MAX_STEPS:
        return ("basin", "failed_noise")
    return (mode, context, next_a0, next_a1, switch, t + 1)


def resolve_iv(behavior, mode: str) -> Optional[str]:
    if isinstance(behavior, dict):
        return behavior.get(mode)
    return behavior


class Chain:
    """Exact TPM of the policy-closed episodic chain, integer-indexed.

    states[i] for i < n_nonterminal are non-terminal tuples; the last
    four indices are the absorbing basin states (self-loops).
    """

    def __init__(self, q_table, regime: Optional[str], behavior,
                 modes: Sequence[str]):
        ctxs = contexts_for(regime)
        self.states: List = []
        for mode in modes:
            for context in ctxs:
                for a0 in positions():
                    for a1 in positions():
                        for sw in (False, True):
                            for t in range(MAX_STEPS):
                                self.states.append(
                                    (mode, context, a0, a1, sw, t))
        self.n_nonterminal = len(self.states)
        for b in BASINS:
            self.states.append(("basin", b))
        self.index = {s: i for i, s in enumerate(self.states)}
        self.rows: List[List[Tuple[int, float]]] = []
        for s in self.states:
            if s[0] == "basin":
                self.rows.append([(self.index[s], 1.0)])
                continue
            iv = resolve_iv(behavior, s[0])
            row: Dict[int, float] = defaultdict(float)
            for action, p in softmax_probs(q_table, s, iv).items():
                row[self.index[transition(s, action)]] += p
            self.rows.append(sorted(row.items()))
        # initial distribution: uniform over allowed (mode, context)
        self.init: Dict[int, float] = {}
        w = 1.0 / (len(modes) * len(ctxs))
        for mode in modes:
            for context in ctxs:
                s0 = (mode, context, A0_START, A1_START, False, 0)
                self.init[self.index[s0]] = w


# Exact EI (Hoel) and coarse-graining

def ei_bits_rows(n: int, rows: Sequence[Sequence[Tuple[int, float]]]) -> float:
    """EI = (1/N) sum_s KL(T(s,.) || Tbar), Tbar the uniform-input marginal
    of X_{t+1}: the exact maximum-entropy-intervention EI, in bits."""
    tbar: Dict[int, float] = defaultdict(float)
    for row in rows:
        for j, p in row:
            tbar[j] += p / n
    total = 0.0
    for row in rows:
        for j, p in row:
            if p > 0:
                total += p * math.log2(p / tbar[j])
    return total / n


def macro_ei_bits(chain: Chain, label) -> float:
    """Exact EI of the macro TPM under coarse-graining `label` (macro row =
    uniform average of member micro rows, Hoel's macro mechanism)."""
    group_of: List[int] = []
    group_index: Dict[object, int] = {}
    for s in chain.states:
        g = label(s)
        if g not in group_index:
            group_index[g] = len(group_index)
        group_of.append(group_index[g])
    n_groups = len(group_index)
    sizes = [0] * n_groups
    for gi in group_of:
        sizes[gi] += 1
    macro_rows: List[Dict[int, float]] = [defaultdict(float)
                                          for _ in range(n_groups)]
    for i, row in enumerate(chain.rows):
        gi = group_of[i]
        for j, p in row:
            macro_rows[gi][group_of[j]] += p / sizes[gi]
    return ei_bits_rows(n_groups,
                        [sorted(r.items()) for r in macro_rows])


def absorption_map(chain: Chain) -> List[Dict[str, float]]:
    """Exact basin-absorption distribution from every state (episodes
    always absorb within MAX_STEPS because t is part of the state)."""
    dist: List[Optional[Dict[str, float]]] = [None] * len(chain.states)
    for i in range(chain.n_nonterminal, len(chain.states)):
        dist[i] = {chain.states[i][1]: 1.0}
    # states with larger t absorb sooner: solve by decreasing t
    order = sorted(range(chain.n_nonterminal),
                   key=lambda i: -chain.states[i][5])
    for i in order:
        row_dist: Dict[str, float] = defaultdict(float)
        for j, p in chain.rows[i]:
            for b, q in dist[j].items():  # type: ignore[union-attr]
                row_dist[b] += p * q
        dist[i] = dict(row_dist)
    return dist  # type: ignore[return-value]


def candidate_partitions(chain: Chain, absorb: List[Dict[str, float]]):
    """Structured family of macro coarse-grainings. Absorbing states keep
    their basin identity in every partition."""
    idx = chain.index

    def base(s):
        return ("B", s[1]) if s[0] == "basin" else None

    def by_switch(s):
        return base(s) or ("sw", s[4])

    def by_mode_switch(s):
        return base(s) or ("ms", s[0], s[4])

    def by_forecast(s):
        b = base(s)
        if b:
            return b
        d = absorb[idx[s]]
        return ("fc", max(d, key=lambda k: d[k]))

    def by_forecast_switch(s):
        b = base(s)
        if b:
            return b
        d = absorb[idx[s]]
        return ("fs", max(d, key=lambda k: d[k]), s[4])

    def by_win_quartile(s):
        b = base(s)
        if b:
            return b
        d = absorb[idx[s]]
        p = d.get("sacrifice_rescue", 0.0) + d.get("team_direct", 0.0)
        return ("wq", min(3, int(p * 4)))

    return {
        "switch_flag": by_switch,
        "mode_x_switch": by_mode_switch,
        "basin_forecast": by_forecast,
        "forecast_x_switch": by_forecast_switch,
        "win_prob_quartile": by_win_quartile,
    }


# Exact Rosas Psi on the behavioral occupancy measure

def occupancy_joint(chain: Chain) -> Dict[Tuple[int, int], float]:
    """Exact pooled one-step joint P(s_t, s_{t+1}) for t = 0..MAX_STEPS-1
    from the episode initial distribution; terminal self-loops excluded
    (the pooled measure conditions on the episode still running)."""
    cur = dict(chain.init)
    joint: Dict[Tuple[int, int], float] = defaultdict(float)
    for _t in range(MAX_STEPS):
        nxt: Dict[int, float] = defaultdict(float)
        for i, w in cur.items():
            if w <= 0 or i >= chain.n_nonterminal:
                continue
            for j, p in chain.rows[i]:
                joint[(i, j)] += w * p
                nxt[j] += w * p
        cur = nxt
    total = sum(joint.values())
    return {k: v / total for k, v in joint.items()}


def mi_bits(joint: Mapping[Tuple, float]) -> float:
    px: Dict[object, float] = defaultdict(float)
    py: Dict[object, float] = defaultdict(float)
    for (x, y), w in joint.items():
        px[x] += w
        py[y] += w
    total = 0.0
    for (x, y), w in joint.items():
        if w > 0:
            total += w * math.log2(w / (px[x] * py[y]))
    return total


def project_joint(joint: Mapping[Tuple[int, int], float], states, fx, fy
                  ) -> Dict[Tuple, float]:
    out: Dict[Tuple, float] = defaultdict(float)
    for (i, j), w in joint.items():
        out[(fx(states[i]), fy(states[j]))] += w
    return dict(out)


def micro_components(s) -> Tuple:
    """Micro decomposition of the closed state: agent-0 position, agent-1
    position, (switch flag, clock), (mode, context). Absorbing states are
    globally visible (episode over, outcome public)."""
    if s[0] == "basin":
        done = ("done", s[1])
        return (done, done, done, done)
    mode, context, a0, a1, sw, t = s
    return (a0, a1, (sw, t), (mode, context))


# Micro decompositions the rival may choose from (fewer components =>
# fewer subtracted MI terms => larger Psi; agents-only is the generous
# reading where only the two agents count as parts of the system).
MICRO_DECOMPOSITIONS: Dict[str, Tuple[int, ...]] = {
    "agents_only": (0, 1),
    "full_state": (0, 1, 2, 3),
}


def psi_bits(joint: Mapping[Tuple[int, int], float], states, v_feature,
             component_ids: Sequence[int]) -> float:
    """Psi(V) = I(V_t; V_{t+1}) - sum_j I(X^j_t; V_{t+1}) -- Rosas et al.
    2020 practical criterion, first-order form. Exact plug-in on the exact
    occupancy joint."""
    i_vv = mi_bits(project_joint(joint, states, v_feature, v_feature))
    total_micro = 0.0
    for j in component_ids:
        fx = lambda s, j=j: micro_components(s)[j]
        total_micro += mi_bits(project_joint(joint, states, fx, v_feature))
    return i_vv - total_micro


def candidate_v_features(chain: Chain, absorb: List[Dict[str, float]]):
    """Supervenient macro features V (each an exact function of the closed
    micro state). The rival gets its best member."""
    idx = chain.index

    def v_forecast(s):
        if s[0] == "basin":
            return ("done", s[1])
        d = absorb[idx[s]]
        return max(d, key=lambda k: d[k])

    def v_win_prob(s):
        if s[0] == "basin":
            return ("done", s[1])
        d = absorb[idx[s]]
        p = d.get("sacrifice_rescue", 0.0) + d.get("team_direct", 0.0)
        return min(3, int(p * 4))

    def v_switch(s):
        if s[0] == "basin":
            return ("done", s[1])
        return s[4]

    def v_joint_progress(s):
        if s[0] == "basin":
            return ("done", s[1])
        _m, _c, a0, a1, sw, _t = s
        d_team = manhattan(a0, TEAM_A0) + manhattan(a1, TEAM_A1)
        return (sw, min(3, d_team // 3))

    return {
        "basin_forecast": v_forecast,
        "win_prob_quartile": v_win_prob,
        "switch_flag": v_switch,
        "joint_progress": v_joint_progress,
    }


# Scoring (mirrors prior_metrics_comparison.py)

def hindsight_best(scores: Dict[str, float]) -> Dict[str, object]:
    best: Dict[str, object] = {"accuracy": -1.0}
    values = sorted(set(scores.values()))
    cuts = [values[0] - 1.0] + [
        0.5 * (a + b) for a, b in zip(values, values[1:])] + [values[-1] + 1.0]
    for direction in (1, -1):
        for cut in cuts:
            preds = {s: int(direction * v > direction * cut)
                     for s, v in scores.items()}
            acc = sum(int(preds[s] == TRUTH[s]) for s in scores) / len(scores)
            if acc > best["accuracy"]:
                best = {
                    "accuracy": acc, "threshold": cut, "direction": direction,
                    "misclassified": sorted(s for s in scores
                                            if preds[s] != TRUTH[s]),
                }
    return best


def natural_threshold(scores: Dict[str, float], cut: float = 0.0
                      ) -> Dict[str, object]:
    preds = {s: int(v > cut) for s, v in scores.items()}
    acc = sum(int(preds[s] == TRUTH[s]) for s in scores) / len(scores)
    return {"accuracy": acc, "threshold": cut,
            "misclassified": sorted(s for s in scores if preds[s] != TRUTH[s])}


def main() -> None:
    policies: Dict[str, Dict] = {}
    for idx, regime in enumerate(("uncertain_preference", "pure_team",
                                  "dense_shaping", "random_noise")):
        print(f"training {regime} ...", flush=True)
        policies[regime] = train_policy(regime, 60000, 6011 + idx * 10_000)

    ce_scores: Dict[str, float] = {}
    psi_scores: Dict[str, float] = {}
    details: Dict[str, Dict] = {}

    for name, (regime, behavior, modes) in SYSTEMS.items():
        q_table = policies[regime] if regime else {}
        chain = Chain(q_table, regime, behavior, modes)
        absorb = absorption_map(chain)

        ei_micro = ei_bits_rows(len(chain.states), chain.rows)
        macro_gains = {
            pname: macro_ei_bits(chain, label) - ei_micro
            for pname, label in candidate_partitions(chain, absorb).items()
        }
        ce = max(macro_gains.values())

        joint = occupancy_joint(chain)
        psis = {
            f"{vname}|{dname}": psi_bits(joint, chain.states, v, comps)
            for vname, v in candidate_v_features(chain, absorb).items()
            for dname, comps in MICRO_DECOMPOSITIONS.items()
        }
        psi = max(psis.values())

        ce_scores[name] = ce
        psi_scores[name] = psi
        details[name] = {
            "n_states": len(chain.states),
            "ei_micro_bits": ei_micro,
            "ce_by_partition": macro_gains,
            "psi_by_feature": psis,
            "ground_truth": TRUTH[name],
        }
        print(f"{name:20s} EI_micro {ei_micro:7.3f}  CE(best) {ce:+8.4f}  "
              f"Psi(best) {psi:+8.4f}  truth {TRUTH[name]}", flush=True)

    out = {
        "note": ("Exact formalisms, zero sampling error, on the exact "
                 "policy-closed chain (state incl. mode, context, t). "
                 "CE = max over the candidate coarse-graining family of "
                 "EI(macro)-EI(micro) (Hoel CE 1.0, max-ent interventions). "
                 "Psi = max over candidate supervenient features (Rosas 2020 "
                 "practical criterion, first-order). Each rival gets its "
                 "best case AND a hindsight-optimal threshold."),
        "temperature": TEMPERATURE,
        "detectors": {
            "causal_emergence_exact": {
                "scores": ce_scores,
                "hindsight_best": hindsight_best(ce_scores),
                "natural_threshold_gt0": natural_threshold(ce_scores),
            },
            "phiid_psi_exact": {
                "scores": psi_scores,
                "hindsight_best": hindsight_best(psi_scores),
                "natural_threshold_gt0": natural_threshold(psi_scores),
            },
        },
        "details": details,
        "truth": TRUTH,
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "exact_prior_formalisms.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    for dname, d in out["detectors"].items():
        hb = d["hindsight_best"]
        nt = d["natural_threshold_gt0"]
        print(f"\n{dname}: hindsight acc {hb['accuracy']:.3f} "
              f"(missed: {hb['misclassified']}); "
              f"natural >0 acc {nt['accuracy']:.3f} "
              f"(missed: {nt['misclassified']})")
    print(f"\nWrote {OUTPUTS / 'exact_prior_formalisms.json'}")


if __name__ == "__main__":
    main()
