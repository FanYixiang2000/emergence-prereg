"""Pre-registered transfer of the five-component criterion to an external system.

See EXTERNAL_TRANSFER_PREREGISTRATION.md for the full protocol registered
before this script produced any measurement. In short:

- Target family: the pre-existing continuous swarm decoy benchmark in
  examples_6.29_MARL_SWARM (N-vs-N combat, local observations, decoy /
  threat / fragile enemy roles). Nothing about its state, action, or
  observation space is shared with the internal gridworld family.
- Added latent context (never shown to any controller as a label):
  `passive` front enemies are classic high-HP decoys (engaging wastes the
  horizon); `aggressive` front enemies are fragile-but-deadly ambushers
  (bypassing costs transit damage; clearing first is better).
- Five systems: a REINFORCE target-selection learner trained with team
  reward only (marl_learned), its untrained twin (marl_untrained), and
  three hand rules from the external project (nearest_only, role_oracle,
  damage_aware).
- All thresholds are copied unchanged from criterion_ablation_battery.
- Ground-truth labels come from a behavioral audit (per-context trigger
  rates), not from the criterion itself.

Trigger event: cumulative damage dealt to decoy-role enemies > 0.3 HP.
Macro-basins: {win, loss} x {engaged, bypassed}.
do_trigger: restrict target candidates to decoy-role enemies while any is
alive and visible. do_non_trigger: remove decoy-role enemies from target
candidates for the whole episode. Both are minimal do-operators on the
target-selection interface; movement and combat dynamics are untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "examples_6.29_MARL_SWARM"
sys.path.insert(0, str(EXTERNAL_DIR))

from swarm_decoy_abstraction import (  # noqa: E402
    DecoyConfig,
    DecoySwarmAbstraction,
    MISSION_ROLES,
    ROLE_DECOY,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"

CONTEXTS = ("passive", "aggressive")
BASINS = ("win_engaged", "win_bypassed", "loss_engaged", "loss_bypassed")
TRIGGER_DAMAGE = 0.3

# Copied unchanged from criterion_ablation_battery.THRESHOLDS.
THRESHOLDS = {
    "potential_bits": 0.5,
    "selectivity_tension": 0.5,
    "specificity_js": 0.2,
    "usefulness_gap": 0.0,
}
COMPONENTS = ("potential", "selectivity", "specificity", "usefulness", "endogeneity")

FEAT_DIM = 6  # [dist_norm, dx_norm, dy_norm, hp_norm, damage_norm, in_range]


def make_config(seed: int, horizon: int = 25) -> DecoyConfig:
    # The external project's own "tough front" calibration
    # (decoy_tough_front_h25_*): short horizon and deadly threats make the
    # decoy trap bind -- engaging high-HP decoys in the passive context
    # genuinely wastes the episode.
    return DecoyConfig(
        n_agents=6,
        n_enemies=6,
        horizon=horizon,
        agent_damage=0.08,
        threat_damage=0.16,
        decoy_period=3,
        front_decoys=True,
        seed=seed,
    )


class ContextualDecoyEnv(DecoySwarmAbstraction):
    """External decoy env plus one latent episode context.

    passive: front decoys are the external benchmark's classic decoys
    (HP 4.0, damage 0.005). aggressive: the same front enemies are fragile
    ambushers (HP 0.9, damage 0.16) that punish transit if left alive.
    The context only changes the front enemies' HP/damage; roles, layout,
    win condition, and every other dynamic stay exactly as in the external
    project.
    """

    def __init__(self, cfg: DecoyConfig, context: str):
        self.context = context
        self.intervention: Optional[str] = None
        self.controller: Optional[Callable[["ContextualDecoyEnv", int, List[int]], Optional[int]]] = None
        super().__init__(cfg, rules={})

    def _role_hp(self, role: str) -> float:  # type: ignore[override]
        if role == ROLE_DECOY:
            return 0.9 if self.context == "aggressive" else 4.0
        if role == "threat":
            return 1.2
        return 0.8

    def _role_damage(self, role: str) -> float:  # type: ignore[override]
        if role == ROLE_DECOY:
            return 0.16 if self.context == "aggressive" else self.cfg.decoy_damage
        return super()._role_damage(role)

    def candidate_enemies(self, agent_id: int) -> List[int]:
        visible = self._visible_enemies(agent_id)
        if not visible:
            return []
        decoys = [e for e in visible if self.enemy_roles[e] == ROLE_DECOY]
        others = [e for e in visible if self.enemy_roles[e] != ROLE_DECOY]
        if self.intervention == "do_trigger" and decoys:
            return decoys
        if self.intervention == "do_non_trigger":
            return others
        return visible

    def choose_target(self, agent_id: int) -> Optional[int]:  # type: ignore[override]
        candidates = self.candidate_enemies(agent_id)
        if not candidates or self.controller is None:
            return None
        return self.controller(self, agent_id, candidates)

    def agent_features(self, agent_id: int, candidates: Sequence[int]) -> np.ndarray:
        pos = self.agent_pos[agent_id]
        feats = np.zeros((len(candidates), FEAT_DIM), dtype=np.float32)
        for k, e in enumerate(candidates):
            rel = self.enemy_pos[e] - pos
            dist = float(np.linalg.norm(rel))
            feats[k, 0] = dist / self.cfg.sight_range
            feats[k, 1] = rel[0] / self.cfg.sight_range
            feats[k, 2] = rel[1] / self.cfg.sight_range
            feats[k, 3] = float(self.enemy_hp[e]) / 4.0
            feats[k, 4] = float(self.enemy_damage[e]) / 0.2
            feats[k, 5] = 1.0 if dist <= self.cfg.attack_range else 0.0
        return feats


# --- controllers -----------------------------------------------------------


def rule_controller(priority: Sequence[str]):
    def controller(env: ContextualDecoyEnv, agent_id: int, candidates: List[int]) -> Optional[int]:
        for mode in priority:
            scored = [(env._score_target(agent_id, e, mode), e) for e in candidates]
            best_score, best_enemy = min(scored)
            if math.isfinite(best_score):
                return int(best_enemy)
        return int(candidates[0])

    return controller


def scorer_controller(scorer, sample: bool = False, rng: Optional[np.random.Generator] = None):
    import torch

    def controller(env: ContextualDecoyEnv, agent_id: int, candidates: List[int]) -> Optional[int]:
        feats = env.agent_features(agent_id, candidates)
        with torch.no_grad():
            scores = scorer(torch.from_numpy(feats))
            if sample and rng is not None:
                probs = torch.softmax(scores, dim=0).numpy()
                idx = int(rng.choice(len(candidates), p=probs / probs.sum()))
            else:
                idx = int(torch.argmax(scores).item())
        return int(candidates[idx])

    return controller


# --- episode rollout -------------------------------------------------------


def run_episode(
    controller,
    context: str,
    seed: int,
    intervention: Optional[str],
) -> Dict[str, Any]:
    env = ContextualDecoyEnv(make_config(seed), context)
    env.controller = controller
    env.intervention = intervention
    mission_before = env.alive_mission_enemies().size

    for _ in range(env.cfg.horizon):
        env.step()
        if env.alive_agents().size == 0 or env.alive_mission_enemies().size == 0:
            break

    win = bool(env.alive_mission_enemies().size == 0 and env.alive_agents().size > 0)
    engaged = env.damage_to_roles[ROLE_DECOY] > TRIGGER_DAMAGE
    mission_hp_dealt = sum(env.damage_to_roles[r] for r in MISSION_ROLES)
    kills = int(mission_before - env.alive_mission_enemies().size)
    deaths = int(env.cfg.n_agents - env.alive_agents().size)
    score = 10.0 * win + 2.0 * mission_hp_dealt + 3.0 * kills - 1.0 * deaths - 0.02 * env.t
    basin = ("win_" if win else "loss_") + ("engaged" if engaged else "bypassed")
    return {
        "basin": basin,
        "score": score,
        "engaged": engaged,
        "win": win,
        "context": context,
    }


def entropy(dist: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def js(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        return sum(a.get(k, 0.0) * math.log2(a.get(k, 0.0) / b[k])
                   for k in keys if a.get(k, 0.0) > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def basin_distribution(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = {b: 0 for b in BASINS}
    for ep in episodes:
        counts[ep["basin"]] += 1
    total = max(sum(counts.values()), 1)
    return {b: c / total for b, c in counts.items()}


def rollout_batch(
    controller,
    n_eval: int,
    seed: int,
    intervention: Optional[str],
) -> List[Dict[str, Any]]:
    episodes = []
    for i in range(n_eval):
        context = CONTEXTS[i % len(CONTEXTS)]
        episodes.append(run_episode(controller, context, seed + i * 97, intervention))
    return episodes


# --- REINFORCE training on the contextual env ------------------------------


def train_marl_scorer(iters: int, batch: int, lr: float, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    scorer = nn.Sequential(
        nn.Linear(FEAT_DIM, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )

    def forward(x: "torch.Tensor") -> "torch.Tensor":
        return scorer(x).squeeze(-1)

    opt = torch.optim.Adam(scorer.parameters(), lr=lr)
    baseline = 0.0
    history: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)

    for it in range(iters):
        opt.zero_grad()
        batch_loss = torch.zeros(())
        ep_returns: List[float] = []
        ep_wins: List[float] = []
        for b in range(batch):
            context = CONTEXTS[int(rng.integers(len(CONTEXTS)))]
            env = ContextualDecoyEnv(make_config(seed + it * 100_003 + b * 17), context)
            env.intervention = None
            logps: List["torch.Tensor"] = []
            rewards: List[float] = []
            mission_prev = env.alive_mission_enemies().size

            for _ in range(env.cfg.horizon):
                alive = env.alive_agents()
                if alive.size == 0 or env.alive_mission_enemies().size == 0:
                    break
                step_logp = torch.zeros(())
                targets: Dict[int, int] = {}
                for a in alive:
                    a = int(a)
                    candidates = env.candidate_enemies(a)
                    if not candidates:
                        continue
                    feats = torch.from_numpy(env.agent_features(a, candidates))
                    probs = torch.softmax(forward(feats), dim=0)
                    dist = torch.distributions.Categorical(probs=probs)
                    idx = dist.sample()
                    step_logp = step_logp + dist.log_prob(idx)
                    targets[a] = int(candidates[int(idx.item())])
                env.controller = lambda _env, agent_id, _cands, _t=targets: _t.get(agent_id)

                hp_before = env.enemy_hp.copy()
                agents_before = env.alive_agents().size
                env.step()

                mission_hp_dealt = sum(
                    max(0.0, float(hp_before[e] - env.enemy_hp[e]))
                    for e in range(env.cfg.n_enemies)
                    if env.enemy_roles[e] in MISSION_ROLES
                )
                mission_now = env.alive_mission_enemies().size
                kills = max(0, mission_prev - mission_now)
                mission_prev = mission_now
                deaths = max(0, agents_before - env.alive_agents().size)
                # Same team reward shape as the external project:
                # decoy-damage coefficient is exactly 0, so the trigger step
                # itself carries no process reward.
                r = 2.0 * mission_hp_dealt + 3.0 * kills - 0.02 - 1.0 * deaths
                rewards.append(float(r))
                logps.append(step_logp)

            win = bool(env.alive_mission_enemies().size == 0 and env.alive_agents().size > 0)
            if win and rewards:
                rewards[-1] += 10.0
            if not logps:
                continue
            running = 0.0
            rtg = [0.0] * len(rewards)
            for t in reversed(range(len(rewards))):
                running = rewards[t] + 0.98 * running
                rtg[t] = running
            rtg_t = torch.tensor(rtg, dtype=torch.float32)
            adv = rtg_t - baseline
            logp_t = torch.stack(logps)
            batch_loss = batch_loss - (logp_t * adv).sum()
            ep_returns.append(float(sum(rewards)))
            ep_wins.append(1.0 if win else 0.0)

        if ep_returns:
            batch_loss = batch_loss / len(ep_returns)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            opt.step()
            mean_ret = float(np.mean(ep_returns))
            baseline = 0.9 * baseline + 0.1 * mean_ret
            if it % max(1, iters // 15) == 0 or it == iters - 1:
                print(f"  iter {it:4d} | return {mean_ret:7.2f} | win {np.mean(ep_wins):.3f}")
                history.append({"iter": it, "return": mean_ret, "win": float(np.mean(ep_wins))})

    return forward, history


def untrained_scorer(seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    scorer = nn.Sequential(
        nn.Linear(FEAT_DIM, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )

    def forward(x: "torch.Tensor") -> "torch.Tensor":
        return scorer(x).squeeze(-1)

    return forward


# --- criterion measurement --------------------------------------------------


def measure_system(
    name: str,
    controller,
    prespecified: bool,
    n_eval: int,
    seed: int,
) -> Dict[str, Any]:
    natural = rollout_batch(controller, n_eval, seed, None)
    do_t = rollout_batch(controller, n_eval, seed + 1_000_000, "do_trigger")
    do_n = rollout_batch(controller, n_eval, seed + 2_000_000, "do_non_trigger")

    nat_dist = basin_distribution(natural)
    p_trigger = float(np.mean([1.0 if ep["engaged"] else 0.0 for ep in natural]))

    def ctx_rate(episodes: List[Dict[str, Any]], context: str) -> float:
        flags = [1.0 if ep["engaged"] else 0.0 for ep in episodes if ep["context"] == context]
        return float(np.mean(flags)) if flags else 0.0

    def ctx_gap(context: str) -> float:
        t_scores = [ep["score"] for ep in do_t if ep["context"] == context]
        n_scores = [ep["score"] for ep in do_n if ep["context"] == context]
        return float(np.mean(t_scores) - np.mean(n_scores))

    return {
        "system": name,
        "prespecified": 1 if prespecified else 0,
        "h0_bits": entropy(nat_dist),
        "natural_trigger_rate": p_trigger,
        "passive_trigger_rate": ctx_rate(natural, "passive"),
        "aggressive_trigger_rate": ctx_rate(natural, "aggressive"),
        "selectivity_tension": 4.0 * p_trigger * (1.0 - p_trigger),
        "specificity_js": js(basin_distribution(do_t), basin_distribution(do_n)),
        "usefulness_gap": float(
            np.mean([ep["score"] for ep in natural]) - np.mean([ep["score"] for ep in do_n])
        ),
        "iv_gap_passive": ctx_gap("passive"),
        "iv_gap_aggressive": ctx_gap("aggressive"),
        "natural_score": float(np.mean([ep["score"] for ep in natural])),
        "natural_win_rate": float(np.mean([1.0 if ep["win"] else 0.0 for ep in natural])),
        "basin_distribution": nat_dist,
    }


def component_passes(row: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "potential": float(row["h0_bits"]) >= THRESHOLDS["potential_bits"],
        "selectivity": float(row["selectivity_tension"]) >= THRESHOLDS["selectivity_tension"],
        "specificity": float(row["specificity_js"]) >= THRESHOLDS["specificity_js"],
        "usefulness": float(row["usefulness_gap"]) > THRESHOLDS["usefulness_gap"],
        "endogeneity": int(row["prespecified"]) == 0,
    }


def audit_label(row: Dict[str, Any]) -> int:
    """Pre-registered audit rule for learned systems: per-context trigger
    rates must separate (aggressive >= 0.7 and passive <= 0.3)."""
    return int(
        float(row["aggressive_trigger_rate"]) >= 0.7
        and float(row["passive_trigger_rate"]) <= 0.3
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-registered external criterion transfer.")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n_eval", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7031)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Training REINFORCE target-selection learner on the contextual decoy env")
    print("=" * 80)
    learned, history = train_marl_scorer(args.iters, args.batch, args.lr, args.seed)
    untrained = untrained_scorer(args.seed + 999)

    systems = (
        ("marl_learned", scorer_controller(learned), False),
        ("marl_untrained", scorer_controller(untrained), False),
        ("nearest_only", rule_controller(["nearest"]), True),
        ("role_oracle", rule_controller(["threat", "fragile", "non_decoy", "nearest"]), True),
        ("damage_aware", rule_controller(["damage", "nearest"]), True),
    )

    rows: List[Dict[str, Any]] = []
    for idx, (name, controller, prespec) in enumerate(systems):
        print(f"\nMeasuring {name} ...")
        row = measure_system(name, controller, prespec, args.n_eval, args.seed + idx * 50_000)
        rows.append(row)
        print(
            f"  H0 {row['h0_bits']:.3f} | p_trig {row['natural_trigger_rate']:.3f} "
            f"(pas {row['passive_trigger_rate']:.2f} / agg {row['aggressive_trigger_rate']:.2f}) "
            f"| JS {row['specificity_js']:.3f} | gap {row['usefulness_gap']:+.3f} "
            f"| win {row['natural_win_rate']:.3f}"
        )

    # Ground-truth labels: audit rule for learned systems, design knowledge
    # for the rest (all registered in the pre-registration document).
    labels: Dict[str, int] = {}
    for row in rows:
        if row["system"] == "marl_learned":
            labels[row["system"]] = audit_label(row)
        elif row["system"] == "marl_untrained":
            labels[row["system"]] = 0
        else:
            labels[row["system"]] = 0

    verdicts: Dict[str, Dict[str, Any]] = {}
    correct = 0
    for row in rows:
        passes = component_passes(row)
        predicted = int(all(passes.values()))
        truth = labels[row["system"]]
        verdicts[row["system"]] = {
            "passes": passes,
            "full_criterion": predicted,
            "audited_label": truth,
            "agrees": predicted == truth,
        }
        correct += int(predicted == truth)

    damage_row = next(row for row in rows if row["system"] == "damage_aware")
    damage_passes = component_passes(damage_row)
    prediction_checks = {
        "p1_full_criterion_matches_all_audited_labels": correct == len(rows),
        "p2_damage_aware_excluded_only_by_endogeneity": (
            all(damage_passes[c] for c in COMPONENTS if c != "endogeneity")
            and not damage_passes["endogeneity"]
        ),
        "p3_forced_engagement_gap_sign_flips": (
            float(rows[0]["iv_gap_aggressive"]) > 0 > float(rows[0]["iv_gap_passive"])
        ),
    }

    summary = {
        "preregistration": "EXTERNAL_TRANSFER_PREREGISTRATION.md",
        "thresholds": THRESHOLDS,
        "n_eval": args.n_eval,
        "train_history": history,
        "audited_labels": labels,
        "verdicts": verdicts,
        "full_criterion_accuracy": correct / len(rows),
        "prediction_checks": prediction_checks,
        "measurements": rows,
    }
    (args.output_dir / "external_transfer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    flat_rows = [{k: v for k, v in row.items() if k != "basin_distribution"} for row in rows]
    with (args.output_dir / "external_transfer_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)

    print("\nsystem,truth,full,agrees,components_failed")
    for row in rows:
        v = verdicts[row["system"]]
        failed = ";".join(c for c in COMPONENTS if not v["passes"][c]) or "-"
        print(f"{row['system']},{v['audited_label']},{v['full_criterion']},{v['agrees']},{failed}")
    print("\nPre-registered prediction checks:")
    for key, ok in prediction_checks.items():
        print(f"  {key}: {'PASS' if ok else 'FAIL'}")
    print(f"\nWrote {args.output_dir / 'external_transfer_summary.json'}")


if __name__ == "__main__":
    main()
