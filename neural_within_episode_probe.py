"""Neural (DQN) version of the within-episode possibility-collapse probe.

Two reviewer objections are addressed at once:

1. "All P_t(B) evidence is tabular." Here a small MLP Q-network is trained with
   DQN on the contextual rescue/bridge task, and the same within-episode
   Monte Carlo estimation of P_t(B | s_t) plus the minimal do-operator contrast
   is repeated with the neural policy.
2. "The representation-jump bridge was never tested on a real learned
   embedding." At training checkpoints we record the penultimate-layer
   activations on a fixed probe-state set and measure embedding jumps J_k,
   aligned against future-basin collapse bursts B_k measured from rollouts of
   the same checkpoints.

The environment, reward regimes, and basin classification are imported from
the existing contextual benchmark, so nothing about the task changes; only the
learner does.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from contextual_sacrifice_gridworld import (
    GRID_SIZE,
    JOINT_ACTIONS,
    MAX_STEPS,
    MODES,
    REGIMES,
    SWITCH,
    ContextualSacrificeEnv,
    classify_basin,
    move_position,
    sample_mode,
    sample_preference_context,
    scalar_reward,
)


OUTPUTS = Path(__file__).resolve().parent / "outputs"
BASINS = ("sacrifice_rescue", "team_direct", "selfish_escape", "failed_noise")
TRIGGER_EVENTS = ("a0_step_on_sacrifice_switch", "a0_step_on_decoy_switch")
CONTEXTS = ("fixed", "self_preservation", "visible_teamwork", "latent_sacrifice")
ACTION_INDEX = {action: idx for idx, action in enumerate(JOINT_ACTIONS)}
STATE_DIM = 13
EMBED_DIM = 64

PROBE_STATES = tuple(
    (mode, a0, a1, gate, used, t)
    for mode in MODES
    for a0, a1, gate, used, t in (
        ((0, 2), (0, 4), False, False, 0),
        (SWITCH, (0, 4), True, True, 2),
        ((4, 2), (0, 4), False, False, 4),
        ((2, 3), (2, 4), False, False, 4),
        ((3, 1), (4, 4), True, True, 5),
    )
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def normalize(counts: Mapping[str, float]) -> Dict[str, float]:
    total = sum(max(counts.get(basin, 0.0), 0.0) for basin in BASINS)
    if total <= 0:
        return {basin: 1.0 / len(BASINS) for basin in BASINS}
    return {basin: max(counts.get(basin, 0.0), 0.0) / total for basin in BASINS}


def entropy(p: Mapping[str, float]) -> float:
    eps = 1e-12
    return -sum(p[b] * math.log(p[b] + eps, 2) for b in BASINS if p[b] > 0)


def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    eps = 1e-12
    return sum(p[b] * math.log((p[b] + eps) / (q[b] + eps), 2) for b in BASINS if p[b] > 0)


def js(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    m = {b: 0.5 * (p[b] + q[b]) for b in BASINS}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return num / (dx * dy)


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def encode_state(state, context: str) -> np.ndarray:
    mode, a0, a1, gate, used, t = state
    features = np.zeros(STATE_DIM, dtype=np.float32)
    features[0] = 1.0 if mode == "rescue" else 0.0
    features[1] = 1.0 if mode == "bridge" else 0.0
    features[2] = a0[0] / (GRID_SIZE - 1)
    features[3] = a0[1] / (GRID_SIZE - 1)
    features[4] = a1[0] / (GRID_SIZE - 1)
    features[5] = a1[1] / (GRID_SIZE - 1)
    features[6] = 1.0 if gate else 0.0
    features[7] = 1.0 if used else 0.0
    features[8] = t / MAX_STEPS
    features[9 + CONTEXTS.index(context)] = 1.0
    return features


class QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, EMBED_DIM),
            nn.ReLU(),
        )
        self.head = nn.Linear(EMBED_DIM, len(JOINT_ACTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def action_mask(state, intervention: Optional[str]) -> List[int]:
    """Indices of allowed actions under the minimal do-operator."""

    _, a0_pos, _, _, switch_used, _ = state
    indices = list(range(len(JOINT_ACTIONS)))
    if switch_used or intervention is None:
        return indices
    if intervention == "do_trigger":
        toward = [
            idx for idx, action in enumerate(JOINT_ACTIONS)
            if manhattan(move_position(a0_pos, action[0]), SWITCH) < manhattan(a0_pos, SWITCH)
        ]
        return toward or indices
    if intervention == "do_non_trigger":
        allowed = [
            idx for idx, action in enumerate(JOINT_ACTIONS)
            if move_position(a0_pos, action[0]) != SWITCH
        ]
        return allowed or indices
    return indices


@torch.no_grad()
def choose_action(
    model: QNet,
    device: torch.device,
    state,
    context: str,
    temperature: float,
    rng: random.Random,
    intervention: Optional[str] = None,
) -> Tuple[str, str]:
    x = torch.from_numpy(encode_state(state, context)).to(device).unsqueeze(0)
    q = model(x).squeeze(0).cpu().numpy()
    allowed = action_mask(state, intervention)
    values = q[allowed]
    if temperature <= 0:
        return JOINT_ACTIONS[allowed[int(values.argmax())]]
    values = values - values.max()
    weights = np.exp(values / temperature)
    weights = weights / weights.sum()
    choice = rng.choices(range(len(allowed)), weights=list(weights))[0]
    return JOINT_ACTIONS[allowed[choice]]


@torch.no_grad()
def embedding_vector(model: QNet, device: torch.device) -> np.ndarray:
    inputs = np.stack(
        [encode_state(state, context) for context in CONTEXTS for state in PROBE_STATES]
    )
    x = torch.from_numpy(inputs).to(device)
    return model.embed(x).cpu().numpy().reshape(-1)


def rollout_basin_distribution(
    model: QNet,
    device: torch.device,
    regime: str,
    episodes: int,
    temperature: float,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    counts = {basin: 0.0 for basin in BASINS}
    for episode in range(episodes):
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context(regime, rng, episode)
        events: List[str] = []
        done = False
        while not done:
            action = choose_action(model, device, state, context, temperature, rng)
            result = env.step(state, action)
            events.extend(result.events)
            state = result.state
            done = result.done
        counts[classify_basin(events)] += 1.0
    return normalize(counts)


def train_dqn(
    regime: str,
    episodes: int,
    seed: int,
    device: torch.device,
    checkpoint_every: int,
    checkpoint_eval_episodes: int,
    eval_temperature: float,
) -> Tuple[QNet, List[Dict[str, float]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer: deque = deque(maxlen=60000)
    gamma = 0.96
    batch_size = 128
    step_count = 0
    checkpoint_rows: List[Dict[str, float]] = []
    prev_embedding: Optional[np.ndarray] = None
    initial_dist: Optional[Dict[str, float]] = None
    prev_collapse = 0.0

    def maybe_checkpoint(episode: int) -> None:
        nonlocal prev_embedding, initial_dist, prev_collapse
        dist = rollout_basin_distribution(
            model, device, regime, checkpoint_eval_episodes, eval_temperature,
            seed + 900_001 + episode,
        )
        if initial_dist is None:
            initial_dist = dict(dist)
        collapse = kl(dist, initial_dist)
        burst = max(collapse - prev_collapse, 0.0) if checkpoint_rows else 0.0
        embedding = embedding_vector(model, device)
        if prev_embedding is None:
            jump = 0.0
        else:
            jump = float(np.sqrt(np.mean((embedding - prev_embedding) ** 2)))
        checkpoint_rows.append(
            {
                "episode": float(episode),
                "collapse_kl": collapse,
                "collapse_burst": burst,
                "embedding_jump": jump,
                **{f"p_{basin}": dist[basin] for basin in BASINS},
            }
        )
        prev_embedding = embedding
        prev_collapse = collapse

    maybe_checkpoint(0)
    for episode in range(episodes):
        epsilon = 0.04 + (0.45 - 0.04) * max(0.0, 1.0 - episode / max(1, episodes))
        mode = sample_mode(rng, episode)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context(regime, rng, episode)
        done = False
        while not done:
            if rng.random() < epsilon:
                action = JOINT_ACTIONS[rng.randrange(len(JOINT_ACTIONS))]
            else:
                action = choose_action(model, device, state, context, 0.0, rng)
            result = env.step(state, action)
            reward = scalar_reward(regime, context, result.rewards, result.events, rng)
            buffer.append(
                (
                    encode_state(state, context),
                    ACTION_INDEX[action],
                    reward,
                    encode_state(result.state, context),
                    1.0 if result.done else 0.0,
                )
            )
            state = result.state
            done = result.done
            step_count += 1

            if len(buffer) >= 1000 and step_count % 2 == 0:
                batch = rng.sample(range(len(buffer)), batch_size)
                s = torch.from_numpy(np.stack([buffer[i][0] for i in batch])).to(device)
                a = torch.tensor([buffer[i][1] for i in batch], device=device)
                r = torch.tensor([buffer[i][2] for i in batch], dtype=torch.float32, device=device)
                s2 = torch.from_numpy(np.stack([buffer[i][3] for i in batch])).to(device)
                d = torch.tensor([buffer[i][4] for i in batch], dtype=torch.float32, device=device)
                q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(s2).max(dim=1).values
                    target_q = r + gamma * (1.0 - d) * q_next
                loss = nn.functional.smooth_l1_loss(q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if step_count % 2000 == 0:
                target.load_state_dict(model.state_dict())

        if (episode + 1) % checkpoint_every == 0:
            maybe_checkpoint(episode + 1)

    return model, checkpoint_rows


def estimate_future(
    model: QNet,
    device: torch.device,
    env: ContextualSacrificeEnv,
    state,
    contexts: Sequence[str],
    events_so_far: Sequence[str],
    temperature: float,
    samples: int,
    rng: random.Random,
    intervention: Optional[str] = None,
) -> Tuple[Dict[str, float], float]:
    counts = {basin: 0.0 for basin in BASINS}
    returns: List[float] = []
    for sample_idx in range(samples):
        context = contexts[sample_idx % len(contexts)]
        events = list(events_so_far)
        current = state
        total = 0.0
        done = current[5] >= MAX_STEPS
        while not done:
            action = choose_action(model, device, current, context, temperature, rng, intervention)
            result = env.step(current, action)
            events.extend(result.events)
            total += result.rewards[0] + result.rewards[1]
            current = result.state
            done = result.done
        counts[classify_basin(events)] += 1.0
        returns.append(total)
    return normalize(counts), mean(returns)


def probe_contexts(regime: str) -> Tuple[str, ...]:
    if regime == "uncertain_preference":
        return ("self_preservation", "visible_teamwork", "latent_sacrifice")
    return ("fixed",)


def within_episode_summary(
    model: QNet,
    device: torch.device,
    regime: str,
    mode: str,
    probe_episodes: int,
    samples: int,
    temperature: float,
    probe_temperature: float,
    seed: int,
) -> Dict[str, float | str]:
    contexts = probe_contexts(regime)
    h0_values: List[float] = []
    iv_js_values: List[float] = []
    iv_gap_values: List[float] = []
    do_trigger_rescue: List[float] = []
    do_non_trigger_rescue: List[float] = []
    trigger_hits = 0
    for episode in range(probe_episodes):
        rng = random.Random(seed + episode * 13)
        env = ContextualSacrificeEnv(mode)
        state = env.reset()
        context = sample_preference_context(regime, rng, episode)
        dist, _ = estimate_future(
            model, device, env, state, contexts, [], probe_temperature, samples, rng
        )
        h0_values.append(entropy(dist))
        do_t_dist, do_t_ret = estimate_future(
            model, device, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_trigger",
        )
        do_n_dist, do_n_ret = estimate_future(
            model, device, env, state, contexts, [], probe_temperature, samples, rng,
            intervention="do_non_trigger",
        )
        iv_js_values.append(js(do_t_dist, do_n_dist))
        iv_gap_values.append(do_t_ret - do_n_ret)
        do_trigger_rescue.append(do_t_dist["sacrifice_rescue"])
        do_non_trigger_rescue.append(do_n_dist["sacrifice_rescue"])

        events: List[str] = []
        done = False
        while not done:
            action = choose_action(model, device, state, context, temperature, rng)
            result = env.step(state, action)
            events.extend(result.events)
            state = result.state
            done = result.done
        if any(event in events for event in TRIGGER_EVENTS):
            trigger_hits += 1
    return {
        "regime": regime,
        "mode": mode,
        "n_episodes": float(probe_episodes),
        "trigger_rate": trigger_hits / max(probe_episodes, 1),
        "initial_future_entropy": mean(h0_values),
        "intervention_js": mean(iv_js_values),
        "intervention_return_gap": mean(iv_gap_values),
        "do_trigger_p_rescue": mean(do_trigger_rescue),
        "do_non_trigger_p_rescue": mean(do_non_trigger_rescue),
    }


def summarize_bridge(regime: str, rows: Sequence[Mapping[str, float]]) -> Dict[str, float | str]:
    bursts = [float(row["collapse_burst"]) for row in rows]
    jumps = [float(row["embedding_jump"]) for row in rows]
    max_burst = max(bursts)
    max_jump = max(jumps)
    burst_t = float(bursts.index(max_burst))
    jump_t = float(jumps.index(max_jump))
    return {
        "regime": regime,
        "n_checkpoints": float(len(rows)),
        "final_collapse_kl": float(rows[-1]["collapse_kl"]),
        "max_collapse_burst": max_burst,
        "max_embedding_jump": max_jump,
        "peak_alignment": 1.0 / (1.0 + abs(burst_t - jump_t)),
        "burst_jump_correlation": pearson(bursts, jumps),
    }


def run_all(
    regimes: Sequence[str],
    train_episodes: int,
    checkpoint_every: int,
    checkpoint_eval_episodes: int,
    probe_episodes: int,
    samples: int,
    temperature: float,
    probe_temperature: float,
    seed: int,
    output_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_rows: List[Dict[str, float | str]] = []
    bridge_summaries: List[Dict[str, float | str]] = []
    probe_summaries: List[Dict[str, float | str]] = []
    for idx, regime in enumerate(regimes):
        model, checkpoints = train_dqn(
            regime,
            episodes=train_episodes,
            seed=seed + idx * 40_000,
            device=device,
            checkpoint_every=checkpoint_every,
            checkpoint_eval_episodes=checkpoint_eval_episodes,
            eval_temperature=temperature,
        )
        for row in checkpoints:
            bridge_rows.append({"regime": regime, **row})
        bridge_summaries.append(summarize_bridge(regime, checkpoints))
        for mode in MODES:
            probe_summaries.append(
                within_episode_summary(
                    model, device, regime, mode,
                    probe_episodes=probe_episodes,
                    samples=samples,
                    temperature=temperature,
                    probe_temperature=probe_temperature,
                    seed=seed + idx * 40_000 + (7 if mode == "rescue" else 11),
                )
            )

    with (output_dir / "neural_checkpoint_bridge_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bridge_rows[0].keys()))
        writer.writeheader()
        for row in bridge_rows:
            writer.writerow(row)
    with (output_dir / "neural_checkpoint_bridge_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bridge_summaries[0].keys()))
        writer.writeheader()
        for row in bridge_summaries:
            writer.writerow(row)
    with (output_dir / "neural_within_episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(probe_summaries[0].keys()))
        writer.writeheader()
        for row in probe_summaries:
            writer.writerow(row)
    (output_dir / "neural_within_episode_summary.json").write_text(
        json.dumps({"bridge": bridge_summaries, "probe": probe_summaries}, indent=2),
        encoding="utf-8",
    )
    print("bridge: regime,final_collapse,max_burst,max_jump,corr,align")
    for row in bridge_summaries:
        print(
            f"{row['regime']},{float(row['final_collapse_kl']):.4f},"
            f"{float(row['max_collapse_burst']):.4f},{float(row['max_embedding_jump']):.4f},"
            f"{float(row['burst_jump_correlation']):.4f},{float(row['peak_alignment']):.4f}"
        )
    print("probe: regime,mode,trig_rate,H0,iv_js,iv_gap,do_trig_p_rescue")
    for row in probe_summaries:
        print(
            f"{row['regime']},{row['mode']},{float(row['trigger_rate']):.3f},"
            f"{float(row['initial_future_entropy']):.4f},{float(row['intervention_js']):.4f},"
            f"{float(row['intervention_return_gap']):.4f},{float(row['do_trigger_p_rescue']):.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural DQN within-episode probe.")
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=["pure_team", "uncertain_preference"],
        choices=list(REGIMES),
    )
    parser.add_argument("--train_episodes", type=int, default=16000)
    parser.add_argument("--checkpoint_every", type=int, default=2000)
    parser.add_argument("--checkpoint_eval_episodes", type=int, default=240)
    parser.add_argument("--probe_episodes", type=int, default=24)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--probe_temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all(
        regimes=args.regimes,
        train_episodes=args.train_episodes,
        checkpoint_every=args.checkpoint_every,
        checkpoint_eval_episodes=args.checkpoint_eval_episodes,
        probe_episodes=args.probe_episodes,
        samples=args.samples,
        temperature=args.temperature,
        probe_temperature=args.probe_temperature,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"\nWrote {args.output_dir / 'neural_checkpoint_bridge_summary.csv'}")
    print(f"Wrote {args.output_dir / 'neural_within_episode_summary.csv'}")


if __name__ == "__main__":
    main()
