"""LEARN-GRIP-TRANSPORT: two-phase learned task for resolvable B5.

Registered in V2_ALIGNMENT_PREREGISTRATION.md before running. Agents
must first collectively grip the object (attachment accumulates slowly;
side choice is mechanically irrelevant in this phase). Only when
attachment crosses a threshold can pushing move the object, so side
commitment is structurally delayed and the within-episode collapse gets
a temporal window that the B5 detector can resolve.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
N_AGENTS = 16
THRESHOLD = 6
GOAL = 6.0
MAX_STEPS = 80
N_SEEDS = 5
UPDATES = 1200
BATCH = 512
LR = 2e-3
DAMP = 0.88
ACCEL = 0.06
GRIP_GAIN = 0.06
GRIP_DECAY = 0.01
GRIP_MIN = 0.5
DIR_BOUND = 2.2
SEED = 116_001


class GripPolicy(nn.Module):
    """Left-right equivariant policy over {left, right, grip}.

    The direction logit is antisymmetrized in (x, v) with attachment as
    an even coordinate; the grip logit is symmetrized. No side is
    architecturally preferred.
    """

    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 32), nn.Tanh(),
                               nn.Linear(32, 32), nn.Tanh(),
                               nn.Linear(32, 2))

    def forward(self, obs):
        # obs columns: x/GOAL, v, attachment
        flipped = torch.stack([-obs[:, 0], -obs[:, 1], obs[:, 2]], dim=1)
        out = self.f(obs)
        out_f = self.f(flipped)
        a = DIR_BOUND * torch.tanh(out[:, 0] - out_f[:, 0])
        grip = out[:, 1] + out_f[:, 1]
        return torch.stack([-a, a, grip], dim=1)


def entropy_curve_norm(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log2(torch.clamp(probs, min=1e-12))).sum(dim=1) / math.log2(3)


def side_openness(probs: torch.Tensor) -> torch.Tensor:
    """Entropy of the renormalized left/right distribution (bits)."""
    lr = probs[:, :2]
    lr = lr / torch.clamp(lr.sum(dim=1, keepdim=True), min=1e-12)
    return -(lr * torch.log2(torch.clamp(lr, min=1e-12))).sum(dim=1)


def entropy_norm(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / math.log2(3))


def rollout_batch(policy: GripPolicy, batch: int, train: bool = True):
    x = torch.zeros(batch)
    v = torch.zeros(batch)
    att = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    total_r = torch.zeros(batch)
    sum_logp = torch.zeros(batch)
    ent_trace, side_ent_trace, att_trace, side_trace = [], [], [], []
    for _ in range(MAX_STEPS):
        obs = torch.stack([x / GOAL, v, att], dim=1)
        logits = policy(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Multinomial(total_count=N_AGENTS, probs=probs)
        counts = dist.sample()
        logp = dist.log_prob(counts)
        grip_frac = counts[:, 2] / N_AGENTS
        att = torch.clamp(att + GRIP_GAIN * grip_frac - GRIP_DECAY, 0.0, 1.0)
        force = counts[:, 1] - counts[:, 0]
        old_abs = torch.abs(x)
        active = (att >= GRIP_MIN) & (torch.abs(force) >= THRESHOLD)
        v = DAMP * v + active.float() * ACCEL * torch.sign(force)
        x = torch.clamp(x + v, -GOAL, GOAL)
        r = (torch.abs(x) - old_abs) - 0.004
        newly = (~done) & (torch.abs(x) >= GOAL - 1e-6)
        r = r + newly.float() * 5.0
        total_r = total_r + torch.where(done, torch.zeros_like(r), r)
        sum_logp = sum_logp + torch.where(done, torch.zeros_like(logp), logp)
        done = done | newly
        ent_trace.append(entropy_curve_norm(probs).detach())
        side_ent_trace.append(side_openness(probs).detach())
        att_trace.append(att.detach().clone())
        side_trace.append(torch.sign(x).detach())
    if train:
        return total_r, sum_logp, done.float()
    return (done.float().detach(), torch.sign(x).detach(),
            torch.stack(ent_trace), torch.stack(side_ent_trace),
            torch.stack(att_trace), torch.stack(side_trace))


def eval_policy(policy: GripPolicy, batch: int = 4096):
    with torch.no_grad():
        obs0 = torch.zeros((1, 3))
        p0 = torch.softmax(policy(obs0), dim=-1).numpy()[0]
        done, final_side, ent, side_ent, att, side = rollout_batch(
            policy, batch, train=False)
    ep_ent = torch.median(ent, dim=1).values.numpy()
    ep_side_ent = torch.median(side_ent, dim=1).values.numpy()
    ep_att = torch.median(att, dim=1).values.numpy()
    adj = adjudicate(range(MAX_STEPS), ep_ent * math.log2(3))
    return {
        "p0": p0,
        "entropy0": entropy_norm(p0),
        "success": float(done.mean().item()),
        "side_mean": float(final_side.mean().item()),
        "episode_entropy_curve": ep_ent,
        "episode_side_openness_curve": ep_side_ent,
        "episode_attachment_curve": ep_att,
        "episode_entropy_drop": float(ep_ent[0] - ep_ent[-1]),
        "episode_adj": adj,
    }


def run_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = GripPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    baseline = 0.0
    for _ in range(UPDATES):
        returns, logp, _done = rollout_batch(policy, BATCH, train=True)
        adv = returns.detach() - baseline
        baseline = 0.98 * baseline + 0.02 * float(returns.mean().item())
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    ev = eval_policy(policy)
    plateau_len = 0
    for val in ev["episode_side_openness_curve"]:
        if val >= 0.8:
            plateau_len += 1
        else:
            break
    return {
        "final_success": round(ev["success"], 5),
        "final_entropy0": round(ev["entropy0"], 5),
        "final_p0": [round(float(x), 5) for x in ev["p0"]],
        "final_side_mean": round(ev["side_mean"], 5),
        "episode_entropy_drop": round(ev["episode_entropy_drop"], 5),
        "side_openness_plateau_len": plateau_len,
        "final_episode_entropy": [round(float(x), 5) for x in ev["episode_entropy_curve"]],
        "side_openness_curve": [round(float(x), 5) for x in ev["episode_side_openness_curve"]],
        "attachment_curve": [round(float(x), 5) for x in ev["episode_attachment_curve"]],
        "episode_adj": ev["episode_adj"],
    }


def main() -> None:
    rows = {}
    for i in range(N_SEEDS):
        row = run_seed(SEED + i * 101)
        rows[str(i)] = row
        h = row["episode_adj"].get("hinge", {})
        print(f"seed={i}: success={row['final_success']} H0={row['final_entropy0']} "
              f"side_mean={row['final_side_mean']} epdrop={row['episode_entropy_drop']} "
              f"plateau={row['side_openness_plateau_len']} "
              f"B5={row['episode_adj']['b5_onset']} dBIC={h.get('delta_bic')}",
              flush=True)
    learned = [r for r in rows.values() if r["final_success"] >= 0.8]
    outcomes = {
        "LGT1_learnability": bool(len(learned) >= 4),
        "LGT2_precommit_plateau": bool(
            learned and float(np.median(
                [r["side_openness_plateau_len"] for r in learned])) >= 5),
        "LGT3_realization_collapse": bool(
            len(learned) >= 4
            and sum(r["episode_entropy_drop"] >= 0.3 for r in learned) >= 3),
        "LGT4_resolvable_onset": bool(
            len(learned) >= 4
            and sum(r["episode_adj"]["b5_onset"] for r in learned) >= 2),
        "LGT5_symmetry": bool(
            learned and all(abs(r["final_side_mean"]) <= 0.4 for r in learned)),
        "n_learned": len(learned),
        "episode_b5_count_learned": sum(
            r["episode_adj"]["b5_onset"] for r in learned),
        "median_plateau_len": None if not learned else float(np.median(
            [r["side_openness_plateau_len"] for r in learned])),
    }
    report = {
        "status": "LEARN-GRIP-TRANSPORT two-phase learned realization; preregistered",
        "config": {"N_agents": N_AGENTS, "threshold": THRESHOLD, "goal": GOAL,
                   "max_steps": MAX_STEPS, "seeds": N_SEEDS,
                   "updates": UPDATES, "batch": BATCH, "lr": LR,
                   "damp": DAMP, "accel": ACCEL, "grip_gain": GRIP_GAIN,
                   "grip_decay": GRIP_DECAY, "grip_min": GRIP_MIN},
        "seeds": rows,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "learn_grip_transport.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
