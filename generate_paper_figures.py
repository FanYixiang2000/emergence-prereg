"""Generate paper-style figures from GOGOGO experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_figures_dir() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    path = FIGURES / name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Wrote {path}")


def figure_possibility_tree() -> None:
    rows = read_csv(OUTPUTS / "possibility_tree_summary.csv")
    policies = [row["policy"] for row in rows]
    expected = [float(row["expected_return"]) for row in rows]
    immediate = [float(row["immediate_reward"]) for row in rows]
    x = np.arange(len(policies))
    width = 0.38
    plt.figure(figsize=(9, 4.6))
    plt.bar(x - width / 2, immediate, width, label="Immediate reward")
    plt.bar(x + width / 2, expected, width, label="Expected final return")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Return")
    plt.title("Local optimality can close better future options")
    plt.xticks(x, policies, rotation=25, ha="right")
    plt.legend(frameon=False)
    savefig("fig1_possibility_tree_returns.png")


def grid_from_rows(
    rows: Sequence[Mapping[str, str]],
    x_key: str,
    y_key: str,
    z_key: str,
    filter_key: str | None = None,
    filter_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = []
    for row in rows:
        if filter_key is not None and abs(float(row[filter_key]) - float(filter_value)) > 1e-9:
            continue
        selected.append(row)
    xs = sorted({float(row[x_key]) for row in selected})
    ys = sorted({float(row[y_key]) for row in selected})
    z = np.zeros((len(ys), len(xs)))
    lookup = {(float(row[x_key]), float(row[y_key])): float(row[z_key]) for row in selected}
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            z[i, j] = lookup.get((x, y), np.nan)
    return np.array(xs), np.array(ys), z


def plot_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    filename: str,
    cmap: str = "viridis",
) -> None:
    plt.figure(figsize=(7.2, 4.8))
    image = plt.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap=cmap,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar(image)
    cbar.set_label(colorbar_label)
    savefig(filename)


def figure_option_value_heatmap() -> None:
    rows = read_csv(OUTPUTS / "possibility_ablation_grid.csv")
    xs, ys, z = grid_from_rows(
        rows,
        x_key="cash_out",
        y_key="p_trigger_needed",
        z_key="option_value",
        filter_key="preserve_cost",
        filter_value=1.0,
    )
    plot_heatmap(
        xs,
        ys,
        z,
        title="Option value of preserving possibility (cost=1)",
        xlabel="Immediate cash-out reward",
        ylabel="P(trigger-needed context)",
        colorbar_label="V(preserve) - V(cash-out)",
        filename="fig2_option_value_heatmap.png",
        cmap="coolwarm",
    )


def figure_horizon_reversal_heatmap() -> None:
    rows = read_csv(OUTPUTS / "planning_horizon_grid.csv")
    xs, ys, z = grid_from_rows(
        rows,
        x_key="cash_out",
        y_key="p_trigger_needed",
        z_key="horizon_reversal",
        filter_key="preserve_cost",
        filter_value=1.0,
    )
    plot_heatmap(
        xs,
        ys,
        z,
        title="Same Bellman solver: horizon-induced action reversal (cost=1)",
        xlabel="Immediate cash-out reward",
        ylabel="P(trigger-needed context)",
        colorbar_label="H1 cash-out, H2 preserve",
        filename="fig3_horizon_reversal_heatmap.png",
        cmap="magma",
    )


def figure_ground_truth_validation() -> None:
    summary = read_json(OUTPUTS / "ptc_ground_truth_validation_summary.json")["summary"]
    labels = ["Structure only", "Option value", "Combined evidence"]
    aucs = [
        float(summary["auc_structure_only"]),
        float(summary["auc_option_value"]),
        float(summary["auc_combined_evidence"]),
    ]
    plt.figure(figsize=(6.8, 4.2))
    bars = plt.bar(labels, aucs, color=["#999999", "#4C78A8", "#F58518"])
    plt.ylim(0, 1.08)
    plt.ylabel("AUC vs analytic ground truth")
    plt.title("Multimodality alone is not sufficient evidence")
    for bar, value in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center")
    savefig("fig4_ground_truth_validation_auc.png")


def figure_performance_closure() -> None:
    summary = read_json(OUTPUTS / "performance_closure_summary.json")["summary"]
    labels = ["Myopic", "No context", "Full capability"]
    returns = [
        float(summary["myopic_mean_return"]),
        float(summary["no_context_mean_return"]),
        float(summary["full_mean_return"]),
    ]
    success = [
        0.0,
        float(summary["no_context_mean_success"]),
        float(summary["full_mean_success"]),
    ]
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(7.2, 4.5))
    plt.bar(x - width / 2, returns, width, label="Expected return")
    plt.bar(x + width / 2, success, width, label="Success rate")
    plt.xticks(x, labels)
    plt.ylabel("Value")
    plt.title("Full option-preserve + context-use capability closes the loop")
    plt.legend(frameon=False)
    savefig("fig5_performance_closure.png")


def figure_spatial_vs_contextual() -> None:
    spatial = read_csv(OUTPUTS / "spatial_sweep_summary.csv")
    contextual = read_csv(OUTPUTS / "contextual_sweep_summary.csv")
    regimes = ["pure_team", "dense_shaping", "random_noise", "uncertain_preference"]

    spatial_score = {
        row["regime"]: float(row["endogenous_emergence_score_mean"])
        for row in spatial
    }
    spatial_return = {
        row["regime"]: float(row["natural_team_return_mean_mean"])
        for row in spatial
    }
    contextual_return = {
        row["regime"]: float(row["natural_team_return_mean_mean"])
        for row in contextual
    }
    contextual_selective = {
        row["regime"]: float(row["selective_trigger_score_mean"])
        for row in contextual
    }

    x = np.arange(len(regimes))
    width = 0.2
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - 1.5 * width, [spatial_score[r] for r in regimes], width, label="Spatial PTC score")
    plt.bar(x - 0.5 * width, [spatial_return[r] for r in regimes], width, label="Spatial return")
    plt.bar(x + 0.5 * width, [contextual_selective[r] for r in regimes], width, label="Contextual selectivity")
    plt.bar(x + 1.5 * width, [contextual_return[r] for r in regimes], width, label="Contextual return")
    plt.xticks(x, regimes, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.title("PTC signature and factual performance are related but distinct")
    plt.legend(frameon=False, ncol=2)
    savefig("fig6_spatial_contextual_summary.png")


def figure_performance_robustness() -> None:
    rows = read_csv(OUTPUTS / "performance_robustness_grid.csv")
    mismatch = sorted({float(row["mismatch_payoff"]) for row in rows})
    asymmetry = sorted({float(row["trigger_payoff"]) - 11.0 for row in rows})
    z = np.zeros((len(mismatch), len(asymmetry)))
    lookup = {
        (float(row["mismatch_payoff"]), float(row["trigger_payoff"]) - 11.0): float(
            row["performance_closure_rate"]
        )
        for row in rows
    }
    for i, m in enumerate(mismatch):
        for j, a in enumerate(asymmetry):
            z[i, j] = lookup[(m, a)]
    plot_heatmap(
        np.array(asymmetry),
        np.array(mismatch),
        z,
        title="Performance closure is robust across payoff settings",
        xlabel="Payoff asymmetry (trigger - direct around 11)",
        ylabel="Mismatch payoff",
        colorbar_label="Performance-closure rate",
        filename="fig7_performance_robustness.png",
        cmap="viridis",
    )


def figure_external_sacrifice_ptc() -> None:
    rows = read_csv(OUTPUTS / "external_sacrifice_ptc_scores.csv")
    methods = [row["method"] for row in rows]
    scores = [float(row["external_ptc_score"]) for row in rows]
    blind = [float(row["low_bc_sacrifice"]) for row in rows]
    high = [float(row["high_bc_sacrifice"]) for row in rows]
    x = np.arange(len(methods))
    width = 0.28
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width, scores, width, label="External PTC score")
    plt.bar(x, high, width, label="High benefit sacrifice")
    plt.bar(x + width, blind, width, label="Low benefit blind sacrifice")
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Metric value")
    plt.title("External MARL sacrifice evidence: conditionality beats blind sacrifice")
    plt.legend(frameon=False)
    savefig("fig8_external_sacrifice_ptc.png")


def figure_collapse_burst() -> None:
    rows = read_csv(OUTPUTS / "collapse_burst_timeseries.csv")
    regimes = ["ordinary_gradual", "reward_shaped", "collapse_emergence", "random_instability"]
    prior = {basin: 0.25 for basin in ("selfish", "direct_team", "trigger_success", "noise")}

    def kl(dist: Mapping[str, float]) -> float:
        eps = 1e-12
        return sum(dist[b] * np.log2((dist[b] + eps) / (prior[b] + eps)) for b in prior)

    plt.figure(figsize=(8.2, 4.6))
    for regime in regimes:
        selected = [row for row in rows if row["regime"] == regime]
        ts = [float(row["t"]) for row in selected]
        cs = [
            kl({b: float(row[f"p_{b}"]) for b in prior})
            for row in selected
        ]
        plt.plot(ts, cs, label=regime)
    plt.xlabel("Time")
    plt.ylabel("Collapse KL from initial basin distribution")
    plt.title("Strong emergence requires burst-like useful collapse, not mere convergence")
    plt.legend(frameon=False)
    savefig("fig9_collapse_burst_timeseries.png")


def figure_synergy_pid_proxy() -> None:
    rows = read_csv(OUTPUTS / "synergy_pid_proxy_summary.csv")
    systems = [row["system"] for row in rows]
    i_joint = [float(row["i_joint_b"]) for row in rows]
    synergy = [float(row["synergy_proxy"]) for row in rows]
    x = np.arange(len(systems))
    width = 0.38
    plt.figure(figsize=(8, 4.4))
    plt.bar(x - width / 2, i_joint, width, label="I(X1,X2;B)")
    plt.bar(x + width / 2, synergy, width, label="Synergy proxy")
    plt.xticks(x, systems, rotation=20, ha="right")
    plt.ylabel("Bits")
    plt.title("Spatial emergence: only joint structure predicts the future basin")
    plt.legend(frameon=False)
    savefig("fig10_synergy_pid_proxy.png")


def figure_external_decoy_ptc() -> None:
    rows = read_csv(OUTPUTS / "external_decoy_ptc_scores.csv")
    controllers = [row["controller"] for row in rows]
    mean_win = [float(row["mean_win"]) for row in rows]
    decoy_damage = [float(row["mean_decoy_damage"]) for row in rows]
    score = [float(row["external_decoy_ptc_score"]) for row in rows]
    x = np.arange(len(controllers))
    width = 0.25
    plt.figure(figsize=(7.8, 4.5))
    plt.bar(x - width, mean_win, width, label="Mean win")
    plt.bar(x, decoy_damage, width, label="Decoy damage")
    plt.bar(x + width, score, width, label="External decoy PTC")
    plt.xticks(x, controllers, rotation=15, ha="right")
    plt.ylabel("Metric value")
    plt.title("External decoy benchmark: role structure avoids local traps")
    plt.legend(frameon=False)
    savefig("fig11_external_decoy_ptc.png")


def figure_external_decoy_trajectory_ptc() -> None:
    rows = read_csv(OUTPUTS / "external_decoy_trajectory_ptc_summary.csv")
    controllers = [row["controller"] for row in rows]
    useful = [float(row["mean_useful_nondecoy_consensus"]) for row in rows]
    trap = [float(row["mean_decoy_trap_consensus"]) for row in rows]
    win = [float(row["mean_win"]) for row in rows]
    score = [float(row["trajectory_level_ptc_score"]) for row in rows]
    x = np.arange(len(controllers))
    width = 0.2
    plt.figure(figsize=(7.8, 4.5))
    plt.bar(x - 1.5 * width, useful, width, label="Useful non-decoy collapse")
    plt.bar(x - 0.5 * width, trap, width, label="Decoy trap collapse")
    plt.bar(x + 0.5 * width, win, width, label="Win rate")
    plt.bar(x + 1.5 * width, score, width, label="Trajectory PTC")
    plt.xticks(x, controllers, rotation=10, ha="right")
    plt.ylabel("Metric value")
    plt.title("Trajectory-level decoy collapse: useful vs trap basins")
    plt.legend(frameon=False)
    savefig("fig12_external_decoy_trajectory_ptc.png")


def figure_representation_jump_bridge() -> None:
    time_rows = read_csv(OUTPUTS / "representation_jump_bridge_timeseries.csv")
    summary_rows = read_csv(OUTPUTS / "representation_jump_bridge_summary.csv")
    regimes = ["ordinary_gradual", "reward_shaped", "collapse_emergence", "random_instability"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for regime in regimes:
        selected = [row for row in time_rows if row["regime"] == regime]
        ts = [float(row["t"]) for row in selected]
        jumps = [float(row["representation_jump"]) for row in selected]
        bursts = [float(row["collapse_burst"]) for row in selected]
        axes[0].plot(ts, bursts, label=f"{regime} collapse")
        axes[0].plot(ts, jumps, linestyle="--", label=f"{regime} rep jump")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Stepwise change")
    axes[0].set_title("Possibility-collapse bursts align with representation jumps")

    labels = [row["regime"] for row in summary_rows]
    max_burst = [float(row["max_collapse_burst"]) for row in summary_rows]
    max_jump = [float(row["max_representation_jump"]) for row in summary_rows]
    corr = [float(row["burst_jump_correlation"]) for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.25
    axes[1].bar(x - width, max_burst, width, label="Max collapse burst")
    axes[1].bar(x, max_jump, width, label="Max rep jump")
    axes[1].bar(x + width, corr, width, label="Burst-jump corr.")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Representation jump is strongest under useful collapse")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].legend(frameon=False, fontsize=8)
    savefig("fig13_representation_jump_bridge.png")


def figure_learned_representation_audit() -> None:
    spatial = read_csv(OUTPUTS / "learned_representation_jump_summary.csv")
    contextual = read_csv(OUTPUTS / "contextual_learned_representation_summary.csv")
    regimes = [row["regime"] for row in contextual]
    spatial_raw = {
        row["regime"]: float(row["learned_representation_bridge_score"])
        for row in spatial
    }
    spatial_gated = {
        row["regime"]: float(row["ptc_gated_representation_bridge_score"])
        for row in spatial
    }
    contextual_raw = {
        row["regime"]: float(row["learned_representation_bridge_score"])
        for row in contextual
    }
    contextual_gated = {
        row["regime"]: float(row["ptc_gated_contextual_bridge_score"])
        for row in contextual
    }
    x = np.arange(len(regimes))
    width = 0.2
    plt.figure(figsize=(9.4, 4.8))
    plt.bar(x - 1.5 * width, [spatial_raw.get(r, 0.0) for r in regimes], width, label="Spatial raw rep jump")
    plt.bar(x - 0.5 * width, [spatial_gated.get(r, 0.0) for r in regimes], width, label="Spatial PTC-gated")
    plt.bar(x + 0.5 * width, [contextual_raw.get(r, 0.0) for r in regimes], width, label="Contextual raw rep jump")
    plt.bar(x + 1.5 * width, [contextual_gated.get(r, 0.0) for r in regimes], width, label="Contextual gated")
    plt.xticks(x, regimes, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Learned Q-representation jumps can create false positives")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    savefig("fig14_learned_representation_false_positive_audit.png")


def figure_within_episode_collapse() -> None:
    rows = read_csv(OUTPUTS / "within_episode_collapse_summary.csv")
    labels = [f"{row['regime']}\n{row['mode']}" for row in rows]
    h0 = [float(row["initial_future_entropy"]) for row in rows]
    iv_js = [float(row["intervention_js"]) for row in rows]
    gap = [float(row["intervention_return_gap"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(len(labels))
    width = 0.38
    axes[0].bar(x - width / 2, h0, width, label="Initial future entropy (bits)")
    axes[0].bar(x + width / 2, iv_js, width, label="Intervention JS (do-trigger vs do-not)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Bits")
    axes[0].set_title("State-level future distributions from learned rollouts")
    axes[0].legend(frameon=False, fontsize=8)

    colors = ["#2E8B57" if value > 0 else "#B22222" for value in gap]
    axes[1].bar(x, gap, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Return(do-trigger) - Return(do-not-trigger)")
    axes[1].set_title("Same action: useful collapse in rescue, harmful in bridge")
    savefig("fig15_within_episode_collapse.png")


def figure_unsupervised_basins() -> None:
    rows = read_csv(OUTPUTS / "unsupervised_basin_summary.csv")
    regimes = [row["regime"] for row in rows]
    purity = [float(row["cluster_purity"]) for row in rows]
    hand = [float(row["hand_effective_modes"]) for row in rows]
    cluster = [float(row["cluster_effective_modes"]) for row in rows]
    x = np.arange(len(regimes))
    width = 0.28
    plt.figure(figsize=(7.6, 4.4))
    plt.bar(x - width, purity, width, label="Cluster purity vs hand basins")
    plt.bar(x, hand, width, label="Effective modes (hand labels)")
    plt.bar(x + width, cluster, width, label="Effective modes (clusters)")
    plt.xticks(x, regimes, rotation=10, ha="right")
    plt.ylabel("Value")
    plt.title("Basins are recoverable without hand labels")
    plt.legend(frameon=False, fontsize=8)
    savefig("fig16_unsupervised_basin_discovery.png")


def figure_within_episode_ci() -> None:
    rows = read_csv(OUTPUTS / "within_episode_sweep_ci.csv")
    labels = [f"{row['regime']}\n{row['mode']}" for row in rows]
    means = [float(row["intervention_return_gap_mean"]) for row in rows]
    lo = [float(row["intervention_return_gap_lo95"]) for row in rows]
    hi = [float(row["intervention_return_gap_hi95"]) for row in rows]
    err_low = [m - l for m, l in zip(means, lo)]
    err_high = [h - m for m, h in zip(means, hi)]
    colors = ["#2E8B57" if value > 0 else "#B22222" for value in means]
    x = np.arange(len(labels))
    plt.figure(figsize=(7.6, 4.4))
    plt.bar(x, means, color=colors)
    plt.errorbar(x, means, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("Return(do-trigger) - Return(do-not-trigger)")
    plt.title("Intervention gap with 95% bootstrap CI over 5 seeds")
    savefig("fig17_within_episode_ci.png")


def figure_neural_probe() -> None:
    probe = read_csv(OUTPUTS / "neural_within_episode_summary.csv")
    bridge = read_csv(OUTPUTS / "neural_checkpoint_bridge_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    labels = [f"{row['regime']}\n{row['mode']}" for row in probe]
    h0 = [float(row["initial_future_entropy"]) for row in probe]
    gap = [float(row["intervention_return_gap"]) for row in probe]
    x = np.arange(len(labels))
    width = 0.38
    axes[0].bar(x - width / 2, h0, width, label="Initial future entropy (bits)")
    colors = ["#2E8B57" if value > 0 else "#B22222" for value in gap]
    axes[0].bar(x + width / 2, gap, width, color=colors, label="Intervention return gap")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_title("DQN policy: same sign-flip as tabular")
    axes[0].legend(frameon=False, fontsize=8)

    regimes = [row["regime"] for row in bridge]
    corr = [float(row["burst_jump_correlation"]) for row in bridge]
    align = [float(row["peak_alignment"]) for row in bridge]
    xb = np.arange(len(regimes))
    axes[1].bar(xb - 0.19, corr, 0.38, label="Burst-jump correlation")
    axes[1].bar(xb + 0.19, align, 0.38, label="Peak alignment")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(regimes, fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Neural embedding jumps track collapse bursts")
    axes[1].legend(frameon=False, fontsize=8)
    savefig("fig18_neural_probe.png")


def figure_criterion_battery() -> None:
    matrix = read_csv(OUTPUTS / "criterion_battery_matrix.csv")
    summary = read_json(OUTPUTS / "criterion_battery_summary.json")
    components = ["potential", "selectivity", "specificity", "usefulness", "endogeneity"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    systems = [row["system"] for row in matrix]
    grid = np.array(
        [[int(row[f"pass_{component}"]) for component in components] for row in matrix]
    )
    truth = np.array([int(row["ground_truth_emergent"]) for row in matrix])
    axes[0].imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(components)))
    axes[0].set_xticklabels(components, rotation=20, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(systems)))
    axes[0].set_yticklabels(
        [f"{'[E] ' if t else ''}{s}" for s, t in zip(systems, truth)], fontsize=8
    )
    axes[0].set_title("Original five-component harness (9 systems)")

    accuracy: Mapping[str, float] = summary["accuracy"]  # type: ignore[assignment]
    names = list(accuracy.keys())
    values = [float(accuracy[name]) for name in names]
    colors = ["#4C78A8" if name == "full" else "#999999" for name in names]
    axes[1].barh(range(len(names)), values, color=colors)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlim(0, 1.05)
    axes[1].axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Classification accuracy vs ground truth")
    axes[1].set_title("Named drop-ablation failures are component-specific")
    savefig("fig19_criterion_battery.png")


def figure_estimator_robustness() -> None:
    rows = read_csv(OUTPUTS / "estimator_robustness_grid.csv")
    temperatures = sorted({float(row["probe_temperature"]) for row in rows})
    plt.figure(figsize=(8.2, 4.6))
    for temperature in temperatures:
        selected = sorted(
            [row for row in rows if abs(float(row["probe_temperature"]) - temperature) < 1e-9],
            key=lambda row: int(row["samples"]),
        )
        samples = [int(row["samples"]) for row in selected]
        rescue = [float(row["uncertain_preference_rescue_gap"]) for row in selected]
        bridge = [float(row["uncertain_preference_bridge_gap"]) for row in selected]
        plt.plot(samples, rescue, marker="o", label=f"rescue gap, T={temperature}")
        plt.plot(samples, bridge, marker="s", linestyle="--", label=f"bridge gap, T={temperature}")
    plt.axhline(0, color="black", linewidth=0.9)
    plt.xscale("log", base=2)
    plt.xlabel("Rollout samples per estimate")
    plt.ylabel("Intervention return gap")
    plt.title("Sign conclusions survive all estimator settings (12/12 cells)")
    plt.legend(frameon=False, fontsize=7, ncol=2)
    savefig("fig20_estimator_robustness.png")


def figure_external_transfer() -> None:
    summary = read_json(OUTPUTS / "external_transfer_summary.json")
    components = ["potential", "selectivity", "specificity", "usefulness", "endogeneity"]
    rows = summary["measurements"]
    verdicts = summary["verdicts"]
    labels = summary["audited_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    systems = [row["system"] for row in rows]
    grid = np.array(
        [[int(verdicts[s]["passes"][component]) for component in components] for s in systems]
    )
    axes[0].imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(components)))
    axes[0].set_xticklabels(components, rotation=20, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(systems)))
    axes[0].set_yticklabels(
        [f"{'[E] ' if labels[s] else ''}{s}" for s in systems], fontsize=8
    )
    axes[0].set_title(
        "External swarm battery: pre-registered thresholds, 5/5 verdicts correct"
    )

    x = np.arange(len(systems))
    passive = [float(row["iv_gap_passive"]) for row in rows]
    aggressive = [float(row["iv_gap_aggressive"]) for row in rows]
    axes[1].bar(x - 0.19, passive, 0.38, color="#B22222", label="Forced engagement gap (passive)")
    axes[1].bar(x + 0.19, aggressive, 0.38, color="#2E8B57", label="Forced engagement gap (aggressive)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems, fontsize=7, rotation=15, ha="right")
    axes[1].set_ylabel("do_trigger - do_non_trigger score")
    axes[1].set_title("Same sign-flipping usefulness as the internal family")
    axes[1].legend(frameon=False, fontsize=8)
    savefig("fig21_external_transfer.png")


def figure_phase_boundary() -> None:
    summary = read_json(OUTPUTS / "phase_boundary_summary.json")
    rows = summary["rows"]
    g = [float(row["goal_reward"]) for row in rows]
    rate = [float(row["natural_trigger_rate"]) for row in rows]
    gap = [float(row["usefulness_gap"]) for row in rows]
    accepted = [int(row["accepted"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    axes[0].plot(g, rate, marker="o", color="#4C78A8", label="Natural trigger rate")
    axes[0].axvline(5.0, color="gray", linestyle=":", label="Predicted onset G=5")
    axes[0].axvline(11.0, color="gray", linestyle="--", label="Predicted 2nd onset G=11")
    axes[0].axhline(1 / 6, color="#4C78A8", linewidth=0.6, linestyle=":")
    axes[0].axhline(1 / 3, color="#4C78A8", linewidth=0.6, linestyle=":")
    axes[0].set_xlabel("High-goal reward G")
    axes[0].set_ylabel("Natural trigger rate")
    axes[0].set_title("Behavioral onset follows the payoff accounting")
    axes[0].legend(frameon=False, fontsize=8)

    colors = ["#2E8B57" if a else "#B22222" for a in accepted]
    axes[1].bar(g, gap, width=1.2, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].axvline(9.0, color="gray", linestyle="--", label="Predicted usefulness boundary G=9")
    axes[1].set_xlabel("High-goal reward G")
    axes[1].set_ylabel("Counterfactual-necessity gap")
    axes[1].set_title(
        "Criterion acceptance (green) flips at the predicted boundary"
    )
    axes[1].legend(frameon=False, fontsize=8)
    savefig("fig22_phase_boundary.png")


def figure_grokking_bridge() -> None:
    rows = read_csv(OUTPUTS / "grokking_collapse_timeseries.csv")
    summary = read_json(OUTPUTS / "grokking_collapse_summary.json")
    runs = ("grokking", "memorizer", "no_structure", "prewired")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    palette = {"grokking": "#4C78A8", "memorizer": "#E45756",
               "no_structure": "#B279A2", "prewired": "#F58518"}
    for run in runs:
        selected = [row for row in rows if row["run"] == run]
        epochs = [int(row["epoch"]) for row in selected]
        axes[0].plot(epochs, [float(r["test_acc"]) for r in selected],
                     color=palette[run], label=run)
        axes[1].plot(epochs, [float(r["test_entropy_bits"]) for r in selected],
                     color=palette[run], label=run)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Held-out accuracy")
    axes[0].set_title("Grokking: delayed sudden generalization")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Held-out predictive entropy (bits)")
    axes[1].set_title("Possibility collapse on unseen inputs")
    axes[1].legend(frameon=False, fontsize=8)

    grok = [row for row in rows if row["run"] == "grokking"]
    epochs = [int(row["epoch"]) for row in grok]
    collapse = [float(row["collapse_bits"]) for row in grok]
    bursts = [0.0] + [max(collapse[i] - collapse[i - 1], 0.0) for i in range(1, len(collapse))]
    acc = [float(row["test_acc"]) for row in grok]
    ax2 = axes[2].twinx()
    axes[2].plot(epochs, bursts, color="#4C78A8", label="Collapse burst B_k")
    ax2.plot(epochs, acc, color="#2E8B57", linestyle="--", label="Test accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Collapse burst (bits)", color="#4C78A8")
    ax2.set_ylabel("Test accuracy", color="#2E8B57")
    window_epoch = summary["runs"]["grokking"]["stats"]["window_epoch"]
    axes[2].axvline(window_epoch, color="gray", linestyle=":", linewidth=1.0)
    axes[2].set_title("Useful burst coincides with the generalization jump")
    savefig("fig23_grokking_bridge.png")


def figure_external_transfer_sweep() -> None:
    summary = read_json(OUTPUTS / "external_transfer_sweep_summary.json")
    rows = read_csv(OUTPUTS / "external_transfer_sweep_per_seed.csv")
    systems = ("marl_learned", "marl_untrained", "nearest_only", "role_oracle", "damage_aware")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    rates = summary["prediction_pass_rates"]
    names = list(rates.keys())
    axes[0].barh(range(len(names)), [float(rates[n]) for n in names], color="#4C78A8")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlim(0, 1.05)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Pass rate across seeds")
    axes[0].set_title("Registered predictions across independent seeds")

    for idx, system in enumerate(systems):
        values = [float(row["usefulness_gap"]) for row in rows if row["system"] == system]
        axes[1].scatter([idx] * len(values), values, color="#4C78A8", alpha=0.7, s=28)
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_xticks(range(len(systems)))
    axes[1].set_xticklabels(systems, fontsize=7, rotation=15, ha="right")
    axes[1].set_ylabel("Counterfactual-necessity gap")
    axes[1].set_title("Per-seed usefulness gaps (external battery)")
    savefig("fig24_external_transfer_sweep.png")


def figure_refined_confirmation() -> None:
    summary = read_json(OUTPUTS / "refined_confirmation_summary.json")
    per_seed = summary["external"]["per_seed"]
    seeds = sorted(per_seed.keys())
    systems = ("marl_learned", "marl_untrained", "nearest_only", "role_oracle", "damage_aware")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    grid = np.array(
        [
            [
                1 if per_seed[seed][system]["verdict"]["emergent"]
                == per_seed[seed][system]["truth"] else 0
                for seed in seeds
            ]
            for system in systems
        ]
    )
    axes[0].imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(seeds)))
    axes[0].set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
    axes[0].set_yticks(range(len(systems)))
    axes[0].set_yticklabels(systems, fontsize=8)
    axes[0].set_title("Frozen refined criterion on fresh seeds: 25/25 correct")

    learned_acq = [float(per_seed[s]["marl_learned"]["acquisition"]) for s in seeds]
    untrained_acq = [float(per_seed[s]["marl_untrained"]["acquisition"]) for s in seeds]
    untrained_sep = [float(per_seed[s]["marl_untrained"]["separation"]) for s in seeds]
    x = np.arange(len(seeds))
    axes[1].bar(x - 0.25, learned_acq, 0.25, color="#2E8B57", label="learned: acquisition")
    axes[1].bar(x, untrained_acq, 0.25, color="#B22222", label="untrained: acquisition")
    axes[1].bar(x + 0.25, untrained_sep, 0.25, color="#F58518",
                label="untrained: separation (accidental)")
    axes[1].axhline(0.3, color="black", linestyle="--", linewidth=0.9,
                    label="acquisition threshold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{s}" for s in seeds], fontsize=8)
    axes[1].set_ylabel("Per-context separation / acquisition")
    axes[1].set_title("Acquisition separates learned structure from accident")
    axes[1].legend(frameon=False, fontsize=7)
    savefig("fig25_refined_confirmation.png")


def figure_scale_decomposition() -> None:
    summary = read_json(OUTPUTS / "scale_emergence_summary.json")
    scales = [float(s) for s in summary["scales"]]
    acc = [float(v) for v in summary["acc_curve"]]
    collapse = [float(v) for v in summary["collapse_curve"]]
    logp = [float(v) for v in summary["logp_curve"]]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(scales, acc, marker="o", color="#B22222", label="Exact-match accuracy (discontinuous)")
    lo, hi = min(collapse), max(collapse)
    collapse_norm = [(c - lo) / (hi - lo + 1e-9) for c in collapse]
    ax.plot(scales, collapse_norm, marker="s", color="#4C78A8",
            label="Held-out collapse C (normalized, continuous)")
    lo2, hi2 = min(logp), max(logp)
    logp_norm = [(v - lo2) / (hi2 - lo2 + 1e-9) for v in logp]
    ax.plot(scales, logp_norm, marker="^", color="#2E8B57", linestyle="--",
            label="Correct-answer log-prob (normalized)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Width scale factor")
    ax.set_ylabel("Normalized value")
    ax.set_title(
        f"Ability jump vs collapse mechanism across scale "
        f"(verdict: {summary['data_verdict']})"
    )
    ax.legend(frameon=False, fontsize=8)
    savefig("fig26_scale_decomposition.png")


def figure_grokking_generality() -> None:
    summary = read_json(OUTPUTS / "grokking_generality_summary.json")
    cells = summary["cells"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    grok = [c for c in cells if c["condition"] == "grokking"]
    mem = [c for c in cells if c["condition"] == "memorizer"]
    labels = [f"{c['op']}\nseed {c['seed']}" for c in grok]
    x = np.arange(len(grok))
    axes[0].bar(x - 0.19, [c["usefulness_acc_gain"] for c in grok], 0.38,
                color="#2E8B57", label="grokking")
    axes[0].bar(x + 0.19, [c["usefulness_acc_gain"] for c in mem], 0.38,
                color="#B22222", label="memorizer")
    axes[0].axhline(0.2, color="black", linestyle="--", linewidth=0.9,
                    label="usefulness threshold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("Accuracy gain at collapse window")
    axes[0].set_title("Useful collapse: grokking vs memorizer, both tasks")
    axes[0].legend(frameon=False, fontsize=8)

    train_ep = [c["train_acc_99_epoch"] for c in grok]
    test_ep = [c["test_acc_90_epoch"] for c in grok]
    axes[1].bar(x - 0.19, train_ep, 0.38, color="#999999", label="train acc 99% epoch")
    axes[1].bar(x + 0.19, test_ep, 0.38, color="#4C78A8", label="test acc 90% epoch")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("Epoch")
    axes[1].set_title("Delayed generalization in every cell")
    axes[1].legend(frameon=False, fontsize=8)
    savefig("fig27_grokking_generality.png")


def figure_robustness_and_priors() -> None:
    sens = read_json(OUTPUTS / "threshold_sensitivity_summary.json")
    priors = read_json(OUTPUTS / "prior_metrics_comparison.json")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for key, curve in sens["external"]["curves"].items():
        if key == "usefulness_gap":
            continue
        xs = [float(point["multiplier"]) for point in curve]
        ys = [float(point["accuracy"]) for point in curve]
        axes[0].plot(xs, ys, marker="o", label=f"external: {key}")
    for key, curve in sens["internal"]["curves"].items():
        if key == "usefulness_gap":
            continue
        xs = [float(point["multiplier"]) for point in curve]
        ys = [float(point["accuracy"]) for point in curve]
        axes[0].plot(xs, ys, marker="s", linestyle="--", label=f"internal: {key}")
    axes[0].axvline(1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_xlabel("Threshold multiplier")
    axes[0].set_ylabel("Battery accuracy")
    axes[0].set_title("Threshold sensitivity: wide plateaus, one honest edge")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)

    names = list(priors["detectors"].keys()) + ["full_criterion"]
    values = [
        float(priors["detectors"][n]["best_accuracy"]) for n in priors["detectors"]
    ] + [1.0]
    colors = ["#999999"] * (len(names) - 1) + ["#4C78A8"]
    axes[1].barh(range(len(names)), values, color=colors)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlim(0, 1.05)
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Battery accuracy (prior detectors get oracle thresholds)")
    axes[1].set_title("Single prior observables cap below the full criterion")
    savefig("fig28_robustness_and_priors.png")


def figure_induction_heads() -> None:
    rows = read_csv(OUTPUTS / "induction_head_timeseries.csv")
    summary = read_json(OUTPUTS / "induction_head_summary.json")
    runs = ("induction_2layer", "induction_1layer", "no_structure", "memorizer")
    palette = {"induction_2layer": "#4C78A8", "induction_1layer": "#E45756",
               "no_structure": "#B279A2", "memorizer": "#F58518"}

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    for run in runs:
        selected = [row for row in rows if row["run"] == run]
        steps = [int(row["epoch"]) for row in selected]
        axes[0].plot(steps, [float(r["test_acc"]) for r in selected],
                     color=palette[run], label=run)
        axes[1].plot(steps, [float(r["test_entropy_bits"]) for r in selected],
                     color=palette[run], label=run)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Induction (copy) accuracy, fresh sequences")
    axes[0].set_title("Induction ability forms abruptly; 1-layer control cannot")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Held-out predictive entropy (bits)")
    axes[1].set_title("Possibility collapse at second-half positions")
    axes[1].legend(frameon=False, fontsize=8)

    two = [row for row in rows if row["run"] == "induction_2layer"]
    steps = [int(row["epoch"]) for row in two]
    collapse = [float(row["collapse_bits"]) for row in two]
    bursts = [0.0] + [max(collapse[i] - collapse[i - 1], 0.0)
                      for i in range(1, len(collapse))]
    acc = [float(row["test_acc"]) for row in two]
    ax2 = axes[2].twinx()
    axes[2].plot(steps, bursts, color="#4C78A8", label="Collapse burst B_k")
    ax2.plot(steps, acc, color="#2E8B57", linestyle="--", label="Copy accuracy")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Collapse burst (bits)", color="#4C78A8")
    ax2.set_ylabel("Copy accuracy", color="#2E8B57")
    window = summary["runs"]["induction_2layer"]["stats"]["window_epoch"]
    axes[2].axvline(window, color="gray", linestyle=":", linewidth=1.0)
    axes[2].set_title("Useful burst coincides with circuit formation")
    savefig("fig29_induction_heads.png")


def figure_transformer_grokking() -> None:
    rows = read_csv(OUTPUTS / "transformer_grokking_timeseries.csv")
    summary = read_json(OUTPUTS / "transformer_grokking_summary.json")
    palette = {"transformer_grokking": "#4C78A8", "transformer_memorizer": "#E45756"}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for run, color in palette.items():
        selected = [row for row in rows if row["run"] == run]
        epochs = [int(row["epoch"]) for row in selected]
        axes[0].plot(epochs, [float(r["test_acc"]) for r in selected],
                     color=color, label=run)
        axes[1].plot(epochs, [float(r["test_entropy_bits"]) for r in selected],
                     color=color, label=run)
    window = summary["runs"]["transformer_grokking"]["stats"]["window_epoch"]
    for ax in axes:
        ax.axvline(window, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Held-out accuracy")
    axes[0].set_title("Grokking replicates in a transformer (412k params)")
    axes[1].set_ylabel("Held-out predictive entropy (bits)")
    axes[1].set_title("Same useful possibility collapse, different architecture")
    savefig("fig30_transformer_grokking.png")


def figure_multiberts() -> None:
    rows = read_csv(OUTPUTS / "multiberts_collapse_timeseries.csv")
    summary = read_json(OUTPUTS / "multiberts_collapse_summary.json")
    agree = [row for row in rows if row["run"] == "multiberts_agreement"]
    rand = [row for row in rows if row["run"] == "multiberts_random_target"]
    perm = [row for row in rows if row["run"] == "shuffled_vocab"]
    steps = [int(row["epoch"]) for row in agree]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    axes[0].plot(steps, [float(r["test_acc"]) for r in agree],
                 color="#4C78A8", marker="o", ms=3, label="agreement (real ability)")
    axes[0].plot(steps, [float(r["test_acc"]) for r in rand],
                 color="#E45756", marker="s", ms=3, label="random-target control")
    axes[0].plot(steps, [float(r["test_acc"]) for r in perm],
                 color="#B279A2", marker="^", ms=3, label="shuffled-vocab control")
    axes[0].axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_xscale("symlog", linthresh=20000)
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel("Pretraining step (published checkpoints)")
    axes[0].set_ylabel("Minimal-pair accuracy")
    axes[0].set_title("Subject-verb agreement forms early in MultiBERTs")
    axes[0].legend(frameon=False, fontsize=7)

    ax_h = axes[1]
    ax_h.plot(steps, [float(r["test_entropy_bits"]) for r in agree],
              color="#4C78A8", marker="o", ms=3)
    ax_h.set_xscale("symlog", linthresh=20000)
    ax_h.set_xlabel("Pretraining step")
    ax_h.set_ylabel("Masked-position entropy (bits)")
    ax_h.set_title("Possibility collapse at the masked verb")

    collapse = [float(r["collapse_bits"]) for r in agree]
    bursts = [0.0] + [max(collapse[i] - collapse[i - 1], 0.0)
                      for i in range(1, len(collapse))]
    acc = [float(r["test_acc"]) for r in agree]
    ax2 = axes[2].twinx()
    axes[2].plot(steps, bursts, color="#4C78A8", label="Collapse burst")
    ax2.plot(steps, acc, color="#2E8B57", linestyle="--", label="Agreement accuracy")
    axes[2].set_xscale("symlog", linthresh=20000)
    window = summary["runs"]["multiberts_agreement"]["stats"]["window_epoch"]
    axes[2].axvline(window, color="gray", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("Pretraining step")
    axes[2].set_ylabel("Collapse burst (bits)", color="#4C78A8")
    ax2.set_ylabel("Agreement accuracy", color="#2E8B57")
    axes[2].set_title("Useful burst in a 110M-param public model")
    savefig("fig31_multiberts_public_series.png")


def figure_phenomena_and_alignment() -> None:
    rows = read_csv(OUTPUTS / "multiberts_phenomena_timeseries.csv")
    align = read_json(OUTPUTS / "burst_alignment_test.json")
    families = ("reflexive", "determiner", "facts", "npi")
    palette = {"reflexive": "#4C78A8", "determiner": "#2E8B57",
               "facts": "#F58518", "npi": "#B22222"}

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    for fam in families:
        selected = [row for row in rows if row["run"] == fam]
        steps = [int(row["epoch"]) for row in selected]
        axes[0].plot(steps, [float(r["test_acc"]) for r in selected],
                     color=palette[fam], marker="o", ms=3, label=fam)
    axes[0].axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_xscale("symlog", linthresh=20000)
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel("Pretraining step")
    axes[0].set_ylabel("Minimal-pair accuracy")
    axes[0].set_title("Four abilities, one public model: all abrupt, NPI last")
    axes[0].legend(frameon=False, fontsize=8)

    npi = [row for row in rows if row["run"] == "npi"]
    steps = [int(row["epoch"]) for row in npi]
    ax2 = axes[1].twinx()
    axes[1].plot(steps, [float(r["test_entropy_bits"]) for r in npi],
                 color="#B22222", label="NPI entropy")
    ax2.plot(steps, [float(r["test_acc"]) for r in npi],
             color="#2E8B57", linestyle="--", label="NPI accuracy")
    axes[1].set_xscale("symlog", linthresh=20000)
    axes[1].set_xlim(left=0)
    axes[1].set_xlabel("Pretraining step")
    axes[1].set_ylabel("Entropy (bits)", color="#B22222")
    ax2.set_ylabel("Accuracy", color="#2E8B57")
    axes[1].set_title("NPI: collapse at 20k precedes usefulness at 40k")

    emergent = align["emergent"]
    controls = align["controls"]
    names = list(emergent.keys()) + list(controls.keys())
    values = [emergent[n]["p_value"] for n in emergent] + \
             [controls[n]["p_value"] for n in controls]
    colors = ["#4C78A8"] * len(emergent) + ["#999999"] * len(controls)
    axes[2].barh(range(len(names)), values, color=colors)
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels(names, fontsize=6)
    axes[2].axvline(3 / 27, color="black", linestyle="--", linewidth=0.8,
                    label="R4 bound (3/27)")
    axes[2].set_xlabel("Empirical window rank (blue = accepted runs)")
    axes[2].set_title("Ability jumps land in top collapse-burst windows")
    axes[2].legend(frameon=False, fontsize=7)
    savefig("fig32_phenomena_and_alignment.png")


def figure_chess_collapse() -> None:
    rows = read_csv(OUTPUTS / "chess_collapse_main_positions.csv")
    summary = read_json(OUTPUTS / "chess_collapse_main_summary.json")
    sac = [r for r in rows if r["kind"] == "sacrifice"]
    qui = [r for r in rows if r["kind"] == "quiet"]

    def vals(rs, field):
        return [float(r[field]) for r in rs if r.get(field)]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    labels = ["key\n(annotated)", "deep alt\n(best other)", "greedy\n(depth 2)",
              "random", "quiet best\n(control)"]
    data = [vals(sac, "p_win_key"), vals(sac, "p_win_deep_alt"),
            vals(sac, "p_win_greedy"), vals(sac, "p_win_random"),
            vals(qui, "p_win_best")]
    colors = ["#B22222", "#4C78A8", "#F58518", "#999999", "#2E8B57"]
    parts = axes[0].violinplot(data, showmeans=True, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("P(win basin | do move)")
    axes[0].set_title("Forced key move vs counterfactuals\n"
                      "(240 sacrifice + 120 quiet positions)")

    cost = vals(sac, "local_cost_key")
    pwin = vals(sac, "p_win_key")
    axes[1].scatter(cost, pwin, s=12, alpha=0.45, color="#B22222")
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Material cost of key move vs best reply (pawns)")
    axes[1].set_ylabel("P(win | do key)")
    axes[1].set_title("Locally costly, globally decisive\n"
                      "(median cost -3.0; 80% strictly costly)")

    gap_sac = [float(r["p_win_key"]) - float(r["p_win_deep_alt"])
               for r in sac if r.get("p_win_deep_alt")]
    base_q = vals(qui, "p_win_base")
    gap_qui = [float(r["p_win_best"]) - b for r, b in zip(qui, base_q)
               if r.get("p_win_best")]
    axes[2].hist(gap_sac, bins=24, alpha=0.65, color="#B22222",
                 label="sacrifice: key - deep alt")
    axes[2].hist(gap_qui, bins=24, alpha=0.65, color="#2E8B57",
                 label="quiet: best - base")
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("P(win) do-contrast")
    axes[2].set_ylabel("Positions")
    c3 = summary["C3_selectivity"]
    axes[2].set_title("Trigger-specific useful collapse\n"
                      f"(mean gap {c3['gap']:.2f}, sign test p = "
                      f"{c3['sign_vs_deep_alt']['p_one_sided']:.0e})")
    axes[2].legend(frameon=False, fontsize=8)
    savefig("fig33_chess_collapse.png")


def _panel_gradualism(ax) -> None:
    rows = read_csv(OUTPUTS / "multiberts_tail_timeseries.csv")
    families = [("head_facts_top1", "#F58518", "head facts (top-1)"),
                ("tail_facts_top1", "#B22222", "tail facts (top-1)"),
                ("tail_words_top1", "#4C78A8", "tail words (top-1)")]
    for run, color, label in families:
        selected = [r for r in rows if r["run"] == run]
        steps = [int(r["epoch"]) for r in selected]
        ax.plot(steps, [float(r["test_acc"]) for r in selected],
                color=color, marker="o", ms=3, label=label)
    ax.axhline(0.2, color="gray", linewidth=0.8, linestyle="--",
               label="usefulness threshold (windowed)")
    ax.set_xscale("symlog", linthresh=20000)
    ax.set_xlim(left=0)
    ax.set_xlabel("Pretraining step")
    ax.set_ylabel("Top-1 recall (chance ~ 3e-5)")
    ax.set_title("Gradualism rejection test: tail words accrue\n"
                 "slowly and are correctly NOT emergent")
    ax.legend(frameon=False, fontsize=7)


def _panel_chess_grid(ax) -> None:
    grid = read_json(OUTPUTS / "chess_robustness_grid.json")
    cells = list(grid["grid"].items())
    names = [n for n, _ in cells]
    gaps = [c["summary"]["C3_selectivity"]["gap"] for _, c in cells]
    colors = ["#2E8B57" if c["checks"]["c3_gap_and_sign"] else "#B22222"
              for _, c in cells]
    y = np.arange(len(names))
    ax.barh(y, gaps, color=colors, alpha=0.8)
    ax.axvline(0.15, color="black", linestyle="--", linewidth=0.8,
               label="C3 threshold 0.15")
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=6)
    ax.set_xlabel("P(win) do-contrast gap (key vs deep alt)")
    ax.set_title("Chess conclusions across 12 estimator /\n"
                 "basin / engine cells (12/12 core pass)")
    ax.legend(frameon=False, fontsize=7)


def _panel_chess_detectors(ax) -> None:
    detect = read_json(OUTPUTS / "chess_prior_detectors.json")
    det = detect["key_identification"]
    dnames = ["useful_collapse", "collapse_only", "local_value", "specificity_only"]
    rates = [det[d]["key_top1_rate"] for d in dnames]
    dcolors = ["#2E8B57", "#4C78A8", "#F58518", "#999999"]
    ax.bar(range(len(dnames)), rates, color=dcolors)
    ax.axhline(0.25, color="gray", linewidth=0.8, linestyle="--",
               label="4-way chance")
    ax.set_xticks(range(len(dnames)))
    ax.set_xticklabels([d.replace("_", "\n") for d in dnames], fontsize=8)
    ax.set_ylabel("Annotated key move ranked first")
    ax.set_title("Single signals cannot find the key move\n"
                 "(240 external sacrifice positions)")
    ax.legend(frameon=False, fontsize=7)


def figure_gradualism_and_robustness() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    _panel_gradualism(axes[0])
    _panel_chess_grid(axes[1])
    _panel_chess_detectors(axes[2])
    savefig("fig34_gradualism_and_robustness.png")

    # Standalone splits consumed by assemble_main_figures.py. The detector
    # diagnostic is retained in the three-panel audit figure but excluded
    # from the main manuscript because its move ranking shares a constant
    # within-position baseline and is not a fair test of the martingale lesson.
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.4))
    _panel_gradualism(ax)
    savefig("fig34a_gradualism.png")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    _panel_chess_grid(axes[0])
    _panel_chess_detectors(axes[1])
    savefig("fig34bc_chess_grid_detectors.png")
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.4))
    _panel_chess_grid(ax)
    savefig("fig34b_chess_grid.png")


def figure_deep_marl() -> None:
    seeds = (11, 22, 33)
    data = {s: read_json(OUTPUTS / f"deep_marl_collapse_mappo_seed{s}.json")
            for s in seeds}
    agg = read_json(OUTPUTS / "deep_marl_collapse_aggregate.json")
    trained = [data[s]["conditions"][f"trained_seed{s}"] for s in seeds]
    controls = data[seeds[0]]["conditions"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    names = [f"trained\nseed {s}" for s in seeds] + ["untrained", "greedy\nnearest", "noise"]
    pots = ([t["early_potential_bits"] for t in trained]
            + [controls["untrained"]["early_potential_bits"],
               controls["greedy_nearest"]["early_potential_bits"],
               controls["noise"]["early_potential_bits"]])
    bij = ([t["final_bijection_rate"] for t in trained]
           + [controls["untrained"]["final_bijection_rate"],
              controls["greedy_nearest"]["final_bijection_rate"],
              controls["noise"]["final_bijection_rate"]])
    x = np.arange(len(names))
    width = 0.38
    colors = ["#2E8B57"] * 3 + ["#999999", "#F58518", "#B22222"]
    axes[0].bar(x - width / 2, pots, width, color=colors, alpha=0.85,
                label="early potential (bits)")
    axes[0].bar(x + width / 2, bij, width, color=colors, alpha=0.45,
                label="final bijection rate")
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].axhline(3 * 2 * 1 / 27, color="gray", linewidth=0.8, linestyle=":",
                    label="uniform-assignment rate (0.22)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_title("Potential and usefulness are different axes\n"
                      "(PPO on PettingZoo simple_spread)")
    axes[0].legend(frameon=False, fontsize=7)

    gaps = []
    for t in trained:
        gaps.extend(e["p_win_do_commit"] - e["p_win_do_block"]
                    for e in t["episodes"])
    axes[1].hist(gaps, bins=21, color="#4C78A8", alpha=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(float(np.median(gaps)), color="#B22222", linewidth=1.2,
                    label=f"median +{np.median(gaps):.3f}")
    d3 = agg["D3_counterfactual"]
    axes[1].set_xlabel("P(bijection | do_commit) - P(bijection | do_block)")
    axes[1].set_ylabel("Episodes (3 seeds pooled)")
    axes[1].set_title("Commitment is counterfactually load-bearing\n"
                      f"(episode-level, conditional on 3 policies: "
                      f"{d3['pooled_sign_wins']}W/"
                      f"{d3['pooled_sign_losses']}L; seed-level "
                      "inference in main Fig. 5)")
    axes[1].legend(frameon=False, fontsize=8)

    shifts = [t["p_win_end"] - t["p_win_start"] for t in trained]
    starts = [t["p_win_start"] for t in trained]
    axes[2].bar(np.arange(3) - 0.2, starts, 0.38, color="#4C78A8",
                label="P0(bijection) under own policy")
    axes[2].bar(np.arange(3) + 0.2, [t["p_win_end"] for t in trained], 0.38,
                color="#2E8B57", label="P_end(bijection)")
    axes[2].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels([f"seed {s}" for s in seeds])
    axes[2].set_title("D2 registered failure: P_t(win) is a martingale\n"
                      "under the behaving policy (win shift ~ 0)")
    axes[2].legend(frameon=False, fontsize=8)
    savefig("fig35_deep_marl.png")


def figure_lbf_cross_task() -> None:
    data = read_json(OUTPUTS / "lbf_collapse_main.json")
    boot = read_json(OUTPUTS / "bootstrap_intervals.json")
    seeds = (11, 22, 33)
    trained = [data["conditions"][f"trained_seed{s}"] for s in seeds]
    controls = data["conditions"]
    verd = data["verdicts"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    names = [f"trained\nseed {s}" for s in seeds] + [
        "untrained", "greedy\nnearest", "noise"]
    pots = ([t["early_potential_bits"] for t in trained]
            + [controls["untrained"]["early_potential_bits"],
               controls["greedy_nearest"]["early_potential_bits"],
               controls["noise"]["early_potential_bits"]])
    wins = ([t["final_win_rate"] for t in trained]
            + [controls["untrained"]["final_win_rate"],
               controls["greedy_nearest"]["final_win_rate"],
               controls["noise"]["final_win_rate"]])
    x = np.arange(len(names))
    width = 0.38
    colors = ["#2E8B57"] * 3 + ["#999999", "#F58518", "#B22222"]
    axes[0].bar(x - width / 2, pots, width, color=colors, alpha=0.85,
                label="early order-openness (bits)")
    axes[0].bar(x + width / 2, wins, width, color=colors, alpha=0.45,
                label="full-clearance rate")
    axes[0].axhline(0.8, color="black", linewidth=0.8, linestyle="--",
                    label="L1 threshold (0.8 bits)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_title("LBF cross-task replication\n"
                      "openness and usefulness dissociate")
    axes[0].legend(frameon=False, fontsize=7)

    gaps = []
    for t in trained:
        gaps.extend(e["p_win_do_commit"] - e["p_win_do_block"]
                    for e in t["episodes"] if "p_win_do_commit" in e)
    l3 = verd["L3_counterfactual"]
    ci = boot["deep_marl_lbf"]["lbf_do_gap_mean_pooled"]
    axes[1].hist(gaps, bins=21, color="#4C78A8", alpha=0.8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(float(np.median(gaps)), color="#B22222", linewidth=1.2,
                    label=f"median +{np.median(gaps):.3f}")
    axes[1].set_xlabel("P(clear | do_commit) - P(clear | do_block)")
    axes[1].set_ylabel("Episodes (3 seeds pooled)")
    axes[1].set_title(
        f"Commitment is counterfactually necessary\n"
        f"(episode-level, conditional on 3 policies: "
        f"{l3['pooled_sign_wins']}W/{l3['pooled_sign_losses']}L; "
        f"mean={ci['point']:+.3f}; seed-level in main Fig. 5)"
    )
    axes[1].legend(frameon=False, fontsize=8)

    labels = ["L1\npotential", "L2\nuseful structure", "L3\ncounterfactual",
              "L4\ngreedy contrast"]
    passes = [verd["L1_potential"]["pass"], verd["L2_useful_structure"]["pass"],
              verd["L3_counterfactual"]["pass"], verd["L4_greedy_contrast"]["pass"]]
    axes[2].bar(range(4), [1] * 4,
                color=["#2E8B57" if p else "#B22222" for p in passes], alpha=0.8)
    for i, p in enumerate(passes):
        axes[2].text(i, 0.5, "PASS" if p else "FAIL", ha="center",
                     va="center", color="white", fontsize=11, fontweight="bold")
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_yticks([])
    axes[2].set_title("Four preregistered predictions\n"
                      "martingale lesson incorporated")
    savefig("fig36_lbf_cross_task.png")


def figure_exact_prior_formalisms() -> None:
    data = read_json(OUTPUTS / "exact_prior_formalisms.json")
    ce = data["detectors"]["causal_emergence_exact"]
    psi = data["detectors"]["phiid_psi_exact"]
    truth = data["truth"]
    systems = list(ce["scores"].keys())
    colors = ["#4C78A8" if truth[s] else "#B0B0B0" for s in systems]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ys = range(len(systems))
    axes[0].barh(ys, [ce["scores"][s] for s in systems], color=colors)
    axes[0].set_yticks(list(ys))
    axes[0].set_yticklabels(systems, fontsize=8)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Exact CE = max EI(macro) - EI(micro)  [bits]")
    axes[0].set_title(
        "Hoel EI, exact form: CE < 0 for every system\n"
        f"best threshold = trivial classifier "
        f"(acc {ce['hindsight_best']['accuracy']:.1f}, misses both positives)")

    axes[1].barh(ys, [psi["scores"][s] for s in systems], color=colors)
    axes[1].set_yticks(list(ys))
    axes[1].set_yticklabels(systems, fontsize=8)
    axes[1].axvline(0.0, color="black", linewidth=0.8,
                    label="published verdict: $\\Psi>0$ (acc 0.3)")
    axes[1].axvline(float(psi["hindsight_best"]["threshold"]), color="#E45756",
                    linestyle="--", linewidth=1.0,
                    label="hindsight best (sign-inverted, acc 0.9)")
    axes[1].set_xlabel("Exact $\\Psi$ = max over supervenient V  [bits]")
    axes[1].set_title(
        "Rosas $\\Psi$, exact form: top scorer is wrong_selector\n"
        "blue = truly emergent, gray = imitation")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    savefig("fig37_exact_prior_formalisms.png")


def figure_pythia_decoder() -> None:
    rows = read_csv(OUTPUTS / "pythia_collapse_timeseries.csv")
    summary = read_json(OUTPUTS / "pythia_collapse_summary.json")
    tail = read_csv(OUTPUTS / "pythia_tail_timeseries.csv")
    agree = [r for r in rows if r["run"] == "pythia_agreement"]
    rand = [r for r in rows if r["run"] == "pythia_random_target"]
    perm = [r for r in rows if r["run"] == "shuffled_vocab"]
    steps = [int(r["epoch"]) for r in agree]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    axes[0].plot(steps, [float(r["test_acc"]) for r in agree],
                 color="#4C78A8", marker="o", ms=3, label="agreement (real ability)")
    axes[0].plot(steps, [float(r["test_acc"]) for r in rand],
                 color="#E45756", marker="s", ms=3, label="random-target control")
    axes[0].plot(steps, [float(r["test_acc"]) for r in perm],
                 color="#B279A2", marker="^", ms=3, label="shuffled-vocab control")
    axes[0].axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_xscale("symlog", linthresh=1000)
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel("Pretraining step (published checkpoints)")
    axes[0].set_ylabel("Minimal-pair accuracy")
    axes[0].set_title("Agreement emerges in Pythia-160m (decoder)")
    axes[0].legend(frameon=False, fontsize=7)

    collapse = [float(r["collapse_bits"]) for r in agree]
    bursts = [0.0] + [max(collapse[i] - collapse[i - 1], 0.0)
                      for i in range(1, len(collapse))]
    acc = [float(r["test_acc"]) for r in agree]
    ax2 = axes[1].twinx()
    axes[1].plot(steps, bursts, color="#4C78A8", label="Collapse burst")
    ax2.plot(steps, acc, color="#2E8B57", linestyle="--", label="Agreement accuracy")
    axes[1].set_xscale("symlog", linthresh=1000)
    axes[1].set_xlim(left=0)
    window = summary["runs"]["pythia_agreement"]["stats"]["window_epoch"]
    axes[1].axvline(window, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Pretraining step")
    axes[1].set_ylabel("Collapse burst (bits)", color="#4C78A8")
    ax2.set_ylabel("Agreement accuracy", color="#2E8B57")
    axes[1].set_title("Foreshadow burst, then the jump (window dotted)")

    palette = {"head_facts": "#F58518", "tail_facts": "#B22222",
               "tail_words": "#7B4173"}
    verdicts = {"head_facts": "emergent", "tail_facts": "rejected (burst)",
                "tail_words": "rejected (usefulness)"}
    for fam, color in palette.items():
        sel = [r for r in tail if r["run"] == fam]
        fam_steps = [int(r["epoch"]) for r in sel]
        axes[2].plot(fam_steps, [float(r["test_acc"]) for r in sel],
                     color=color, marker="o", ms=3,
                     label=f"{fam}: {verdicts[fam]}")
    axes[2].axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_xscale("symlog", linthresh=1000)
    axes[2].set_xlim(left=0)
    axes[2].set_xlabel("Pretraining step")
    axes[2].set_ylabel("Minimal-pair accuracy")
    axes[2].set_title("Head facts accepted; tail abilities rejected")
    axes[2].legend(frameon=False, fontsize=7)
    savefig("fig38_pythia_decoder.png")


def main() -> None:
    ensure_figures_dir()
    figure_possibility_tree()
    figure_option_value_heatmap()
    figure_horizon_reversal_heatmap()
    figure_ground_truth_validation()
    figure_performance_closure()
    figure_spatial_vs_contextual()
    figure_performance_robustness()
    figure_external_sacrifice_ptc()
    figure_collapse_burst()
    figure_synergy_pid_proxy()
    figure_external_decoy_ptc()
    figure_external_decoy_trajectory_ptc()
    figure_representation_jump_bridge()
    figure_learned_representation_audit()
    figure_within_episode_collapse()
    figure_unsupervised_basins()
    figure_within_episode_ci()
    figure_neural_probe()
    figure_criterion_battery()
    figure_estimator_robustness()
    figure_external_transfer()
    figure_phase_boundary()
    figure_grokking_bridge()
    figure_external_transfer_sweep()
    figure_refined_confirmation()
    figure_scale_decomposition()
    figure_grokking_generality()
    figure_robustness_and_priors()
    figure_induction_heads()
    figure_transformer_grokking()
    figure_multiberts()
    figure_phenomena_and_alignment()
    figure_chess_collapse()
    figure_gradualism_and_robustness()
    figure_deep_marl()
    figure_lbf_cross_task()
    figure_exact_prior_formalisms()
    figure_pythia_decoder()


if __name__ == "__main__":
    main()
