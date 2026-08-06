"""Generate the Contextual LBF six-component confirmation figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"
COMPONENTS = (
    "potential", "conditional_selectivity", "specificity",
    "usefulness", "endogeneity", "acquisition",
)


def main() -> None:
    data = json.loads(
        (OUTPUTS / "contextual_lbf_confirmation.json").read_text())
    analysis = json.loads(
        (OUTPUTS / "contextual_lbf_confirmation_analysis.json").read_text())
    seed_names = list(data["seeds"])
    learned = [
        data["seeds"][seed]["systems"]["learned"] for seed in seed_names
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for seed, item in zip(seed_names, learned):
        rates = item["metrics"]["trigger_rates"]
        color = "#c23b3b" if not item["verdict"]["emergent"] else "#3973ac"
        ax.plot(
            [0, 1], [rates["0"], rates["1"]], marker="o",
            color=color, alpha=0.72, linewidth=1.5)
        if not item["verdict"]["emergent"]:
            ax.annotate(
                f"seed {seed}", (1, rates["1"]), xytext=(5, 4),
                textcoords="offset points", fontsize=8, color=color)
    ax.set_xticks([0, 1], ["food 0 nearer", "food 1 nearer"])
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("$P$(food 0 collected first)")
    ax.set_title("a  Acquired conditional selection", loc="left",
                 fontweight="bold")
    ax.text(
        0.04, 0.05, "10/10 context ordering",
        transform=ax.transAxes, fontsize=10)

    ax = axes[1]
    learned_matrix = np.asarray([
        [item["verdict"]["passes"][component] for component in COMPONENTS]
        for item in learned
    ], dtype=float)
    control_names = ("initial_twin", "team_nearest",
                     "fixed_food0", "fixed_food1")
    control_matrix = np.asarray([
        [
            np.mean([
                data["seeds"][seed]["systems"][name]["verdict"]["passes"][
                    component]
                for seed in seed_names
            ])
            for component in COMPONENTS
        ]
        for name in control_names
    ])
    matrix = np.vstack([learned_matrix, control_matrix])
    row_labels = [
        f"L-{seed}" for seed in seed_names
    ] + ["init (mean)", "nearest (mean)", "fixed-0 (mean)", "fixed-1 (mean)"]
    image = ax.imshow(
        matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn",
        interpolation="nearest")
    ax.set_xticks(
        range(len(COMPONENTS)),
        ["pot.", "select.", "spec.", "use.", "endo.", "acq."],
        rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7.5)
    ax.axhline(9.5, color="black", linewidth=1.2)
    ax.set_title("b  Six-component falsification matrix", loc="left",
                 fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("pass rate")

    ax = axes[2]
    acquisition = np.asarray([item["acquisition"] for item in learned])
    usefulness = np.asarray([
        item["metrics"]["usefulness_gap"] for item in learned])
    passed = np.asarray([item["verdict"]["emergent"] for item in learned])
    ax.scatter(
        acquisition[passed == 1], usefulness[passed == 1],
        color="#3973ac", s=58, label="full pass (9)")
    ax.scatter(
        acquisition[passed == 0], usefulness[passed == 0],
        color="#c23b3b", marker="X", s=82, label="selectivity miss (1)")
    for seed, x, y, ok in zip(seed_names, acquisition, usefulness, passed):
        if not ok:
            ax.annotate(
                seed, (x, y), xytext=(4, 4), textcoords="offset points",
                fontsize=8)
    ax.axvline(0.3, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    acq_ci = analysis["seed_bootstrap_intervals"]["acquisition"]
    use_ci = analysis["seed_bootstrap_intervals"]["usefulness_gap"]
    ax.errorbar(
        acq_ci["point"], use_ci["point"],
        xerr=[[acq_ci["point"] - acq_ci["ci95"][0]],
              [acq_ci["ci95"][1] - acq_ci["point"]]],
        yerr=[[use_ci["point"] - use_ci["ci95"][0]],
              [use_ci["ci95"][1] - use_ci["point"]]],
        marker="D", color="black", capsize=4, label="seed bootstrap mean")
    ax.set_xlabel("Acquisition over initialization twin")
    ax.set_ylabel("Natural $-$ do-non-trigger value")
    ax.set_title("c  Seed-level value and acquisition", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "Prospectively frozen Contextual LBF confirmation: "
        "9/10 learned policies pass, 40/40 controls reject",
        fontweight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "ed_fig8_contextual_lbf.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
