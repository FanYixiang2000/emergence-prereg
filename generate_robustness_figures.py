"""Generate supplementary figures for new robustness/seed-aware analyses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"


def short(label: str) -> str:
    replacements = {
        "multiberts": "MB",
        "agreement": "agr.",
        "pythia": "Pythia",
        "vocabulary": "vocab.",
        "transformer": "tr.",
        "induction": "ind.",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def main() -> None:
    process = json.loads(
        (OUTPUTS / "process_proxy_robustness.json").read_text())
    marl = json.loads(
        (OUTPUTS / "hierarchical_marl_analysis.json").read_text())

    fig = plt.figure(figsize=(15, 8.2))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.0, 1.45), height_ratios=(1, 1),
        hspace=0.38, wspace=0.32)

    ax = fig.add_subplot(grid[0, 0])
    for name, item in process["runs"].items():
        metrics = item["primary"]["metrics"]
        color = "#cf3f3f" if item["expected"] else "#61788a"
        marker = "o" if item["status"] == "confirmatory" else "^"
        ax.scatter(
            metrics["raw_burstiness_ratio"],
            metrics["bounded_burst_concentration"],
            color=color, marker=marker, s=42, alpha=0.85)
    ax.axvline(5, color="black", linestyle="--", linewidth=1)
    ax.axhline(5 / 6, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Legacy burst ratio (log scale)")
    ax.set_ylabel("Bounded transform $b/(b+m)$")
    ax.set_title("a  Exact bounded reparameterization", loc="left",
                 fontweight="bold")
    ax.text(
        0.03, 0.04, "27/27 primary verdicts unchanged",
        transform=ax.transAxes, fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})

    ax = fig.add_subplot(grid[:, 1])
    labels = list(process["runs"])
    columns = ["radius 0", "radius 1", "radius 2",
               "thin 2", "thin 3", "thin 4"]
    matrix = np.zeros((len(labels), len(columns)))
    for row, name in enumerate(labels):
        item = process["runs"][name]
        for radius in range(3):
            matrix[row, radius] = (
                item["radius_sensitivity"][str(radius)]["verdict"]["emergent"]
                == item["expected"]
            )
        for offset, factor in enumerate((2, 3, 4), start=3):
            cells = item["thinning_sensitivity"][str(factor)]
            matrix[row, offset] = np.mean([
                cell["verdict"]["emergent"] == item["expected"]
                for cell in cells
            ])
    image = ax.imshow(
        matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn",
        interpolation="nearest")
    ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), [short(label) for label in labels],
                  fontsize=7.5)
    ax.set_title(
        "b  Verdict stability under windows and checkpoint thinning",
        loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("Fraction matching full-grid expected verdict")

    ax = fig.add_subplot(grid[1, 0])
    y_positions = {"simple_spread": 1, "lbf": 0}
    colors = {"simple_spread": "#4c78a8", "lbf": "#f28e2b"}
    combined_path = OUTPUTS / "hierarchical_marl_analysis_combined.json"
    combined = (json.loads(combined_path.read_text())
                if combined_path.exists() else None)
    for domain, y in y_positions.items():
        entry = marl[domain]
        means = entry["seed_level"]["mean_effects"]
        ci = entry["cluster_bootstrap"][
            "mean_of_seed_means_cluster_ci95"]
        point = entry["cluster_bootstrap"]["mean_of_seed_means"]
        ax.scatter(
            means, [y + 0.12] * len(means), color=colors[domain],
            s=52, zorder=3, label=f"{domain} (registered)")
        ax.errorbar(
            point, y + 0.12, xerr=[[point - ci[0]], [ci[1] - point]],
            fmt="D", color="black", capsize=4, zorder=4)
        if combined is not None:
            registered = set(entry["seed_level"]["mean_effects"])
            all_means = combined[domain]["seed_level"]["mean_effects"]
            ext_means = [m for m in all_means if m not in registered]
            cci = combined[domain]["cluster_bootstrap"][
                "mean_of_seed_means_cluster_ci95"]
            cpoint = combined[domain]["cluster_bootstrap"][
                "mean_of_seed_means"]
            ax.scatter(
                ext_means, [y - 0.12] * len(ext_means),
                facecolors="none", edgecolors=colors[domain],
                s=52, zorder=3, label=f"{domain} (extension)")
            ax.errorbar(
                cpoint, y - 0.12,
                xerr=[[cpoint - cci[0]], [cci[1] - cpoint]],
                fmt="s", color="black", capsize=4, zorder=4)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks([1, 0], ["simple_spread", "LBF"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Mean do-contrast across evaluation episodes")
    ax.set_title("c  Policy-seed effects and cluster intervals", loc="left",
                 fontweight="bold")
    if combined is not None:
        ax.text(
            0.35, 0.76,
            "Registered 3 seeds: $p=0.125$\n"
            "Combined 6/8 seeds: $p=0.016$ / $p=0.004$",
            transform=ax.transAxes, fontsize=9)
    else:
        ax.text(
            0.43, 0.82, "Exact seed sign test:\n$p=0.125$ (both)",
            transform=ax.transAxes, fontsize=10)

    fig.suptitle(
        "Exploratory robustness audits separate measurement stability "
        "from training-population inference",
        fontsize=14, fontweight="bold", y=0.995)
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "ed_fig7_robustness_hierarchy.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
