"""Generate ED figure: Pythia scaling family (160m-2.8B) under the frozen
process proxy, including the registered 2.8B burstiness failure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"

SCALES = [
    ("160m", "pythia_collapse_timeseries.csv", "pythia_collapse_summary.json"),
    ("410m", "pythia_collapse_timeseries_410m.csv", "pythia_collapse_summary_410m.json"),
    ("1B", "pythia_collapse_timeseries_1b.csv", "pythia_collapse_summary_1b.json"),
    ("1.4B", "pythia_collapse_timeseries_1.4b.csv", "pythia_collapse_summary_1.4b.json"),
    ("2.8B", "pythia_collapse_timeseries_2.8b.csv", "pythia_collapse_summary_2.8b.json"),
]
COLORS = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#c23b3b"]


def series(csv_name: str, run: str):
    rows = [r for r in csv.DictReader((OUTPUTS / csv_name).open())
            if r["run"] == run]
    rows.sort(key=lambda r: int(float(r["epoch"])))
    steps = np.array([int(float(r["epoch"])) for r in rows])
    acc = np.array([float(r["test_acc"]) for r in rows])
    col = np.array([float(r["collapse_bits"]) for r in rows])
    return steps, acc, col


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for (label, csv_name, _), color in zip(SCALES, COLORS):
        steps, acc, _ = series(csv_name, "pythia_agreement")
        x = np.arange(len(steps))
        ax.plot(x, acc, marker="o", ms=3.5, color=color, label=label)
        ax.set_xticks(x[::4], [str(s) for s in steps[::4]], fontsize=8)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Published checkpoint (step)")
    ax.set_ylabel("Agreement pair accuracy")
    ax.set_title("a  One ability curve, five scales", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="center right")

    ax = axes[1]
    for (label, csv_name, _), color in zip(SCALES, COLORS):
        steps, _, col = series(csv_name, "pythia_agreement")
        x = np.arange(len(steps))
        bursts = np.maximum(np.diff(col), 0.0)
        ax.plot(x[1:], bursts, marker="o", ms=3.5, color=color, label=label)
    ax.set_xlabel("Checkpoint index")
    ax.set_ylabel("Collapse burst (bits)")
    ax.set_title("b  Collapse bursts: concentrated at small scale,\n"
                 "    spread out at 2.8B", loc="left", fontweight="bold")

    ax = axes[2]
    labels = []
    ratios = []
    gains = []
    colors_used = []
    for (label, _, summary_name), color in zip(SCALES, COLORS):
        summary = json.loads((OUTPUTS / summary_name).read_text())
        stats = summary["runs"]["pythia_agreement"]["stats"]
        labels.append(label)
        ratios.append(min(stats["burstiness_ratio"], 40))
        gains.append(stats["usefulness_acc_gain"])
        colors_used.append(color)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, ratios, width=0.36, color=colors_used,
           label="burstiness ratio (cap 40)")
    ax.axhline(5, color="black", linestyle="--", linewidth=1)
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, gains, width=0.36, color="#888888", alpha=0.6,
            label="usefulness gain")
    ax2.axhline(0.2, color="#555555", linestyle=":", linewidth=1)
    ax2.set_ylabel("Usefulness gain (right, gray)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Burstiness ratio (left, colored)")
    ax.set_title("c  Frozen thresholds across scales:\n"
                 "    registered 2.8B burstiness failure", loc="left",
                 fontweight="bold")
    ax.annotate("3.2 < 5\nregistered\nfailure", (4 - 0.18, 5.5),
                fontsize=8, ha="center", color="#c23b3b")

    fig.suptitle(
        "Held-out scaling of the frozen process proxy (agreement condition, "
        "160m$\\rightarrow$2.8B)", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "ed_fig9_pythia_scaling.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
