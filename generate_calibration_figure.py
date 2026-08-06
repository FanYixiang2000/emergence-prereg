"""fig46: six-knob ground-truth generator calibration.

Panel a: sensitivity matrix (response range of each measured dimension
         per generator knob), showing diagonal dominance.
Panel b: selected sweeps -- each knob moves its matched dimension
         across its full range while the others stay flat.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"

KNOBS = ("s", "b", "v", "q", "a", "r")
KNOB_LABELS = ("selectivity $s$", "reorganization $b$", "value $v$",
               "acquired fraction $q$", "steepness $a$",
               "retention $r$")
DIMS = ("S", "M", "V", "Q_rel", "A", "R")
DIM_LABELS = ("$S$", "$M$", "$V$", "$Q$", "$A$", "$R$")
MATCH = {"s": "S", "b": "M", "v": "V", "q": "Q_rel", "a": "A", "r": "R"}


def main() -> None:
    data = json.loads((OUTPUTS / "generator_calibration.json").read_text())
    J = data["sensitivity_matrix_range"]
    mat = np.array([[J[d][k] for k in KNOBS] for d in DIMS])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    ax = axes[0]
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    for i in range(len(DIMS)):
        for j, k in enumerate(KNOBS):
            diag = MATCH[k] == DIMS[i]
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if mat[i, j] < mat.max() * 0.6
                    else "black",
                    fontweight="bold" if diag else "normal")
            if diag:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="red",
                                           linewidth=1.6))
    ax.set_xticks(range(len(KNOBS)))
    ax.set_xticklabels(KNOB_LABELS, fontsize=7.5, rotation=25)
    ax.set_yticks(range(len(DIMS)))
    ax.set_yticklabels(DIM_LABELS, fontsize=10)
    ax.set_title("a  sensitivity matrix $J_{ij}$ = range of dimension $i$\n"
                 "under knob $j$ (red boxes: matched constructs)",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1]
    sweeps = data["sweeps"]
    grid = data["grid"]
    colors = ("#4C78A8", "#F58518", "#2E8B57", "#B22222", "#B279A2",
              "#7F7F7F")
    for j, k in enumerate(KNOBS):
        xs = np.linspace(0, 1, len(grid[k]))
        target = MATCH[k]
        vals = sweeps[k][target]
        ax.plot(xs, vals, marker="o", markersize=3.5, color=colors[j],
                label=f"{KNOB_LABELS[j]} $\\to$ "
                      f"{DIM_LABELS[DIMS.index(target)]}")
    ax.set_xlabel("knob position (normalized grid)")
    ax.set_ylabel("matched dimension value")
    ax.set_title("b  each knob sweeps its matched dimension\n"
                 "(GC-1 diagonal dominance: no violation; GC-3/4/5 pass;\n"
                 "GC-2 off-diagonal rule: retained miss, follow-up "
                 "disclosed)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    path = FIGURES / "fig46_generator_calibration.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
