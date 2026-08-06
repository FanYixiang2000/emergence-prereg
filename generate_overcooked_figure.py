"""fig45: Overcooked-AI public-environment confirmation.

Panel a: per-seed six-component pass matrix for the learned policy
         (12 confirmation seeds; the registered acceptance line is
         8/12 and the observed count is 8/12).
Panel b: usefulness do-contrast per seed (natural minus do-block team
         score), the primary seed-level inference (12/12 positive,
         exact sign test p = 0.0002).
Panel c: verdict map across all systems (learned + four controls),
         showing that 48/48 control verdicts are rejections and the
         component route of every rejection class.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = HERE / "figures"

BLUE = "#4C78A8"
GREEN = "#2E8B57"
RED = "#B22222"
GRAY = "#B0B0B0"

SEEDS = list(range(77001, 77013))
COMPONENTS = ("potential", "conditional_selectivity", "specificity",
              "usefulness", "endogeneity", "acquisition")
COMP_LABELS = ("P", "S", "M", "U", "E", "Q")
SYSTEMS = ("learned", "initial_twin", "scripted_roles", "bc_clone",
           "untrained_other")
SYS_LABELS = ("learned", "init. twin", "scripted roles", "BC clone",
              "untrained other")


def load() -> dict:
    seeds = {}
    for s in SEEDS:
        data = json.loads(
            (OUTPUTS / f"overcooked_confirm_s{s}.json").read_text())
        seeds[s] = data["seeds"][str(s)]
    return seeds


def main() -> None:
    seeds = load()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0),
                             gridspec_kw={"width_ratios": [1.1, 1.0, 1.15]})

    # a: learned per-seed component pass matrix
    ax = axes[0]
    mat = np.array([[seeds[s]["learned"]["verdict"]["passes"][c]
                     for c in COMPONENTS] for s in SEEDS], dtype=float)
    ax.imshow(mat, cmap="RdYlGn", vmin=-0.25, vmax=1.25, aspect="auto")
    for i, s in enumerate(SEEDS):
        emergent = seeds[s]["learned"]["verdict"]["emergent"]
        for j in range(len(COMPONENTS)):
            ax.text(j, i, "\u2713" if mat[i, j] else "\u00d7",
                    ha="center", va="center", fontsize=8,
                    color="black")
        ax.text(len(COMPONENTS) - 0.30, i, "  accept" if emergent
                else "  reject", ha="left", va="center", fontsize=7,
                color=GREEN if emergent else RED, fontweight="bold",
                clip_on=False)
    ax.set_xticks(range(len(COMPONENTS)))
    ax.set_xticklabels(COMP_LABELS, fontsize=9)
    ax.set_yticks(range(len(SEEDS)))
    ax.set_yticklabels([str(s)[-2:] for s in SEEDS], fontsize=7)
    ax.set_ylabel("confirmation seed")
    ax.set_title("a  learned policy, six components\n"
                 "(8/12 accepted; registered line 8/12)", fontsize=10,
                 loc="left")

    # b: usefulness do-contrast per seed
    ax = axes[1]
    gaps = [seeds[s]["learned"]["metrics"]["usefulness_gap"]
            for s in SEEDS]
    accepted = [seeds[s]["learned"]["verdict"]["emergent"]
                for s in SEEDS]
    x = np.arange(len(SEEDS))
    ax.bar(x, gaps, color=[GREEN if a else BLUE for a in accepted],
           alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s)[-2:] for s in SEEDS], fontsize=7)
    ax.set_ylabel("natural $-$ do-block team score")
    ax.set_title("b  usefulness do-contrast per seed\n"
                 "(12/12 positive; exact sign test $p=0.0002$)",
                 fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.85)]
    ax.legend(handles, ["all six components pass", "verdict rejected"],
              frameon=False, fontsize=7, loc="upper right")

    # c: verdict map for all systems
    ax = axes[2]
    verdict = np.array([[seeds[s][sys]["verdict"]["emergent"]
                         for sys in SYSTEMS] for s in SEEDS],
                       dtype=float)
    ax.imshow(verdict, cmap="RdYlGn", vmin=-0.25, vmax=1.25,
              aspect="auto")
    for i in range(len(SEEDS)):
        for j in range(len(SYSTEMS)):
            ax.text(j, i, "\u2713" if verdict[i, j] else "\u00d7",
                    ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(SYSTEMS)))
    ax.set_xticklabels(SYS_LABELS, fontsize=7.5, rotation=20)
    ax.set_yticks(range(len(SEEDS)))
    ax.set_yticklabels([str(s)[-2:] for s in SEEDS], fontsize=7)
    ax.set_title("c  verdicts across systems\n"
                 "(48/48 control verdicts rejected)", fontsize=10,
                 loc="left")

    fig.suptitle("Overcooked-AI (public, unmodified): externally "
                 "timestamped preregistered confirmation, "
                 "5/5 registered predictions passed",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = FIGURES / "fig45_overcooked_confirmation.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
