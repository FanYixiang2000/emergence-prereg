"""Figure 1 (concept): four regimes + a measured six-component walkthrough.

Row 1 is schematic: the observer's future-basin distribution P_t(B) as a
stacked area over time, in the four regimes the criterion must separate.

Row 2 is measured, not schematic: one full identification case from the
stored Contextual LBF confirmation (seed 1101), showing where each of the
six components is read off -- selectivity and acquisition from per-context
trigger rates against the initialization twin, specificity and usefulness
from the do-contrast, endogeneity from provenance -- and the resulting
component checklist for the learned system and three controls.

Basins: high-value (green), mediocre (gray), trap (red), diffuse/other
(light blue).
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES = Path(__file__).resolve().parent / "figures"
OUTPUTS = Path(__file__).resolve().parent / "outputs"

C_WIN = "#2E8B57"
C_MED = "#B8B8B8"
C_TRAP = "#C0504D"
C_OTHER = "#9DC3E6"
T = np.linspace(0, 1, 200)
TRIGGER_T = 0.45


def sigmoid(t, center, steep):
    return 1.0 / (1.0 + np.exp(-steep * (t - center)))


def normalize(rows):
    total = np.sum(rows, axis=0)
    return [r / total for r in rows]


def regime_single_mode():
    win = 0.82 + 0.06 * sigmoid(T, 0.3, 10)
    med = 0.5 * (1 - win)
    trap = 0.2 * (1 - win)
    other = 1 - win - med - trap
    return normalize([win, med, trap, other])


def regime_noise():
    rng = np.random.default_rng(7)
    base = np.array([0.25, 0.3, 0.2, 0.25])
    rows = []
    for i, b in enumerate(base):
        wiggle = 0.08 * np.sin(2 * np.pi * (3 + i) * T + rng.uniform(0, 6)) \
            + 0.05 * np.sin(2 * np.pi * (7 + i) * T + rng.uniform(0, 6))
        rows.append(np.clip(b + wiggle, 0.05, None))
    return normalize(rows)


def regime_trap():
    s = sigmoid(T, TRIGGER_T, 22)
    trap = 0.2 + 0.72 * s
    win = 0.28 * (1 - s) + 0.03
    med = 0.30 * (1 - s) + 0.03
    other = 0.22 * (1 - s) + 0.02
    return normalize([win, med, trap, other])


def regime_useful():
    s = sigmoid(T, TRIGGER_T, 22)
    win = 0.26 + 0.68 * s
    med = 0.30 * (1 - s) + 0.03
    trap = 0.22 * (1 - s) + 0.02
    other = 0.22 * (1 - s) + 0.03
    return normalize([win, med, trap, other])


def draw(ax, rows, title, verdict, show_trigger, annotate_counterfactual=False):
    stack = np.cumsum(np.vstack(rows), axis=0)
    colors = [C_WIN, C_MED, C_TRAP, C_OTHER]
    bottom = np.zeros_like(T)
    for row, top, color in zip(rows, stack, colors):
        ax.fill_between(T, bottom, top, color=color, alpha=0.85, linewidth=0)
        bottom = top
    if show_trigger:
        ax.axvline(TRIGGER_T, color="black", linewidth=1.2, linestyle="--")
        label = "trigger (locally costly)" if annotate_counterfactual else "event"
        ax.text(TRIGGER_T - 0.03, 0.04, label, rotation=90, fontsize=8,
                ha="right", va="bottom", color="black")
    if annotate_counterfactual:
        s = sigmoid(T, TRIGGER_T, 22)
        counter = 0.26 + 0.02 * s
        ax.plot(T, counter, color="black", linewidth=1.8, linestyle=":")
        ax.annotate("without trigger\n(do-block)", xy=(0.97, counter[-1]),
                    xytext=(0.96, 0.52), fontsize=8, ha="right",
                    arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=11)
    ax.text(0.5, -0.10, "time $\\rightarrow$", transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.text(0.5, -0.21, verdict, transform=ax.transAxes, ha="center",
            fontsize=8.8, style="italic")


def draw_walkthrough(fig, gs_row) -> None:
    """Measured six-component walkthrough (Contextual LBF seed 1101)."""
    data = json.loads(
        (OUTPUTS / "contextual_lbf_confirmation.json").read_text())
    systems = data["seeds"]["1101"]["systems"]
    th = data["thresholds"]

    ax1 = fig.add_subplot(gs_row[0])
    names = ("learned", "initial_twin")
    labels = ("learned policy", "initialization twin")
    colors = (C_WIN, "#888888")
    x = np.arange(2)
    width = 0.36
    for k, (name, color) in enumerate(zip(names, colors)):
        rates = systems[name]["metrics"]["trigger_rates"]
        ax1.bar(x + (k - 0.5) * width, [rates["0"], rates["1"]], width,
                color=color, alpha=0.88, label=labels[k])
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["context A\n(food 0 nearer)",
                         "context B\n(food 1 nearer)"], fontsize=8)
    ax1.set_ylabel("P(trigger)", fontsize=9)
    ax1.set_ylim(0, 1.12)
    sel = systems["learned"]["metrics"]["conditional_selectivity"]
    acq = systems["learned"]["acquisition"]
    ax1.set_title("SELECTIVITY + ACQUISITION\n"
                  f"separation {sel:.2f} (twin: 0.01); "
                  f"acquired gain {acq:.2f}", fontsize=9)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax2 = fig.add_subplot(gs_row[1])
    m = systems["learned"]["metrics"]
    vals = [m["potential_bits"], m["specificity_js_bits"],
            m["usefulness_gap"]]
    cuts = [th["potential_bits"], th["specificity_js_bits"],
            th["usefulness_gap"]]
    bars = ax2.bar(range(3), vals, color=[C_OTHER, C_WIN, "#F58518"],
                   alpha=0.88)
    for i, cut in enumerate(cuts):
        ax2.hlines(cut, i - 0.42, i + 0.42, color=C_TRAP, linestyle="--",
                   linewidth=1.3)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                 f"{val:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["POTENTIAL\nH(P$_t$(B)) [bits]",
                         "SPECIFICITY\nJS(do-trigger,\ndo-block) [bits]",
                         "USEFULNESS\nvalue do-contrast"], fontsize=7.5)
    ax2.set_title("Open future, causal load, positive value\n"
                  "(dashed: frozen thresholds)", fontsize=9)

    ax3 = fig.add_subplot(gs_row[2])
    harmful = json.loads(
        (OUTPUTS / "learned_harmful_emergence.json").read_text())
    he = harmful["seeds"]["9301"]
    comps = ("potential", "conditional_selectivity", "specificity",
             "usefulness", "endogeneity", "acquisition")
    comp_labels = ("Pot.", "Sel.", "Spec.", "Use.", "Endo.", "Acq.")
    rows = []
    for name, label in (("learned", "learned policy"),
                        ("initial_twin", "init. twin"),
                        ("team_nearest", "scripted nearest"),
                        ("fixed_food0", "fixed rule")):
        passes = systems[name]["verdict"]["passes"]
        rows.append((label, [passes[c] for c in comps],
                     "ACCEPT" if systems[name]["verdict"]["emergent"]
                     else "reject"))
    he_struct = [he["potential_bits"] >= 0.5,
                 he["selectivity_separation"] >= 0.5,
                 he["specificity_js_bits"] >= 0.2]
    rows.append(("learned exploiter\n(team value)",
                 he_struct + [he["usefulness_team"] > 0, True,
                              he["acquisition"] >= 0.3],
                 "structural\nonly"))
    rows.append(("same exploiter\n(beneficiary value)",
                 he_struct + [he["usefulness_private"] > 0, True,
                              he["acquisition"] >= 0.3],
                 "ACCEPT"))
    grid = np.array([[1.0 if ok else 0.0 for ok in flags]
                     for _, flags, _ in rows])
    ax3.imshow(grid, cmap="RdYlGn", vmin=-0.15, vmax=1.15, aspect="auto")
    for i, (_, flags, _) in enumerate(rows):
        for j, ok in enumerate(flags):
            ax3.text(j, i, "\u2713" if ok else "\u2717",
                     ha="center", va="center", fontsize=9,
                     color="white" if ok else "#7a1010")
    for i, (_, _, verdict_text) in enumerate(rows):
        accept = verdict_text == "ACCEPT"
        ax3.text(len(comps) - 0.28, i, verdict_text,
                 ha="left", va="center", fontsize=7,
                 fontweight="bold" if accept else "normal",
                 color=C_WIN if accept else "#8a6d00"
                 if "structural" in verdict_text else "#666666")
    ax3.set_xlim(-0.5, len(comps) + 1.6)
    ax3.set_xticks(range(len(comps)))
    ax3.set_xticklabels(comp_labels, fontsize=8)
    ax3.set_yticks(range(len(rows)))
    ax3.set_yticklabels([lbl for lbl, _, _ in rows], fontsize=7)
    ax3.set_title("Layered verdict: structural (Pot./Sel./Spec.)\n"
                  "+ adaptive (Use./Endo./Acq.); adaptivity is\n"
                  "relative to the declared value", fontsize=8.5)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16.0, 8.2))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.05],
                          hspace=0.62, wspace=2.4)
    top = [fig.add_subplot(gs[0, 3 * i:3 * i + 3]) for i in range(4)]

    draw(top[0], regime_single_mode(),
         "Forced convergence\n(reward shaping / pure team)",
         "fails POTENTIAL: the future was never open",
         show_trigger=False)
    draw(top[1], regime_noise(),
         "Structureless openness\n(noise, random policies)",
         "fails SELECTIVITY + USEFULNESS: open but never collapses",
         show_trigger=False)
    draw(top[2], regime_trap(),
         "Harmful collapse (decoy traps):\nstructural, not adaptive",
         "fails USEFULNESS: collapse without value",
         show_trigger=True)
    draw(top[3], regime_useful(),
         "Useful possibility collapse\n(= adaptive emergence)",
         "all components pass, incl. counterfactual necessity",
         show_trigger=True, annotate_counterfactual=True)

    bottom = [gs[1, 0:4], gs[1, 4:8], gs[1, 8:12]]
    draw_walkthrough(fig, bottom)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in (C_WIN, C_MED, C_TRAP, C_OTHER)]
    fig.legend(handles, ["high-value basin", "mediocre basin", "trap basin",
                         "other/diffuse"],
               loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.975,
             "a   Four regimes of the future-basin distribution "
             "$P_t(B\\,|\\,s_t)$ that performance or representation jumps "
             "conflate", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.475,
             "b   One measured identification case (Contextual LBF, fresh "
             "seed 1101): where each component is read off",
             ha="center", fontsize=12, fontweight="bold")
    out = FIGURES / "figure1_concept.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
