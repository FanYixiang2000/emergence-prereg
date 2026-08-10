"""Overview schematic: the whole story in one drawing, no data panels."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLUE = "#4477AA"
CYAN = "#66CCEE"
GREEN = "#228833"
YELLOW = "#CCBB44"
RED = "#EE6677"
PURPLE = "#AA3377"
GREY = "#9DA7B1"
DARK = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

t_star = 4.6  # breakpoint position in axes units
rng = np.random.default_rng(7)
x = np.linspace(0.6, 9.4, 400)

def branch(y_end, seed):
    r = np.random.default_rng(seed)
    y = np.full_like(x, 5.0)
    pre = x < t_star
    frac = (x[pre] - x[0]) / (t_star - x[0])
    wig = 0.35 * np.sin(x[pre] * (1.5 + r.uniform(-0.4, 0.4)) + r.uniform(0, 6))
    y[pre] = 5.0 + frac ** 0.8 * (y_end - 5.0) + frac * wig
    return y

# open fan: many joint futures, all still possible
ends = np.linspace(1.6, 8.4, 13)
for i, ye in enumerate(ends):
    y = branch(ye, 30 + i)
    pre = x < t_star
    ax.plot(x[pre], y[pre], color=GREY, lw=0.9, alpha=0.55, zorder=1)
    # abandoned branches fade out just past the breakpoint
    tail = (x >= t_star) & (x < t_star + 0.7)
    if abs(ye - 5.0) > 0.9:
        yt = np.full(tail.sum(), y[pre][-1])
        ax.plot(x[tail], yt, color=GREY, lw=0.9, alpha=0.18, zorder=1)

# the committed regime: one thick surviving path
post = x >= t_star
y_comm = 5.0 + 0.18 * np.sin(x[post] * 1.1 + 0.4)
ax.plot(x[post], y_comm, color=BLUE, lw=2.6, zorder=3,
        solid_capstyle="round")
pre_comm = branch(5.0, 99)
ax.plot(x[x < t_star], pre_comm[x < t_star], color=BLUE, lw=1.2,
        alpha=0.8, zorder=2)

# breakpoint marker
ax.axvline(t_star, color=RED, lw=1.1, ls=(0, (4, 3)), ymin=0.30,
           ymax=0.92, zorder=2)
ax.text(t_star, 9.55, "breakpoint $t^{*}$", ha="center", fontsize=8.5,
        color=RED)

# capability arrives after the space has already closed
cap_x = 7.6
ax.plot([cap_x], [5.0 + 0.18 * np.sin(cap_x * 1.1 + 0.4)], marker="*",
        ms=13, color=YELLOW, mec=DARK, mew=0.5, zorder=4)
ax.text(cap_x, 6.15, "capability", ha="center", fontsize=8, color=DARK)
ax.annotate("", xy=(cap_x - 0.15, 4.15), xytext=(t_star + 0.15, 4.15),
            arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
ax.text((t_star + cap_x) / 2, 3.55, "collapse comes first",
        ha="center", fontsize=7.5, color=DARK, style="italic")

# stage labels
ax.text(2.4, 9.55, "open: many joint futures", ha="center", fontsize=8.5,
        color=DARK)
ax.text(7.6, 9.55, "committed: one regime", ha="center", fontsize=8.5,
        color=DARK)

# instrument bracket under the breakpoint
ax.annotate("", xy=(t_star - 1.1, 2.55), xytext=(t_star + 1.1, 2.55),
            arrowprops=dict(arrowstyle="-", color=DARK, lw=0.8))
ax.plot([t_star - 1.1, t_star - 1.1], [2.55, 2.8], color=DARK, lw=0.8)
ax.plot([t_star + 1.1, t_star + 1.1], [2.55, 2.8], color=DARK, lw=0.8)
ax.text(t_star, 1.95, "measured: amount $\\cdot$ abruptness $\\cdot$ timing "
        "$\\cdot$ source", ha="center", fontsize=7.5, color=DARK)

# three findings
chips = [
    ("collapse precedes capability", BLUE),
    ("barriers make it abrupt", PURPLE),
    ("openness = steerability", GREEN),
]
for i, (label, c) in enumerate(chips):
    cx = 1.75 + i * 3.25
    box = FancyBboxPatch((cx - 1.48, 0.12), 2.96, 1.05,
                         boxstyle="round,pad=0.06,rounding_size=0.18",
                         facecolor=c, edgecolor="none", alpha=0.14)
    ax.add_patch(box)
    ax.text(cx, 0.65, label, ha="center", va="center", fontsize=7.2,
            color=DARK, fontweight="bold")

fig.savefig(os.path.join(FIG, "schematic_overview.pdf"),
            bbox_inches="tight", pad_inches=0.03)
fig.savefig(os.path.join(FIG, "schematic_overview.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.03)
print("wrote schematic_overview")
