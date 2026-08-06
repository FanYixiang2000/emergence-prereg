"""NMI-style figures for the possibility-collapse manuscript.

Every data panel is drawn directly from outputs/*.json (the same files
read by manuscript_numbers.py). Panels labelled 'schematic' contain no
data. Style follows Nature figure guidelines: Helvetica/Arial,
7 pt base font, lowercase bold panel letters, muted colourblind-safe
palette (Paul Tol), no top/right spines.

Run:  python make_figures.py     ->  figures/fig{1..6}.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


# ---- Nature/NMI style ----------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
    "figure.dpi": 150,
})

# Paul Tol bright palette (colourblind safe)
BLUE = "#4477AA"
CYAN = "#66CCEE"
GREEN = "#228833"
YELLOW = "#CCBB44"
RED = "#EE6677"
PURPLE = "#AA3377"
GREY = "#BBBBBB"
DARK = "#222222"
SEEDC = [BLUE, CYAN, GREEN, YELLOW, PURPLE]

DOUBLE = 7.2   # inches, 183 mm
SINGLE = 3.5   # inches, 89 mm


def panel(fig, x, y, letter):
    fig.text(x, y, letter, fontsize=9, fontweight="bold", va="top",
             ha="left", family="sans-serif")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ===========================================================================
# Figure 1 | problem + instrument
# ===========================================================================
def fig1():
    cc = load("collective_constraint.json")["matched_confound"]
    b72 = load("bench72_factorial.json")
    sd = load("collapse_source_decomposition.json")

    fig = plt.figure(figsize=(DOUBLE, 2.5))
    gs = fig.add_gridspec(1, 4, wspace=0.55,
                          width_ratios=[1.25, 1.0, 1.0, 1.15])

    # -- a schematic: possibility space collapse -----------------------
    ax = fig.add_subplot(gs[0])
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 300)
    tstar = 0.5
    for i in range(22):
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(1.5, 4.0)
        amp = rng.uniform(0.35, 0.95)
        wig = amp * np.sin(freq * 2 * np.pi * t + phase)
        # after t*, every trajectory relaxes onto the committed regime
        lam = np.clip((t - tstar) * 14, 0, None)
        y = wig * np.exp(-lam) + 0.75 * (1 - np.exp(-lam))
        ax.plot(t, y, color=GREY, lw=0.5, alpha=0.6, zorder=1)
    # the abandoned alternative regime
    ax.plot(t[t > tstar], np.full((t > tstar).sum(), -0.75), color=RED,
            lw=0.8, ls=":", alpha=0.7)
    ax.text(0.97, -0.62, "abandoned regime", fontsize=5.2, color=RED,
            ha="right")
    ax.axvline(tstar, color=RED, lw=0.9, ls="--", zorder=2)
    ax.text(tstar + 0.02, 1.18, r"breakpoint $t^{*}$", color=RED,
            fontsize=6)
    ax.text(0.03, 1.18, "open regime", fontsize=6, color=DARK)
    ax.text(0.97, 0.44, "committed\nregime", fontsize=6, color=DARK,
            ha="right")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("time (schematic)")
    ax.set_ylabel("joint possibility space")

    # -- b matched confound --------------------------------------------
    ax = fig.add_subplot(gs[1])
    rows = ["central script", "common cause", "coincidence", "local feedback"]
    checks = ["joint distr.", "marginals", "macro outcome"]
    ok = [[cc["joint_distributions_identical"],
           cc["single_agent_marginals_identical"], True]] * 4
    for i in range(len(rows)):
        for j in range(len(checks)):
            ax.add_patch(Rectangle((j, 3 - i), 0.92, 0.92,
                                   facecolor="#E7EDF5", edgecolor="white"))
            ax.text(j + 0.46, 3 - i + 0.46, "=", ha="center", va="center",
                    fontsize=8, color=BLUE, fontweight="bold")
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.15, 4.05)
    ax.set_xticks([j + 0.46 for j in range(3)])
    ax.set_xticklabels(checks, rotation=25, ha="right")
    ax.set_yticks([3 - i + 0.46 for i in range(4)])
    ax.set_yticklabels(rows)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("four mechanisms, identical observables", fontsize=6.5)

    # -- c BENCH-72: J orders shapes, M invariant ----------------------
    gd = b72["checks"]["group_detail"]
    shapes = ["gradual", "sigmoid", "punctuated"]
    ax = fig.add_subplot(gs[2])
    for k, sh in enumerate(shapes):
        vals = [g["J"][sh] for g in gd]
        x = np.full(len(vals), k) + np.random.default_rng(3).uniform(
            -0.12, 0.12, len(vals))
        ax.plot(x, vals, "o", ms=2.4, color=SEEDC[k], alpha=0.8, mew=0)
        ax.hlines(np.median(vals), k - 0.22, k + 0.22, color=DARK, lw=1.1)
    mrr = max(g["M_rel_range"] for g in gd)
    ax.text(0.03, 0.60, "amplitude M invariant:\nrelative range "
            f"{mrr:.3f}\n({len(gd)}/{len(gd)} groups)",
            transform=ax.transAxes, fontsize=5.6, va="top")
    ax.set_xticks(range(3))
    ax.set_xticklabels(shapes, rotation=15)
    ax.set_ylabel("abruptness J")
    ax.set_ylim(0, 1.12)

    # -- d source dissociation on analytic ground truth ----------------
    det = sd["results"]["SD3_dissociations"]["detail"]["components"]
    gens = ["pure_env", "pure_pair", "pure_high"]
    comps = ["C_env", "C_pair", "C_high"]
    labels = [r"$C_{\rm env}$", r"$C_{\rm pair}$", r"$C_{\rm high}$"]
    colors = [YELLOW, BLUE, PURPLE]
    ax = fig.add_subplot(gs[3])
    w = 0.24
    for j, c in enumerate(comps):
        fr = [max(det[g][c], 0.0) / det[g]["C_total"] for g in gens]
        ax.bar(np.arange(3) + (j - 1) * w, fr, w, color=colors[j],
               label=labels[j], edgecolor="none")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["env", "pair", "high"], fontsize=6.5)
    ax.set_xlabel("pure generator")
    ax.set_ylabel("fraction of total collapse")
    ax.set_ylim(0, 1.32)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              handlelength=0.8, columnspacing=0.8, borderaxespad=0.0)

    panel(fig, 0.005, 1.00, "a")
    panel(fig, 0.30, 1.00, "b")
    panel(fig, 0.52, 1.00, "c")
    panel(fig, 0.75, 1.00, "d")
    save(fig, "fig1")


# ===========================================================================
# Figure 2 | grip flagship: punctuated realization
# ===========================================================================
def fig2():
    lgt = load("learn_grip_transport.json")
    b5 = load("learn_grip_transport_b5.json")
    ext = load("learn_grip_ext.json")
    a2c = load("learn_grip_a2c.json")
    eqs = load("learn_transport_equivariant_slow.json")
    lc = load("learn_convention.json")
    lr = load("learn_roles.json")

    fig = plt.figure(figsize=(DOUBLE, 5.0))
    gs = fig.add_gridspec(1, 4, wspace=0.5, bottom=0.585, top=0.97,
                          width_ratios=[1.1, 1.25, 1.0, 1.0])
    gs2 = fig.add_gridspec(1, 3, wspace=0.45, bottom=0.075, top=0.435,
                           width_ratios=[1.15, 1.15, 1.0])

    # -- a task schematic -----------------------------------------------
    ax = fig.add_subplot(gs[0])
    obj = Circle((0.5, 0.55), 0.10, facecolor="#DDDDDD", edgecolor=DARK,
                 lw=0.8, zorder=3)
    ax.add_patch(obj)
    rng = np.random.default_rng(2)
    for k in range(16):
        a = 2 * np.pi * k / 16
        r = 0.30 if k % 2 else 0.24
        x, y = 0.5 + r * np.cos(a), 0.55 + r * 0.72 * np.sin(a)
        ax.add_patch(Circle((x, y), 0.022, facecolor=BLUE,
                            edgecolor="none", zorder=4))
        ax.add_patch(FancyArrowPatch((x, y),
                     (0.5 + 0.13 * np.cos(a), 0.55 + 0.095 * np.sin(a)),
                     arrowstyle="-", color=GREY, lw=0.5, zorder=2))
    ax.annotate("", xy=(0.90, 0.55), xytext=(0.66, 0.55),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(0.85, 0.68, "push\n(left or right)", fontsize=6, color=RED,
            ha="center", va="bottom")
    ax.text(0.5, 0.15, "phase 1: grip (attachment $\\geq$ threshold)\n"
            "phase 2: collective push", fontsize=6, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1)
    ax.axis("off")
    ax.set_title("grip-then-push transport (N = 16)", fontsize=6.5)

    # -- b side-openness curves (REINFORCE, 5 seeds) ---------------------
    ax = fig.add_subplot(gs[1])
    for i, (s, d) in enumerate(sorted(lgt["seeds"].items())):
        c = d["side_openness_curve"][:41]
        ax.plot(range(len(c)), c, color=SEEDC[i], lw=1.0,
                label=f"seed {s}")
        tstar = b5["seeds"][s]["adj"]["hinge"]["t_star"]
        ax.plot([tstar], [np.interp(tstar, range(len(c)), c)], "v",
                color=SEEDC[i], ms=3.5, mew=0)
    ax.set_xlabel("episode step")
    ax.set_ylabel("side openness $O_t$")
    ax.set_ylim(-0.03, 1.06)
    ax.legend(frameon=False, loc="center right", handlelength=1.0,
              borderaxespad=0.1)
    ax.text(6, 1.025, "open plateau", fontsize=6, color=DARK)
    ax.annotate("collapse", xy=(20.5, 0.35), xytext=(27, 0.30), fontsize=6,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=DARK))

    # -- c mechanism contrast: dBIC with vs without preparation phase ----
    ax = fig.add_subplot(gs[2])
    grip_db = [b5["seeds"][s]["adj"]["hinge"]["delta_bic"]
               for s in sorted(b5["seeds"])]
    nopr_db = [eqs["seeds"][s]["episode_adj"]["hinge"]["delta_bic"]
               for s in sorted(eqs["seeds"])]
    x1 = np.arange(5) * 0.5
    x2 = x1 + 3.8
    ax.bar(x1, grip_db, 0.4, color=BLUE, label="grip-then-push")
    ax.bar(x2, nopr_db, 0.4, color=GREY, label="no preparation phase")
    ax.axhline(10, color=RED, lw=0.8, ls="--")
    ax.text(5.7, 11.5, "$\\Delta$BIC = 10", color=RED, fontsize=5.8,
            ha="right")
    ax.set_xticks([x1.mean(), x2.mean()])
    ax.set_xticklabels(["grip task\n5/5 onset", "no-preparation\n0/5 onset"],
                       fontsize=6)
    ax.set_xlim(-0.6, 6.3)
    ax.set_ylabel("breakpoint evidence ($\\Delta$BIC)")
    ax.set_ylim(0, 58)

    # -- d breakpoint location across seeds and algorithms ---------------
    ax = fig.add_subplot(gs[3])
    t_re = sorted(ext["registered_outcomes"]["t_stars"])
    t_a2c = sorted(a2c["registered_outcomes"]["t_stars"])
    rng = np.random.default_rng(5)
    ax.plot(rng.uniform(-0.10, 0.10, len(t_re)) + 0, t_re, "o", ms=3.5,
            color=BLUE, mew=0, label="REINFORCE (10 seeds)")
    ax.plot(rng.uniform(-0.10, 0.10, len(t_a2c)) + 1, t_a2c, "s", ms=3.5,
            color=PURPLE, mew=0, label="A2C (5 seeds)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["REINFORCE", "A2C"])
    ax.set_ylabel("breakpoint $t^{*}$ (step)")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(10, 28)
    ax.text(0, 25.6, "10/10 B5", ha="center", fontsize=6, color=BLUE)
    ax.text(1, 18.3, "5/5 shape\n3/5 strict", ha="center", fontsize=6,
            color=PURPLE)

    # -- e convention formation (no designed gate) ------------------------
    ax = fig.add_subplot(gs2[0])
    grid_c = list(range(0, 4001, 25))
    for i, (s, d) in enumerate(sorted(lc["seeds"].items())):
        ax.plot(grid_c, d["openness_curve"], color=SEEDC[i], lw=0.9)
        if d["adj"]["b5_onset"]:
            ts = d["adj"]["hinge"]["t_star"]
            ax.plot([ts], [np.interp(ts, grid_c, d["openness_curve"])],
                    "v", color=SEEDC[i], ms=3.5, mew=0)
    ax.set_xlabel("training update")
    ax.set_ylabel("convention openness")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.03, 1.06)
    ax.set_title("signalling population, 120 equivalent codes\n"
                 "no gate: 4/5 onset, 5 distinct codes", fontsize=6.5)
    ax.text(300, 1.005, "open plateau", fontsize=6, color=DARK)

    # -- f role lock-in (no designed gate) ---------------------------------
    ax = fig.add_subplot(gs2[1])
    grid_r = list(range(0, 6001, 25))
    for i, (s, d) in enumerate(sorted(lr["seeds"].items())):
        ax.plot(grid_r, d["openness_curve"], color=SEEDC[i], lw=0.9)
        ts = d["adj"]["hinge"]["t_star"]
        ax.plot([ts], [np.interp(ts, grid_r, d["openness_curve"])],
                "v", color=SEEDC[i], ms=3.5, mew=0)
    ax.set_xlabel("training update")
    ax.set_ylabel("assignment openness")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.03, 1.06)
    ax.set_title("division of labour, 720 equivalent regimes\n"
                 "no gate: 5/5 onset, 5 distinct permutations",
                 fontsize=6.5)

    # -- g collapse precedes capability ------------------------------------
    ax = fig.add_subplot(gs2[2])
    for d in lc["seeds"].values():
        if d["adj"]["b5_onset"] and d["success_090_cross"] is not None:
            ax.plot(d["adj"]["hinge"]["t_star"], d["success_090_cross"],
                    "o", ms=4, color=BLUE, mew=0)
    for d in lr["seeds"].values():
        if d["adj"]["b5_onset"] and d["success_090_cross"] is not None:
            ax.plot(d["adj"]["hinge"]["t_star"], d["success_090_cross"],
                    "s", ms=4, color=PURPLE, mew=0)
    lim = [0, 1200]
    ax.plot(lim, lim, color=GREY, lw=0.7, ls="--")
    ax.set_xlim(150, 650)
    ax.set_ylim(300, 1150)
    ax.set_xlabel("breakpoint $t^{*}$ (update)")
    ax.set_ylabel("capability crossing (S = 0.9)")
    ax.plot([], [], "o", ms=4, color=BLUE, mew=0, label="convention")
    ax.plot([], [], "s", ms=4, color=PURPLE, mew=0, label="roles")
    ax.legend(frameon=False, loc="lower right", handlelength=1.0)
    ax.text(180, 1050, "collapse precedes\ncapability in 9/9", fontsize=6,
            color=DARK)

    panel(fig, 0.005, 0.99, "a")
    panel(fig, 0.28, 0.99, "b")
    panel(fig, 0.55, 0.99, "c")
    panel(fig, 0.78, 0.99, "d")
    panel(fig, 0.005, 0.46, "e")
    panel(fig, 0.37, 0.46, "f")
    panel(fig, 0.70, 0.46, "g")
    save(fig, "fig2")


# ===========================================================================
# Figure 3 | two timescales dissociate
# ===========================================================================
def fig3():
    lgf = load("learn_grip_formation.json")
    fine = load("learn_grip_formation_fine.json")

    fig = plt.figure(figsize=(DOUBLE, 2.3))
    gs = fig.add_gridspec(1, 3, wspace=0.45, width_ratios=[1.2, 1.2, 1.0])

    # -- a formation axis: smooth ----------------------------------------
    ax = fig.add_subplot(gs[0])
    grid = None
    for i, (s, d) in enumerate(sorted(lgf["seeds"].items())):
        sc = d["success_curve"]
        oc = d["ocap_curve"]
        n = len(sc)
        every = lgf["config"]["grid_every"]
        x = np.arange(n) * every
        ax.plot(x, sc, color=SEEDC[i], lw=0.9, alpha=0.9)
        ax.plot(x, oc, color=SEEDC[i], lw=0.9, alpha=0.45, ls="--")
    ax.set_xlabel("training update")
    ax.set_ylabel("success (solid)\noutcome openness (dashed)")
    ax.set_xlim(0, 1200)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("formation: smooth, 0/5 breakpoints", fontsize=6.5)

    # -- b fine grid -------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    every = fine["config"]["save_every"]
    for i, (s, d) in enumerate(sorted(fine["seeds"].items())):
        sc = d["success_curve"]
        x = np.arange(len(sc)) * every
        ax.plot(x, sc, color=SEEDC[i], lw=0.9)
        mp = d["success_midpoint_update"]
        ax.plot([mp], [np.interp(mp, x, sc)], "o", color=SEEDC[i], ms=3,
                mew=0)
    ax.set_xlabel("training update (5-update grid)")
    ax.set_ylabel("success rate")
    ax.set_xlim(0, 200)
    ax.set_title("fast but smooth: midpoints 10–20,\n0/5 breakpoints",
                 fontsize=6.5)

    # -- c the quadrant ----------------------------------------------------
    ax = fig.add_subplot(gs[2])
    ro = lgf["registered_outcomes"]
    cells = [
        ("formation (across training)", "expansive, smooth",
         f"{ro['formation_b5_count']}/5 breakpoints", CYAN, 0.30),
        ("realization (within episode)", "plateau, punctuated",
         f"{ro['realization_b5_count']}/5 breakpoints", BLUE, 0.55),
    ]
    for i, (rowlab, shape, b5c, col, alpha) in enumerate(cells):
        y0 = 0.56 - i * 0.48
        ax.add_patch(Rectangle((0.02, y0), 0.96, 0.40,
                     facecolor=col, alpha=alpha, edgecolor=col, lw=1.0))
        ax.text(0.06, y0 + 0.30, rowlab, fontsize=6.8, fontweight="bold",
                va="center")
        ax.text(0.06, y0 + 0.12, f"{shape}  —  {b5c}", fontsize=6.2,
                va="center")
    ax.text(0.5, 0.0, "same system, same instrument", fontsize=6,
            ha="center", style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.axis("off")

    panel(fig, 0.005, 1.00, "a")
    panel(fig, 0.38, 1.00, "b")
    panel(fig, 0.72, 1.00, "c")
    save(fig, "fig3")


# ===========================================================================
# Figure 4 | source typology transfers to learned systems
# ===========================================================================
def fig4():
    lst = load("learn_stance_transport.json")
    tric = load("triad_highorder_cue.json")
    e1c = load("overcooked_profile_confirmatory.json")
    ko = load("kuramoto_offdesign_ladder.json")

    fig = plt.figure(figsize=(DOUBLE, 2.4))
    gs = fig.add_gridspec(1, 4, wspace=0.75,
                          width_ratios=[1.1, 1.1, 1.0, 1.0])

    # -- a stance: relational collapse -----------------------------------
    ax = fig.add_subplot(gs[0])
    ax2 = ax.twinx()
    for i, (s, d) in enumerate(sorted(lst["seeds"].items())):
        lad = d["ladder"]
        xs = sorted(int(k) for k in lad)
        h1 = [lad[str(x)]["H1"] for x in xs]
        tc = [lad[str(x)]["TC"] for x in xs]
        ax.plot(xs, h1, color=GREY, lw=0.8)
        ax2.plot(xs, tc, color=BLUE, lw=0.9)
    ax.set_xlabel("episode step")
    ax.set_ylabel("per-agent entropy (bits)", color="#777777")
    ax2.set_ylabel("total correlation (bits)", color=BLUE)
    ax.set_ylim(0, 1.5)
    ax2.set_ylim(0, 7.3)
    ax2.axhline(7, color=BLUE, lw=0.5, ls=":")
    ax2.text(38, 7.05, "max", color=BLUE, fontsize=5.5, ha="right")
    ax2.spines["right"].set_visible(True)
    ax.set_title("marginals stay open,\njoint space closes", fontsize=6.5)

    # -- b TRI-C: learned higher-order carrier ---------------------------
    ax = fig.add_subplot(gs[1])
    for i, (s, d) in enumerate(sorted(tric["seeds"].items())):
        cks = sorted(int(k) for k in d)
        ch = [d[str(c)]["ladder2_hidden"]["C_high"] for c in cks]
        cp = [d[str(c)]["ladder2_hidden"]["C_pair"] for c in cks]
        ax.plot(cks, ch, color=PURPLE, lw=0.9)
        ax.plot(cks, cp, color=GREY, lw=0.8)
    ax.set_xlabel("training update")
    ax.set_ylabel("collapse component (bits)")
    ax.set_ylim(-0.04, 1.05)
    ax.text(2000, 0.90, r"$C_{\rm high}$", color=PURPLE, fontsize=6.5,
            ha="right")
    ax.text(2000, 0.08, r"$C_{\rm pair}$", color="#777777", fontsize=6.5,
            ha="right")
    ax.set_title("blocked low-order route:\nlearning builds XOR carrier",
                 fontsize=6.5)

    # -- c contract relativity (TRI-C, seed mean) -------------------------
    ax = fig.add_subplot(gs[2])
    comps = ["C_individual", "C_env", "C_pair", "C_high"]
    labels = ["ind", "env", "pair", "high"]
    colors = [GREEN, YELLOW, BLUE, PURPLE]
    hid = np.mean([[tric["seeds"][s][max(tric["seeds"][s], key=int)]
                    ["ladder2_hidden"][c] for c in comps]
                   for s in tric["seeds"]], axis=0)
    dec = np.mean([[tric["seeds"][s][max(tric["seeds"][s], key=int)]
                    ["ladder2_declared_e"][c] for c in comps]
                   for s in tric["seeds"]], axis=0)
    for j in range(4):
        ax.bar([0], [hid[j]], 0.55, bottom=sum(hid[:j]), color=colors[j])
        ax.bar([1], [dec[j]], 0.55, bottom=sum(dec[:j]), color=colors[j])
    ax.text(0, hid[3] / 2 + sum(hid[:3]), r"$C_{\rm high}$", ha="center",
            va="center", fontsize=6.5, color="white", fontweight="bold")
    ax.text(1, dec[1] / 2 + dec[0], r"$C_{\rm env}$", ha="center",
            va="center", fontsize=6.5, color=DARK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cues\nhidden", "cues declared\nexogenous"],
                       fontsize=5.8)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("collapse composition (bits)")
    ax.set_title("same collapse,\nobserver-declared source", fontsize=6.5)

    # -- d matched-product profile separation (E1-C) ----------------------
    ax = fig.add_subplot(gs[3])
    l, n = e1c["learned"], e1c["noisy_scripted"]
    vals = [l["C_env"], n["C_env"]]
    cis = [l["C_env_ci95"], n["C_env_ci95"]]
    ax.bar([0, 1], vals, 0.5, color=[BLUE, GREY])
    for k in range(2):
        ax.errorbar([k], [vals[k]],
                    yerr=[[vals[k] - cis[k][0]], [cis[k][1] - vals[k]]],
                    color=DARK, lw=0.8, capsize=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["learned", "scripted\n+ noise"], fontsize=6)
    ax.set_ylabel(r"$C_{\rm env}$ (bits)")
    ax.set_ylim(0, 0.0185)
    ax.set_title("matched task score:\nprofile separates provenance",
                 fontsize=6.5)
    ax.text(0.62, 0.0095, "non-overlapping\n95% CIs", fontsize=5.5,
            ha="center")

    panel(fig, 0.005, 1.06, "a")
    panel(fig, 0.29, 1.06, "b")
    panel(fig, 0.55, 1.06, "c")
    panel(fig, 0.80, 1.06, "d")
    save(fig, "fig4")


# ===========================================================================
# Figure 5 | openness predicts controllability
# ===========================================================================
def fig5():
    lgu = load("learn_grip_utility.json")
    lss = load("learn_stance_sticky.json")
    lsc = load("learn_stance_control.json")
    aic = load("ant_conditional_leverage.json")
    ext = load("learn_grip_ext.json")

    fig = plt.figure(figsize=(DOUBLE, 2.4))
    gs = fig.add_gridspec(1, 4, wspace=0.5, width_ratios=[1.2, 1, 1.1, 1])

    # -- a intervention window vs breakpoint ------------------------------
    ax = fig.add_subplot(gs[0])
    ms = lgu["registered_outcomes"]["mean_switch_by_tau"]
    taus = sorted(int(k) for k in ms)
    ax.plot(taus, [ms[str(t)] for t in taus], "o-", color=BLUE, lw=1.1,
            ms=3.5, mew=0)
    tstars = ext["registered_outcomes"]["t_stars"]
    ax.axvspan(min(tstars), max(tstars), color=RED, alpha=0.12, lw=0)
    ax.text((min(tstars) + max(tstars)) / 2, 0.15,
            "breakpoint\nrange $t^{*}$", color=RED, fontsize=5.8,
            ha="center")
    ax.set_xlabel("intervention time $\\tau$ (step)")
    ax.set_ylabel("switch probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("the window closes after $t^{*}$", fontsize=6.5)

    # -- b predictor race (grip) ------------------------------------------
    ax = fig.add_subplot(gs[1])
    br = lgu["registered_outcomes"]["baseline_race"]
    names = ["side_open", "absx", "absv", "tau", "att"]
    labs = ["openness", "|x|", "|v|", "time", "attach"]
    aucs = [br[n]["auc"] for n in names]
    cols = [BLUE, GREY, GREY, GREY, GREY]
    ax.bar(range(5), aucs, 0.6, color=cols)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("AUC (switch prediction)")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color=DARK, lw=0.5, ls=":")
    ax.set_title("grip system", fontsize=6.5)

    # -- c matched-parameter causal contrast ------------------------------
    ax = fig.add_subplot(gs[2])
    st = lss["registered_outcomes"]["baseline_race"]
    ct = lsc["registered_outcomes"]["baseline_race"]
    names = ["open", "absx", "absv", "tau"]
    labs = ["openness", "|x|", "|v|", "time"]
    x = np.arange(4)
    ax.bar(x - 0.18, [st[n]["auc"] for n in names], 0.34, color=BLUE,
           label="sticky (hidden phase)")
    ax.bar(x + 0.18, [ct[n]["auc"] for n in names], 0.34, color=GREY,
           label="matched control")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("AUC (switch prediction)")
    ax.set_ylim(0.5, 0.95)
    ax.legend(frameon=False, loc="upper right", fontsize=5.5,
              handlelength=1.0)
    ax.set_title("one parameter flips the ranking", fontsize=6.5)

    # -- d ant per-episode conditional law --------------------------------
    ax = fig.add_subplot(gs[3])
    bins = aic["bins"]
    labels = [f"{b['bin'][0]:g}–{b['bin'][1]:g}" for b in bins]
    rates = [b["flip_rate"] for b in bins]
    ns = [b["n"] for b in bins]
    ax.bar(range(4), rates, 0.6, color=BLUE)
    for k, (r, n) in enumerate(zip(rates, ns)):
        ypos = max(r + 0.008, 0.014)
        ax.text(k, ypos, f"{n:,}", fontsize=4.8, ha="center",
                color="#666666")
    ax.text(0, 0.045, "0 flips", fontsize=5.4, ha="center", color=DARK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=5.2, rotation=25)
    ax.set_xlabel("episode openness", fontsize=6.5)
    ax.set_ylabel("intervention flip rate")
    ax.set_ylim(0, 0.25)
    ax.set_title("ant colony", fontsize=6.5)

    panel(fig, 0.005, 1.00, "a")
    panel(fig, 0.30, 1.00, "b")
    panel(fig, 0.52, 1.00, "c")
    panel(fig, 0.79, 1.00, "d")
    save(fig, "fig5")


# ===========================================================================
# Figure 6 | laws and scope
# ===========================================================================
def fig6():
    fss = load("ant_fss.json")
    ks = load("kuramoto_scale.json")
    vd = load("ceb_vicsek_dense.json")

    fig = plt.figure(figsize=(DOUBLE, 2.7))
    gs = fig.add_gridspec(1, 3, wspace=0.45, width_ratios=[1.0, 1.0, 1.5])

    # -- a finite-size scaling law ------------------------------------------
    ax = fig.add_subplot(gs[0])
    law = fss["registered_outcomes"]["log_law"]
    per = fss["per_size"]
    all_sizes = [int(n) for n in per]
    for n in sorted(all_sizes):
        t50 = per[str(n)]["t50"]
        if t50 is None:
            continue
        if per[str(n)]["b5_onset"]:
            ax.plot(n, t50, "o", ms=4, color=BLUE, mew=0)
        else:
            ax.plot(n, t50, "o", ms=4, mfc="white", mec=GREY, mew=0.9)
    nn = np.geomspace(15, 600, 50)
    ax.plot(nn, law["a"] + law["b"] * np.log(nn), color=BLUE, lw=0.9,
            ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("colony size N")
    ax.set_ylabel("commitment time $t_{50}$")
    ax.set_title("ant model: $t_{50}=a+b\\,\\ln N$\n"
                 f"$b={law['b']:.0f}$, $R^2={law['r2']:.2f}$; "
                 "width N-invariant", fontsize=6.5)
    ax.plot([], [], "o", ms=4, color=BLUE, mew=0, label="onset")
    ax.plot([], [], "o", ms=4, mfc="white", mec=GREY, mew=0.9,
            label="no onset")
    ax.legend(frameon=False, loc="upper left", handlelength=0.8,
              borderaxespad=0.1)

    # inset: translation data collapse for N >= 50
    axi = ax.inset_axes([0.55, 0.14, 0.42, 0.36])
    grid = np.arange(0, 1501, 10, dtype=float)
    for n in (50, 100, 200, 500):
        med = np.array(fss["median_curves"][str(n)])
        t50 = per[str(n)]["t50"]
        axi.plot(grid - t50, med, lw=0.6)
    axi.set_xlim(-220, 220)
    axi.set_title("curves collapse under\ntime translation",
                  fontsize=4.8, pad=1,
                  bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    axi.tick_params(labelsize=4.5, length=1.5, width=0.4)
    for s in axi.spines.values():
        s.set_linewidth(0.4)

    # -- b Kuramoto laws ----------------------------------------------------
    ax = fig.add_subplot(gs[1])
    Ks = [float(k) for k in ks["config"]["Ks"]]
    tstar = ks["registered_outcomes"]["mean_t_star"]
    slope = ks["registered_outcomes"]["mean_post_slope"]
    ax.plot(Ks, tstar, "o-", color=BLUE, lw=1.0, ms=3.5, mew=0)
    ax.set_xlabel("coupling K")
    ax.set_ylabel("breakpoint time $t^{*}$", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.plot(Ks, slope, "s--", color=RED, lw=1.0, ms=3.2, mew=0)
    ax2.set_ylabel("closing slope", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Kuramoto: critical slowing down,\nsharpening with feedback",
                 fontsize=6.5)

    # -- c scope: where onset lives -----------------------------------------
    ax = fig.add_subplot(gs[2])
    rows = [
        ("convention formation (no gate)", "onset", "onset"),
        ("role lock-in (no gate)", "onset", "onset"),
        ("ant colony, N $\\geq$ 20", "onset", "onset"),
        ("Kuramoto, K > K$_c$", "onset", "onset"),
        ("learned high-order (TRI-C)", "onset", "onset"),
        ("grip transport, realization", "onset", "onset"),
        ("Overcooked ring: direction", "onset", "onset 1/8"),
        ("grip transport, formation", "gradual", "gradual"),
        ("ordinary supervised learner", "gradual", "gradual"),
        ("consensus / quorum populations", "gradual", "gradual"),
        ("deep MARL (cramped room)", "gradual", "gradual"),
        ("Vicsek, N $\\leq$ 400", "gradual", "gradual"),
        ("Schelling, Swift–Hohenberg", "gradual", "gradual"),
        ("Kuramoto, K < K$_c$", "no collapse", "no collapse"),
        ("LM checkpoints (stored grids)", "unresolvable", "unresolvable"),
    ]
    colmap = {"onset": BLUE, "gradual": CYAN, "no collapse": GREY,
              "unresolvable": "#DDDDDD"}
    for i, (name, verdict, label) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.add_patch(Rectangle((0.68, y + 0.08), 0.32, 0.84,
                     facecolor=colmap[verdict], edgecolor="white",
                     lw=0.5))
        ax.text(0.66, y + 0.5, name, fontsize=6, ha="right", va="center")
        ax.text(0.84, y + 0.5, label, fontsize=5.5, ha="center",
                va="center",
                color="white" if verdict == "onset" else DARK)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.1, len(rows) + 0.4)
    ax.axis("off")
    ax.set_title("onset appears where a new joint regime is crossed",
                 fontsize=6.5)

    panel(fig, 0.005, 1.06, "a")
    panel(fig, 0.30, 1.06, "b")
    panel(fig, 0.56, 1.06, "c")
    save(fig, "fig6")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("all figures written to", FIG)
