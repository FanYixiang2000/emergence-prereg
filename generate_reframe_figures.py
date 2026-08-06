"""Panels added in the reframing revision.

fig39: seed-level MARL do-contrast effects (replaces episode-level p and
       PASS-bar displays as the main statistical exhibit).
fig40: fair multivariate baseline comparison (in-sample vs LOOCV vs
       frozen transfer to the fresh battery).
fig41: latent-context sequence model, full six-component confirmation.
fig42: uncurated chess discovery (AUROC by month, flag precision lift,
       referee family checks including the engine-free realized-outcome
       referee).
fig43: dual plausible observer contracts (verdict agreement and the
       component route of every flip).
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
ORANGE = "#F58518"
GRAY = "#B0B0B0"
PURPLE = "#B279A2"


def read(name: str):
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Wrote {path}")


def figure_marl_seed_level() -> None:
    data = read("hierarchical_marl_analysis_combined.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, task, title in (
        (axes[0], "simple_spread",
         "simple_spread: per-seed mean do-contrast\n(6 policy seeds; "
         "seed-level sign test $p=0.016$)"),
        (axes[1], "lbf",
         "Level-Based Foraging: per-seed mean do-contrast\n(8 policy "
         "seeds; seed-level sign test $p=0.004$)"),
    ):
        seeds = list(data[task]["per_seed"].keys())
        means = data[task]["seed_level"]["mean_effects"]
        ci = data[task]["cluster_bootstrap"][
            "mean_of_seed_means_cluster_ci95"]
        grand = data[task]["cluster_bootstrap"]["mean_of_seed_means"]
        x = np.arange(len(seeds))
        registered = 3
        colors = [GREEN if i < registered else BLUE
                  for i in range(len(seeds))]
        ax.bar(x, means, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhspan(ci[0], ci[1], color=GRAY, alpha=0.25,
                   label=f"cluster-bootstrap 95% CI "
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
        ax.axhline(grand, color=RED, linewidth=1.2,
                   label=f"mean of seed means {grand:+.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
        ax.set_ylabel("mean per-episode do-contrast")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.85)]
    axes[0].legend(handles + axes[0].get_legend_handles_labels()[0],
                   ["registered seeds", "post-hoc extension seeds"]
                   + axes[0].get_legend_handles_labels()[1],
                   frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig39_marl_seed_level.png")


def figure_fair_baselines() -> None:
    data = read("fair_baseline_comparison.json")
    orig = data["original_battery"]
    fresh = data["fresh_battery"]["frozen_baselines"]

    entries = [
        ("2-signal AND\n(prior)",
         orig["prior5_conjunctions"]["conj_2"]["accuracy"], None,
         fresh["conj_2_frozen"]["accuracy"]),
        ("3-signal AND\n(prior)",
         orig["prior5_conjunctions"]["conj_3"]["accuracy"], None,
         fresh["conj_3_frozen"]["accuracy"]),
        ("logistic\n(5 prior signals)",
         orig["learned_models"]["prior5"]["logistic_in_sample"],
         orig["learned_models"]["prior5"]["logistic_loocv"],
         fresh["logistic_frozen"]["accuracy"]),
        ("tree depth-2\n(5 prior signals)",
         orig["learned_models"]["prior5"]["tree_in_sample"],
         orig["learned_models"]["prior5"]["tree_loocv"],
         fresh["tree_frozen"]["accuracy"]),
        ("specificity+\nusefulness AND",
         orig["own_two_component"]["specificity_js+usefulness_gap"]
         ["accuracy"], None,
         fresh["two_component_specificity_js+usefulness_gap_frozen"]
         ["accuracy"]),
    ]
    labels = [e[0] for e in entries]
    x = np.arange(len(entries))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.bar(x - width, [e[1] for e in entries], width, color=GRAY,
           alpha=0.9, label="hindsight fit (original battery)")
    loocv = [e[2] if e[2] is not None else np.nan for e in entries]
    ax.bar(x, loocv, width, color=ORANGE, alpha=0.9,
           label="leave-one-out CV")
    ax.bar(x + width, [e[3] for e in entries], width, color=BLUE,
           alpha=0.9, label="frozen, fresh-seed battery")
    ax.axhline(1.0, color=GREEN, linewidth=1.6,
               label="six-component protocol, frozen thresholds,\n"
                     "same fresh seed (10/10, stored confirmation)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("battery accuracy")
    ax.set_title("Multivariate prior-signal baselines with equal freedom: "
                 "fit does not transfer\n(every frozen baseline "
                 "misclassifies the true conditional emergence)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig40_fair_baselines.png")


def figure_latent_context_lm() -> None:
    data = read("latent_context_lm_confirmation.json")
    seeds = sorted(data["seeds"].keys())
    systems = ["learned", "initial_twin", "router", "fixed_R0", "fixed_R1"]
    sys_labels = ["learned", "init twin", "oracle router",
                  "fixed rule 0", "fixed rule 1"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))

    matrix = np.zeros((len(systems), len(seeds)))
    for j, seed in enumerate(seeds):
        for i, name in enumerate(systems):
            matrix[i, j] = data["seeds"][seed]["systems"][name][
                "verdict"]["emergent"]
    expected = np.array([[1] + [0] * 4]).T.repeat(len(seeds), axis=1)
    correct = (matrix == expected)
    ax = axes[0]
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(len(systems)):
        for j in range(len(seeds)):
            mark = "ACC" if matrix[i, j] else "rej"
            color = "white" if matrix[i, j] else "#555555"
            ax.text(j, i, mark, ha="center", va="center", fontsize=6.5,
                    color=color,
                    fontweight="bold" if matrix[i, j] else "normal")
    assert correct.all()
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([s[-2:] for s in seeds], fontsize=7)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(sys_labels, fontsize=8)
    ax.set_xlabel("fresh seed (21xx)")
    ax.set_title("Sequence domain: six-component verdicts\n"
                 "(10/10 learned accepted; 40/40 controls rejected)",
                 fontsize=10)

    ax = axes[1]
    comp = {"selectivity": "conditional_selectivity",
            "specificity (JS bits)": "specificity_js_bits",
            "usefulness gap": "usefulness_gap"}
    xs = np.arange(len(comp) + 1)
    for j, (label, key) in enumerate(comp.items()):
        vals = [data["seeds"][s]["systems"]["learned"]["metrics"][key]
                for s in seeds]
        ax.scatter(np.full(len(vals), xs[j]) +
                   np.linspace(-0.12, 0.12, len(vals)), vals, s=18,
                   color=BLUE, zorder=3)
    acq = [data["seeds"][s]["systems"]["learned"]["acquisition"]
           for s in seeds]
    ax.scatter(np.full(len(acq), xs[-1]) +
               np.linspace(-0.12, 0.12, len(acq)), acq, s=18,
               color=GREEN, zorder=3)
    th = data["thresholds"]
    for j, cutoff in enumerate([th["conditional_selectivity"],
                                th["specificity_js_bits"],
                                th["usefulness_gap"], th["acquisition"]]):
        ax.hlines(cutoff, xs[j] - 0.3, xs[j] + 0.3, color=RED,
                  linestyle="--", linewidth=1.1)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(comp.keys()) + ["acquisition"], fontsize=8)
    ax.set_ylabel("component value (per seed)")
    ax.set_title("Learned components vs frozen thresholds (dashed)",
                 fontsize=10)
    save(fig, "fig41_latent_context_lm.png")


def figure_chess_discovery() -> None:
    main = read("chess_discovery_main.json")["analysis"]
    repl = read("chess_discovery_replication_2016_03.json")["analysis"]
    cross = read("chess_discovery_cross_engine.json")
    outcome = read("chess_realized_outcome.json")

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

    ax = axes[0]
    scores = ["do_gap", "shallow_gap_cp", "tactical_density",
              "material_imbalance", "shallow_eval_abs_cp"]
    labels = ["collapse\ndo-gap", "engine\neval gap", "tactical\ndensity",
              "material\nimbalance", "|eval|"]
    x = np.arange(len(scores))
    width = 0.38
    ax.bar(x - width / 2, [main["auroc"][s] for s in scores], width,
           color=BLUE, alpha=0.9, label="2015-08 (n=400)")
    ax.bar(x + width / 2, [repl["auroc"][s] for s in scores], width,
           color=GREEN, alpha=0.9, label="2016-03 replication (n=400)")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0.70, color=RED, linewidth=1.0, linestyle="--",
               label="registered CD1 cutoff (0.70)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUROC vs value-critical label")
    ax.set_title("Uncurated discovery: the do-gap is the\n"
                 "distribution-stable predictor across months", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    months = ["2015-08", "2016-03"]
    prec = [main["flag"]["precision"], repl["flag"]["precision"]]
    base = [main["referee_base_rate"], repl["referee_base_rate"]]
    x = np.arange(2)
    ax.bar(x - 0.19, prec, 0.34, color=BLUE, alpha=0.9,
           label="frozen-flag precision")
    ax.bar(x + 0.19, base, 0.34, color=GRAY, alpha=0.9,
           label="base rate")
    for i in range(2):
        ax.text(i, prec[i] + 0.012, f"{prec[i]/base[i]:.1f}x lift",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("P(value-critical)")
    ax.set_title("Frozen binary flag (cutoffs reused from the\n"
                 "curated study, never refit)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    fams = ["NNUE referee\n(depth 18)", "classical SF11\nreferee",
            "realized game\noutcome (engine-free)"]
    m1 = [main["auroc"]["do_gap"],
          cross["months"]["main"]["auroc_do_gap_vs_sf11"],
          np.nan]
    m2 = [repl["auroc"]["do_gap"],
          cross["months"]["replication_2016_03"]["auroc_do_gap_vs_sf11"],
          np.nan]
    x = np.arange(len(fams))
    ax.bar(x - 0.19, m1, 0.34, color=BLUE, alpha=0.9, label="2015-08")
    ax.bar(x + 0.19, m2, 0.34, color=GREEN, alpha=0.9, label="2016-03")
    d1 = outcome["per_month"]["2015_08"]["delta_flagged"]
    d2 = outcome["per_month"]["2016_03"]["delta_flagged"]
    ax.text(2, 0.55,
            f"directionally consistent\n($\\Delta$ = {d1:+.3f} / "
            f"{d2:+.3f})\nbut underpowered:\nregistered interaction "
            f"null\n(p = "
            f"{outcome['pooled']['permutation_p_one_sided']:.2f})",
            ha="center", va="center", fontsize=7.5, color="#333333")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(fams, fontsize=8)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUROC of collapse do-gap")
    ax.set_title("Referee-family checks: NNUE, classical,\n"
                 "and the engine-free realized outcome", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig42_chess_discovery.png")


def figure_dual_observer() -> None:
    data = read("dual_observer_contracts.json")
    systems = ["learned", "initial_twin", "team_nearest",
               "fixed_food0", "fixed_food1"]
    sys_labels = ["learned", "init twin", "team nearest",
                  "fixed food0", "fixed food1"]
    seeds = sorted(data["seeds"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))

    ax = axes[0]
    grid = np.zeros((len(systems), len(seeds)))
    for j, seed in enumerate(seeds):
        for i, name in enumerate(systems):
            b = data["seeds"][seed][name]["verdict"]["emergent"]
            expected = data["expected"][name]
            grid[i, j] = 1 if b == expected else 0.5 if b == 0 else 0
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for j, seed in enumerate(seeds):
        for i, name in enumerate(systems):
            b = data["seeds"][seed][name]["verdict"]["emergent"]
            expected = data["expected"][name]
            if b != expected:
                failed = data["seeds"][seed][name]["verdict"]["failed"]
                route = ("U" if "usefulness" in failed else "S")
                ax.text(j, i, route, ha="center", va="center",
                        fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([s[-2:] for s in seeds], fontsize=7)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(sys_labels, fontsize=8)
    ax.set_xlabel("policy seed (11xx/12xx)")
    ax.set_title("Contract B verdicts vs contract A expectations\n"
                 "(green = agree; amber = conservative flip; route: "
                 "U=usefulness, S=selectivity)", fontsize=10)

    ax = axes[1]
    outcomes = data["registered_outcomes"]
    labels = ["controls\nrejected", "learned\naccepted",
              "verdict\nagreement"]
    fracs = [60 / 60, 10 / 15, 70 / 75]
    texts = [outcomes["DO2_controls_rejected"],
             outcomes["DO1_learned_accepted"],
             outcomes["DO3_contract_agreement"]]
    colors = [GREEN, ORANGE, BLUE]
    ax.bar(range(3), fracs, color=colors, alpha=0.85)
    for i, (frac, text) in enumerate(zip(fracs, texts)):
        ax.text(i, frac + 0.02, text, ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("fraction under contract B")
    ax.set_title("Structural verdicts travel; value verdicts are\n"
                 "contract-relative (all flips conservative)", fontsize=10)
    save(fig, "fig43_dual_observer.png")


def figure_strength_gradient() -> None:
    grad = read("strength_gradient_battery.json")
    fine = read("strength_gradient_fine.json")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))

    ax = axes[0]
    names = ["scripted", "shaped", "outcome_only"]
    labels = ["scripted\n(prescribed)", "process-shaped\n(trigger named)",
              "outcome-only\n(discovered)"]
    colors = [GRAY, ORANGE, GREEN]
    means = [0.0,
             grad["systems"]["shaped"]["seed_mean_c_prov_bits"],
             grad["systems"]["outcome_only"]["seed_mean_c_prov_bits"]]
    ax.bar(range(3), means, color=colors, alpha=0.9)
    for i, name in enumerate(names[1:], start=1):
        seeds = grad["systems"][name]["seeds"]
        vals = [seeds[s]["c_prov_bits"] for s in seeds]
        ax.scatter(np.full(len(vals), i)
                   + np.linspace(-0.1, 0.1, len(vals)), vals, s=16,
                   color="#333333", zorder=3)
    ax.axhline(grad["open_prior"]["c_open_bits"], color=BLUE,
               linestyle="--", linewidth=1.2,
               label=f"open-space rarity "
                     f"{grad['open_prior']['c_open_bits']:.1f} bits "
                     "(same pattern, all systems)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("provenance rarity $C$ [bits]")
    ax.set_title("Same macro-pattern, three provenances:\n"
                 "rarity under each provenance's own search prior",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    for name, color, label in (("shaped", ORANGE, "process-shaped"),
                               ("outcome_only", GREEN, "outcome-only")):
        seeds = fine["systems"][name]["seeds"]
        for j, s in enumerate(sorted(seeds)):
            trace = seeds[s]["trace"]
            x = np.arange(len(trace)) * 250
            ax.plot(x, trace, color=color, alpha=0.55, linewidth=1.1,
                    label=label if j == 0 else None)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("training episode (fine grid)")
    ax.set_ylabel("pattern probability $p_t$")
    ax.set_title("Discovery is ~2x later without process shaping\n"
                 "(five seeds each; dotted: discovery threshold)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    disc_s = [fine["systems"]["shaped"]["seeds"][s]["discovery_episode"]
              for s in fine["systems"]["shaped"]["seeds"]]
    disc_o = [fine["systems"]["outcome_only"]["seeds"][s]
              ["discovery_episode"]
              for s in fine["systems"]["outcome_only"]["seeds"]]
    sud_s = [grad["systems"]["shaped"]["seeds"][s]["suddenness_ratio"]
             for s in grad["systems"]["shaped"]["seeds"]]
    sud_o = [grad["systems"]["outcome_only"]["seeds"][s]
             ["suddenness_ratio"]
             for s in grad["systems"]["outcome_only"]["seeds"]]
    ax.scatter(disc_s, sud_s, color=ORANGE, s=40, label="process-shaped")
    ax.scatter(disc_o, sud_o, color=GREEN, s=40, label="outcome-only")
    ax.set_xlabel("discovery episode (fine grid)")
    ax.set_ylabel("suddenness ratio (coarse grid)")
    ax.set_title("Retained ST-3 miss: suddenness does not\n"
                 "order provenances (both step-like)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig44_strength_gradient.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    figure_marl_seed_level()
    figure_fair_baselines()
    figure_latent_context_lm()
    figure_chess_discovery()
    figure_dual_observer()
    figure_strength_gradient()


if __name__ == "__main__":
    main()
