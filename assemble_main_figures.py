"""Assembles the six MAIN composite figures from the panel PNGs.

Layouts follow the "Main figures (6)" section of MANUSCRIPT.md. Every
panel PNG is produced by generate_paper_figures.py (dpi 220) or
generate_figure1_concept.py; this script only composes them (scaling
rows to a common width, adding row labels), so the underlying data
path stays single-sourced.

fig34's panels are re-rendered as standalone splits by
generate_paper_figures.py so main Fig. 4 can take the gradualism panel
and main Fig. 5 the chess robustness panel without pixel cropping.

Output: figures/main_fig1.png ... main_fig6.png
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

TARGET_WIDTH = 3200          # px; every row is scaled to this width
ROW_GAP = 36                 # px between rows
LABEL_SIZE = 64              # px font height for row labels
MARGIN = 20                  # px page margin


def load(name: str) -> Image.Image:
    return Image.open(FIGURES / name).convert("RGB")


def hstack(images: List[Image.Image], gap: int = 24) -> Image.Image:
    """Side-by-side composition at a common height (max of members)."""
    height = max(im.height for im in images)
    scaled = []
    for im in images:
        if im.height != height:
            im = im.resize((round(im.width * height / im.height), height),
                           Image.LANCZOS)
        scaled.append(im)
    width = sum(im.width for im in scaled) + gap * (len(scaled) - 1)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for im in scaled:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def font() -> ImageFont.FreeTypeFont:
    import matplotlib.font_manager as fm
    path = fm.findfont("DejaVu Sans:bold")
    return ImageFont.truetype(path, LABEL_SIZE)


def compose(rows: List[Tuple[str, Image.Image]], out_name: str) -> None:
    fnt = font()
    scaled_rows = []
    for label, im in rows:
        if im.width != TARGET_WIDTH:
            im = im.resize(
                (TARGET_WIDTH, round(im.height * TARGET_WIDTH / im.width)),
                Image.LANCZOS)
        scaled_rows.append((label, im))
    label_h = LABEL_SIZE + 16
    total_h = (sum(im.height + label_h for _, im in scaled_rows)
               + ROW_GAP * (len(scaled_rows) - 1) + 2 * MARGIN)
    page = Image.new("RGB", (TARGET_WIDTH + 2 * MARGIN, total_h), "white")
    draw = ImageDraw.Draw(page)
    y = MARGIN
    for label, im in scaled_rows:
        draw.text((MARGIN, y), label, fill="black", font=fnt)
        y += label_h
        page.paste(im, (MARGIN, y))
        y += im.height + ROW_GAP
    out = FIGURES / out_name
    page.save(out, dpi=(220, 220))
    print(f"Wrote {out} ({page.width}x{page.height})")


def main() -> None:
    # Fig. 1 -- concept: four regimes of P_t(B); the editor's picture.
    compose([("", load("figure1_concept.png"))], "main_fig1.png")

    # Fig. 2 -- criterion vs imitations: battery matrix + named
    # counterexamples (a), threshold plateau + prior detectors (b),
    # exact rival formalisms within declared candidate families (c),
    # fair multivariate baselines with equal freedom (d).
    compose([
        ("a", load("fig19_criterion_battery.png")),
        ("b", load("fig28_robustness_and_priors.png")),
        ("c", load("fig37_exact_prior_formalisms.png")),
        ("d", load("fig40_fair_baselines.png")),
    ], "main_fig2.png")

    # Fig. 3 -- full six-component confirmation across three domains:
    # external swarm (a), Contextual LBF (b), latent-context sequence
    # model (c). The claim hierarchy puts these first among transfers.
    compose([
        ("a", load("fig25_refined_confirmation.png")),
        ("b", load("ed_fig8_contextual_lbf.png")),
        ("c", load("fig41_latent_context_lm.png")),
    ], "main_fig3.png")

    # Fig. 4 -- chess: curated event-level recovery (a) and prospective
    # discovery on uncurated months with referee-family checks (b).
    compose([
        ("a", load("fig33_chess_collapse.png")),
        ("b", load("fig42_chess_discovery.png")),
    ], "main_fig4.png")

    # Fig. 5 -- deep MARL seed-level inference (a) and the prospective
    # phase-boundary prediction (b). Episode-level detail is ED.
    compose([
        ("a", load("fig39_marl_seed_level.png")),
        ("b", load("fig22_phase_boundary.png")),
    ], "main_fig5.png")

    # ---------------- Extended Data composites ----------------

    # ED Fig. 1 -- analytic core: possibility tree returns, option-value
    # boundary, horizon reversal.
    compose([
        ("a", load("fig1_possibility_tree_returns.png")),
        ("b", hstack([load("fig2_option_value_heatmap.png"),
                      load("fig3_horizon_reversal_heatmap.png")])),
    ], "ed_fig1.png")

    # ED Fig. 2 -- representation-jump bridge and false-positive audit.
    compose([
        ("a", load("fig13_representation_jump_bridge.png")),
        ("b", load("fig14_learned_representation_false_positive_audit.png")),
    ], "ed_fig2.png")

    # ED Fig. 3 -- grokking bridge, generality sweep, scale decomposition.
    compose([
        ("a", load("fig23_grokking_bridge.png")),
        ("b", hstack([load("fig27_grokking_generality.png"),
                      load("fig26_scale_decomposition.png")])),
    ], "ed_fig3.png")

    # ED Fig. 4 -- induction heads and transformer grokking replication.
    compose([
        ("a", load("fig29_induction_heads.png")),
        ("b", load("fig30_transformer_grokking.png")),
    ], "ed_fig4.png")

    # ED Fig. 5 -- external swarm transfer: main, sweep, refined
    # out-of-sample confirmation.
    compose([
        ("a", load("fig21_external_transfer.png")),
        ("b", hstack([load("fig24_external_transfer_sweep.png"),
                      load("fig25_refined_confirmation.png")])),
    ], "ed_fig5.png")

    # ED Fig. 6 -- estimator robustness grid.
    compose([
        ("", load("fig20_estimator_robustness.png")),
    ], "ed_fig6.png")

    # ED Fig. 10 -- mechanism within single episodes (moved from main).
    compose([
        ("a", hstack([load("fig15_within_episode_collapse.png")])),
        ("b", hstack([load("fig17_within_episode_ci.png"),
                      load("fig16_unsupervised_basin_discovery.png")])),
        ("c", load("fig18_neural_probe.png")),
    ], "ed_fig10_mechanism.png")

    # ED Fig. 11 -- public checkpoint series (moved from main).
    compose([
        ("a", load("fig31_multiberts_public_series.png")),
        ("b", load("fig32_phenomena_and_alignment.png")),
        ("c", hstack([load("fig34a_gradualism.png"),
                      load("fig38_pythia_decoder.png")])),
    ], "ed_fig11_public.png")

    # ED Fig. 12 -- episode-level MARL detail (conditional tests) and
    # the chess robustness grid.
    compose([
        ("a", load("fig35_deep_marl.png")),
        ("b", load("fig36_lbf_cross_task.png")),
        ("c", load("fig34b_chess_grid.png")),
    ], "ed_fig12_episode_detail.png")

    # ED Fig. 13 -- dual plausible observer contracts on Contextual LBF.
    compose([
        ("", load("fig43_dual_observer.png")),
    ], "ed_fig13_dual_observer.png")

    # ED Fig. 14 -- emergence-strength gradient across provenances.
    compose([
        ("", load("fig44_strength_gradient.png")),
    ], "ed_fig14_strength_gradient.png")


if __name__ == "__main__":
    main()
