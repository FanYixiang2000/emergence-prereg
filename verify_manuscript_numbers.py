"""Cross-check headline manuscript numbers against stored output files.

Read-only audit: loads the JSON outputs that back the manuscript's headline
claims and compares them with the values asserted in paper/main.tex. Any
mismatch is reported as FAIL. This does not modify any experiment output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, bool(ok), detail))


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


# CLBF
clbf = json.loads((OUTPUTS / "contextual_lbf_confirmation_analysis.json").read_text())
counts = clbf["counts"]
check("CLBF learned passes 9/10", counts["learned_full_passes"] == 9,
      f"observed {counts['learned_full_passes']}")
check("CLBF controls rejected 40/40",
      counts["nonlearned_controls_rejected"] == 40,
      f"observed {counts['nonlearned_controls_rejected']}")
check("CLBF team_nearest exact route 10/10",
      counts["team_nearest_exact_failure_route"] == 10,
      f"observed {counts['team_nearest_exact_failure_route']}")
iv = clbf["seed_bootstrap_intervals"]
for name, point, lo, hi in (
    ("conditional_selectivity", 0.793, 0.644, 0.939),
    ("specificity_js_bits", 0.795, 0.769, 0.822),
    ("usefulness_gap", 0.079, 0.053, 0.108),
    ("acquisition", 0.784, 0.634, 0.933),
):
    item = iv[name]
    ok = (close(item["point"], point, 1e-3)
          and close(item["ci95"][0], lo, 1e-3)
          and close(item["ci95"][1], hi, 1e-3))
    check(f"CLBF {name} {point} [{lo},{hi}]", ok,
          f"observed {item['point']:.4f} "
          f"[{item['ci95'][0]:.4f},{item['ci95'][1]:.4f}]")

raw = json.loads((OUTPUTS / "contextual_lbf_confirmation.json").read_text())
miss = raw["seeds"]["1104"]["systems"]["learned"]["metrics"][
    "conditional_selectivity"]
check("CLBF seed-1104 selectivity 0.4875", close(miss, 0.4875),
      f"observed {miss}")

ext = json.loads((OUTPUTS / "contextual_lbf_extension_analysis.json").read_text())
check("CLBF extension 4/5 passes",
      ext["counts"]["learned_full_passes"] == 4,
      f"observed {ext['counts']['learned_full_passes']}")
check("CLBF extension 20/20 controls rejected",
      ext["counts"]["nonlearned_controls_rejected"] == 20,
      f"observed {ext['counts']['nonlearned_controls_rejected']}")
check("CLBF extension all positive usefulness",
      ext["counts"]["learned_positive_usefulness"] == 5,
      f"observed {ext['counts']['learned_positive_usefulness']}")
check("CLBF extension all positive acquisition",
      ext["counts"]["learned_positive_acquisition"] == 5,
      f"observed {ext['counts']['learned_positive_acquisition']}")

# Pythia 160m
py = json.loads((OUTPUTS / "pythia_collapse_summary.json").read_text())
agree = py["runs"]["pythia_agreement"]["stats"]
check("Pythia-160m agreement burstiness 27.6",
      close(agree["burstiness_ratio"], 27.6, 0.05),
      f"observed {agree['burstiness_ratio']:.3f}")
check("Pythia-160m agreement usefulness 0.47",
      close(agree["usefulness_acc_gain"], 0.47, 0.005),
      f"observed {agree['usefulness_acc_gain']:.4f}")
check("Pythia-160m potential 8.86 bits",
      close(agree["h_pre_burst_bits"], 8.86, 0.005),
      f"observed {agree['h_pre_burst_bits']:.4f}")

tail = json.loads((OUTPUTS / "pythia_tail_summary.json").read_text())
head = tail["runs"]["head_facts"]["stats"]
tfacts = tail["runs"]["tail_facts"]["stats"]
twords = tail["runs"]["tail_words"]["stats"]
check("Pythia-160m head facts burstiness 16.4",
      close(head["burstiness_ratio"], 16.4, 0.05),
      f"observed {head['burstiness_ratio']:.3f}")
check("Pythia-160m tail facts burstiness 3.2",
      close(tfacts["burstiness_ratio"], 3.2, 0.05),
      f"observed {tfacts['burstiness_ratio']:.3f}")
check("Pythia-160m tail words usefulness 0.056",
      close(twords["usefulness_acc_gain"], 0.056, 0.001),
      f"observed {twords['usefulness_acc_gain']:.4f}")

# Pythia 1b
py1b = json.loads((OUTPUTS / "pythia_collapse_summary_1b.json").read_text())
a1b = py1b["runs"]["pythia_agreement"]
check("Pythia-1B agreement emergent",
      a1b["verdict"]["emergent"] == 1, str(a1b["verdict"]["passes"]))
check("Pythia-1B controls rejected",
      py1b["runs"]["pythia_random_target"]["verdict"]["emergent"] == 0
      and py1b["runs"]["shuffled_vocab"]["verdict"]["emergent"] == 0,
      "both controls")
t1b = json.loads((OUTPUTS / "pythia_tail_summary_1b.json").read_text())
h1b = t1b["runs"]["head_facts"]
check("Pythia-1B head facts rejected via burstiness",
      h1b["verdict"]["emergent"] == 0
      and not h1b["verdict"]["passes"]["burstiness"]
      and h1b["stats"]["final_test_acc"] == 1.0,
      f"burstiness {h1b['stats']['burstiness_ratio']:.3f}, "
      f"final acc {h1b['stats']['final_test_acc']}")

# Pythia 1.4b
py14 = json.loads((OUTPUTS / "pythia_collapse_summary_1.4b.json").read_text())
a14 = py14["runs"]["pythia_agreement"]
check("Pythia-1.4B agreement emergent, window 1000",
      a14["verdict"]["emergent"] == 1
      and a14["stats"]["window_epoch"] == 1000
      and close(a14["stats"]["burstiness_ratio"], 7.3, 0.05)
      and close(a14["stats"]["usefulness_acc_gain"], 0.43, 0.005),
      f"burst {a14['stats']['burstiness_ratio']:.3f}, "
      f"gain {a14['stats']['usefulness_acc_gain']:.4f}, "
      f"window {a14['stats']['window_epoch']}")
check("Pythia-1.4B controls rejected",
      py14["runs"]["pythia_random_target"]["verdict"]["emergent"] == 0
      and py14["runs"]["shuffled_vocab"]["verdict"]["emergent"] == 0,
      "both controls")
t14 = json.loads((OUTPUTS / "pythia_tail_summary_1.4b.json").read_text())
check("Pythia-1.4B tails rejected, head fails burstiness at acc 1.0",
      all(t14["runs"][f]["verdict"]["emergent"] == 0
          for f in ("head_facts", "tail_facts", "tail_words"))
      and t14["runs"]["head_facts"]["stats"]["final_test_acc"] == 1.0,
      f"head burst {t14['runs']['head_facts']['stats']['burstiness_ratio']:.2f}")
hashes = json.loads(
    (OUTPUTS / "pythia_1.4b_checkpoint_hashes.json").read_text())
check("Pythia-1.4B 21 revisions pairwise distinct",
      hashes["n_complete"] == 21 and hashes["all_revisions_distinct"],
      f"complete {hashes['n_complete']}")

# Pythia 2.8b
py28 = json.loads((OUTPUTS / "pythia_collapse_summary_2.8b.json").read_text())
a28 = py28["runs"]["pythia_agreement"]
check("Pythia-2.8B registered S1 failure via burstiness",
      a28["verdict"]["emergent"] == 0
      and not a28["verdict"]["passes"]["burstiness"]
      and a28["verdict"]["passes"]["usefulness"]
      and a28["stats"]["window_epoch"] == 1000
      and close(a28["stats"]["burstiness_ratio"], 3.2, 0.05)
      and close(a28["stats"]["usefulness_acc_gain"], 0.417, 0.005)
      and close(a28["stats"]["final_test_acc"], 0.951, 0.005),
      f"burst {a28['stats']['burstiness_ratio']:.3f}, "
      f"gain {a28['stats']['usefulness_acc_gain']:.4f}")
check("Pythia-2.8B controls rejected",
      py28["runs"]["pythia_random_target"]["verdict"]["emergent"] == 0
      and py28["runs"]["shuffled_vocab"]["verdict"]["emergent"] == 0,
      "both controls")
t28 = json.loads((OUTPUTS / "pythia_tail_summary_2.8b.json").read_text())
check("Pythia-2.8B all tail families rejected",
      all(t28["runs"][f]["verdict"]["emergent"] == 0
          for f in ("head_facts", "tail_facts", "tail_words")),
      "3 families")
scale = json.loads((OUTPUTS / "pythia_scaling_summary.json").read_text())
pc = scale["prediction_counts"]
check("scaling totals 4/5 agreement, 10/10 controls, 4+4 tails",
      pc["agreement_passes"] == [4, 5]
      and pc["controls_rejected"] == [10, 10]
      and pc["tail_rejections"] == {"tail_facts": 4, "tail_words": 4},
      str(pc))
# discovery
disc = json.loads((OUTPUTS / "chess_discovery_main.json").read_text())
da = disc["analysis"]
check("discovery AUROC 0.730, 4/4 predictions",
      close(da["auroc"]["do_gap"], 0.730, 0.001)
      and da["all_pass"] and da["n_positions"] == 400,
      f"auroc {da['auroc']['do_gap']:.4f}, n {da['n_positions']}")
check("discovery precision 0.24 vs base 0.095, 75 flagged",
      close(da["flag"]["precision"], 0.24, 0.001)
      and close(da["referee_base_rate"], 0.095, 0.001)
      and da["flag"]["n_flagged"] == 75
      and close(da["flag"]["recall"], 0.4737, 0.001),
      f"prec {da['flag']['precision']:.3f}, flagged {da['flag']['n_flagged']}")
check("discovery flagged median potential 1.92 bits",
      close(da["flagged_median_potential"], 1.925, 0.005),
      f"observed {da['flagged_median_potential']:.4f}")

rep = json.loads(
    (OUTPUTS / "chess_discovery_replication_2016_03.json").read_text())
ra = rep["analysis"]
check("discovery replication 4/4, AUROC 0.725, precision 0.347",
      ra["all_pass"] and close(ra["auroc"]["do_gap"], 0.725, 0.001)
      and close(ra["flag"]["precision"], 0.3467, 0.001)
      and close(ra["referee_base_rate"], 0.1175, 0.001)
      and close(ra["auroc"]["shallow_gap_cp"], 0.652, 0.001),
      f"auroc {ra['auroc']['do_gap']:.4f}")

coup = json.loads((OUTPUTS / "trajectory_basin_coupling.json").read_text())
check("trajectory-basin coupling: DPI clean, retentions as stated",
      not coup["dpi_violations"]
      and coup["max_rarity_identity_gap"] < 1e-9
      and close(coup["systems"]["latent_conditional"]["do_trigger"][
          "basin_retention"], 0.320, 0.001)
      and close(coup["systems"]["converged_team"]["do_trigger"][
          "kl_trajectory_bits"], 32.155, 0.01)
      and close(coup["systems"]["converged_team"]["do_trigger"][
          "basin_retention"], 0.032, 0.001),
      f"lc retention "
      f"{coup['systems']['latent_conditional']['do_trigger']['basin_retention']:.3f}")

lm = json.loads((OUTPUTS / "latent_context_lm_confirmation.json").read_text())
lm_learned = [e["systems"]["learned"] for e in lm["seeds"].values()]
lm_controls_rejected = sum(
    1 - e["systems"][n]["verdict"]["emergent"]
    for e in lm["seeds"].values()
    for n in ("initial_twin", "router", "fixed_R0", "fixed_R1"))
router_exact = sum(
    set(k for k, ok in e["systems"]["router"]["verdict"]["passes"].items()
        if not ok) == {"endogeneity", "acquisition"}
    for e in lm["seeds"].values())
check("LM domain: 10/10 learned pass, 40/40 controls, router route miss",
      lm["summary"]["learned_passes"] == 10
      and lm_controls_rejected == 40 and router_exact == 0
      and all(s["acquisition"] > 0 for s in lm_learned)
      and all(s["metrics"]["usefulness_gap"] > 0 for s in lm_learned),
      f"passes {lm['summary']['learned_passes']}, "
      f"controls {lm_controls_rejected}, router exact {router_exact}")

xeng = json.loads((OUTPUTS / "chess_discovery_cross_engine.json").read_text())
check("cross-engine referee: AUROC 0.743/0.669, transfers",
      close(xeng["months"]["main"]["auroc_do_gap_vs_sf11"], 0.743, 0.001)
      and close(xeng["months"]["replication_2016_03"][
          "auroc_do_gap_vs_sf11"], 0.669, 0.001)
      and xeng["do_gap_transfers_across_engine_families"],
      "both months")

obs = json.loads((OUTPUTS / "adversarial_observer_audit.json").read_text())
check("adversarial observer: bijection exact, declared at 100th pct, "
      "random spec pass 80%",
      obs["bijection_max_deviation"] < 1e-12
      and obs["declared_partition"]["specificity_percentile_vs_random"] == 1.0
      and close(obs["random_observers"]["mean_specificity_pass_rate"],
                0.8015, 0.001),
      f"pct {obs['declared_partition']['specificity_percentile_vs_random']}")

ol = json.loads((OUTPUTS / "ordinary_learner_control.json").read_text())
check("ordinary learner: proxy accepts 6/6 at acc >= 0.9",
      ol["finding"]["runs_accepted_by_proxy"] == "6/6"
      and ol["finding"]["runs_with_final_acc_ge_0.9"] == "6/6",
      ol["finding"]["runs_accepted_by_proxy"])

cn = json.loads((OUTPUTS / "capability_novelty_boundary.json").read_text())
cnv = cn["values"]
check("capability novelty: ordinary rejected, grokking/induction accepted",
      cn["registered_outcomes"] == {
          "CN1_ordinary_low_order_suffices": True,
          "CN2_modular_rule_requires_interaction": True,
          "CN3_induction_requires_composition": True,
          "CN4_old_proxy_false_positive_repaired": True,
      }
      and cn["novelty_qualified_verdict"] == {
          "ordinary": False,
          "modular_grokking": True,
          "induction": True,
      }
      and abs(cnv["ordinary"]["novelty_gap"]) < 0.01
      and cnv["modular_addition"]["novelty_gap"] >= 0.99
      and cnv["induction"]["novelty_gap"] >= 0.87,
      f"gaps {cnv['ordinary']['novelty_gap']:.3f}/"
      f"{cnv['modular_addition']['novelty_gap']:.3f}/"
      f"{cnv['induction']['novelty_gap']:.3f}")

bb = json.loads((OUTPUTS / "burst_boundary_audit.json").read_text())
bbv = bb["values"]
check("burst boundary: not sufficient, not necessary, grid-relative",
      bb["registered_outcomes"] == {
          "BB1_burst_not_sufficient": True,
          "BB2_burst_not_necessary": True,
          "BB3_burst_grid_relative": True,
          "BB4_burst_needs_utility_controls": True,
      }
      and bbv["ordinary_burst_not_sufficient"]["min_burstiness_ratio"] > 5
      and bbv["ordinary_burst_not_sufficient"]["novelty_verdict"] is False
      and bbv["ant_gradual_but_collective"]["span_frac"] >= 0.10
      and bbv["pythia_grid_relative"]["thinning_agreement"] == 0.0
      and len(bbv["bursty_failed_controls"]) >= 1,
      f"ordinary min burst "
      f"{bbv['ordinary_burst_not_sufficient']['min_burstiness_ratio']:.1f}; "
      f"ant span {bbv['ant_gradual_but_collective']['span_frac']:.3f}; "
      f"2.8B thin {bbv['pythia_grid_relative']['thinning_agreement']:.1f}")

pers = json.loads((OUTPUTS / "contextual_lbf_persistence.json").read_text())
pp = pers["predictions"]
check("persistence: horizon/noise 40/40, novel layouts 5/10, twins 70/70",
      not pp["PS1_structure_persists"]["pass"]
      and pp["PS1_structure_persists"]["counts"]["P1_novel_layouts"] == 5
      and all(pp["PS1_structure_persists"]["counts"][k] == 10
              for k in ("P2_horizon12", "P3_horizon18",
                        "P4_noise005", "P5_noise010"))
      and pp["PS3_twins_stay_below"]["pass"]
      and not pp["PS4_graceful_noise_degradation"]["pass"],
      str(pp["PS1_structure_persists"]["counts"]))

impl = json.loads(
    (OUTPUTS / "trajectory_kl_implementation_check.json").read_text())
check("trajectory KL implementation cross-check passes",
      impl["pass"] and impl["max_abs_gap_bits"] < 1e-9
      and impl["n_trials"] == 2000,
      f"max gap {impl['max_abs_gap_bits']:.2e}")

wit = json.loads((OUTPUTS / "component_witness_matrix.json").read_text())
check("all six components have measured witnesses",
      wit["all_components_witnessed"], "matrix assembled")

# ablation
abl = json.loads((OUTPUTS / "component_ablation_witnesses.json").read_text())
check("CLBF leave-one-out: only selectivity non-redundant",
      abl["components"]["conditional_selectivity"]["newly_accepted_total"] == 2
      and not abl["components"]["conditional_selectivity"][
          "false_positive_controls"]
      and all(abl["components"][c]["empirically_redundant_here"]
              for c in abl["components"] if c != "conditional_selectivity"),
      f"n_systems {abl['n_systems']}")

s7 = json.loads((OUTPUTS / "held_out_scaling_robustness.json").read_text())
check("S7 registered failure 130/162 = 80.2%",
      s7["S7"]["cells_agreeing"] == 130 and s7["S7"]["cells_total"] == 162
      and not s7["S7"]["pass"],
      f"{s7['S7']['cells_agreeing']}/{s7['S7']['cells_total']}")
flip = s7["scales"]["2.8b"]["pythia_agreement"]
check("2.8B agreement flips to accept in all thinning cells",
      flip["full_grid_emergent"] == 0 and flip["thinning_agreement"] == 0.0
      and flip["n_thinning_cells"] == 9,
      f"thin_agree={flip['thinning_agreement']}, "
      f"cells={flip['n_thinning_cells']}")

# deep MARL
marl = json.loads((OUTPUTS / "deep_marl_collapse_aggregate.json").read_text())
d3 = marl["D3_counterfactual"]
check("simple_spread pooled do-gap median +0.083",
      close(d3["pooled_do_gap_median"], 0.083, 0.001),
      f"observed {d3['pooled_do_gap_median']:.4f}")
check("simple_spread sign 74/44 p=0.0037",
      d3["pooled_sign_wins"] == 74 and d3["pooled_sign_losses"] == 44
      and close(d3["pooled_sign_p"], 0.0037, 2e-4),
      f"observed {d3['pooled_sign_wins']}/{d3['pooled_sign_losses']} "
      f"p={d3['pooled_sign_p']:.5f}")

lbf = json.loads((OUTPUTS / "lbf_collapse_main.json").read_text())
l3 = lbf["verdicts"]["L3_counterfactual"]
check("LBF pooled do-gap median +0.042",
      close(l3["pooled_do_gap_median"], 0.042, 0.001),
      f"observed {l3['pooled_do_gap_median']:.4f}")
check("LBF 69 wins / 0 losses",
      l3["pooled_sign_wins"] == 69 and l3["pooled_sign_losses"] == 0,
      f"observed {l3['pooled_sign_wins']}/{l3['pooled_sign_losses']}")

# chess
chess = json.loads((OUTPUTS / "chess_collapse_main_summary.json").read_text())


def chess_field(data, *names):
    for name in names:
        if name in data:
            return data[name]
    return None


boot = json.loads((OUTPUTS / "bootstrap_intervals.json").read_text())
gap = boot["chess"]["do_key_minus_do_alt_win_gap_mean"]
check("chess do-gap 0.539 [0.508, 0.570]",
      close(gap["point"], 0.539, 0.001)
      and close(gap["ci_lo"], 0.508, 0.001)
      and close(gap["ci_hi"], 0.570, 0.001),
      f"observed {gap['point']:.4f} [{gap['ci_lo']:.4f},{gap['ci_hi']:.4f}]")

# reframing-revision audits
fair = json.loads((OUTPUTS / "fair_baseline_comparison.json").read_text())
frozen = fair["fresh_battery"]["frozen_baselines"]
best_frozen = max(v["accuracy"] for v in frozen.values())
check("fair baselines: best frozen transfer 0.9",
      close(best_frozen, 0.9, 1e-9),
      f"observed {best_frozen}")
check("fair baselines: every frozen baseline misses latent_conditional",
      all("latent_conditional" in v["misclassified"]
          for v in frozen.values()),
      "misclassified lists checked")
loocv = fair["original_battery"]["learned_models"]
check("fair baselines: LOOCV 0.7-0.8",
      all(0.7 - 1e-9 <= loocv[t][k] <= 0.8 + 1e-9
          for t in ("prior5", "prior7")
          for k in ("logistic_loocv", "tree_loocv")),
      f"prior5 {loocv['prior5']}, prior7 {loocv['prior7']}")

dual = json.loads((OUTPUTS / "dual_observer_contracts.json").read_text())
outcomes = dual["registered_outcomes"]
check("dual observer: 10/15 learned, 60/60 controls, 70/75 agreement",
      outcomes["DO1_learned_accepted"] == "10/15"
      and outcomes["DO2_controls_rejected"] == "60/60"
      and outcomes["DO3_contract_agreement"] == "70/75",
      f"observed {outcomes}")
flip_routes = []
for seed, systems in dual["seeds"].items():
    verdict = systems["learned"]["verdict"]
    if verdict["emergent"] == 0:
        flip_routes.append(tuple(verdict["failed"]))
check("dual observer: 4 usefulness flips + 1 selectivity flip",
      sorted(flip_routes) == sorted(
          [("usefulness",)] * 4 + [("conditional_selectivity",)]),
      f"observed {flip_routes}")

ro = json.loads((OUTPUTS / "chess_realized_outcome.json").read_text())
check("realized outcome: interaction null p=0.54",
      close(ro["pooled"]["permutation_p_one_sided"], 0.54, 0.01)
      and not ro["registered_outcomes"]["RO1_interaction_positive_pooled"],
      f"observed p={ro['pooled']['permutation_p_one_sided']:.4f}")
check("realized outcome: delta flagged positive both months",
      ro["registered_outcomes"]["RO2_delta_flagged_positive_each_month"],
      f"deltas {ro['per_month']['2015_08']['delta_flagged']:.4f}, "
      f"{ro['per_month']['2016_03']['delta_flagged']:.4f}")
check("realized outcome: played-rate contrast 0.53/0.34 and 0.43/0.27",
      close(ro["per_month"]["2015_08"]["played_rate_flagged"], 0.533, 0.001)
      and close(ro["per_month"]["2015_08"]["played_rate_unflagged"],
                0.338, 0.001)
      and close(ro["per_month"]["2016_03"]["played_rate_flagged"],
                0.427, 0.001)
      and close(ro["per_month"]["2016_03"]["played_rate_unflagged"],
                0.274, 0.001),
      "rates checked")

grad = json.loads((OUTPUTS / "strength_gradient_battery.json").read_text())
check("strength gradient: open-space rarity 6.3 bits",
      close(grad["open_prior"]["c_open_bits"], 6.28, 0.05),
      f"observed {grad['open_prior']['c_open_bits']:.3f}")
check("strength gradient: seed-mean ordering 0 < 0.39 < 0.65",
      close(grad["systems"]["shaped"]["seed_mean_c_prov_bits"], 0.391, 0.005)
      and close(grad["systems"]["outcome_only"]["seed_mean_c_prov_bits"],
                0.648, 0.005)
      and grad["registered_outcomes"]["ST2_rarity_ordering"],
      f"shaped {grad['systems']['shaped']['seed_mean_c_prov_bits']:.3f}, "
      f"outcome {grad['systems']['outcome_only']['seed_mean_c_prov_bits']:.3f}")
check("strength gradient: ST-3 retained miss + competence pass",
      not grad["registered_outcomes"]["ST3_suddenness_ordering"]
      and grad["registered_outcomes"]["ST4_competence_comparable"],
      "ST3 false, ST4 true")
fine = json.loads((OUTPUTS / "strength_gradient_fine.json").read_text())
check("strength gradient fine grid: discovery ~1600 vs ~3150 episodes",
      close(fine["systems"]["shaped"]["mean_discovery_checkpoint"] * 250,
            1600, 25)
      and close(fine["systems"]["outcome_only"]["mean_discovery_checkpoint"]
                * 250, 3150, 25)
      and fine["follow_up_outcomes"]["later_discovery_outcome_only"]
      and fine["follow_up_outcomes"][
          "higher_pre_discovery_rarity_outcome_only"],
      f"shaped {fine['systems']['shaped']['mean_discovery_checkpoint']}, "
      f"outcome "
      f"{fine['systems']['outcome_only']['mean_discovery_checkpoint']}")

cluster = json.loads((OUTPUTS / "chess_clustered_inference.json").read_text())
c1 = cluster["months"]["2015_08"]["auroc_mover_cluster_ci95"]
c2 = cluster["months"]["2016_03"]["auroc_mover_cluster_ci95"]
check("clustered inference: mover-cluster CIs [0.615,0.829]/[0.613,0.834]",
      close(c1[0], 0.615, 0.001) and close(c1[1], 0.829, 0.001)
      and close(c2[0], 0.613, 0.001) and close(c2[1], 0.834, 0.001)
      and c1[0] > 0.5 and c2[0] > 0.5,
      f"observed [{c1[0]:.3f},{c1[1]:.3f}] [{c2[0]:.3f},{c2[1]:.3f}]")

lb = json.loads((OUTPUTS / "learned_basin_clbf.json").read_text())
lbo = lb["registered_outcomes"]
check("learned basins: 60/60, 14/15, 74/75, 15/15 identifiable",
      lbo["LB1_controls_rejected"] == "60/60"
      and lbo["LB2_learned_accepted"] == "14/15"
      and lbo["LB3_agreement_with_hand_basins"] == "74/75"
      and lbo["learned_partitions_identifiable"] == "15/15",
      f"observed {lbo}")

ir = json.loads((OUTPUTS / "independent_rollout_audit.json").read_text())
iro = ir["registered_outcomes"]
check("rollout models: IR-1 15/15, IR-2 miss 8/15, IR-3 15/15",
      iro["IR1_potential_survives_greedy"] == "15/15"
      and iro["IR2_learned_accepted_both_models"] == "8/15"
      and not iro["IR2_pass"]
      and iro["IR3_twins_rejected_both_models"] == "15/15",
      f"observed {iro}")

toga = json.loads((OUTPUTS / "chess_discovery_toga_referee.json").read_text())
t1 = toga["months"]["main"]["frozen_rule"]
t2 = toga["months"]["replication_2016_03"]["frozen_rule"]
check("Toga referee: TG-1 retained miss (0.663/0.534, base 0.0075/0.0125)",
      not toga["registered_outcomes"]["TG1_auroc_gt_0.60_both_months"]
      and close(t1["auroc_do_gap_vs_toga"], 0.663, 0.001)
      and close(t2["auroc_do_gap_vs_toga"], 0.534, 0.001)
      and close(t1["toga_base_rate"], 0.0075, 1e-4)
      and close(t2["toga_base_rate"], 0.0125, 1e-4),
      f"observed {t1['auroc_do_gap_vs_toga']:.3f}, "
      f"{t2['auroc_do_gap_vs_toga']:.3f}")


xfit = json.loads((OUTPUTS / "crossfit_lowlevel_basins.json").read_text())
xo = xfit["registered_outcomes"]
check("cross-fit low-level basins: XF1-3 pass, mean agreement 0.957",
      xo["XF1_pass"] and xo["XF2_pass"] and xo["XF3_pass"]
      and close(xo["XF3_mean_agreement"], 0.9567, 0.001),
      f"mean agreement {xo['XF3_mean_agreement']:.4f}")

he = json.loads((OUTPUTS / "learned_harmful_emergence.json").read_text())
heo = he["registered_outcomes"]
check("learned harmful emergence: HE1-4 pass 5/5",
      all(heo[f"HE{i}_pass"] for i in (1, 2, 3, 4)),
      f"{ {k: v for k, v in heo.items() if not k.endswith('_pass')} }")
u_priv = [m["usefulness_private"] for m in he["seeds"].values()]
u_team = [m["usefulness_team"] for m in he["seeds"].values()]
check("harmful emergence: U_private +7.5, U_team -2.0",
      all(close(u, 7.5, 0.2) for u in u_priv)
      and all(close(u, -2.0, 0.2) for u in u_team),
      f"priv {u_priv[0]:.2f}, team {u_team[0]:.2f}")

mp = json.loads((OUTPUTS / "matched_provenance.json").read_text())
mpo = mp["registered_outcomes"]
check("matched provenance: MP1-5 all pass",
      all(mpo[f"MP{i}_" + k] for i, k in (
          (1, "behaviour_matched"), (2, "structural_magnitude_matched"),
          (3, "acquisition_boundary"), (4, "provenance_rarity_ordering"),
          (5, "clone_counterfactually_distinguishable"))),
      f"{mpo}")
cp = mp["seed_mean_c_prov_bits"]
check("matched provenance: rarity 0.33 < 0.39 < 0.65",
      close(cp["bc_clone"], 0.327, 0.005)
      and close(cp["shaped"], 0.391, 0.005)
      and close(cp["outcome_only"], 0.648, 0.005),
      f"{cp}")

p2d = json.loads((OUTPUTS / "phase_2d_prediction.json").read_text())
po = p2d["registered_outcomes"]
check("2-D phase surface: 14/15 non-fragile match, non-rectangular",
      po["P2D1_nonfragile_match"] == "14/15" and not po["P2D1_pass"]
      and po["P2D2_nonrectangular"],
      f"{po}")

ce = json.loads((OUTPUTS / "contract_ensemble_analysis.json").read_text())
head = ce["headline"]
check("contract ensemble: 420/420 control rejections, 10/15 learned >= 6/7",
      head["controls_contract_invariant_negative"].startswith("420/420")
      and head["learned_at_least_6_of_7"] == 10
      and close(head["learned_mean_R"], 0.8667, 0.001),
      f"mean R {head['learned_mean_R']:.4f}")

wm = json.loads((OUTPUTS / "world_model_closure.json").read_text())
check("world-model closure: WM-1/2 pass, WM-3 retained miss",
      wm["registered_outcomes"]["WM1_calibration_monotone"]
      and wm["registered_outcomes"]["WM2_no_silent_wrong_verdict_at_K20000"]
      and not wm["registered_outcomes"]["WM3_margin_rule_catches_all_mismatches"],
      json.dumps({k: v for k, v in wm["registered_outcomes"].items()
                  if k.startswith("WM")}))
wmf = json.loads((OUTPUTS / "world_model_closure_followup.json").read_text())
check("world-model follow-up: 20/20 mismatches caught, abstains at K=20000",
      wmf["F1_all_mismatches_caught"]
      and not wmf["F2_no_abstention_at_K20000"]
      and wmf["mismatches_caught_by_augmented_rule"] == 20,
      f"caught {wmf['mismatches_caught_by_augmented_rule']}/"
      f"{wmf['mismatches_total']}")

br = json.loads((OUTPUTS / "bridge_identity_verification.json").read_text())
check("bridge identity: max gap < 1e-15 over 10k systems; det boundary 0",
      br["pass"]
      and br["random_systems"]["max_identity_gap"] < 1e-15
      and br["deterministic_boundary"][
          "max_action_attributable_collapse"] == 0.0,
      f"gap {br['random_systems']['max_identity_gap']:.2e}")

cal = json.loads((OUTPUTS / "profile_calibration.json").read_text())
dose = cal["part2_dose_response"]
mono = all(
    all(x <= y + 1e-12 for x, y in
        zip(s["M_vs_forced_trigger"], s["M_vs_forced_trigger"][1:]))
    for s in dose["seeds"].values())
check("profile calibration: dose-response monotone on 5/5 seeds",
      mono and len(dose["seeds"]) == 5,
      f"{len(dose['seeds'])} seeds")
prof = json.loads((OUTPUTS / "profile_existing_systems.json").read_text())
check("taxonomy profiles: 5/5 axis-separation checks",
      all(prof["axis_separation_check"].values()),
      json.dumps(prof["axis_separation_check"]))

rs = json.loads((OUTPUTS / "contract_ranking_stability.json").read_text())
check("ranking stability: mean rho 0.76, RS-2 retained miss, RS-2b 15/15",
      close(rs["mean_pairwise_spearman"], 0.759, 0.005)
      and rs["declared_outcomes"]["RS1_mean_spearman_ge_0.5"]
      and not rs["declared_outcomes"][
          "RS2_class_separation_every_contract"]
      and rs["declared_outcomes"][
          "RS2b_followup_E_adapt_positive_learned"].startswith("15/15"),
      f"rho {rs['mean_pairwise_spearman']:.3f}")

pv = json.loads((OUTPUTS / "predictive_validity.json").read_text())
check("predictive validity: PV-1..3 retained misses; U_early 0.81",
      not any(pv["registered_outcomes"].values())
      and close(pv["auroc_M_early"], 0.626, 0.005)
      and close(pv["auroc_perf_early"], 0.783, 0.005)
      and close(pv["auroc_U_early"], 0.808, 0.005),
      f"M {pv['auroc_M_early']:.3f} perf {pv['auroc_perf_early']:.3f} "
      f"U {pv['auroc_U_early']:.3f}")

gc = json.loads((OUTPUTS / "generator_calibration.json").read_text())
gco = gc["registered_outcomes"]
jm = gc["sensitivity_matrix_range"]
check("generator calibration: GC-1 dominance, GC-2 retained miss, GC-3..5 pass",
      gco["GC1_diagonal_dominance"] and not gco["GC2_offdiagonal_lt_quarter"]
      and gco["GC3_nullity"] and gco["GC4_value_separability"]
      and gco["GC5_provenance_separability"]
      and close(jm["V"]["v"], 1.52, 0.02) and close(jm["S"]["s"], 0.99, 0.02)
      and close(jm["Q_rel"]["q"], 0.985, 0.02),
      f"J[V][v]={jm['V']['v']:.3f} J[S][s]={jm['S']['s']:.3f}")

ax = json.loads((OUTPUTS / "record_axioms_verification.json").read_text())
check("record axioms A1-A8 all pass", ax["all_pass"],
      ", ".join(k for k, v in ax["results"].items() if v["pass"]))

cv = json.loads((OUTPUTS / "convergent_validity.json").read_text())
cvo = cv["registered_outcomes"]
check("convergent validity: CV-1 pass (0.56 vs -0.09), CV-2/3 retained",
      cvo["CV1_M_matched_beats_crossed"]
      and not cvo["CV2_U_matched_beats_crossed"]
      and not cvo["CV3_record_adds_loo_r2"]
      and close(cv["spearman"]["M_early->M_final"], 0.561, 0.005)
      and close(cv["spearman"]["perf_early->M_final"], -0.089, 0.005),
      f"rho matched {cv['spearman']['M_early->M_final']:.3f}")

oc = json.loads((OUTPUTS / "overcooked_confirmation_pooled.json").read_text())
check("Overcooked: all five registered predictions passed",
      all(oc["registered_outcomes"].values()),
      json.dumps(oc["registered_outcomes"]))
check("Overcooked: learned 8/12, controls 48/48, useful+ 12/12 p=2.4e-4",
      oc["learned_accepted"] == 8 and oc["n_seeds"] == 12
      and oc["control_rejections"] == "48/48"
      and oc["learned_usefulness_positive"] == "12/12"
      and close(oc["oc5_sign_test_p"], 2.44e-4, 1e-5),
      f"accepted {oc['learned_accepted']}/{oc['n_seeds']}; "
      f"controls {oc['control_rejections']}; "
      f"p {oc['oc5_sign_test_p']:.2e}")
op = json.loads((OUTPUTS / "overcooked_profiles.json").read_text())
opo = op["declared_outcomes"]
check("Overcooked profiles: OP-1/2/3 hold (0.653 max rej < 0.695 min acc)",
      opo["OP1_accepted_learned_E_adapt_positive"].endswith("True")
      and opo["OP2_all_controls_E_adapt_zero"] is True
      and opo["OP3_rejected_below_accepted_E_struct"].startswith("True")
      and "0.653" in opo["OP3_rejected_below_accepted_E_struct"]
      and "0.695" in opo["OP3_rejected_below_accepted_E_struct"],
      opo["OP3_rejected_below_accepted_E_struct"])

oc_routes = oc["control_failure_routes"]
check("Overcooked: scripted/clone fail endogeneity+acquisition 12/12",
      all(oc_routes[c]["endogeneity"] == 12
          and oc_routes[c]["acquisition"] == 12
          for c in ("scripted_roles", "bc_clone")),
      json.dumps({c: oc_routes[c] for c in ("scripted_roles", "bc_clone")}))

cr = json.loads((OUTPUTS / "crowd_vote_domain.json").read_text())
cro = cr["registered_outcomes"]
check("crowd domain: CR-1 7/10 retained miss, CR-2 50/50, CR-3 10/10, "
      "CR-4 10/10, CR-5 retained miss",
      cro["CR1_learned_ge_8_of_10"].startswith("7/10")
      and cro["CR2_all_50_controls_rejected"].startswith("50/50")
      and cro["CR3_useful_positive_ge_9"].startswith("10/10")
      and cro["CR4_doblock_fall_shift_ge_0.3_all"].startswith("10/10")
      and cro["CR5_field_anarchy_ge_0.7_all"].startswith("7/10"),
      json.dumps(cro))
crowd_routes_ok = all(
    set(s["scripted_switcher"]["verdict"]["failed"])
    == {"endogeneity", "acquisition"}
    and set(s["bc_clone"]["verdict"]["failed"])
    == {"endogeneity", "acquisition"}
    for s in cr["seeds"].values())
check("crowd domain: switcher/clone fail exactly endogeneity+acquisition",
      crowd_routes_ok, "10/10 seeds each")

bf = json.loads((OUTPUTS / "convention_bifurcation.json").read_text())
bfo = bf["registered_outcomes"]
fr = bfo["selective_fractions_by_cost"]
check("bifurcation: BF-1/2 pass, BF-3 4/5, BF-4 retained miss (0.50)",
      bfo["BF1_monotone_nondecreasing"] and bfo["BF2_endpoints"]
      and bfo["BF3_gap_sign_predicts_majority"].startswith("4/5")
      and bfo["BF4_early_snapshot_predicts_basin"].startswith("0.50")
      and close(fr["1.0"], 0.1) and close(fr["3.0"], 0.9),
      json.dumps(fr))
gap15 = bf["costs"]["1.5"][
    "reference_value_gap_selective_minus_blanket"]
check("bifurcation: transition-region gap +0.12 at d=1.5",
      close(gap15, 0.12, 0.01), f"{gap15:+.3f}")

mc = json.loads((OUTPUTS / "meta_collapse_commitment.json").read_text())
mco = mc["registered_outcomes"]
check("meta-collapse: MC-2/3/4 retained misses; stable 26/22/2 split",
      mco["MC2_iqr_ge_0.20"].endswith("False")
      and mco["MC3_blind_fewer_anarchy_field_successes"] == "False"
      and mco["MC4_entropy_decline_mostly_after_25pct"].endswith("False")
      and mc["final_classes"]["selective"] == 26
      and mc["final_classes"]["blind_dem"] == 22,
      json.dumps(mc["final_classes"]))

mf = json.loads((OUTPUTS / "meta_collapse_margins.json").read_text())
mfo = mf["declared_outcomes"]
check("meta-collapse margins: F-1 retained miss, F-3 24/24 field-state",
      mfo["F1_moat_after_last_flip"].startswith("0.00")
      and mf["f3_far_margin_separates"].startswith("24/24")
      and 0.8 <= mf["f2_soft_commit_median"] <= 0.85
      and mf["f2_soft_commit_iqr"] >= 0.15,
      mf["f3_far_margin_separates"])

sp = json.loads((OUTPUTS / "spatial_bridge_verification.json").read_text())
check("spatial bridge: S-A..S-D all pass (TC identity, N-growth, "
      "monotone coupling, provenance-blind max)",
      sp["all_pass"]
      and sp["results"]["SA_identity"]["max_gap_tc_vs_kl"] < 1e-12
      and abs(sp["results"]["SD_provenance_blind"]["tc_script"]
              - sp["results"]["SD_provenance_blind"]["max_theory"]) < 1e-9,
      f"max TC gap {sp['results']['SA_identity']['max_gap_tc_vs_kl']:.1e}")

ce = json.loads((OUTPUTS / "canonical_exemplars.json").read_text())
ceo = ce["registered_outcomes"]
check("canonical exemplars: CE-1..5 all pass",
      all(ceo.values()), json.dumps(ceo))
check("exemplars numbers: contraction 0->1.64, TC3 0.04->2.70, "
      "Schelling JS 0.33 at 0.7, Life deterministic",
      close(ce["boids"]["1.0"]["mean_heading_entropy_contraction_bits"],
            1.64, 0.02)
      and close(ce["boids"]["1.0"]["tc3_bits"], 2.70, 0.02)
      and close(ce["boids"]["0.0"]["tc3_bits"], 0.04, 0.02)
      and close(ce["schelling"]["0.7"]["js_do_freeze_bits"], 0.331, 0.005)
      and ce["life"]["deterministic_replays_distinct_finals"] == 1,
      f"TC3 max {ce['boids']['1.0']['tc3_bits']:.3f}")

uo = json.loads((OUTPUTS / "universal_observer.json").read_text())
uoo = uo["registered_outcomes"]
check("universal observer: U-1 8/9 (positives accepted), U-2 18/18 + "
      "15/15, U-3 identical code",
      uoo["U1_battery"].startswith("8/9")
      and uoo["U1_battery"].endswith("True")
      and uoo["U2_crowd"].endswith("True")
      and uoo["U3_no_per_domain_tuning"] is True
      and uo["battery"]["rows"]["latent_conditional"][
          "universal_verdict"] == 1
      and uo["battery"]["rows"]["noise_policy"][
          "universal_verdict"] == 1,
      json.dumps({k: v for k, v in uoo.items()}))

ou = json.loads((OUTPUTS / "overcooked_universal_observer.json").read_text())
ouo = ou["registered_outcomes"]
check("Overcooked universal: U-4a 19/20, U-4b 16/16, U-4c retained miss",
      ouo["U4a_agreement"].startswith("19/20")
      and ouo["U4b_controls_rejected"].startswith("16/16")
      and ouo["U4c_learned_pattern_preserved"] is False,
      json.dumps(ouo))

ld = json.loads((OUTPUTS / "live_demonstration.json").read_text())
ldo = ld["registered_outcomes"]
rho = ld["part1_blind_accuracy"]["rho_matched"]
check("live demo: LD-1 blind recovery >= 0.95 all dims, LD-2 retained "
      "miss, LD-3 verdicts correct",
      ldo["LD1_matched_spearman_ge_0.9"]
      and not ldo["LD2_cross_leakage_le_0.35"]
      and ldo["LD3_verdicts_match_known_identities"]
      and all(v >= 0.95 for v in rho.values()),
      json.dumps({k: round(v, 3) for k, v in rho.items()}))

pe = json.loads((OUTPUTS / "emergence_promoting_selection.json").read_text())
peo = pe["registered_outcomes"]
check("emergence-promoting selection: PE-1..3 all pass (0.78/0.39/0.53)",
      all(v.endswith("True") for v in peo.values())
      and close(pe["arms"]["E"]["selective_fraction_mean"], 0.775, 0.005)
      and close(pe["arms"]["N"]["selective_fraction_mean"], 0.3875, 0.005)
      and close(pe["arms"]["V"]["selective_fraction_mean"], 0.525, 0.005),
      json.dumps(peo))

ec = json.loads((OUTPUTS / "emergence_coordinates.json").read_text())
eco = ec["registered_outcomes"]
ng = ec["calibration"]["N_gates"]
check("emergence coordinates: EC-1/2/3 all pass (final run)",
      eco["EC1_calibration_recovery"] is True
      and eco["EC2_blind_lattice"].startswith("4/4")
      and eco["EC3_adversarial"].startswith("8/8")
      and ng["xor2"]["abs_err"] < 0.01
      and close(ng["majority3"]["analytic"], 0.4338, 0.001)
      and close(ec["calibration"]["A_chains"]["A_pos"]["A_exact"],
                0.189, 0.001),
      json.dumps(eco))

ac = json.loads((OUTPUTS / "ant_contrast.json").read_text())
aco = ac["registered_outcomes"]
check("ant double-bridge: ANT-1..3 all pass "
      "(solo not collective / trail weak / gradual collapse)",
      all(aco.values())
      and ac["SOLO"]["D"] < 0.5
      and ac["SOLO"]["consolidation_rate"] < 0.3
      and ac["TRAIL"]["D"] >= 0.5
      and ac["TRAIL"]["R"] >= 0.6
      and ac["TRAIL"]["gradualism"]["total_collapse"] >= 0.5
      and ac["TRAIL"]["gradualism"]["span_frac"] >= 0.10,
      json.dumps(aco))

ir = json.loads((OUTPUTS / "emergence_irreducibility.json").read_text())
iro = ir["registered_outcomes"]
irs = ir["systems"]
check("irreducibility: IR-1..6 all pass "
      "(no-go real / defeated / pairwise-reducible / role-lock strong / "
      "magnitude!=strength / functional!=pathological)",
      all(iro.values())
      # env_synergy distributionally identical to role_parity yet rejected
      and close(irs["env_synergy"]["C_irr_marginal"], 1.0, 0.01)
      and close(irs["env_synergy"]["O_information"], -1.0, 0.01)
      and irs["env_synergy"]["C_irr_given_env"] < 0.05
      and irs["env_synergy"]["D_higher"] < 0.5
      and close(irs["role_parity"]["C_irr_given_env"], 1.0, 0.01)
      and close(irs["role_parity"]["D_higher"], 1.0, 0.01)
      and irs["role_parity"]["strong_emergence"] is True
      and irs["consensus"]["C_total"] > irs["role_parity"]["C_total"]
      and ir["magnitude_vs_strength"]["spearman"] <= 0.0
      and ir["gaussian_common_cause"]["O_information_bits"] > 0,
      json.dumps(iro))

cc = json.loads((OUTPUTS / "collective_constraint.json").read_text())
cco = cc["registered_outcomes"]
ccm = cc["mechanisms"]
check("collective constraint: CC-1..5 all pass "
      "(matched confound / endogeneity separates / micro-down macro-up / "
      "persistence / four-quadrant only local feedback)",
      all(cco.values())
      and cc["matched_confound"]["joint_distributions_identical"] is True
      and ccm["local_feedback"]["accept"] is True
      and ccm["central_script"]["accept"] is False
      and ccm["common_cause"]["accept"] is False
      and ccm["independent_coincidence"]["accept"] is False
      and close(ccm["local_feedback"]["C_constraint"], 1.585, 0.01)
      and close(ccm["local_feedback"]["M_endogenous_macro_gain"], 0.667, 0.01)
      and ccm["central_script"]["C_constraint"] < 0.01
      and ccm["independent_coincidence"]["R_persistence"] < 0.6,
      json.dumps(cco))

occ = json.loads((OUTPUTS / "overcooked_collective_constraint.json").read_text())
occo = occ["registered_outcomes"]
occ_summary = occ["summary"]
check("Overcooked collective-constraint bridge: OCC-1..5 "
      "(real stored roles + retained full-certificate limitation)",
      all(occo.values())
      and occ_summary["accepted_CG_positive"] == "8/8"
      and occ_summary["learned_M_positive"] == "12/12"
      and occ_summary["accepted_contractB_persistent"] == "8/8"
      and occ_summary["control_C_zero"]["scripted_roles"] == "12/12"
      and occ_summary["control_C_zero"]["bc_clone"] == "12/12"
      and occ["limitations"]["full_interaction_broken_certificate_available"]
      is False,
      json.dumps(occo))

otc_s = json.loads((OUTPUTS / "overcooked_transition_certificate_smoke_scripted.json").read_text())
otc_i = json.loads((OUTPUTS / "overcooked_transition_certificate_smoke_initial.json").read_text())
check("Overcooked transition scaffold: real-vs-ghost smoke passes, no learned claim",
      all(otc_s["registered_smoke_outcomes"].values())
      and all(otc_i["registered_smoke_outcomes"].values())
      and otc_s["policy"]["learned_checkpoint_supplied"] is False
      and otc_i["policy"]["learned_checkpoint_supplied"] is False
      and close(sum(otc_s["overall"]["P_real"].values()), 1.0, 1e-9)
      and close(sum(otc_s["overall"]["P_cut"].values()), 1.0, 1e-9)
      and otc_s["overall"]["G_js_bits"] >= 0.0
      and otc_i["overall"]["G_js_bits"] >= 0.0
      and "partner_action_tv" in otc_s["overall"]
      and "partner_action_tv" in otc_i["overall"],
      f"scripted G {otc_s['overall']['G_js_bits']:.3f}; "
      f"initial G {otc_i['overall']['G_js_bits']:.3f}")

otc_a = json.loads((OUTPUTS / "overcooked_transition_pilot_audit.json").read_text())
rows = {r["label"]: r for r in otc_a["rows"] if r["available"]}
check("Overcooked learned transition pilots audited: 2M pilot positive",
      otc_a["summary"]["learned_available"] >= 2
      and otc_a["summary"]["any_learned_positive_M"] is True
      and rows["learned_40k"]["M_score_gain"] == 0.0
      and rows["learned_500k"]["M_score_gain"] < 0.0
      and rows["learned_2m"]["M_score_gain"] > 10.0
      and rows["learned_2m"]["G_js_bits"] > rows["learned_500k"]["G_js_bits"]
      and rows["learned_500k"]["policy"]["learned_checkpoint_supplied"]
      is True
      and rows["learned_2m"]["policy"]["learned_checkpoint_supplied"]
      is True
      and rows["learned_2m"]["partner_action_tv"] < 0.05,
      f"learned max G {otc_a['summary']['max_learned_G']:.3f}; "
      f"2M M {rows['learned_2m']['M_score_gain']:.3f}")

cpc = json.loads((OUTPUTS / "canonical_possibility_collapse.json").read_text())
cpco = cpc["registered_outcomes"]
check("canonical possibility-collapse validation: CPC-1..6 all pass "
      "(classic positives / pseudo negatives / capability cases / "
      "Overcooked bridge / complete matrix / provenance tiers)",
      all(cpco.values())
      and cpc["counts"]["n_rows"] == 19
      and cpc["counts"]["matched"] == 19
      and any(r["name"] == "Induction head 2-layer"
              and r["verdict_matches_ground_truth"] for r in cpc["rows"])
      and any(r["name"] == "Game-of-Life glider"
              and r["expected_status"].startswith("weak emergence")
              for r in cpc["rows"])
      and any(r["name"] == "Overcooked learned accepted seeds"
              and "limitation" in r for r in cpc["rows"]),
      json.dumps(cpco))

# report
failures = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")
print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
if failures:
    raise SystemExit(1)
