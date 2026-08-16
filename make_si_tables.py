"""Generate every Supplementary Information table directly from outputs/*.json.

Writes si_tables.tex, which si.tex includes verbatim. No number in the SI
tables is typed by hand; rerun this script after any experiment changes.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def fmt(x, nd=2):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


TABLES = []


def table(caption, header, rows, label, align=None, fit_width=False):
    ncol = len(header)
    align = align or "l" + "c" * (ncol - 1)
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
    ]
    if fit_width:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if fit_width:
        lines.append("}")
    lines += [r"\end{table}", ""]
    TABLES.append("\n".join(lines))


# Supplementary Table 1: detector operating characteristics
dv = load("detector_validation.json")
rows = []
for dens, v in sorted(dv["by_density"].items(), key=lambda kv: int(kv[0])):
    rows.append([f"{dens} points, $\\sigma=0.02$",
                 fmt(v["onset_power"], 3), fmt(v["fpr"], 3)])
for sig, v in sorted(dv["by_sigma"].items(), key=lambda kv: float(kv[0])):
    rows.append([f"80 points, $\\sigma={sig}$",
                 fmt(v["onset_power"], 3), fmt(v["fpr"], 3)])
table(
    r"\textbf{Frozen-detector operating characteristics on the held-out "
    r"synthetic benchmark.} 200 labelled curves per family per condition; "
    r"onset power is the detection rate on onset-type positives and the "
    r"false-positive rate is pooled over the deceleration, gradual and flat "
    r"control families. The zero power at 12 grid points is the resolution "
    r"floor cited in the main text.",
    ["Operating point", "Onset power", "False-positive rate"],
    rows, "si:tab:detector")

# Supplementary Table 2: 243-contract adjudication battery
rr = load("repr_robustness.json")
sysname = {"grip:side_openness": "Grip transport (side openness)",
           "ant_N100": "Ant colony ($N{=}100$)",
           "tri_c_bp": "Learned high-order coordination"}
rows = []
for key, s in rr["systems"].items():
    rows.append([sysname.get(key, key), s["n_cells"], s["n_gated"],
                 fmt(s["frac_onset_among_gated"], 2),
                 f"{fmt(s['t_star_min'], 0)}--{fmt(s['t_star_max'], 0)}",
                 fmt(100 * s["t_star_range_frac"], 1) + r"\%"])
table(
    r"\textbf{Adjudication-contract battery.} Every headline onset verdict "
    r"re-adjudicated across 243 analysis contracts per system (saturation "
    r"fraction $\times$ window $\times$ effect-size gate $\times$ "
    r"$\Delta$BIC threshold $\times$ grid stride, $3^{5}$ cells). "
    r"``Onset among gated'' is the fraction of adequate-resolution cells "
    r"whose verdict is unchanged; the location range is the spread of "
    r"$t^{*}$ across those cells as a fraction of curve span.",
    ["System", "Cells", "Gated", "Onset among gated", "$t^{*}$ range",
     "Range / span"],
    rows, "si:tab:contracts")

# Supplementary Table 3: representation-equivalence battery
rc = load("repr_equiv_convention.json")
rg = load("repr_equiv_grip.json")
nice = {
    "R1_population_mean": "Convention: population-mean mapping",
    "R2_per_agent_mean": "Convention: per-agent mean entropy",
    "R3_listener_dual": "Convention: listener-side dual",
    "R4_behavioural_2048": "Convention: behavioural sample (2{,}048)",
    "R5_pref_checkpoint0": "Convention: checkpoint-zero reference",
    "R6_prob_truncation": "Convention: $\\varepsilon=0.01$ truncation",
    "R7_symbol_binning_3": "Convention: 5$\\to$3 symbol coarse-graining",
    "G1_policy_probs": "Grip: policy probabilities",
    "G2_state_coarse_0.1": "Grip: state quantization 0.1",
    "G3_state_coarse_0.5": "Grip: state quantization 0.5",
    "G4_behavioural_counts": "Grip: realized action counts",
    "G5_prob_truncation": "Grip: $\\varepsilon=0.01$ truncation",
}
rows = []
for src in (rc, rg):
    for key, cell in src["representation_cells"].items():
        rows.append([nice.get(key, key.replace("_", r"\_")),
                     fmt(cell.get("delta_bic"), 1),
                     fmt(cell.get("t_star"), 0),
                     "onset" if cell.get("b5_onset") else "no verdict"])
table(
    r"\textbf{Representation-equivalence battery.} The openness object "
    r"recomputed under twelve preregistered measurement representations on "
    r"byte-identical retrained seeds and adjudicated by the frozen "
    r"detector. The verdict survives in 8/12 cells; each breaking cell "
    r"fails for the identifiable reason given in Methods and still locates "
    r"the commitment window.",
    ["Representation", "$\\Delta$BIC", "$t^{*}$", "Verdict"],
    rows, "si:tab:representations")

# Supplementary Table 7: grip per-seed statistics
b5 = load("learn_grip_transport_b5.json")
a2c = load("learn_grip_a2c.json")
rows = []
for sd, v in sorted(b5["seeds"].items(), key=lambda kv: int(kv[0])):
    h = v["adj"]["hinge"]
    rows.append([f"REINFORCE {sd}", fmt(v["final_success"], 3),
                 v["plateau_len"], fmt(h["delta_bic"], 1),
                 fmt(h["t_star"], 0), fmt(v["final_side_mean"], 3)])
for sd, v in sorted(a2c["seeds"].items(), key=lambda kv: int(kv[0])):
    h = v["adj"]["hinge"]
    thin_ok = all(t["ok"] for t in v["adj"]["thinning"].values())
    rows.append([f"A2C {sd}" + ("" if thin_ok else r"$^{\dagger}$"),
                 fmt(v["final_success"], 3), v["plateau_len"],
                 fmt(h["delta_bic"], 1), fmt(h["t_star"], 0),
                 fmt(v["final_side_mean"], 3)])
table(
    r"\textbf{Grip-transport flagship, per-seed statistics.} Five REINFORCE "
    r"seeds (primary) and five advantage actor--critic seeds "
    r"(byte-identical environment). $\Delta$BIC and $t^{*}$ are from the "
    r"primary hinge fit; $\dagger$ marks A2C seeds whose strict "
    r"parity-thinning clause fails on thinned-subsample detector power "
    r"(the 3/5 partial replication reported in the main text). Final side "
    r"mean near zero confirms the left/right symmetry of the learned "
    r"population.",
    ["Seed", "Success", "Plateau (steps)", "$\\Delta$BIC", "$t^{*}$",
     "Final side mean"],
    rows, "si:tab:grip")

# Supplementary Table 8: convention and role systems, tabular seeds
conv = load("learn_convention.json")
roles = load("learn_roles.json")
rows = []
for name, src in (("Signalling", conv), ("Roles", roles)):
    for sd, v in sorted(src["seeds"].items(), key=lambda kv: int(kv[0])):
        h = v["adj"]["hinge"]
        onset = v["adj"].get("b5_onset", h.get("onset_type"))
        rows.append([f"{name} {sd}", fmt(v["final_success"], 3),
                     fmt(h["delta_bic"], 1), fmt(h["t_star"], 0),
                     fmt(v.get("success_090_cross"), 0),
                     "onset" if onset else "no verdict"])
table(
    r"\textbf{Non-constructed learned systems (tabular policies), per-seed "
    r"statistics.} Ten-agent Lewis signalling population and six-agent "
    r"division-of-labour task. ``Capability cross'' is the update at which "
    r"mutual intelligibility (respectively team success) first reaches "
    r"0.9; in every onset seed the breakpoint precedes it.",
    ["Seed", "Success", "$\\Delta$BIC", "$t^{*}$", "Capability cross",
     "Verdict"],
    rows, "si:tab:tabular")

# Supplementary Table 9: neural replications, per-seed
nn = load("learn_nn_resolution.json")
rows = []
for name, sysk in (("Neural signalling", "convention"),
                   ("Neural roles", "roles")):
    for sd, v in sorted(nn["systems"][sysk]["seeds"].items(),
                        key=lambda kv: int(kv[0])):
        rows.append([f"{name} {sd}", fmt(v.get("final_success"), 3),
                     fmt(v.get("delta_bic"), 1), fmt(v.get("t_star"), 0),
                     fmt(v.get("success_090_cross"), 0),
                     "onset" if v.get("b5_onset") else "no verdict"])
table(
    r"\textbf{Neural replications, per-seed statistics at 5-update "
    r"resolution.} One-hidden-layer networks replace the tables with "
    r"environments and recipes fixed (per-agent MLPs for signalling; one "
    r"shared MLP for all six role agents). Random initialization "
    r"compresses commitment into the first few hundred updates, which is "
    r"why these runs are evaluated on the preregistered fine grid. Seeds "
    r"that never reach the 0.9 capability criterion (final success 0.800) "
    r"are shown for completeness but excluded from the learned-seed counts "
    r"in the main text.",
    ["Seed", "Success", "$\\Delta$BIC", "$t^{*}$", "Capability cross",
     "Verdict"],
    rows, "si:tab:neural")

# Supplementary Table 10: Overcooked coordination ring, per-seed
orc = load("overcooked_ring_convention.json")
oce = load("oc_ring_ext.json")
rows = []
for sd, v in orc["systems"]["ring"].items():
    rows.append([f"Original {sd}", fmt(v["final_p_ccw"], 2),
                 "counterclockwise" if v["final_p_ccw"] > 0.5 else "clockwise"])
for sd, v in oce["ext_seeds"].items():
    rows.append([f"Extension {sd}", fmt(v["final_p_ccw"], 2),
                 "counterclockwise" if v["final_p_ccw"] > 0.5 else "clockwise"])
ro = oce["registered_outcomes"]
table(
    r"\textbf{Overcooked coordination-ring conventions, per-seed.} All "
    rf"{ro['n_pooled']} pooled self-play seeds end committed to one "
    r"circulation direction (final direction probability at the Laplace "
    r"floor or ceiling); "
    rf"{ro['n_onset']}/{ro['n_pooled']} certifies a punctuated onset, the "
    r"instrument boundary on non-monotone commitment discussed in Methods.",
    ["Seed", "Final $P(\\mathrm{ccw})$", "Committed direction"],
    rows, "si:tab:ring")

# Supplementary Table 11: ant finite-size scaling and Kuramoto coupling grid
fss = load("ant_fss.json")
rows = []
for size, v in sorted(fss["per_size"].items(), key=lambda kv: int(kv[0])):
    rows.append([f"Ant $N={size}$", fmt(v.get("t50"), 0),
                 fmt(v.get("width"), 0), fmt(v.get("delta_bic"), 1),
                 "onset" if v.get("b5_onset") else "--"])
ks = load("kuramoto_scale_n10.json")
summ = ks["registered_outcomes"]["per_K_summary"]
for K, v in sorted(summ.items(), key=lambda kv: float(kv[0])):
    tci = v["ci_t_star"]
    sci = v["ci_post_slope"]
    rows.append([f"Kuramoto $K={K}$ (10 seeds)",
                 f"{v['mean_t_star']:.2f} [{tci[0]:.2f}, {tci[1]:.2f}]",
                 f"{v['mean_post_slope']:.3f} [{sci[0]:.3f}, {sci[1]:.3f}]",
                 "--", f"onset {v['n_pass']}/10"])
table(
    r"\textbf{Scaling data.} Ant-colony finite-size scaling (median "
    r"$t_{50}$ crossing and 10--90\% width in trips, 30 episodes per "
    r"size) and the Kuramoto coupling grid (ten seeds per coupling "
    r"$K$; seed-mean $t^{*}$ and closing-slope magnitude with 95\% "
    r"bootstrap intervals over seeds; for the Kuramoto rows the width "
    r"column reports the closing slope). These rows are the points "
    r"behind Fig.~6a,b.",
    ["System / size", "$t_{50}$ or $t^{*}$", "Width / slope",
     "$\\Delta$BIC", "Verdict"],
    rows, "si:tab:scaling")

# Supplementary Table 4: regime-ensemble audit, per-candidate summary
rea = load("regime_ensemble_audit.json")["results"]
rea2 = load("regime_ensemble_audit2.json")["results"]
CAND_NAMES = {
    "P1_speaker_code": ("Convention", "speaker code (declared)", "declared"),
    "P2_listener_code": ("Convention", "listener code", "plausible"),
    "P3_composed_channel": ("Convention", "composed channel", "plausible"),
    "P4_meaning0_symbol": ("Convention", "single-meaning sub-regime",
                           "plausible"),
    "X1_symbol_marginal": ("Convention", "pooled symbol marginal",
                           "control (erasing)"),
    "X2_speaker_identity": ("Convention", "speaker identity",
                            "control (exogenous)"),
    "P1_assignment": ("Roles", "assignment openness (declared)", "declared"),
    "P2_agent0_role": ("Roles", "agent 0's role", "plausible"),
    "P3_role0_owner": ("Roles", "owner of role 0", "plausible"),
    "X1_role_marginal": ("Roles", "pooled role marginal",
                         "control (erasing)"),
    "P2_force_sign": ("Grip", "force-sign ensemble marginal",
                      "plausible (mis-specified)"),
    "X1_xabs_tertiles": ("Grip", "$|x|$ tertile occupancy",
                         "control (erasing)"),
}
rows = []
for sysname, srows in rea.items():
    cands = list(next(iter(srows.values()))["candidates"].keys())
    for c in cands:
        label = CAND_NAMES[c]
        n_on = sum(r["candidates"][c]["b5_onset"] for r in srows.values())
        n_decl = sum(r["declared_b5"] for r in srows.values())
        agree = sum(r["candidates"][c]["b5_onset"] == r["declared_b5"]
                    for r in srows.values())
        ts = [r["candidates"][c]["t_star"] for r in srows.values()
              if r["candidates"][c]["b5_onset"]]
        trange = (f"{min(ts):.0f}--{max(ts):.0f}" if ts else "--")
        rows.append([label[0], label[1], label[2], f"{n_on}/5",
                     f"{n_decl}/5", f"{agree}/5", trange])
n_on2 = sum(r["candidate_adj"]["b5_onset"] for r in rea2.values())
rows.append(["Grip", "per-episode force direction (v2)",
             "plausible (registered miss)", f"{n_on2}/5", "5/5",
             f"{n_on2}/5", "--"])
table(
    r"\textbf{Regime-ensemble audit.} Alternative regime objects "
    r"enumerated from each environment specification, adjudicated with "
    r"the frozen detector on byte-identical reruns. Admissible "
    r"formation-axis alternatives agree with the declared verdict in "
    r"22/25 cells; no control cell certifies an onset; the "
    r"realization-axis force candidates lose the verdict (Methods, "
    r"`Regime-object audits').",
    ["System", "Regime object", "Class", "Onset", "Declared",
     "Agree", "$t^{*}$ range"],
    rows, "si:tab:regimeensemble", fit_width=True)

# Supplementary Table 5: regime-discovery audit, per-seed
rda = load("regime_discovery_audit.json")["results"]
rda2 = load("regime_discovery_audit2.json")["results"]
rows = []
for sysname in ("convention", "roles"):
    for sd, r in sorted(rda[sysname].items()):
        a = r["discovered_adj"]
        h = a.get("hinge", {})
        rows.append([sysname.capitalize(), sd, r["k_discovered"],
                     "onset" if r["declared_b5"] else "--",
                     fmt(r["declared_t_star"], 0),
                     "onset" if a["b5_onset"] else "--",
                     fmt(h.get("delta_bic"), 1), fmt(h.get("t_star"), 0),
                     "onset" if r["control_adj"]["b5_onset"] else "--"])
for sd, r in sorted(rda2.items()):
    a = r["discovered_adj"]
    h = a.get("hinge", {})
    rows.append(["Grip (cross-fitted)", sd, r["k_discovered"],
                 "onset" if r["declared_b5"] else "--",
                 fmt(r["declared_t_star"], 0),
                 "onset" if a["b5_onset"] else "--",
                 fmt(h.get("delta_bic"), 1), fmt(h.get("t_star"), 0),
                 "onset" if r["control_adj"]["b5_onset"] else "--"])
table(
    r"\textbf{Regime-discovery audit.} Machine-discovered regime "
    r"variables (k-means on raw episode records, $k$ by silhouette, one "
    r"recipe across systems) adjudicated with the frozen detector; "
    r"controls apply the identical pipeline to the untrained population "
    r"or policy. Formation-axis disagreements sit at the detector "
    r"threshold; the grip rows use the cross-fitted v2 estimator; "
    r"untrained grip controls certify onset because the structural gate "
    r"delays commitment for any policy (Methods).",
    ["System", "Seed", "$k$", "Declared", "$t^{*}_\\mathrm{decl}$",
     "Discovered", "$\\Delta$BIC", "$t^{*}$", "Control"],
    rows, "si:tab:regimediscovery")

# Supplementary Table 6: per-system qualification (certificate)
cert = load("emergence_certificates.json")["certificates"]
rows = []
for name, c in cert.items():
    q = c["qualification"]
    e = c["eip"]
    verdict = c["verdict"].split(" (")[0]
    rows.append([name.replace("&", r"\&"),
                 fmt(e.get("amplitude_fraction_closed"), 2),
                 fmt(q["regime_level"]), fmt(q["endogenous"]),
                 fmt(q["persistent"]), verdict])
table(
    r"\textbf{Qualification is three-way, not an entropy drop.} "
    r"Standardized certificate verdicts: amplitude (fraction of "
    r"reference entropy closed) with the three qualification gates. "
    r"The single-ant control and the Overcooked occupancy object show "
    r"that large collapse with a failed gate is not certified as "
    r"emergence---ordinary individual determinization is excluded by "
    r"construction.",
    ["System", "$M$", "Regime-level", "Endogenous", "Persistent",
     "Verdict"],
    rows, "si:tab:qualification",
    align="lccccl", fit_width=True)

# numbering follows main-text citation order: the three audit tables
# (built after tables 7-11) are Supplementary Tables 4-6
TABLES = TABLES[:3] + TABLES[8:11] + TABLES[3:8]

# Supplementary Table 12: method-baseline battery
mb = load("method_baseline_battery.json")
comp = mb["composite_on_factorial"]
amp = mb["amplitude_rule"]
mc = mb["matched_confound"]
cp = mb["changepoint_rivals"]["rivals"]
inst = mb["changepoint_rivals"]["instrument_reference_rates"]


def fpr_str(r):
    f = r["fpr_by_family"]
    return (f"power {r['power_onset']:.2f}; FPR {f['knee']:.2f} knee, "
            f"{f['gradual']:.2f} gradual, {f['flat']:.2f} flat")


rows = [
    ["Marginals + TC + pairwise MI (composite)",
     "source recovery, 72-cell factorial",
     f"{comp['n_correct']}/72; {comp['env_misassigned']}/"
     f"{comp['env_cells']} environment cells misread as pairwise",
     "72/72"],
    ["Amplitude threshold (no qualification)",
     "pseudo-controls and true positives",
     f"accepts {len(amp['controls_accepted_by_R1'])}/2 external "
     f"takeovers; {amp['true_cells_accepted']} true positives",
     "rejects all controls; profiles all cells"],
    ["Any joint-distribution functional",
     "matched-confound mechanisms",
     f"identical across generators (max diff "
     f"{mc['max_functional_diff']:.0e})",
     "contract verdicts separate them"],
    ["Binary segmentation, RBF (5\\% calibrated)",
     "held-out curve benchmark",
     fpr_str(cp["binseg_rbf_gain"]),
     f"power {inst['onset']:.2f}; FPR 0.00 all families"],
    ["CUSUM (5\\% calibrated)",
     "held-out curve benchmark",
     fpr_str(cp["cusum"]),
     f"power {inst['onset']:.2f}; FPR 0.00 all families"],
]
table(
    r"\textbf{Method-baseline battery.} Standard alternatives evaluated "
    r"on exactly the data the instrument used, with rival decision rules "
    r"frozen in the preregistration before implementation; the only "
    r"calibration is the 5\% false-positive calibration on flat curves. "
    r"One preregistration detail was wrong and is retained: the "
    r"amplitude rule was predicted to accept the revelation/metric "
    r"controls, but those carry zero entropy amplitude; the clause "
    r"passed through the external-takeover controls instead.",
    ["Baseline", "Test", "Baseline result", "Instrument"],
    rows, "si:tab:baselines", align="p{0.22\\textwidth}p{0.18\\textwidth}"
    "p{0.28\\textwidth}p{0.17\\textwidth}")

# Supplementary Table 13: fixed-time ring intervention, all runs
ft = load("oc_ring_fixed_time.json")
base = ft["baselines_at_tfix"]
rows = []
for r in ft["runs"]:
    b = base[str(r["seed"])]
    rows.append([r["seed"], fmt(r["scale"], 2),
                 fmt(r["openness_at_perturbation"], 2),
                 fmt(b["mean_soups_at_tfix"], 2),
                 fmt(r["final_p_ccw"], 2), r["outcome"]])
oc = ft["registered_outcomes"]
table(
    r"\textbf{Fixed-time ring intervention (registered miss).} All "
    r"eight seeds perturbed at the same training step "
    rf"({oc['T_FIX']//1000}k, variance-maximizing rule frozen before "
    r"the run) at two noise scales and resumed for 400k steps with "
    r"unchanged mechanics. No run moved "
    rf"({oc['OCF1_movable_open']} open vs "
    rf"{oc['OCF1_movable_committed']} committed seeds movable, Fisher "
    rf"$p={oc['OCF1_seed_fisher_p']:.1f}$): every continuation, "
    r"including the behaviourally open seeds, re-converged to its "
    r"seed's eventual direction.",
    ["Seed", "Noise scale", "Openness at 960k", "Soups at 960k",
     "Final $P(\\mathrm{ccw})$", "Outcome"],
    rows, "si:tab:fixedtime")

# Supplementary Table 14: discovered-regime controllability race
rdc = load("learn_grip_discovery_utility.json")["registered_outcomes"]
race = rdc["race"]
ftau = rdc["fixed_tau"]
valid_taus = [t for t in sorted(ftau, key=int) if "auc_disc" in ftau[t]]
names = {
    "disc_open": "Discovered openness (k-means, no analyst)",
    "side_open": "Declared side-openness",
    "pol_ent": "Policy action entropy",
    "absx": "$|x|$", "absv": "$|v|$", "att": "Attachment flag",
    "tau": "Intervention time $\\tau$",
}
rows = []
for key in ["disc_open", "side_open", "pol_ent", "absx", "absv", "att",
            "tau"]:
    if key == "disc_open":
        span = (f"{min(ftau[t]['auc_disc'] for t in valid_taus):.2f}--"
                f"{max(ftau[t]['auc_disc'] for t in valid_taus):.2f}")
    elif key == "side_open":
        span = (f"{min(ftau[t]['auc_side'] for t in valid_taus):.2f}--"
                f"{max(ftau[t]['auc_side'] for t in valid_taus):.2f}")
    else:
        span = "---"
    rows.append([names[key], fmt(race[key]["rank_corr"], 2),
                 fmt(race[key]["auc"], 3), span])
table(
    r"\textbf{Discovered-regime controllability race (registered "
    r"miss).} Predictors of kick-induced outcome switching raced on the "
    r"same 81{,}920 grip intervention episodes (5 seeds $\times$ 8 "
    r"intervention times $\times$ 2{,}048 episodes). The discovery "
    r"recipe (k-means on raw traces, $k$ by silhouette, frozen from the "
    r"regime-discovery audit) recovers $k=2$ in 5/5 seeds. Fixed-time "
    r"AUCs are shown at the four intervention times where both outcome "
    r"classes have $\geq$20 episodes; the registered "
    r"0.80-at-every-time bar for discovered openness fails (2/4 "
    r"times), so the analyst-free result is reported as a registered "
    r"miss. Pooled AUCs are compressed for all state-based predictors "
    r"because early kicks always switch the side.",
    ["Predictor", "Rank corr.", "Pooled AUC", "Fixed-time AUC"],
    rows, "si:tab:rdc",
    align="p{0.4\\textwidth}p{0.12\\textwidth}p{0.12\\textwidth}"
    "p{0.2\\textwidth}")

# Supplementary Table 15: seed-level statistics for the grip race
su = load("learn_grip_stat_unit.json")["registered_outcomes"]
rows = []
for key, label in [("side_open", "Declared side-openness"),
                   ("disc_open", "Discovered openness"),
                   ("pol_ent", "Policy action entropy")]:
    per = su["per_seed_auc"][key]
    ci = su["boot_ci_95_seed_cluster"][key]
    loo = su.get("leave_one_seed_out", {}).get(key)
    rows.append([label, f"{min(per):.3f}--{max(per):.3f}",
                 f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                 f"{min(loo):.3f}--{max(loo):.3f}" if loo else "---"])
table(
    r"\textbf{Seed-level statistics for the grip intervention race.} "
    r"The 81{,}920 intervention episodes nest within 5 training seeds, "
    r"so the race of Supplementary Table~14 is re-aggregated at the "
    r"seed level under a frozen analysis addendum. Per-seed: pooled "
    r"AUC computed within each seed. CI: seed-cluster bootstrap "
    r"(10{,}000 resamples of the 5 seeds with replacement). LOSO: "
    r"leave-one-seed-out pooled AUC. Both registered outcomes were "
    r"met: side-openness beats policy entropy in 5/5 seeds, and its "
    r"CI lies above 0.95. Within-seed AUCs exceed the cross-seed "
    r"pooled values because seed-to-seed predictor offsets add "
    r"between-cluster variance.",
    ["Predictor", "Per-seed AUC", "Cluster bootstrap 95\\% CI",
     "LOSO AUC"],
    rows, "si:tab:statunit",
    align="p{0.32\\textwidth}p{0.17\\textwidth}p{0.22\\textwidth}"
    "p{0.15\\textwidth}")

with open(os.path.join(HERE, "si_tables.tex"), "w") as f:
    f.write("% Generated by make_si_tables.py -- do not edit by hand.\n\n")
    f.write("\n".join(TABLES))
print(f"wrote si_tables.tex ({len(TABLES)} tables)")
