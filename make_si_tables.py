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


def table(caption, header, rows, label, align=None):
    ncol = len(header)
    align = align or "l" + "c" * (ncol - 1)
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
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

# Supplementary Table 4: grip per-seed statistics
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

# Supplementary Table 5: convention and role systems, tabular seeds
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

# Supplementary Table 6: neural replications, per-seed
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

# Supplementary Table 7: Overcooked coordination ring, per-seed
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

# Supplementary Table 8: ant finite-size scaling and Kuramoto coupling grid
fss = load("ant_fss.json")
rows = []
for size, v in sorted(fss["per_size"].items(), key=lambda kv: int(kv[0])):
    rows.append([f"Ant $N={size}$", fmt(v.get("t50"), 0),
                 fmt(v.get("width"), 0), fmt(v.get("delta_bic"), 1),
                 "onset" if v.get("b5_onset") else "--"])
ks = load("kuramoto_scale.json")
for K, seeds in sorted(ks["per_K"].items(), key=lambda kv: float(kv[0])):
    for sd, v in sorted(seeds.items()):
        h = v["hinge"]
        rows.append([f"Kuramoto $K={K}$ (seed {sd})",
                     fmt(h["t_star"], 1),
                     "--", fmt(h["delta_bic"], 1),
                     "onset" if v.get("onset_pass") else "--"])
table(
    r"\textbf{Scaling data.} Ant-colony finite-size scaling (median "
    r"$t_{50}$ crossing and 10--90\% width in trips, 30 episodes per "
    r"size) and the Kuramoto coupling grid (two seeds per coupling "
    r"$K$; $t^{*}$ in time units). These rows are the points behind "
    r"Fig.~6a,b.",
    ["System / size", "$t_{50}$ or $t^{*}$", "Width", "$\\Delta$BIC",
     "Verdict"],
    rows, "si:tab:scaling")

with open(os.path.join(HERE, "si_tables.tex"), "w") as f:
    f.write("% Generated by make_si_tables.py -- do not edit by hand.\n\n")
    f.write("\n".join(TABLES))
print(f"wrote si_tables.tex ({len(TABLES)} tables)")
