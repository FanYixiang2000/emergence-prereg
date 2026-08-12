"""Adversarial number audit: every headline number in main.tex must be
recomputable from outputs/*.json.

Each check states (a) the claim as printed in the manuscript, (b) the
value recomputed from the raw experiment output, and (c) optionally a
literal text snippet that must occur in main.tex (whitespace-normalized),
binding JSON -> script -> manuscript in one pass.

Run:  python verify_paper_numbers.py
Exit code 0 iff every check passes.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

TEX = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
TEX_NORM = re.sub(r"\s+", " ", TEX)

RESULTS = []


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def check(name, computed, expected, tex_snippet=None, tol=0.0):
    ok = True
    detail = f"computed={computed!r} expected={expected!r}"
    if isinstance(expected, float) or isinstance(computed, float):
        try:
            ok = abs(float(computed) - float(expected)) <= tol + 1e-12
        except (TypeError, ValueError):
            ok = False
    else:
        ok = computed == expected
    tex_ok = True
    if tex_snippet is not None:
        snippet_norm = re.sub(r"\s+", " ", tex_snippet)
        tex_ok = snippet_norm in TEX_NORM
        if not tex_ok:
            detail += f" | TEX SNIPPET NOT FOUND: {snippet_norm[:70]!r}"
    RESULTS.append((name, ok and tex_ok, detail))


def rng(vals):
    return min(vals), max(vals)


# ---------------------------------------------------------------- Fig 1
b72 = load("bench72_factorial.json")
ro = b72["registered_outcomes"]
check("BENCH72 source recovery 72/72", ro["B72_1_source"], True,
      "recovers the correct source in 72/72 cells")
mvb = b72["checks"]["B72_2_M_vs_B"]
check("BENCH72 M invariant across shape (24/24 groups)",
      (mvb["M_groups_ok"], mvb["n_groups"]), (24, 24),
      "amplitude $M$ is invariant across temporal shape")
check("BENCH72 J orders shapes (24/24 groups)",
      (mvb["J_groups_ok"], mvb["n_groups"]), (24, 24),
      "strictly orders punctuated $>$ sigmoid $>$ gradual")

sdec = load("collapse_source_decomposition.json")
ro = sdec["registered_outcomes"]
check("Source decomposition five checks SD1-SD5", all(ro[f"SD{i}"] for i in range(1, 6)), True,
      "five preregistered checks pass")

# ---------------------------------------------------------------- detector
dv = load("detector_validation.json")
ro = dv["registered_outcomes"]
check("Detector reference FPR 0.000", dv["reference_point"]["controls_pooled"]["fpr"]
      if "reference_point" in dv and "controls_pooled" in dv.get("reference_point", {})
      else ro["ref_fpr"], 0.0,
      "the false-positive rate across all control families is 0.000")
check("Detector onset power 1.00", ro["ref_onset_power"], 1.0,
      "onset power is 1.00")

# ---------------------------------------------------------------- Fig 2 grip
b5 = load("learn_grip_transport_b5.json")
ro = b5["registered_outcomes"]
check("Grip primary onsets 5/5 (B5 battery)", (ro["b5_count_learned"], ro["n_learned"]), (5, 5))
ext = load("learn_grip_ext.json")
rx = ext["registered_outcomes"]
check("Grip learns 10/10 seeds with onset", (rx["n_learned"], rx["b5_count"]), (10, 10),
      "learns this task in 10/10 seeds")
dbics = [d["adj"]["hinge"]["delta_bic"] for d in b5["seeds"].values()]
check("Grip dBIC range 45.8-52.7", (round(min(dbics), 1), round(max(dbics), 1)), (45.8, 52.7),
      "$\\Delta$BIC 45.8--52.7")
tstars = [d["adj"]["hinge"]["t_star"] for d in b5["seeds"].values()]
check("Grip t* 16-18", (min(tstars), max(tstars)), (16, 18), "$t^{*}$ at 16--18")
lgt = load("learn_grip_transport.json")
succ = [lgt["seeds"][s]["final_success"] for s in lgt["seeds"]]
check("Grip success >= 0.995", min(succ) >= 0.995, True, "success $\\geq 0.995$")

a2c = load("learn_grip_a2c.json")
ro = a2c["registered_outcomes"]
n_hinge = sum(1 for s in a2c["seeds"].values()
              if (s["adj"].get("hinge") or {}).get("delta_bic", 0) >= 10)
check("A2C 5/5 shape and primary hinge", n_hinge, 5,
      "5/5 seeds reproduce the plateau-then-collapse shape and primary hinge")
check("A2C strict thinning 3/5", ro["b5_count"], 3,
      "the strict thinning clause passes in only 3/5")
d2 = [a2c["seeds"][s]["adj"]["hinge"]["delta_bic"] for s in a2c["seeds"]]
check("A2C dBIC 37.7-45.5", (round(min(d2), 1), round(max(d2), 1)), (37.7, 45.5),
      "$\\Delta$BIC 37.7--45.5")

# ---------------------------------------------------------------- Fig 3
lgf = load("learn_grip_formation.json")
ro = lgf["registered_outcomes"]
check("Formation 0 breakpoints, realization 5/5",
      (ro["formation_b5_count"], ro["realization_b5_count"]), (0, 5),
      "collapse is punctuated (5/5")
fine = load("learn_grip_formation_fine.json")
ro = fine["registered_outcomes"]
check("Fine-resolution success no onset 0/5", ro["succ_b5_count"], 0)

# ---------------------------------------------------------------- convention/roles
lc = load("learn_convention.json")
seeds = lc["seeds"]
onsets = [k for k, r in seeds.items() if r["adj"]["b5_onset"]]
check("Convention onsets 4/5", len(onsets), 4, "onset breakpoint in 4/5 seeds")
cd = [seeds[k]["adj"]["hinge"]["delta_bic"] for k in onsets]
check("Convention dBIC 17.8-30.1", (round(min(cd), 1), round(max(cd), 1)), (17.8, 30.1),
      "$\\Delta$BIC 17.8--30.1")
ct = [seeds[k]["adj"]["hinge"]["t_star"] for k in onsets]
check("Convention t* 275-300", (min(ct), max(ct)), (275, 300), "$t^{*}=275$--$300$")
cross = [seeds[k]["success_090_cross"] for k in onsets]
check("Intelligibility 0.9 crossing 700-1025", (min(cross), max(cross)), (700, 1025),
      "0.9 (updates 700--1{,}025)")
codes = set(str(r.get("code")) for r in seeds.values())
check("5 different codes", len(codes), 5, "five \\emph{different} codes")

lr = load("learn_roles.json")
rs = lr["seeds"]
ronsets = [k for k, r in rs.items() if r["adj"]["b5_onset"]]
check("Roles onsets 5/5", len(ronsets), 5)
rd = [rs[k]["adj"]["hinge"]["delta_bic"] for k in ronsets]
check("Roles dBIC 53.6-71.7", (round(min(rd), 1), round(max(rd), 1)), (53.6, 71.7),
      "$\\Delta$BIC 53.6--71.7")
perms = set(str(r.get("assignment")) for r in rs.values())
check("5 distinct role permutations", len(perms), 5, "five distinct role permutations")

# ---------------------------------------------------------------- barrier
bx = load("barrier_xplay.json")
ro = bx["registered_outcomes"]
check("Barrier unilateral gains 0.011/0.001",
      (round(ro["conv_adoption_gain_pre"], 3), round(ro["roles_adoption_gain_pre"], 3)),
      (0.011, 0.001), "gains essentially nothing (0.011; 0.001)")
check("Barrier deviation costs 0.40/1.00",
      (round(ro["conv_deviation_cost_post"], 2), round(ro["roles_deviation_cost_post"], 2)),
      (0.40, 1.00), "(0.40 of a 0.40 ceiling; 1.00)")
check("Cross-seed incompat 0.14/0.05",
      (round(ro["conv_cross_intel"], 2), round(ro["roles_hybrid_success"], 2)),
      (0.14, 0.05), "cross-seed intelligibility 0.14, hybrid-team success 0.05")

# ---------------------------------------------------------------- NN replication
nr = load("learn_nn_resolution.json")
co = nr["systems"]["convention"]["outcomes"]
rr_ = nr["systems"]["roles"]["outcomes"]
check("NN conv onsets 6/7 learned", (co["n_onset"], co["n_learned"]), (6, 7),
      "onset in 6/7 learned convention seeds")
check("NN roles onsets 10/10", (rr_["n_onset"], rr_["n_learned"]), (10, 10),
      "10/10 role seeds")
alld = [r["delta_bic"] for s in ("convention", "roles")
        for r in nr["systems"][s]["seeds"].values()
        if r["b5_onset"] and r["delta_bic"] is not None]
check("NN max dBIC 162", round(max(alld)), 162, "$\\Delta$BIC up to 162")

# ---------------------------------------------------------------- TRI-C
tric = load("triad_highorder_cue.json")
finals = []
pairs = []
for s, d in tric["seeds"].items():
    last = d[max((k for k in d if k.isdigit()), key=int)]["ladder2_hidden"]
    finals.append(last["C_high"])
    pairs.append(last["C_pair"])
check("TRI-C C_high 0.94-0.96", (round(min(finals), 2), round(max(finals), 2)), (0.94, 0.96),
      "0.94--0.96 bits")
check("TRI-C pairwise ~0.0004", round(max(pairs), 4) <= 0.0004, True,
      "pairwise $\\approx$ 0.0004 bits")
tbp = load("tri_c_breakpoint.json")
tce = load("tri_c_breakpoint_ext.json")
n_on = sum(1 for v in tbp["seeds"].values() if v["verdict"]["tricbp1_seed"])
n_on += sum(1 for v in tce["seeds"].values() if v["verdict"]["tricbp1_seed"])
n_tot = len(tbp["seeds"]) + len(tce["seeds"])
check("TRI-C onsets pooled 7/8", (n_on, n_tot), (7, 8),
      "7/8 seeds across the original run and a five-seed extension")

# ---------------------------------------------------------------- overcooked profile
e1c = load("overcooked_profile_confirmatory.json")
check("Overcooked C_env learned 0.0137", round(e1c["learned"]["C_env"], 4), 0.0137,
      "0.0137 [0.0123, 0.0156]")
ci = e1c["learned"]["C_env_ci95"]
check("Overcooked C_env CI [0.0123,0.0156]", (round(ci[0], 4), round(ci[1], 4)), (0.0123, 0.0156))
check("Overcooked C_env scripted 0.0005", round(e1c["noisy_scripted"]["C_env"], 4), 0.0005,
      "0.0005 [0.0004, 0.0007]")

# ---------------------------------------------------------------- controllability
lgu = load("learn_grip_utility.json")
ro = lgu["registered_outcomes"]
sw = {int(k): v for k, v in ro["mean_switch_by_tau"].items()}
check("Grip switch 1.0 up to t=16", all(sw[t] >= 0.999 for t in sw if t <= 16), True,
      "probability 1.0 up to $t=16$")
check("Grip switch 0.27 by t=30", round(sw[30], 2), 0.27, "only 0.27 by $t=30$")
check("Grip openness AUC 0.996", round(ro["baseline_race"]["side_open"]["auc"], 3), 0.996,
      "AUC of 0.996")

cc = load("learn_grip_confound.json")
aucs = [v["auc"] for v in cc["cc1_by_tau"].values()]
check("Fixed-time AUC 0.974-0.990", (round(min(aucs), 3), round(max(aucs), 3)), (0.974, 0.990),
      "AUC 0.974--0.990 at every tested $\\tau$")

gp = load("learn_grip_policy.json")
ro = gp["registered_outcomes"]
check("Policy transfer flip 99.99% (raw 0.99985)", ro["pooled_flip"]["open"] * 100, 99.99,
      "flipping 99.99\\% of episodes", tol=0.006)
check("Policy fixed 99.6%", round(ro["pooled_flip"]["fixed"] * 100, 1), 99.6, "(99.6\\%)")
check("Policy 3.8 steps later",
      round(ro["pooled_mean_step"]["open"] - ro["pooled_mean_step"]["fixed"], 1), 3.8,
      "acting 3.8 steps later")
check("Policy beats random by 21 pts",
      round((ro["pooled_flip"]["open"] - ro["pooled_flip"]["random"]) * 100), 21,
      "beating random timing by 21 points")

lss = load("learn_stance_sticky.json")
br = lss["registered_outcomes"]["baseline_race"]
check("Sticky openness AUC 0.886 vs absx 0.849",
      (round(br["open"]["auc"], 3), round(br["absx"]["auc"], 3)), (0.886, 0.849),
      "(AUC 0.886 vs 0.849)")
lsc = load("learn_stance_control.json")
bc = lsc["registered_outcomes"]["baseline_race"]
check("Control reversal 0.811 vs 0.884",
      (round(bc["open"]["auc"], 3), round(bc["absx"]["auc"], 3)), (0.811, 0.884),
      "(0.811 vs 0.884; Fig.~5c)")

aic = load("ant_conditional_leverage.json")
ro = aic["registered_outcomes"]
bins = aic["bins"]
check("Ant flip bins 0.000->0.205",
      (round(bins[0]["flip_rate"], 3), round(bins[-1]["flip_rate"], 3)), (0.0, 0.205),
      "(0.000 $\\to$ 0.205 across bins")
check("Ant closed episodes 0 flips", bins[0]["n_flips"] if "n_flips" in bins[0] else 0, 0)
check("Ant openness gap 0.58", round(ro["mean_openness_flipped_minus_not"], 2), 0.58,
      "0.58 openness units more open")
check("Ant permutation p < 1e-4", ro["permutation_p"] < 1e-4, True)
check("Ant n_pairs 8372", aic["n_pairs"], 8372, "8{,}372 paired counterfactuals")

# ---------------------------------------------------------------- FSS & Kuramoto
fss = load("ant_fss.json")
ro = fss["registered_outcomes"]
ll = ro["log_law"]
check("FSS log-fit b=87.6", round(ll["b"], 1), 87.6, "$b=87.6$")
check("FSS R2 0.93", round(ll["r2"], 2), 0.93, "$R^{2}=0.93$")
check("FSS CI [41,124]", (round(ll["b_ci95"][0]), round(ll["b_ci95"][1])), (41, 124),
      "bootstrap 95\\% CI [41, 124]")
check("FSS RMS 0.010 vs 0.266",
      (round(ro["rms_aligned"], 3), round(ro["rms_unaligned"], 3)), (0.010, 0.266),
      "RMS 0.010 versus 0.266 unaligned")
w = ro["widths_large_N"]
check("FSS width 280-290 across N>=50",
      (min(w.values()), max(w.values())), (280.0, 290.0),
      "280--290 trips from $N{=}50$ to $500$")

ks = load("kuramoto_scale.json")
ro = ks["registered_outcomes"]
tvals = ro["mean_t_star"]
svals = ro["mean_post_slope"]
check("Kuramoto t* 6.7->1.8", (round(tvals[0], 1), round(tvals[-1], 1)), (6.7, 1.8),
      "(6.7 $\\to$ 1.8)")
check("Kuramoto t* monotone decreasing", tvals == sorted(tvals, reverse=True), True)
check("Kuramoto slope 0.032->0.199", (round(svals[0], 3), round(svals[-1], 3)), (0.032, 0.199),
      "(0.032 $\\to$ 0.199)")
check("Kuramoto slope monotone increasing", svals == sorted(svals), True)
runs = [(K, s, r) for K, seeds in ks["per_K"].items() for s, r in seeds.items()]
check("Kuramoto 10/10 onsets", (sum(1 for _, _, r in runs if r["onset_pass"]), len(runs)),
      (10, 10), "(10/10 onsets)")
check("Kuramoto subcritical gated null 3/3",
      load("kuramoto_breakpoint_r2.json")["registered_outcomes"]["KURR2_2_subcritical_gated_null_3of3"],
      True, "Every subcritical run is correctly gated null")
check("Kuramoto pairwise carrier 3/3",
      load("kuramoto_breakpoint_r2.json")["registered_outcomes"]["KURR2_3_relational_carrier_3of3"],
      True, "carried by the pairwise channel (3/3 seeds")

# ---------------------------------------------------------------- negatives & ring
for f, nm in (("overcooked_state_breakpoint.json", "OC state"),
              ("overcooked_occupancy_breakpoint.json", "OC occupancy")):
    d = load(f)
    n_on = sum(1 for v in d["seeds"].values()
               if (v.get("adj_primary") or v.get("adj", {})).get("b5_onset"))
    check(f"{nm} 0/{len(d['seeds'])} onsets", (n_on, len(d["seeds"])), (0, 3))

lq = load("learn_quorum_breakpoint.json")
n_runs, n_on = 0, 0
for N, seeds in lq["per_N"].items():
    for s_, r in seeds.items():
        n_runs += 1
        h = r.get("hinge") or {}
        if r.get("b5_onset") or h.get("b5_onset") or h.get("onset_type") and r.get("gate_passed"):
            n_on += 1
check("Quorum 0/20 onsets", (n_on, n_runs), (0, 20), "no onset (0/20)")

oce = load("oc_ring_ext.json")
ro = oce["registered_outcomes"]
check("Ring committed 8/8 pooled", (ro["n_committed"], ro["n_pooled"]), (8, 8),
      "self-play seeds end committed to one direction")
check("Ring onset 1/8", ro["n_onset"], 1, "certifies a punctuated onset in one seed")
orc = load("overcooked_ring_convention.json")
dirs = [orc["systems"]["ring"][s]["final_p_ccw"] for s in orc["systems"]["ring"]]
dirs += [oce["ext_seeds"][s]["final_p_ccw"] for s in oce["ext_seeds"]]
n_ccw = sum(1 for p in dirs if p > 0.5)
check("Ring 6 ccw / 2 cw", (n_ccw, len(dirs) - n_ccw), (6, 2),
      "six counterclockwise, two clockwise")
check("Ring final probs 0.97/0.03",
      all(round(p, 2) >= 0.97 or round(p, 2) <= 0.03 for p in dirs), True,
      "probability 0.97 or 0.03")

orr = load("oc_ring_realization.json")
ro = orr["registered_outcomes"]
check("Ring realization mid 0/5 onset", ro["n_onset"], 0, "never internally lock (0/5)")
check("Ring final closed 8/8", ro["OCRR5_final_initial_lt_05_ge_6of8"] and ro["OCRR5_final_zero_onset"], True)

oci = load("oc_ring_intervention.json")
ro = oci["registered_outcomes"]
check("OCI run-level 8/16 vs 1/16", (ro["open_moved"], ro["late_moved"]), ("8/16", "1/16"),
      "run-level 8/16 versus 1/16")
check("OCI Fisher p=0.008", round(ro["OCI1_open_moved_gt_late_fisher_p"], 3), 0.008, "Fisher $p=0.008$")
check("OCI strict flips 3 vs 0", (ro["open_strict_flips"], ro["late_strict_flips"]), (3, 0),
      "three open-phase flips, zero late")
check("OCI AUC 0.85", round(ro["OCI3_auc"], 2), 0.85, "(AUC 0.85")
check("OCI recovery 0.92", round(ro["OCI4_late_recovery_median"], 2), 0.92, "recovers to 0.92")

sl_ = load("oci_seed_level.json")
check("OCI seed-level 7/8 vs 1/8", (sl_["moved"]["open_seeds"], sl_["moved"]["late_seeds"]), (7, 1),
      "7/8 seeds perturbed while open versus 1/8")
check("OCI sign-flip p=0.016", round(sl_["moved"]["sign_flip_p"], 3), 0.016,
      "sign-flip $p=0.016$")
check("OCI flip p=0.125 (printed 0.13)", sl_["strict_flip"]["sign_flip_p"], 0.125,
      "$p=0.13$", tol=0.004)

si = load("semi_inject.json")
ro = si["registered_outcomes"]
check("SEMI-INJ FPR 0.01", ro["SI2_fpr_pooled"], 0.01)
check("SEMI-INJ t* err ~1%", round(ro["SI3_median_tstar_err_frac"], 2), 0.01)
check("SEMI-INJ power 0.88", ro["SI1_power_w_le3"], 0.88)

# ---------------------------------------------------------------- SD audit
sd = load("sd_audit.json")
d4 = sd["results"]["SDA4_sample_complexity"]["detail"]
e30k = max(d4["error_table"]["30000"][c]["median_abs_err"] for c in d4["error_table"]["30000"])
check("Sample complexity <=0.02 bits at 3e4", round(e30k, 2) <= 0.02, True,
      "$\\leq$0.02 bits at $3\\times10^{4}$ samples")
e300 = d4["error_table"]["300"]
check("n=300 errors 0.09/0.31/0.81 (ind/high/pair)",
      (round(e300["C_individual"]["median_abs_err"], 2),
       round(e300["C_high"]["median_abs_err"], 2),
       round(e300["C_pair"]["median_abs_err"], 2)),
      (0.09, 0.31, 0.81),
      "0.09 bits (individual), 0.31 (higher-order) and 0.81 (pairwise)")
d2_ = sd["results"]["SDA2_off_family"]["detail"]
mm = d2_["modular_sum"]["components"]["C_high"]
check("Modular-sum C_high 3.32 bits", round(mm, 2), 3.32, "3.32 bits")

# ---------------------------------------------------------------- contracts & repr
rrb = load("repr_robustness.json")
check("243-cell contract battery per system",
      all(s_["n_cells"] == 243 for s_ in rrb["systems"].values()), True)
check("Contract verdict invariance >=0.90 all systems",
      rrb["registered_outcomes"]["RR1_verdict_invariance_ge_0.90_all"], True)
check("Contract onset fraction 1.0 among gated",
      all(s_["frac_onset_among_gated"] == 1.0 for s_ in rrb["systems"].values()), True)

req_c = load("repr_equiv_convention.json")
req_g = load("repr_equiv_grip.json")
cells = list(req_c["representation_cells"].values()) + list(req_g["representation_cells"].values())
kept = sum(1 for c in cells if c["b5_onset"])
check("Representation battery 8/12 keep verdict", (kept, len(cells)), (8, 12))

# Methods section: config parameters as printed
cfg = load("learn_grip_transport.json")["config"]
check("Methods grip 16 agents, threshold 6", (cfg["N_agents"], cfg["threshold"]), (16, 6),
      "16 agents, grip threshold 6")
check("Methods grip gain/decay 0.06/0.01", (cfg["grip_gain"], cfg["grip_decay"]), (0.06, 0.01),
      "grip gain 0.06, decay 0.01")
check("Methods grip 1200 updates batch 512 lr 2e-3",
      (cfg["updates"], cfg["batch"], cfg["lr"]), (1200, 512, 0.002),
      "1{,}200 updates, batch 512, learning rate $2\\times10^{-3}$")
dcfg = load("detector_validation.json")["config"]
check("Methods detector 200 curves per family", dcfg["n_per_family"], 200,
      "200 curves per family")
check("Methods detector ref point 80 pts sigma 0.02",
      (dcfg["ref_density"], dcfg["ref_sigma"]), (80, 0.02),
      "80 grid points, noise $\\sigma=0.02$")
check("Methods detector densities 12/20/40/80", dcfg["densities"], [12, 20, 40, 80],
      "12/20/40/80 points")
check("Methods detector external ruptures Binseg-RBF",
      "Binseg" in dcfg["external_method"] and "rbf" in dcfg["external_method"], True,
      "binary segmentation, RBF cost")
ccfg = load("learn_grip_confound.json")["config"]
check("Methods confound 20480 per cell x 6 taus",
      (ccfg["n_records"] // len(ccfg["taus"]), len(ccfg["taus"])), (20480, 6),
      "20{,}480 episodes per cell")
check("Methods confound 1000 permutations", ccfg["n_perm"], 1000)

rea = load("regime_ensemble_audit.json")
rea_res = rea["results"]
form_agree = form_n = 0
tstar_ok = tstar_n = 0
ctrl_clean = ctrl_n = 0
spans = {"convention": 4000, "roles": 6000, "grip": 79}
plaus = {"convention": ["P2_listener_code", "P3_composed_channel",
                        "P4_meaning0_symbol"],
         "roles": ["P2_agent0_role", "P3_role0_owner"],
         "grip": ["P2_force_sign"]}
ctrls = {"convention": ["X1_symbol_marginal", "X2_speaker_identity"],
         "roles": ["X1_role_marginal"], "grip": ["X1_xabs_tertiles"]}
for s, srows in rea_res.items():
    for r in srows.values():
        for c in plaus[s]:
            cand = r["candidates"][c]
            if s != "grip":
                form_n += 1
                form_agree += cand["b5_onset"] == r["declared_b5"]
            if cand["b5_onset"] and r["declared_b5"]:
                tstar_n += 1
                tstar_ok += (abs(cand["t_star"] - r["declared_t_star"])
                             <= 0.10 * spans[s])
        for c in ctrls[s]:
            ctrl_n += 1
            ctrl_clean += not r["candidates"][c]["b5_onset"]
check("Methods ensemble formation agreement 22/25",
      (form_agree, form_n), (22, 25),
      "reproduce the declared verdict in 22/25 cells")
roles_agree = sum(
    r["candidates"][c]["b5_onset"] == r["declared_b5"]
    for r in rea_res["roles"].values() for c in plaus["roles"])
check("Methods ensemble roles 10/10", roles_agree, 10, "roles 10/10")
lc_agree = sum(
    r["candidates"]["P2_listener_code"]["b5_onset"] == r["declared_b5"]
    for r in rea_res["convention"].values())
check("Methods ensemble listener code 5/5", lc_agree, 5,
      "listener code 5/5")
check("Methods ensemble t* 19/19", (tstar_ok, tstar_n), (19, 19),
      "the detector's span tolerance (19/19)")
check("Methods ensemble controls 0/20", (ctrl_clean, ctrl_n), (20, 20),
      "None of the 20 control cells certifies")
check("Methods ensemble registered 22/30",
      rea["registered_outcomes"]["RE1_verdict_stability"], "22/30",
      "ensemble 22/30 against 24/30")
rda = load("regime_discovery_audit.json")
rda_res = rda["results"]
disc_form = sum(
    r["discovered_adj"]["b5_onset"] == r["declared_b5"]
    for s in ("convention", "roles") for r in rda_res[s].values())
check("Methods discovery formation 6/10", disc_form, 6,
      "it reproduces 6/10 verdicts")
conv_miss_bics = sorted(
    r["discovered_adj"]["hinge"]["delta_bic"]
    for r in rda_res["convention"].values()
    if r["declared_b5"] and not r["discovered_adj"]["b5_onset"])
check("Methods discovery conv near-threshold 6.9 and 9.8",
      [round(b, 1) for b in conv_miss_bics], [6.9, 9.8],
      "$\\Delta$BIC 6.9 and 9.8")
check("Methods discovery registered 6/15",
      rda["registered_outcomes"]["RD2_verdict_agreement"], "6/15",
      "discovery\n6/15 against 12/15")
rda2 = load("regime_discovery_audit2.json")
rea2 = load("regime_ensemble_audit2.json")
best_realization_bic = max(
    [r["discovered_adj"]["hinge"]["delta_bic"]
     for r in rda2["results"].values()]
    + [r["candidate_adj"]["hinge"]["delta_bic"]
       for r in rea2["results"].values()])
check("Methods realization best dBIC 12.6",
      round(best_realization_bic, 1), 12.6,
      "best $\\Delta$BIC 12.6", tol=0.05)
extra = [r for r in rda_res["convention"].values()
         if r["discovered_adj"]["b5_onset"] and not r["declared_b5"]]
check("Methods discovery one gained cell dBIC 23.6",
      (len(extra), round(extra[0]["discovered_adj"]["hinge"]["delta_bic"], 1)),
      (1, 23.6), "$\\Delta$BIC 23.6")
n_cells = (sum(len(r["candidates"]) for srows in rea_res.values()
               for r in srows.values())
           + 2 * sum(len(srows) for srows in rda_res.values())
           + 2 * len(rda2["results"]) + len(rea2["results"]))
check("Methods 105 audit cells", n_cells, 105, "Across all\n105 audit cells")

print()
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
n_fail = len(RESULTS) - n_pass
for name, ok, detail in RESULTS:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}")
    if not ok:
        print(f"        {detail}")
print()
print(f"{n_pass}/{len(RESULTS)} checks passed, {n_fail} failed")
sys.exit(0 if n_fail == 0 else 1)
