"""Extract every number cited in the NMI manuscript from outputs/*.json.

Run:  python manuscript_numbers.py
Every value printed here is read directly from the experiment output
files; the manuscript text must match this report exactly.
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def sec(title):
    print()
    print("=" * 8, title)


# ---------------------------------------------------------------- Fig 2
sec("GRIP FLAGSHIP (realization)")
b5 = load("learn_grip_transport_b5.json")
ro = b5["registered_outcomes"]
print("LGT-B b5_count/learned:", ro["b5_count_learned"], "/", ro["n_learned"])
print("LGT-B median plateau:", ro["median_plateau_len"])
for s, d in b5["seeds"].items():
    h = d["adj"]["hinge"]
    print(f"  seed {s}: dBIC={h['delta_bic']:.1f} t*={h['t_star']} "
          f"slopes {h['slope_before']:.4f}->{h['slope_after']:.4f} "
          f"drop={d['adj']['drop']:.3f}")

lgt = load("learn_grip_transport.json")
succ = [lgt["seeds"][s]["final_success"] for s in lgt["seeds"]]
print("LGT success range:", min(succ), max(succ))
print("LGT config:", lgt["config"])

ext = load("learn_grip_ext.json")
ro = ext["registered_outcomes"]
print("EXT n_learned:", ro["n_learned"], "/", ro["n_total"],
      " b5:", ro["b5_count"], " t_stars:", sorted(ro["t_stars"]))

a2c = load("learn_grip_a2c.json")
ro = a2c["registered_outcomes"]
print("A2C learned:", ro["n_learned"], " b5:", ro["b5_count"],
      " t_stars:", sorted(ro["t_stars"]))
dbics = [a2c["seeds"][s]["adj"]["hinge"]["delta_bic"] for s in a2c["seeds"]]
plats = []
for s in a2c["seeds"]:
    adj = a2c["seeds"][s]["adj"]
    print(f"  A2C seed {s}: dBIC={adj['hinge']['delta_bic']:.1f} "
          f"t*={adj['hinge']['t_star']} thinned "
          f"p0={adj['thinning']['parity0']['delta_bic']:.1f} "
          f"p1={adj['thinning']['parity1']['delta_bic']:.1f}")
print("A2C primary dBIC range:", min(dbics), max(dbics))
a2csucc = [a2c["seeds"][s]["final_success"] for s in a2c["seeds"]]
print("A2C success range:", min(a2csucc), max(a2csucc))

# openness curve numbers for figure text
s0 = b5["seeds"]["0"]
curve = s0.get("side_openness_curve")
if curve:
    print("seed0 curve first5:", [round(c, 3) for c in curve[:5]],
          "last3:", [round(c, 3) for c in curve[-3:]])

# ---------------------------------------------------------------- Fig 3
sec("TWO TIMESCALES (formation vs realization)")
lgf = load("learn_grip_formation.json")
ro = lgf["registered_outcomes"]
print("formation_b5:", ro["formation_b5_count"], " realization_b5:",
      ro["realization_b5_count"], " learned:", ro["n_learned"])
fine = load("learn_grip_formation_fine.json")
ro = fine["registered_outcomes"]
print("FINE ocap_b5:", ro["ocap_b5_count"], " succ_b5:", ro["succ_b5_count"],
      " midpoints:", ro["midpoints"])

# ---------------------------------------------------------------- Fig 4
sec("SOURCE TYPOLOGY on learned systems")
lst = load("learn_stance_transport.json")
for s, d in lst["seeds"].items():
    lad = d.get("ladder") or {}
    if lad:
        print(f"  stance seed {s}: ladder keys {list(lad.keys())}")
        print("   ", {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in lad.items()})
        break
ro = lst["registered_outcomes"]
print("LST outcomes:", {k: ro[k] for k in
      ["LST1_learnability", "LST3_relational_collapse", "n_learned"]})

tric = load("triad_highorder_cue.json")
ro = tric["registered_outcomes"]
print("TRI-C:", {k: ro[k] for k in ro if k.startswith("TRIC")})
for s, d in tric["seeds"].items():
    last = d[max(d.keys(), key=int)]
    print(f"  tric seed {s} final-ckpt: { {k: (round(v,4) if isinstance(v,float) else v) for k,v in last.items()} }")

e1c = load("overcooked_profile_confirmatory.json")
print("E1-C learned C_env:", e1c["learned"]["C_env"])
print("E1-C scripted C_env:", e1c["noisy_scripted"]["C_env"])
print("E1-C outcomes:", e1c["registered_outcomes"])

# ---------------------------------------------------------------- Fig 5
sec("CONTROLLABILITY")
lgu = load("learn_grip_utility.json")
ro = lgu["registered_outcomes"]
print("mean_switch_by_tau:", {k: round(v, 3)
      for k, v in ro["mean_switch_by_tau"].items()})
print("baseline race:", ro["baseline_race"])
print("align hits:", ro["align_hits"])

lss = load("learn_stance_sticky.json")
print("STICKY baseline race:", lss["registered_outcomes"]["baseline_race"])
lsc = load("learn_stance_control.json")
print("CONTROL baseline race:", lsc["registered_outcomes"]["baseline_race"])
print("CONTROL stick_p:", lsc["config"]["stick_p"],
      " STICKY stick_p:", lss["config"]["stick_p"])

aic = load("ant_conditional_leverage.json")
print("ANT-INT-C bins:", aic["bins"])
ro = aic["registered_outcomes"]
print("ANT-INT-C:", {k: ro[k] for k in ro})
print("n_pairs:", aic["n_pairs"])

# ---------------------------------------------------------------- Fig 6
sec("LAWS AND BREADTH")
acb = load("ant_colony_breakpoint.json")
for size, d in acb["per_size"].items():
    h = d.get("hinge") or {}
    print(f"  ant N={size}: dBIC={h.get('delta_bic')} "
          f"slopes {h.get('slope_before')}->{h.get('slope_after')} "
          f"onset={d.get('b5_onset', d.get('onset'))}")
print("ACB outcomes:", acb["registered_outcomes"])

ks = load("kuramoto_scale.json")
print("KUR-SCALE mean_t_star:", ks["registered_outcomes"]["mean_t_star"])
print("KUR-SCALE mean_post_slope:",
      ks["registered_outcomes"]["mean_post_slope"])
print("KUR-SCALE passing_K:", ks["registered_outcomes"]["passing_K"])

k2 = load("kuramoto_breakpoint_r2.json")
print("KUR-BP-R2:", k2["registered_outcomes"])

tbp = load("tri_c_breakpoint.json")
print("TRI-C-BP:", tbp["registered_outcomes"])

b72 = load("bench72_factorial.json")
print("BENCH-72:", b72["registered_outcomes"])

dc = load("definition_calibration.json")
print("DEF-CAL:", dc["registered_outcomes"])

sd = load("collapse_source_decomposition.json")
print("SD:", sd["registered_outcomes"])

# scope boundary: learned-population negatives
for f, key in [("learn_n_exact.json", None), ("learn_eta_breakpoint.json",
               None), ("learn_quorum_breakpoint.json", None),
               ("overcooked_state_breakpoint.json", None),
               ("overcooked_occupancy_breakpoint.json", None)]:
    try:
        d = load(f)
        ro = d.get("registered_outcomes", {})
        print(f"  {f}: {ro}")
    except FileNotFoundError:
        print(f"  {f}: MISSING")

sec("GENESIS / EARLY WARNING")
for s in ["93001", "93002", "93003"]:
    g = load(f"overcooked_genesis_curve_curve_s{s}.json")
    keys = list(g.keys())
    print(f"  seed {s}: t_seed={g['t_seed_steps']} t_visible={g['t_visible_steps']}")

sec("REVISION BATCH: DETECTOR-VAL / REPR-ROBUST / CONVENTION / ROLES / FSS / CONFOUND")
dv = load("detector_validation.json")
print("DETECTOR-VAL:", dv["registered_outcomes"])
print("  by_density:", dv["by_density"])
print("  by_sigma:", {k: v for k, v in dv["by_sigma"].items()})
print("  ref loc err (onset):", dv["reference_point"]["onset"]["median_loc_err_frac"])

rr = load("repr_robustness.json")
print("REPR-ROBUST:", rr["registered_outcomes"])
for name, s in rr["systems"].items():
    print(f"  {name}: onset_frac={s['frac_onset_among_gated']} "
          f"t*=[{s['t_star_min']},{s['t_star_max']}] range_frac={s['t_star_range_frac']}")
print("  object note:", rr["object_semantics_note"])

lc = load("learn_convention.json")
print("CONVENTION:", lc["registered_outcomes"])
for k, r in lc["seeds"].items():
    h = r["adj"].get("hinge", {})
    print(f"  seed {k}: S={r['final_success']} B5={r['adj']['b5_onset']} "
          f"dBIC={h.get('delta_bic')} t*={h.get('t_star')} s090={r['success_090_cross']}")

lr = load("learn_roles.json")
print("ROLES:", lr["registered_outcomes"])
for k, r in lr["seeds"].items():
    h = r["adj"].get("hinge", {})
    print(f"  seed {k}: S={r['final_success']} B5={r['adj']['b5_onset']} "
          f"dBIC={h.get('delta_bic')} t*={h.get('t_star')} s090={r['success_090_cross']}")

fss = load("ant_fss.json")
print("ANT-FSS:", fss["registered_outcomes"])

try:
    cc = load("learn_grip_confound.json")
    print("GRIP-CONFOUND:", cc["registered_outcomes"])
    print("  cc1_by_tau:", cc["cc1_by_tau"])
    print("  cc3:", cc["cc3_logistic"])
except FileNotFoundError:
    print("GRIP-CONFOUND: pending")

tce = load("tri_c_breakpoint_ext.json")
print("TRI-C-EXT:", tce["registered_outcomes"])

try:
    gp = load("learn_grip_policy.json")
    print("GRIP-POLICY:", gp["registered_outcomes"])
except FileNotFoundError:
    print("GRIP-POLICY: pending")

# ---- revision round 2 (representation, NN replication, SD audit) ----

rec = load("repr_equiv_convention.json")
print("REPR-EQUIV-CONV:", rec["registered_outcomes"])
for r, c in rec["representation_cells"].items():
    print(f"  {r}: B5={c['b5_onset']} t*={c['t_star']} dBIC={c['delta_bic']}")

reg = load("repr_equiv_grip.json")
print("REPR-EQUIV-GRIP:", reg["registered_outcomes"])
for r, c in reg["representation_cells"].items():
    print(f"  {r}: B5={c['b5_onset']} t*={c['t_star']} dBIC={c['delta_bic']}")

for fn, tag in (("learn_convention_nn.json", "CONV-NN(coarse grid)"),
                ("learn_roles_nn.json", "ROLES-NN(coarse grid)")):
    d = load(fn)
    print(tag + ":", d["registered_outcomes"])

nr = load("learn_nn_resolution.json")
print("NN-RES:", nr["registered_outcomes"])
for s in ("convention", "roles"):
    print(f"  {s}:", nr["systems"][s]["outcomes"])
    dbics = [r["delta_bic"] for r in nr["systems"][s]["seeds"].values()
             if r["b5_onset"] and r["delta_bic"] is not None]
    print(f"    onset dBIC range: {min(dbics):.1f}-{max(dbics):.1f}")

ni = load("learn_nn_init.json")
print("NN-INIT:", ni["registered_outcomes"])
print("  monotonicity:", ni["monotonicity"])

sd = load("sd_audit.json")
print("SD-AUDIT:", sd["registered_outcomes"])
d2 = sd["results"]["SDA2_off_family"]["detail"]
print("  modular_sum:", d2["modular_sum"]["components"])
print("  markov_chain:", d2["markov_chain"]["components"])
print("  dirichlet:", d2["dirichlet_50"])
d3 = sd["results"]["SDA3_nesting_order"]["detail"]
print("  order shifts: ind", d3["max_individual_shift_bits"],
      "high", d3["max_high_shift_bits"],
      "env/pair", round(d3["max_env_pair_split_shift_bits"], 4))
d4 = sd["results"]["SDA4_sample_complexity"]["detail"]
print("  n=30000 errors:", {c: d4["error_table"]["30000"][c]["median_abs_err"]
                            for c in d4["error_table"]["30000"]})
print("  n=300 C_high err:", d4["error_table"]["300"]["C_high"]["median_abs_err"])

# ---- revision round 3 (standard-environment mechanism recovery) ----

oc = load("overcooked_ring_convention.json")
print("OC-RING:", oc["registered_outcomes"])
for grp in ("ring", "cramped"):
    for s, r in oc["systems"][grp].items():
        print(f"  {grp} {s}: circB5={r['circ_adj']['b5_onset']} "
              f"t*={r['circ_adj']['t_star']} p_ccw={r['final_p_ccw']} "
              f"soups={r['final_soups']} cross={r['capability_crossing']}")

oce = load("oc_ring_ext.json")
print("OC-RING-EXT pooled:", oce["registered_outcomes"])

mp = load("mpe_spread_ppo.json")
print("MPE-PPO:", mp["registered_outcomes"])

ec = load("emergence_certificates.json")
print("CERTIFICATES:")
for name, c in ec["certificates"].items():
    e = c["eip"] or {}
    print(f"  {name}: {c['verdict']} ampl={e.get('amplitude_fraction_closed')}")

# ---- revision round 4 (realization probe, semi-synthetic injection) ----

rr = load("oc_ring_realization.json")
print("OC-RING-REAL:", rr["registered_outcomes"])

si = load("semi_inject.json")
print("SEMI-INJ:", si["registered_outcomes"])

oi = load("oc_ring_intervention.json")
print("OC-RING-INT:", oi["registered_outcomes"])


def barrier_xplay():
    d = load("barrier_xplay.json")
    o = d["registered_outcomes"]
    print("== BARRIER-XPLAY ==")
    for k in ("determinism_ok", "BX1_regime_exclusivity",
              "BX2_no_unilateral_gradient_pre", "BX3_lockin_post",
              "BX4_barrier_asymmetry", "conv_within_intel",
              "conv_cross_intel", "conv_adoption_gain_pre",
              "conv_deviation_cost_post", "roles_within_success",
              "roles_hybrid_success", "roles_adoption_gain_pre",
              "roles_deviation_cost_post", "n_cross_pairs_conv",
              "n_hybrid_teams_roles"):
        print(f"  {k}: {o[k]}")


barrier_xplay()


def oci_seed_level():
    d = load("oci_seed_level.json")
    print("== OCI seed-level (post-hoc) ==")
    print("  moved:", d["moved"])
    print("  strict_flip:", d["strict_flip"])


oci_seed_level()
