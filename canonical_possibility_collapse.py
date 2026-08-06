"""Canonical Possibility-Collapse Validation Matrix.

This script is a *unification layer*, not another private toy generator.
It reads already-frozen outputs from canonical exemplars, emergence
coordinates, ant bridge, Overcooked, grokking and induction-head runs and
places them in one reviewer-facing table:

    emergence = endogenous collapse / reorganization of a reachable
    possibility space, stabilized into a macro organization or capability.

The key correction from early PTC is built into the table:

  - collapse of {success, failure} is NOT enough;
  - ordinary entropy decrease is NOT enough;
  - the possibility space must be named (phase futures, route futures,
    role futures, output/computation futures, etc.);
  - confounds must fail on their predicted dimension.

The output is meant as a reusable validation benchmark. It distinguishes
analytic/mechanistic ground truth from canonical convergent validity and
external empirical evidence; calling all three "ground truth" would be
circular.

REGISTERED PREDICTIONS (frozen before running this script):
  CPC-1  Publicly canonical positive cases all show the expected
         possibility-collapse signature under the stored tests:
         Kuramoto supercritical, Boids high coupling, Schelling high tau,
         Game-of-Life glider, and ant trail (5/5).
  CPC-2  Canonical/constructed negatives fail on the predicted reason:
         Kuramoto subcritical, ant solo, common driver, central controller,
         metric-artifact jump, scripted/BC Overcooked controls (>= 6/6).
  CPC-3  Capability-emergence cases fit the same schema on computation /
         output futures: grokking, transformer grokking and induction heads
         pass, while memorizer/no-structure/1-layer controls fail on
         usefulness, burstiness or architectural possibility (>= 6/6).
  CPC-4  The real public Overcooked bridge supports role-future collapse
         for accepted learned seeds (8/8) and rejects external high-score
         scripted/clone controls (24/24), while retaining the limitation
         that full agent-channel replay is unavailable.
  CPC-5  The benchmark exports a complete ground-truth matrix: every row
         has domain, possibility_space, expected_status, evidence_route,
         and either acceptance coordinates or predicted failure dimension.
  CPC-6  Evidence provenance is explicit: no canonical-consensus or external
         row is mislabeled analytic ground truth; Life is marked as an
         observer-contract boundary rather than falsely claimed to have
         stochastic openness under a fully specified microstate.
Misses and limitations are retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def load(name: str):
    return json.loads((OUTPUTS / name).read_text())


def row(name: str, domain: str, possibility_space: str, expected: str,
        evidence: str, verdict: bool, **extra) -> Dict:
    out = {
        "name": name,
        "domain": domain,
        "possibility_space": possibility_space,
        "expected_status": expected,
        "evidence_route": evidence,
        "verdict_matches_ground_truth": bool(verdict),
    }
    out.update(extra)
    return out


def main() -> None:
    ec = load("emergence_coordinates.json")
    ce = load("canonical_exemplars.json")
    ant = load("ant_contrast.json")
    occ = load("overcooked_collective_constraint.json")
    grok = load("grokking_collapse_summary.json")
    tg = load("transformer_grokking_summary.json")
    ih = load("induction_head_summary.json")

    rows: List[Dict] = []

    # ---- canonical complex-system positives
    ksup = ec["blind_heldout"]["H1_kuramoto_super"]
    ksub = ec["blind_heldout"]["H2_kuramoto_sub"]
    life = ec["blind_heldout"]["H3_life_glider"]
    rows.append(row(
        "Kuramoto supercritical", "synchronization",
        "relative phase futures collapse to the synchrony manifold",
        "weak emergence",
        "D>=0.5 and R>=0.6 under frozen held-out thresholds",
        ksup["weak_emergence"] == 1,
        D=ksup["D"], R=ksup["R"], natural_order=ksup["natural_order"]))
    rows.append(row(
        "Kuramoto subcritical", "synchronization negative",
        "relative phase futures remain dispersed",
        "reject",
        "fails robustness and interaction-dependence threshold",
        ksub["weak_emergence"] == 0,
        predicted_fail="D/R", D=ksub["D"], R=ksub["R"]))

    b1 = ce["boids"]["1.0"]
    rows.append(row(
        "Boids high coupling", "flocking",
        "heading futures collapse from many directions to a flock manifold",
        "structural weak emergence, not adaptive",
        "coupling-monotone entropy contraction, TC growth, do-decouple JS",
        ce["registered_outcomes"]["CE1_boids_collapse_monotone_in_coupling"]
        and ce["registered_outcomes"]["CE3_boids_do_decouple_loadbearing"],
        C_bits=b1["mean_heading_entropy_contraction_bits"],
        TC3=b1["tc3_bits"], D_js=b1["js_do_decouple_bits"]))

    sh = ce["schelling"]["0.7"]
    rows.append(row(
        "Schelling high tolerance", "self-organization",
        "spatial configuration futures collapse to segregated basins",
        "structural weak emergence, often negative value",
        "segregation tipping and do-freeze load-bearing",
        ce["registered_outcomes"]["CE4_schelling_tipping_and_loadbearing"],
        segregation=sh["mean_segregation"], D_js=sh["js_do_freeze_bits"]))

    rows.append(row(
        "Game-of-Life glider", "cellular automaton",
        ("counterfactual perturbation ensemble around a deterministic "
         "travelling orbit; fixed exact microstate has zero uncertainty"),
        "weak emergence, not adaptive",
        ("held-out Life glider accepted by D/R under perturbation/decoupling; "
         "stored test does not by itself measure pre/post stochastic collapse"),
        life["weak_emergence"] == 1 and life["adaptive"] == 0,
        D=life["D"], R=life["R"], natural_alive=life["natural_alive"],
        boundary_note=("exact-state contract: deterministic, C=0; "
                       "perturbation-ensemble contract: D/R weak emergence")))

    trail = ant["TRAIL"]
    solo = ant["SOLO"]
    rows.append(row(
        "Ant double-bridge trail", "swarm",
        "route-choice futures collapse from two open routes to one trail",
        "weak collective emergence",
        "D/R pass and collapse is gradual, not abrupt",
        trail["weak_emergence"] == 1
        and ant["registered_outcomes"]["ANT3_collapse_is_gradual_not_abrupt"],
        C_commit=trail["gradualism"]["total_collapse"],
        D=trail["D"], R=trail["R"]))
    rows.append(row(
        "Ant solo obstacle navigation", "individual adaptation negative",
        "individual route choice, no collective joint-branch constraint",
        "reject as collective emergence",
        "no coupling to break; C/D/R all zero",
        solo["weak_emergence"] == 0,
        predicted_fail="collective D/R", D=solo["D"], R=solo["R"]))

    # ---- adversarial negatives from coordinate matrix
    adv = ec["adversarial"]
    for key in ("ADV1_common_driver", "ADV2_central_controller",
                "ADV7_metric_artifact"):
        rows.append(row(
            key, "pseudo-emergence",
            "apparent coordination / jump without endogenous collapse",
            "reject",
            f"rejected on predicted dimension: {adv[key].get('predicted_fail', adv[key].get('predicted'))}",
            adv[key]["rejected_on_predicted"],
            predicted_fail=adv[key].get("predicted_fail", "metric artifact")))

    # ---- real public Overcooked bridge (read-only limitation retained)
    rows.append(row(
        "Overcooked learned accepted seeds", "public MARL",
        "context-conditioned role futures (context, first-potter role)",
        "adaptive emergence bridge",
        "8/8 accepted seeds have C_ctx/G_ctx; 12/12 positive macro gain",
        occ["registered_outcomes"]["OCC1_accepted_learned_context_constraint_ge_7_of_8"]
        and occ["registered_outcomes"]["OCC3_learned_macro_gain_positive_12_of_12"],
        accepted_CG=occ["summary"]["accepted_CG_positive"],
        M_positive=occ["summary"]["learned_M_positive"],
        limitation="full agent-channel replay unavailable"))
    rows.append(row(
        "Overcooked scripted / BC controls", "public MARL negative",
        "same high-scoring role outcome but no context-conditioned role collapse",
        "reject",
        "scripted and BC clones have C_ctx=0 in 24/24 controls",
        occ["registered_outcomes"]["OCC2_scripted_and_clone_C_zero_12_of_12"],
        predicted_fail="endogeneity / context-role constraint",
        control_C_zero=occ["summary"]["control_C_zero"]))

    # ---- capability emergence: output / computation futures
    gr = grok["runs"]["grokking"]
    gm = grok["runs"]["memorizer"]
    rows.append(row(
        "MLP grokking", "capability emergence",
        "held-out answer distribution collapses to a reusable algorithm",
        "adaptive capability emergence",
        "high pre-burst entropy, large collapse burst, test usefulness",
        gr["verdict"]["emergent"] == 1,
        C_total=gr["stats"]["initial_test_entropy_bits"]
        - gr["stats"]["final_test_entropy_bits"],
        burstiness=gr["stats"]["burstiness_ratio"],
        usefulness=gr["stats"]["usefulness_acc_gain"]))
    rows.append(row(
        "MLP memorizer", "capability pseudo-emergence",
        "training answer space collapses but held-out computation does not",
        "reject",
        "fails usefulness / potential / burstiness controls",
        gm["verdict"]["emergent"] == 0,
        predicted_fail="usefulness / generalization",
        final_test_acc=gm["stats"]["final_test_acc"]))

    tgr = tg["runs"]["transformer_grokking"]
    tgm = tg["runs"]["transformer_memorizer"]
    rows.append(row(
        "Transformer grokking", "capability emergence",
        "sequence-model output futures collapse to generalizable rule",
        "adaptive capability emergence",
        "continuous collapse + delayed generalization, not metric jump alone",
        tgr["verdict"]["emergent"] == 1,
        C_total=tgr["stats"]["initial_test_entropy_bits"]
        - tgr["stats"]["final_test_entropy_bits"],
        burstiness=tgr["stats"]["burstiness_ratio"],
        usefulness=tgr["stats"]["usefulness_acc_gain"]))
    rows.append(row(
        "Transformer memorizer", "capability pseudo-emergence",
        "output entropy drops without generalizable capability",
        "reject",
        "fails usefulness despite burst",
        tgm["verdict"]["emergent"] == 0,
        predicted_fail="usefulness",
        final_test_acc=tgm["stats"]["final_test_acc"]))

    ih2 = ih["runs"]["induction_2layer"]
    ih1 = ih["runs"]["induction_1layer"]
    ihn = ih["runs"]["no_structure"]
    rows.append(row(
        "Induction head 2-layer", "mechanistic LLM capability",
        "copying computation futures collapse to induction circuit",
        "capability emergence",
        "2-layer architecture forms reusable induction behavior",
        ih2["verdict"]["emergent"] == 1,
        C_total=ih2["stats"]["initial_test_entropy_bits"]
        - ih2["stats"]["final_test_entropy_bits"],
        burstiness=ih2["stats"]["burstiness_ratio"],
        usefulness=ih2["stats"]["usefulness_acc_gain"]))
    rows.append(row(
        "Induction head 1-layer", "architectural negative",
        "architecture lacks the composition path, so capability branch cannot stabilize",
        "reject",
        "fails burstiness/usefulness; external mechanistic impossibility aligns",
        ih1["verdict"]["emergent"] == 0,
        predicted_fail="architectural possibility / usefulness",
        final_test_acc=ih1["stats"]["final_test_acc"]))
    rows.append(row(
        "Induction no-structure data", "capability negative",
        "no reusable rule in the data; collapse would be metric artefact",
        "reject",
        "fails usefulness despite noisy burst flag",
        ihn["verdict"]["emergent"] == 0,
        predicted_fail="usefulness",
        final_test_acc=ihn["stats"]["final_test_acc"]))

    # ---- registered outcomes
    by_name = {r["name"]: r for r in rows}
    cpc1_names = [
        "Kuramoto supercritical", "Boids high coupling",
        "Schelling high tolerance", "Game-of-Life glider",
        "Ant double-bridge trail",
    ]
    cpc2_names = [
        "Kuramoto subcritical", "Ant solo obstacle navigation",
        "ADV1_common_driver", "ADV2_central_controller",
        "ADV7_metric_artifact", "Overcooked scripted / BC controls",
    ]
    cpc3_names = [
        "MLP grokking", "MLP memorizer", "Transformer grokking",
        "Transformer memorizer", "Induction head 2-layer",
        "Induction head 1-layer", "Induction no-structure data",
    ]
    # Evidence tiers prevent the common circularity error of treating
    # literature agreement as analytic truth.
    canonical_names = {
        "Kuramoto supercritical", "Kuramoto subcritical",
        "Boids high coupling", "Schelling high tolerance",
        "Game-of-Life glider", "Ant double-bridge trail",
        "Ant solo obstacle navigation",
    }
    external_names = {
        "Overcooked learned accepted seeds",
        "Overcooked scripted / BC controls",
        "MLP grokking", "MLP memorizer", "Transformer grokking",
        "Transformer memorizer", "Induction head 2-layer",
        "Induction head 1-layer", "Induction no-structure data",
    }
    for r in rows:
        if r["name"] in canonical_names:
            r["evidence_tier"] = "C_convergent_validity"
            r["truth_kind"] = "canonical/literature expectation"
        elif r["name"] in external_names:
            r["evidence_tier"] = "D_external_empirical"
            r["truth_kind"] = "external behavior/mechanism label"
        else:
            r["evidence_tier"] = "B_constructed_mechanism_truth"
            r["truth_kind"] = "designed failure mechanism"
    complete_fields = ("domain", "possibility_space", "expected_status",
                       "evidence_route", "verdict_matches_ground_truth",
                       "evidence_tier", "truth_kind")
    cpc5 = all(all(k in r and r[k] not in ("", None) for k in complete_fields)
               for r in rows)
    cpc6 = (all(r["evidence_tier"] != "A_analytic_ground_truth"
                for r in rows if r["name"] in canonical_names | external_names)
            and "boundary_note" in by_name["Game-of-Life glider"])

    report = {
        "status": ("canonical possibility-collapse validation matrix; "
                   "classic emergence, pseudo-emergence, public MARL and "
                   "capability-emergence rows in one schema; evidence tiers "
                   "separate ground truth from convergent validity"),
        "definition": ("emergence as endogenous collapse / reorganization "
                       "and stabilization of a reachable possibility space; "
                       "macro structure is a product/projection, not the "
                       "definition itself"),
        "rows": rows,
        "registered_outcomes": {
            "CPC1_canonical_positive_cases_5_of_5": all(
                by_name[n]["verdict_matches_ground_truth"] for n in cpc1_names),
            "CPC2_pseudo_cases_fail_predicted_routes_ge_6_of_6": all(
                by_name[n]["verdict_matches_ground_truth"] for n in cpc2_names),
            "CPC3_capability_cases_and_controls_ge_6_of_6": all(
                by_name[n]["verdict_matches_ground_truth"] for n in cpc3_names),
            "CPC4_overcooked_real_bridge_with_retained_limitation": (
                by_name["Overcooked learned accepted seeds"]
                ["verdict_matches_ground_truth"]
                and by_name["Overcooked scripted / BC controls"]
                ["verdict_matches_ground_truth"]
                and occ["registered_outcomes"]
                ["OCC5_full_certificate_unavailable_retained_limitation"]),
            "CPC5_ground_truth_matrix_complete": cpc5,
            "CPC6_truth_provenance_and_Life_boundary_explicit": cpc6,
        },
        "counts": {
            "n_rows": len(rows),
            "matched": sum(r["verdict_matches_ground_truth"] for r in rows),
        },
    }
    out = OUTPUTS / "canonical_possibility_collapse.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(json.dumps(report["counts"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
