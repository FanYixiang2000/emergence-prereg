"""Continuous emergence profiles of the known system taxonomy.

Read-only assembly: recomputes the declared continuous record
(P, S, M, V, Q, E_struct, E_adapt) from STORED outputs for the systems
whose types the framework distinguishes, plus the learned-harmful and
matched-provenance constructions. Declared expectation (before
assembly): the profiles separate along different axes, not along one
ranking -- learned adaptive is high on all; learned harmful is high on
structure with negative V; the script and clone are high on structure
with Q = 0 or reduced M; the twin is near zero on structure.

Value scale sigma_V is fixed per domain as the task's standing reward
scale, declared here: gridworld team-return units sigma_V = 5; CLBF
discounted-reward units sigma_V = 0.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import emergence_profile as ep

OUTPUTS = Path(__file__).resolve().parent / "outputs"

SIGMA_GRID = 5.0
SIGMA_CLBF = 0.1


def clbf_profiles() -> dict:
    data = json.loads(
        (OUTPUTS / "contextual_lbf_confirmation.json").read_text())
    out = {}
    for name in ("learned", "initial_twin", "team_nearest", "fixed_food0"):
        vals = {"P": [], "S": [], "M": [], "V": [], "Q": [],
                "E_struct": [], "E_adapt": []}
        for seed, entry in data["seeds"].items():
            m = entry["systems"][name]["metrics"]
            init = entry["systems"]["initial_twin"]["metrics"]
            prof = ep.profile(
                h_bits=m["potential_bits"], n_basins=4,
                selectivity=m["conditional_selectivity"],
                js_do_bits=m["specificity_js_bits"],
                do_contrast=m["usefulness_gap"], sigma_v=SIGMA_CLBF,
                m_init=ep.magnitude_norm(init["specificity_js_bits"])
                if name == "learned" else ep.magnitude_norm(
                    m["specificity_js_bits"]),
                s_init=init["conditional_selectivity"]
                if name == "learned" else m["conditional_selectivity"],
            )
            for k, key in (("P", "P_potential"), ("S", "S_selectivity"),
                           ("M", "M_causal_magnitude"),
                           ("V", "V_signed_value"),
                           ("Q", "Q_acquisition"),
                           ("E_struct", "E_struct"),
                           ("E_adapt", "E_adapt")):
                vals[k].append(prof[key])
        out[name] = {k: sum(v) / len(v) for k, v in vals.items()}
    return out


def harmful_profile() -> dict:
    data = json.loads(
        (OUTPUTS / "learned_harmful_emergence.json").read_text())
    rows = {"u_team": [], "u_private": []}
    for seed, m in data["seeds"].items():
        base = dict(
            h_bits=m["potential_bits"], n_basins=5,
            selectivity=m["selectivity_separation"],
            js_do_bits=m["specificity_js_bits"],
            m_init=0.0, s_init=m["untrained_separation"],
        )
        rows["u_team"].append(ep.profile(
            **base, do_contrast=m["usefulness_team"],
            sigma_v=SIGMA_GRID))
        rows["u_private"].append(ep.profile(
            **base, do_contrast=m["usefulness_private"],
            sigma_v=SIGMA_GRID))
    out = {}
    for tag, profs in rows.items():
        out[tag] = {k: sum(p[k] for p in profs) / len(profs)
                    for k in profs[0] if profs[0][k] is not None}
    return out


def provenance_profiles() -> dict:
    data = json.loads((OUTPUTS / "matched_provenance.json").read_text())
    out = {}
    script = data["systems"]["script"]["metrics"]
    out["script"] = ep.profile(
        h_bits=1.0, n_basins=4, selectivity=1.0,
        js_do_bits=script["specificity_js_bits"],
        do_contrast=script["usefulness_do_gap"], sigma_v=SIGMA_GRID,
        m_init=ep.magnitude_norm(script["specificity_js_bits"]),
        s_init=1.0)
    for prov in ("bc_clone", "shaped", "outcome_only"):
        seeds = data["systems"][prov]["seeds"]
        profs = []
        for seed, entry in seeds.items():
            m = entry["metrics"]
            profs.append(ep.profile(
                h_bits=1.0, n_basins=4, selectivity=1.0,
                js_do_bits=m["specificity_js_bits"],
                do_contrast=m["usefulness_do_gap"], sigma_v=SIGMA_GRID,
                m_init=0.0, s_init=0.0,
                discovery_surprise_bits=entry["c_prov_bits"]))
        out[prov] = {k: sum(p[k] for p in profs) / len(profs)
                     for k in profs[0] if profs[0][k] is not None}
    return out


def main() -> None:
    report = {
        "status": ("continuous profiles of the known taxonomy, "
                   "assembled read-only from stored outputs; "
                   "expectation declared in the docstring"),
        "value_scales": {"gridworld": SIGMA_GRID, "clbf": SIGMA_CLBF},
        "clbf": clbf_profiles(),
        "harmful_two_values": harmful_profile(),
        "matched_provenance": provenance_profiles(),
    }
    c = report["clbf"]
    h = report["harmful_two_values"]
    report["axis_separation_check"] = {
        "learned_high_struct_positive_V": bool(
            c["learned"]["E_struct"] > 0.6 and c["learned"]["V"] > 0),
        "twin_near_zero_struct": bool(
            c["initial_twin"]["E_struct"] < 0.3),
        "scripted_struct_without_acquisition": bool(
            c["team_nearest"]["E_struct"] > 0.6
            and c["team_nearest"]["Q"] < 0.1),
        "harmful_same_struct_opposite_V": bool(
            abs(h["u_team"]["E_struct"] - h["u_private"]["E_struct"])
            < 1e-9 and h["u_team"]["V_signed_value"] < 0
            < h["u_private"]["V_signed_value"]),
        "clone_reduced_M_vs_script": bool(
            report["matched_provenance"]["bc_clone"]["M_causal_magnitude"]
            < report["matched_provenance"]["script"]["M_causal_magnitude"]
            - 0.15),
    }
    out = OUTPUTS / "profile_existing_systems.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["axis_separation_check"], indent=1))
    for name, prof in {**report["clbf"],
                       **report["matched_provenance"]}.items():
        print(f"{name:14s} " + " ".join(
            f"{k}={v:+.2f}" for k, v in prof.items()
            if isinstance(v, float)))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
