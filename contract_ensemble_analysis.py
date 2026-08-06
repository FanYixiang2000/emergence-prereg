"""Observer-contract ensemble: R_contract per system across stored audits.

Instead of claiming observer independence, this analysis reports, for
every stored Contextual-LBF system, the fraction of reasonable declared
contracts under which it is accepted:

    R_contract(system) = Pr[verdict = 1 | contract in ensemble]

The ensemble pools the SEVEN contracts already measured with frozen
thresholds (no new episodes; read-only over stored outputs):

    A   hand basins, horizon 15, discounted reward, T=1 rollouts
        (registered confirmation/extension);
    B   speed basins, horizon 12, success value, T=1 (dual-contract);
    C   k-means summary-feature basins (machine-partitioned);
    D-F cross-fitted LOW-LEVEL basins: k-means, GMM, Ward
        (silhouette-selected k, train/test split);
    G   hand basins under near-greedy (T=0.2) rollouts.

(The diffuse T=2.0 model is excluded by the declared epistemic/behavioural
rollout distinction: it perturbs the behaving policy itself; its results
remain reported in the rollout audit as the retained IR-2 miss.)

Declared classification:
    contract-invariant positive   R = 1
    contract-stable positive      R >= 6/7
    value/contract-relative       1/7 <= R < 6/7
    contract-invariant negative   R = 0

Expectation stated before running the summary: every control is
contract-invariant negative; most learned seeds are contract-stable
positive; the known borderline seeds (1104 and the contract-B usefulness
flips) land in the relative band with identifiable routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

OUTPUTS = Path(__file__).resolve().parent / "outputs"

CONF_SEEDS = tuple(str(s) for s in range(1101, 1111))
EXT_SEEDS = tuple(str(s) for s in range(1201, 1206))
SYSTEMS = ("learned", "initial_twin", "team_nearest",
           "fixed_food0", "fixed_food1")


def stored_hand() -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for path, seeds in (
        (OUTPUTS / "contextual_lbf_confirmation.json", CONF_SEEDS),
        (OUTPUTS / "contextual_lbf_extension.json", EXT_SEEDS),
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        for seed in seeds:
            out[seed] = {
                name: int(s["verdict"]["emergent"])
                for name, s in data["seeds"][seed]["systems"].items()}
    return out


def main() -> None:
    seeds = list(CONF_SEEDS) + list(EXT_SEEDS)
    contracts: Dict[str, Dict[str, Dict[str, int]]] = {}

    contracts["A_hand_registered"] = stored_hand()

    dual = json.loads((OUTPUTS / "dual_observer_contracts.json").read_text())
    contracts["B_speed_success_h12"] = {
        seed: {name: int(dual["seeds"][seed][name]["verdict"]["emergent"])
               for name in SYSTEMS} for seed in seeds}

    lb = json.loads((OUTPUTS / "learned_basin_clbf.json").read_text())
    contracts["C_summary_kmeans"] = {
        seed: {name: int(lb["seeds"][seed][name]["verdict"]["emergent"])
               for name in SYSTEMS} for seed in seeds}

    xf = json.loads((OUTPUTS / "crossfit_lowlevel_basins.json").read_text())
    for method, tag in (("kmeans", "D_lowlevel_kmeans"),
                        ("gmm", "E_lowlevel_gmm"),
                        ("ward", "F_lowlevel_ward")):
        contracts[tag] = {
            seed: {name: int(xf["seeds"][seed][method]["verdicts"][name])
                   for name in SYSTEMS} for seed in seeds}

    ir = json.loads((OUTPUTS / "independent_rollout_audit.json").read_text())
    contracts["G_hand_neargreedy"] = {}
    for seed in seeds:
        near = ir["seeds"][seed]["near_greedy"]
        row = {"learned": int(near["learned"]["verdict"]["emergent"]),
               "initial_twin": int(
                   near["initial_twin"]["verdict"]["emergent"])}
        # scripted controls were not re-rolled under G; their rejections
        # are partition- and rollout-independent (endogeneity/acquisition
        # fail by provenance), so contract A verdicts carry over.
        for name in ("team_nearest", "fixed_food0", "fixed_food1"):
            row[name] = contracts["A_hand_registered"][seed][name]
        contracts["G_hand_neargreedy"][seed] = row

    n_contracts = len(contracts)
    r_contract: Dict[str, Dict[str, Any]] = {}
    bands = {"invariant_positive": 0, "stable_positive": 0,
             "relative": 0, "invariant_negative": 0}
    for seed in seeds:
        r_contract[seed] = {}
        for name in SYSTEMS:
            accepts = sum(contracts[c][seed][name] for c in contracts)
            r = accepts / n_contracts
            if r == 1.0:
                band = "invariant_positive"
            elif r >= 6 / 7 - 1e-9:
                band = "stable_positive"
            elif r > 0:
                band = "relative"
            else:
                band = "invariant_negative"
            bands[band] += 1
            r_contract[seed][name] = {
                "accepts": accepts, "n_contracts": n_contracts,
                "R_contract": r, "band": band}

    learned_r = [r_contract[s]["learned"]["R_contract"] for s in seeds]
    control_accepts = sum(
        r_contract[s][n]["accepts"] for s in seeds for n in SYSTEMS
        if n != "learned")
    summary = {
        "status": ("observer-contract ensemble over seven stored "
                   "frozen-threshold contracts; read-only analysis; "
                   "expectation declared in the docstring"),
        "contracts": list(contracts),
        "R_contract": r_contract,
        "bands": bands,
        "headline": {
            "controls_contract_invariant_negative":
                f"{60 * n_contracts - control_accepts}"
                f"/{60 * n_contracts} rejections "
                f"({control_accepts} acceptances)",
            "learned_mean_R": sum(learned_r) / len(learned_r),
            "learned_at_least_6_of_7":
                sum(1 for r in learned_r if r >= 6 / 7 - 1e-9),
            "learned_relative_band":
                [s for s in seeds
                 if r_contract[s]["learned"]["band"] == "relative"],
        },
    }
    out = OUTPUTS / "contract_ensemble_analysis.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["bands"], indent=1))
    print(json.dumps(summary["headline"], indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
