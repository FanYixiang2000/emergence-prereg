"""Leave-one-component-out audit with named witnesses, across the measured
six-component domains (Contextual LBF confirmation + extension).

Reviewer question addressed: are all six components individually load-bearing,
or are some empirically redundant? For each component we recompute every
system's verdict with that component removed and report:

- how many additional systems become (falsely) accepted, with names;
- whether any true positive is lost (cannot happen: dropping a conjunct can
  only add acceptances);
- which components are redundant on this data (no verdict changes).

Pure re-analysis of stored evaluation JSONs. No stored output is modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

THRESHOLDS = {
    "potential_bits": 0.5,
    "conditional_selectivity": 0.5,
    "specificity_js_bits": 0.2,
    "usefulness_gap": 0.0,
    "acquisition": 0.3,
}
COMPONENTS = ("potential", "conditional_selectivity", "specificity",
              "usefulness", "endogeneity", "acquisition")


def component_passes(system: Dict[str, Any]) -> Dict[str, bool]:
    metrics = system["metrics"]
    frozen = system["verdict"]["passes"]
    return {
        "potential": metrics["potential_bits"] >= THRESHOLDS["potential_bits"],
        "conditional_selectivity": (
            metrics["conditional_selectivity"]
            >= THRESHOLDS["conditional_selectivity"]),
        "specificity": (
            metrics["specificity_js_bits"]
            >= THRESHOLDS["specificity_js_bits"]),
        "usefulness": metrics["usefulness_gap"] > THRESHOLDS["usefulness_gap"],
        "endogeneity": bool(frozen["endogeneity"]),
        "acquisition": (
            float(system.get("acquisition", 0.0)) >= THRESHOLDS["acquisition"]),
    }


def load_systems(path: Path, tag: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for seed, entry in data["seeds"].items():
        for name, system in entry["systems"].items():
            out.append({
                "id": f"{tag}:seed{seed}:{name}",
                "is_learned": name == "learned",
                "passes": component_passes(system),
                "frozen_emergent": int(system["verdict"]["emergent"]),
            })
    return out


def main() -> None:
    systems = (
        load_systems(OUTPUTS / "contextual_lbf_confirmation.json", "conf")
        + load_systems(OUTPUTS / "contextual_lbf_extension.json", "ext")
    )
    for s in systems:
        full = all(s["passes"].values())
        assert int(full) == s["frozen_emergent"], f"mismatch on {s['id']}"

    report: Dict[str, Any] = {
        "status": "leave-one-component-out audit, second full-criterion domain",
        "n_systems": len(systems),
        "n_frozen_accepts": sum(s["frozen_emergent"] for s in systems),
        "components": {},
    }
    for drop in COMPONENTS:
        newly_accepted = []
        for s in systems:
            reduced = all(v for k, v in s["passes"].items() if k != drop)
            if reduced and not s["frozen_emergent"]:
                newly_accepted.append(s["id"])
        control_leaks = [n for n in newly_accepted if ":learned" not in n]
        learned_gains = [n for n in newly_accepted if ":learned" in n]
        report["components"][drop] = {
            "newly_accepted_total": len(newly_accepted),
            "false_positive_controls": control_leaks,
            "borderline_learned_admitted": learned_gains,
            "empirically_redundant_here": len(newly_accepted) == 0,
        }

    # Witness table: for each component, a measured system that passes the
    # other five but fails exactly this one (five-of-six adversarial witness).
    witnesses: Dict[str, List[str]] = {c: [] for c in COMPONENTS}
    for s in systems:
        failed = [c for c, ok in s["passes"].items() if not ok]
        if len(failed) == 1:
            witnesses[failed[0]].append(s["id"])
    report["five_of_six_witnesses"] = {
        c: {"n": len(ids), "examples": ids[:6]} for c, ids in witnesses.items()
    }

    out = OUTPUTS / "component_ablation_witnesses.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for c, item in report["components"].items():
        print(f"drop {c:24s}: +{item['newly_accepted_total']} accepts "
              f"(controls: {len(item['false_positive_controls'])}) "
              f"redundant={item['empirically_redundant_here']}")
    print("five-of-six witnesses:",
          {c: v["n"] for c, v in report["five_of_six_witnesses"].items()})
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
