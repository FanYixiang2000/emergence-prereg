"""Generate the reproducibility manifest for the current manuscript.

Derives the set of stored outputs the manuscript depends on directly
from the three consumer scripts (number audit, figures, Supplementary
Tables), records each file's SHA-256 and generating script, hashes
every protocol document and every file in outputs/, and carries the
invalid/quarantined-artifact registry. Read-only apart from
manifest.json itself.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

CONSUMERS = {
    "number_audit": "verify_paper_numbers.py",
    "figures": "make_figures.py",
    "si_tables": "make_si_tables.py",
}

# Output stems whose generating script is not the same-named .py file.
GENERATOR_EXCEPTIONS = {
    "emergence_certificates": "emergence_certificate.py",
    "method_baseline_battery": "bench_baselines.py",
}

INVALID = [
    {"artifact": "outputs/pythia_collapse_summary_2.8b_mirror_invalid.json",
     "reason": "mirror served one weight object for early revisions "
               "(detected 2026-07-13 before any manuscript use)",
     "in_final_statistics": False},
    {"artifact": "outputs/pythia_tail_summary_2.8b_mirror_invalid.json",
     "reason": "same mirror defect", "in_final_statistics": False},
    {"artifact": "external_pythia_2.8b/step64000.invalid_duplicates_step143000",
     "reason": "upstream repo defect: both weight formats duplicate "
               "step143000 (detected by hash audit before the reported run)",
     "in_final_statistics": False},
    {"artifact": "external_pythia_2.8b/step32000/model.safetensors"
                 ".stale_final_weights",
     "reason": "upstream stale single-file object (= final weights); "
               "rebuilt from bin-verified pytorch_model.bin before the "
               "reported run", "in_final_statistics": False},
    {"artifact": "outputs/contextual_lbf_persistence_layoutbug.json",
     "reason": "perturbation layout spec violated lexicographic food-identity "
               "convention; fixed and rerun (amendment recorded)",
     "in_final_statistics": False},
    {"artifact": "outputs/ordinary_learner_control_attempt1_failed_design.json",
     "reason": "97-class ordinal task unlearnable by the frozen architecture "
               "(final acc 0.07): failed design, not an informative control",
     "in_final_statistics": False},
    {"artifact": "outputs/lbf_prior_detectors_round1.json",
     "reason": "registered round-1 set-composition failure, archived; "
               "round 2 frozen before rerun", "in_final_statistics": False},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def referenced_outputs() -> dict[str, list[str]]:
    """Map output filename -> list of consumer roles that read it."""
    refs: dict[str, list[str]] = {}
    for role, script in CONSUMERS.items():
        src = (HERE / script).read_text(encoding="utf-8")
        for name in sorted(set(re.findall(r"[\w.]+\.json", src))):
            if (OUTPUTS / name).exists():
                refs.setdefault(name, [])
                if role not in refs[name]:
                    refs[name].append(role)
    return refs


def generator_of(fname: str) -> str | None:
    stem = fname[: -len(".json")]
    script = GENERATOR_EXCEPTIONS.get(stem, f"{stem}.py")
    return script if (HERE / script).exists() else None


def main() -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "manuscript": ["main.tex", "si.tex"],
        "consumers": CONSUMERS,
        "dependencies": [],
        "invalid_data_registry": INVALID,
        "protocols": {},
        "outputs": {},
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                                capture_output=True, text=True, timeout=10)
        manifest["git_commit"] = (commit.stdout.strip()
                                  if commit.returncode == 0 else None)
    except Exception:
        manifest["git_commit"] = None

    missing_generators = []
    for fname, roles in sorted(referenced_outputs().items()):
        gen = generator_of(fname)
        if gen is None:
            missing_generators.append(fname)
        manifest["dependencies"].append({
            "output": fname,
            "sha256": sha256(OUTPUTS / fname),
            "generating_script": gen,
            "read_by": roles,
        })

    for proto in sorted(HERE.glob("*PREREGISTRATION*.md")) + [
            HERE / "PREDICTION_LEDGER.md",
            HERE / "THEORY.md",
            HERE / "ANALYSIS_FREEZE.md",
            HERE / "CLAIM_EVIDENCE_MAP.md",
            HERE / "REPRODUCIBILITY.md",
            HERE / "INDEPENDENT_AUDIT_INSTRUCTIONS.md",
            HERE / "INVALID_DATA_REGISTRY.md",
            HERE / "NMI_READINESS_AUDIT.md",
            HERE / "EVIDENCE_AUDIT.md",
    ]:
        if proto.exists():
            manifest["protocols"][proto.name] = sha256(proto)

    for out_file in sorted(OUTPUTS.glob("*.json")) + sorted(
            OUTPUTS.glob("*.csv")):
        manifest["outputs"][out_file.name] = sha256(out_file)

    path = HERE / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manuscript dependencies: {len(manifest['dependencies'])}  "
          f"protocols: {len(manifest['protocols'])}  "
          f"outputs hashed: {len(manifest['outputs'])}  "
          f"generator missing: {missing_generators or 'none'}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
