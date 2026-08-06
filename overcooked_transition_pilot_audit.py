"""Aggregate Overcooked transition pilot outputs.

This is an audit helper for the real-vs-ghost transition scaffold. It compares
scripted nulls, untrained policies and learned pilot checkpoints without
promoting any pilot to a confirmatory flagship result.
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def maybe_load(name: str):
    path = OUTPUTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row(label: str, data: dict | None):
    if data is None:
        return {"label": label, "available": False}
    overall = data["overall"]
    return {
        "label": label,
        "available": True,
        "policy": data["policy"],
        "n": overall["n"],
        "G_js_bits": overall["G_js_bits"],
        "C_signed_bits": overall["C_signed_bits"],
        "M_score_gain": overall["M_score_gain"],
        "real_score": overall["real_score"],
        "cut_score": overall["cut_score"],
        "partner_action_tv": overall["partner_action_tv"],
        "J_temporal_concentration": data["J_temporal_concentration"],
        "smoke_outcomes_all_pass": all(
            data["registered_smoke_outcomes"].values()),
    }


def main() -> None:
    labels = [
        ("scripted_null", "overcooked_transition_certificate_smoke_scripted.json"),
        ("initial_untrained", "overcooked_transition_certificate_smoke_initial.json"),
        ("learned_40k", "overcooked_transition_certificate_pilot40k_s92001.json"),
        ("learned_500k", "overcooked_transition_certificate_pilot500k_s92002.json"),
        ("learned_2m", "overcooked_transition_certificate_pilot2m_s92003.json"),
    ]
    rows = [row(label, maybe_load(name)) for label, name in labels]
    available = [r for r in rows if r["available"]]
    learned = [r for r in available if r["label"].startswith("learned")]
    report = {
        "status": ("Overcooked transition pilot audit; descriptive only, "
                   "not a confirmatory learned flagship"),
        "rows": rows,
        "interpretation_rules": [
            "A learned positive requires non-trivial G/N/R plus macro effect; "
            "G alone is not enough.",
            "Partner-action TV is a diagnostic for cut mismatch and must be "
            "kept small or modelled with a marginal null.",
            "M=0 means no functional positive claim, even if G is finite.",
        ],
        "summary": {
            "n_available": len(available),
            "learned_available": len(learned),
            "any_learned_positive_M": any(
                r.get("M_score_gain", 0.0) > 0 for r in learned),
            "max_learned_G": max(
                [r.get("G_js_bits", 0.0) for r in learned] or [0.0]),
        },
    }
    out = OUTPUTS / "overcooked_transition_pilot_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    for r in rows:
        print(json.dumps(r, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
