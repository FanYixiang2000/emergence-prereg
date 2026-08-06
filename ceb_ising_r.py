"""CEB-ISING-R: dense control-axis Ising battery.

Registered after the coarse-grid near miss and before this run.
Same simulator/contract as CEB-ISING, with a denser temperature grid
around Tc.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ceb_ising import GATE, OUTPUTS, TC, h2_from_abs_m, run_temp
from tri_c_breakpoint import hinge_linear

TEMPS = (4.0, 3.2, 2.8, 2.6, 2.45, 2.35, 2.30, 2.25,
         2.20, 2.15, 2.10, 2.0, 1.8, 1.5, 1.3)
SEED = 90_501


def main() -> None:
    m_abs, openness = [], []
    for i, temp in enumerate(TEMPS):
        m = run_temp(temp, SEED + i * 17)
        o = h2_from_abs_m(m)
        m_abs.append(m)
        openness.append(o)
        print(f"T={temp}: |m|={m:.4f} O={o:.4f}", flush=True)

    drop = openness[0] - openness[-1]
    adj = {"drop": round(float(drop), 4),
           "gate_passed": bool(drop >= GATE)}
    if adj["gate_passed"]:
        x = np.arange(len(TEMPS), dtype=float)
        h = hinge_linear(x, np.array(openness))
        adj["hinge"] = h
        adj["b5_control"] = bool(h["delta_bic"] >= 10)
    else:
        adj["b5_control"] = False

    dm = np.diff(m_abs)
    steep_i = int(np.argmax(dm)) + 1
    hinge_i = int(adj.get("hinge", {}).get("t_star", -99))
    tc_i = min(range(len(TEMPS)), key=lambda i: abs(TEMPS[i] - TC))
    isg1 = bool(adj.get("b5_control"))
    isg2 = bool(isg1 and abs(hinge_i - steep_i) <= 1
                and abs(hinge_i - tc_i) <= 2)
    isg3 = bool(m_abs[0] < 0.2 and m_abs[-1] > 0.8)
    outcomes = {
        "ISGR1_control_collapse": isg1,
        "ISGR2_native_alignment": isg2,
        "ISGR3_high_low_contrast": isg3,
        "hinge_index": hinge_i,
        "steepest_m_index": steep_i,
        "tc_grid_index": tc_i,
    }
    report = {
        "status": "CEB-ISING-R dense Ising battery; preregistered",
        "temperature_high_to_low": TEMPS,
        "abs_magnetization": [round(v, 5) for v in m_abs],
        "openness": [round(v, 5) for v in openness],
        "adj": adj,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "ceb_ising_r.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
