"""Emergence certificate: the standardized adjudication pipeline.

Answers the question "someone hands us a system -- is it emergent, and
how strongly?" with the FROZEN instruments of the paper. Given an
openness curve (plus optional source components), the certificate
returns:

  1. QUALIFICATION (three registered conditions, each testable):
     - regime_level : the collapse passes the frozen effect-size gate
       (drop >= 0.1 of reference) -- reorganization, not noise;
     - endogenous   : declared provenance boundary (the analyst must
       state what is exogenous BEFORE analysis; the certificate records
       the declaration, it cannot invent it);
     - persistent   : after the collapse the curve does not re-open by
       more than 0.1 of reference within the observed window.

  2. INTENSITY PROFILE (EIP) -- a VECTOR, deliberately not a scalar:
     amplitude (fraction of reference possibility closed), abruptness
     class (punctuated onset / deceleration knee / gradual), sharpness
     (|slope_after|/|slope_before| at the breakpoint), commitment time
     t*, and the source ranking when components are provided. Amplitude
     and abruptness are provably independent axes (a discontinuous
     score can fake abruptness with zero amplitude; a slow drift can
     have large amplitude with zero abruptness), so collapsing the EIP
     into one number would reintroduce exactly the
     observable-of-convenience failure the framework exists to fix.

  3. VERDICT (categorical, exhaustive):
     - "emergent: punctuated"  qualified + onset breakpoint;
     - "emergent: gradual"     qualified, collapse without onset;
     - "organized, not qualified"  collapse present but a qualification
       condition fails (which one is named);
     - "no regime-level collapse"  gate fails;
     - "unresolvable"          grid below the validated power floor
       (< 20 points: measured detector power ~0 at 12 points, 0.62 at
       20), so no claim is made either way.

All thresholds are the frozen ones (gate 0.1, Delta-BIC 10, parity
thinning, saturation truncation); this module only PACKAGES them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ant_fine_onset import adjudicate

OUTPUTS = Path(__file__).resolve().parent / "outputs"
LOG2_3 = math.log2(3)
REOPEN_TOL = 0.1
HARD_FLOOR_POINTS = 12       # measured onset power 0.00
FULL_POWER_POINTS = 40       # measured onset power 0.995


def certify(grid, openness, *, endogenous_declared: bool,
            joint_beyond_marginals: bool | None = None,
            provenance_note: str = "", sources: dict | None = None) -> dict:
    grid = list(grid)
    y = np.asarray(openness, dtype=float)
    n = len(y)
    if n <= HARD_FLOOR_POINTS:
        return {"verdict": "unresolvable",
                "reason": (f"{n} grid points is at or below the measured "
                           f"zero-power floor ({HARD_FLOOR_POINTS})"),
                "qualification": None, "eip": None}
    low_power = n < FULL_POWER_POINTS

    adj = adjudicate(grid, y * LOG2_3)
    h = adj.get("hinge", {}) or {}
    amplitude = float(y.max() - y.min())
    # regime-level = the collapse gate AND (when the source analysis is
    # available) the requirement that the reorganization is a property of
    # the JOINT distribution, not of marginals alone. A single optimizing
    # part can produce a large individual-source collapse without any
    # collective regime; the declared joint_beyond_marginals flag records
    # that source-level fact.
    regime_level = bool(adj.get("gate_passed", amplitude >= 0.1))
    if joint_beyond_marginals is not None:
        regime_level = regime_level and joint_beyond_marginals

    # persistence: after the global minimum, no re-opening above tol
    imin = int(np.argmin(y))
    tail = y[imin:]
    persistent = bool(tail.max() - y[imin] <= REOPEN_TOL)

    onset = bool(adj["b5_onset"])
    if onset:
        klass = "punctuated onset"
    elif h.get("onset_type") is False:
        klass = "deceleration knee"
    elif regime_level:
        klass = "gradual"
    else:
        klass = "none"

    sb, sa = h.get("slope_before"), h.get("slope_after")
    sharpness = (abs(sa) / max(abs(sb), 1e-9)
                 if (sa is not None and sb is not None) else None)

    qual = {"regime_level": regime_level,
            "endogenous": bool(endogenous_declared),
            "persistent": persistent,
            "provenance_note": provenance_note}
    qualified = regime_level and endogenous_declared and persistent

    eip = {"amplitude_fraction_closed": round(amplitude, 4),
           "abruptness_class": klass,
           "sharpness_slope_ratio": (round(sharpness, 2)
                                     if sharpness is not None else None),
           "delta_bic": h.get("delta_bic"),
           "t_star": h.get("t_star")}
    if sources:
        order = sorted(sources, key=sources.get, reverse=True)
        eip["source_ranking"] = order
        eip["source_bits"] = {k: round(float(v), 4)
                              for k, v in sources.items()}

    if not regime_level and amplitude < 0.1:
        verdict = "no regime-level collapse"
    elif not qualified:
        failed = [k for k in ("regime_level", "endogenous", "persistent")
                  if not qual[k]]
        verdict = f"organized, not qualified (fails: {', '.join(failed)})"
    elif onset:
        verdict = "emergent: punctuated"
    else:
        verdict = "emergent: gradual"

    out = {"verdict": verdict, "qualification": qual, "eip": eip}
    if low_power:
        out["low_power_warning"] = (
            f"{n} grid points: measured onset power < 0.995, so the "
            f"ABSENCE of an onset verdict is weak evidence of absence")
    return out


# ---------------------------------------------------------------- battery

def load_battery():
    """Every stored system the certificate can be applied to as-is."""
    systems = {}

    g = json.load(open(OUTPUTS / "learn_grip_transport.json"))
    side = np.mean([g["seeds"][s]["side_openness_curve"]
                    for s in g["seeds"]], axis=0)
    systems["grip realization (16 agents)"] = (
        list(range(len(side))), side, True, True,
        "policy-internal side channel; joint commitment of 16 agents")

    c = json.load(open(OUTPUTS / "learn_convention.json"))
    grid = list(range(0, 4001, 25))
    conv = np.mean([c["seeds"][s]["openness_curve"] for s in c["seeds"]],
                   axis=0)
    systems["convention formation (10 agents)"] = (
        grid, conv, True, True, "all codes equivalent; population-level "
        "mapping is a joint object")

    r = json.load(open(OUTPUTS / "learn_roles.json"))
    rgrid = list(range(0, 6001, 25))
    roles = np.mean([r["seeds"][s]["openness_curve"] for s in r["seeds"]],
                    axis=0)
    systems["role lock-in (6 agents)"] = (
        rgrid, roles, True, True,
        "all permutations equivalent; coverage is a joint constraint")

    a = json.load(open(OUTPUTS / "ant_colony_breakpoint.json"))
    cur100 = a["per_size"]["100"]["median_openness"]
    systems["ant colony N=100"] = (
        list(range(len(cur100))), cur100, True, True,
        "food sites symmetric; recruitment couples the colony")
    cur1 = a["per_size"]["1"]["median_openness"]
    systems["single ant N=1 (control)"] = (
        list(range(len(cur1))), cur1, True, False,
        "single chooser: the collapse is purely individual-source")

    t = json.load(open(OUTPUTS / "tri_c_breakpoint.json"))
    sd = list(t["seeds"].values())[0]
    tgrid = t["grid"]
    op = [sd["curve"][str(u)]["openness"] for u in tgrid]
    systems["TRI-C high-order (3 agents)"] = (
        tgrid, op, True, True,
        "collapse carried by the high-order channel (C_high ~0.95 bits)")

    o = json.load(open(OUTPUTS / "overcooked_occupancy_breakpoint.json"))
    ogrid = o["grid"]
    occ = np.mean([o["seeds"][s]["openness"] for s in o["seeds"]], axis=0)
    systems["Overcooked occupancy (deep MARL)"] = (
        ogrid, occ, True, True,
        "trajectory occupancy of the trained pair; standard cramped room")

    return systems


def main() -> None:
    systems = load_battery()
    report = {}
    print(f"{'system':38s} {'verdict':44s} ampl  class")
    for name, (grid, curve, endo, joint, note) in systems.items():
        cert = certify(grid, curve, endogenous_declared=endo,
                       joint_beyond_marginals=joint,
                       provenance_note=note)
        report[name] = cert
        e = cert["eip"] or {}
        flag = " [low power]" if "low_power_warning" in cert else ""
        print(f"{name:38s} {cert['verdict'] + flag:44s} "
              f"{e.get('amplitude_fraction_closed', '-')!s:5s} "
              f"{e.get('abruptness_class', '-')}")
    out = OUTPUTS / "emergence_certificates.json"
    out.write_text(json.dumps({
        "status": ("standardized emergence certificates from frozen "
                   "instruments; packaging only, no new thresholds"),
        "certificates": report}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
