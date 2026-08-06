"""Unified continuous emergence profile (certificate + graded record).

The binary six-component certificate answers WHETHER an adaptive
emergent structure is identified; this module defines the CONTINUOUS
record that answers how strongly, how abruptly, in which value
direction, and with how much acquired content. Declared before the
construct-calibration experiments; aggregation forms are fixed here and
never fitted to labels.

Per-system record (all under a declared observer contract):

    P   potential            H(B_pre) / log|B|                [0, 1]
    S   context selectivity  |trigger-rate separation|        [0, 1]
    M   causal structural magnitude
                             JS(do-trigger, do-block) / 1 bit clipped,
                             or normalized I_do(A;B|C)        [0, 1]
    V   signed value effect  tanh(do-contrast / sigma_V)      [-1, 1]
    Q   acquisition          clip(M_trained - M_init
                             + S_trained - S_init, 0, 2)/2    [0, 1]
    A   temporal abruptness  1 - H(collapse increments)/log(T-1)
                                                              [0, 1]
    D   discovery surprise   -log2 P(pattern | provenance prior)  bits

Summary scalars (secondary; the vector is the record):

    E_struct  = (P * S * M) ** (1/3)          multiplicative: no
                                              dimension can compensate
                                              a missing one
    E_adapt   = E_struct * sqrt(Q) * tanh(V/sigma_V)   signed
    cert_margin = soft-min over standardized component margins
                  (lambda = 1): > 0 iff every component clears its
                  frozen threshold, magnitude = distance to the
                  decision boundary (evidence strength, NOT
                  phenomenon strength)

Terminology: none of these is called "strong/weak emergence"; the
philosophical term is not a score.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

SIGMA_V_DEFAULT = 1.0
LAMBDA_SOFTMIN = 1.0


def potential_norm(h_bits: float, n_basins: int) -> float:
    if n_basins <= 1:
        return 0.0
    return max(0.0, min(1.0, h_bits / math.log2(n_basins)))


def magnitude_norm(js_bits: float) -> float:
    """JS between the two do-laws is bounded by 1 bit."""
    return max(0.0, min(1.0, js_bits))


def value_signed(do_contrast: float, sigma_v: float) -> float:
    return math.tanh(do_contrast / max(sigma_v, 1e-9))


def acquisition_norm(m_trained: float, m_init: float,
                     s_trained: float, s_init: float) -> float:
    gain = (m_trained - m_init) + (s_trained - s_init)
    return max(0.0, min(1.0, gain / 2.0))


def abruptness(collapse_series: List[float]) -> Optional[float]:
    """1 - normalized entropy of positive collapse increments."""
    drops = [max(0.0, collapse_series[i - 1] - collapse_series[i])
             for i in range(1, len(collapse_series))]
    total = sum(drops)
    if total <= 0 or len(drops) < 2:
        return None
    q = [d / total for d in drops]
    h = -sum(p * math.log(p) for p in q if p > 0)
    return 1.0 - h / math.log(len(drops))


def e_struct(p: float, s: float, m: float) -> float:
    return (max(p, 0.0) * max(s, 0.0) * max(m, 0.0)) ** (1.0 / 3.0)


def e_adapt(e_str: float, q: float, v_signed: float) -> float:
    return e_str * math.sqrt(max(q, 0.0)) * v_signed


def certificate_margin(margins: Dict[str, float],
                       lam: float = LAMBDA_SOFTMIN) -> float:
    """Soft-min of standardized margins; > 0 iff all components pass."""
    vals = list(margins.values())
    return -lam * math.log(
        sum(math.exp(-m / lam) for m in vals) / len(vals))


def profile(*, h_bits: float, n_basins: int, selectivity: float,
            js_do_bits: float, do_contrast: float, sigma_v: float,
            m_init: float = 0.0, s_init: float = 0.0,
            collapse_series: Optional[List[float]] = None,
            discovery_surprise_bits: Optional[float] = None,
            margins: Optional[Dict[str, float]] = None) -> Dict:
    p = potential_norm(h_bits, n_basins)
    s = max(0.0, min(1.0, selectivity))
    m = magnitude_norm(js_do_bits)
    v = value_signed(do_contrast, sigma_v)
    q = acquisition_norm(m, m_init, s, s_init)
    a = abruptness(collapse_series) if collapse_series else None
    es = e_struct(p, s, m)
    out = {
        "P_potential": p,
        "S_selectivity": s,
        "M_causal_magnitude": m,
        "V_signed_value": v,
        "Q_acquisition": q,
        "A_abruptness": a,
        "D_discovery_surprise_bits": discovery_surprise_bits,
        "E_struct": es,
        "E_adapt": e_adapt(es, q, v),
    }
    if margins is not None:
        out["certificate_margin"] = certificate_margin(margins)
    return out
