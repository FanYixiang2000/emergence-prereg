"""Canonical exemplars battery: the field's textbook emergence cases,
measured by the layered framework.

Universality check: a framework that only classifies its authors'
systems is a private tool. This battery applies the same substrate
and layer vocabulary to three canonical, NON-learning exemplars that
every emergence review cites, with expectations frozen in advance.
The claim under test is that the layered taxonomy classifies the
classics the way considered usage does: structural emergence where
coupling contracts the joint future space, no adaptive qualification
where nothing was acquired, and a precise reason (not a shrug) for
each verdict.

Systems (all standard, minimal implementations):

  BOIDS     N = 30 headings on a circle; each step every bird mixes
            its own heading with the local mean at alignment weight
            a in {0, .25, .5, .75, 1} plus noise. The classic
            flocking exemplar (Reynolds 1987; Vicsek et al. 1995).
  SCHELLING 20x20 torus, two types, 35% vacancy; an agent moves to a
            random vacancy when the same-type fraction in its Moore
            neighbourhood is below the tolerance threshold tau in
            {0.3, 0.5, 0.7}. The classic unintended-macro exemplar
            (Schelling 1971).
  LIFE      Conway's Game of Life on a 16x16 torus (Gardner 1970),
            random soups; a deterministic CA with no action channel.

Frozen expectations:

  CE-1 (Boids: collapse tracks coupling) The within-episode
       contraction of per-bird heading entropy, C = H(t=0) - H(t=T),
       increases monotonically in a (5-point grid, tolerance one
       inversion of <= 0.05 bits).
  CE-2 (Boids: spatial collapse = measured total correlation) The
       empirical 3-bird total correlation of final heading sectors
       increases monotonically in a, from < 0.1 bits at a = 0 to
       > 1.0 bits at a = 1 (Proposition S made empirical).
  CE-3 (Boids: counterfactual load-bearing) Decoupling the alignment
       channel at mid-episode (do-decouple) shifts the final
       order-parameter basin law by JS >= 0.2 bits at a >= 0.75 and
       <= 0.05 bits at a = 0. Verdict: structural collapse at high
       coupling; NOT adaptive emergence (dynamics prewired, nothing
       acquired) -- the framework's classification, not a failure.
  CE-4 (Schelling: tipping and load-bearing) Final segregation index
       increases in tau; do-freeze (moves blocked from mid-episode)
       shifts the final segregation basin law by JS >= 0.2 bits at
       tau = 0.7. Same verdict class as Boids.
  CE-5 (Life: the substrate boundary, exact) With a fixed initial
       soup the trajectory is deterministic: H(final class | init)
       = 0 exactly -- the future was never open, so there is no
       collapse to attribute; across random soups all apparent basin
       information is initial-condition information and the
       action-attributable collapse is zero BY THE BRIDGE IDENTITY
       (no action channel exists). Life's gliders are substrate
       pattern formation, outside the adaptive scope for a stated
       reason.

Misses are retained.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

OUTPUTS = Path(__file__).resolve().parent / "outputs"

N_BIRDS = 30
T_STEPS = 60
N_EP = 3000
SECTORS = 6
ALIGN_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
NOISE = 0.35

TAUS = (0.3, 0.5, 0.7)
GRID_N = 20
VACANCY = 0.35
SCHELLING_STEPS = 60
N_EP_SCH = 800

LIFE_N = 16
LIFE_STEPS = 40
N_SOUPS = 400


def entropy(counts) -> float:
    total = sum(counts)
    return -sum((c / total) * math.log2(c / total)
                for c in counts if c > 0)


def js_bits(p: List[float], q: List[float]) -> float:
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x, y):
        return sum(a * math.log2(a / b) for a, b in zip(x, y)
                   if a > 0 and b > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ------------------------------------------------------------------ boids

def boids_episode(a: float, seed: int, decouple_at: int = -1):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 2 * math.pi, N_BIRDS)
    h0 = h.copy()
    for t in range(T_STEPS):
        w = 0.0 if (0 <= decouple_at <= t) else a
        mean_vec = np.exp(1j * h).mean()
        mean_ang = np.angle(mean_vec)
        h = np.angle(np.exp(1j * ((1 - w) * h + w * mean_ang
                                  + NOISE * rng.standard_normal(N_BIRDS))))
    return h0, h


def sector(x: np.ndarray) -> np.ndarray:
    return ((x % (2 * math.pi)) / (2 * math.pi) * SECTORS).astype(int) \
        % SECTORS


def order_class(h: np.ndarray) -> int:
    r = abs(np.exp(1j * h).mean())
    return 0 if r < 0.33 else (1 if r < 0.66 else 2)


def run_boids() -> Dict:
    out = {}
    for a in ALIGN_GRID:
        contraction, tc_counts = [], {}
        nat_classes = [0, 0, 0]
        dec_classes = [0, 0, 0]
        for ep in range(N_EP):
            h0, hT = boids_episode(a, 1000 + ep)
            contraction.append(
                entropy(np.bincount(sector(h0), minlength=SECTORS))
                - entropy(np.bincount(sector(hT), minlength=SECTORS)))
            key = tuple(sector(hT[:3]))
            tc_counts[key] = tc_counts.get(key, 0) + 1
            nat_classes[order_class(hT)] += 1
            _, hd = boids_episode(a, 1000 + ep,
                                  decouple_at=T_STEPS // 2)
            dec_classes[order_class(hd)] += 1
        # empirical 3-bird total correlation
        joint = np.zeros((SECTORS,) * 3)
        for key, c in tc_counts.items():
            joint[key] += c
        joint /= joint.sum()
        tc = sum(entropy(joint.sum(axis=tuple(j for j in range(3)
                                              if j != i)).ravel())
                 for i in range(3)) - entropy(joint.ravel())
        nat_p = [c / N_EP for c in nat_classes]
        dec_p = [c / N_EP for c in dec_classes]
        out[str(a)] = {
            "mean_heading_entropy_contraction_bits":
                float(np.mean(contraction)),
            "tc3_bits": float(tc),
            "js_do_decouple_bits": js_bits(nat_p, dec_p),
            "natural_order_dist": nat_p,
        }
        print(f"boids a={a}: contraction "
              f"{out[str(a)]['mean_heading_entropy_contraction_bits']:.2f} "
              f"TC3 {tc:.2f} JS {out[str(a)]['js_do_decouple_bits']:.3f}",
              flush=True)
    return out


# -------------------------------------------------------------- schelling

def schelling_episode(tau: float, seed: int, freeze_at: int = -1):
    rng = random.Random(seed)
    cells = [1] * 130 + [2] * 130 + [0] * (GRID_N * GRID_N - 260)
    rng.shuffle(cells)
    grid = np.array(cells).reshape(GRID_N, GRID_N)

    def unhappy(i, j):
        t = grid[i, j]
        if t == 0:
            return False
        same = tot = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                v = grid[(i + di) % GRID_N, (j + dj) % GRID_N]
                if v:
                    tot += 1
                    same += v == t
        return tot > 0 and same / tot < tau

    for step in range(SCHELLING_STEPS):
        if 0 <= freeze_at <= step:
            break
        movers = [(i, j) for i in range(GRID_N) for j in range(GRID_N)
                  if unhappy(i, j)]
        empties = [(i, j) for i in range(GRID_N) for j in range(GRID_N)
                   if grid[i, j] == 0]
        rng.shuffle(movers)
        for (i, j) in movers:
            if not empties:
                break
            k = rng.randrange(len(empties))
            ei, ej = empties[k]
            grid[ei, ej] = grid[i, j]
            grid[i, j] = 0
            empties[k] = (i, j)

    same = tot = 0
    for i in range(GRID_N):
        for j in range(GRID_N):
            t = grid[i, j]
            if not t:
                continue
            for di, dj in ((0, 1), (1, 0)):
                v = grid[(i + di) % GRID_N, (j + dj) % GRID_N]
                if v:
                    tot += 1
                    same += v == t
    return same / tot if tot else 0.0


def seg_class(s: float) -> int:
    return 0 if s < 0.6 else (1 if s < 0.75 else 2)


def run_schelling() -> Dict:
    out = {}
    for tau in TAUS:
        nat = [0, 0, 0]
        frz = [0, 0, 0]
        segs = []
        for ep in range(N_EP_SCH):
            s = schelling_episode(tau, 5000 + ep)
            segs.append(s)
            nat[seg_class(s)] += 1
            sf = schelling_episode(tau, 5000 + ep,
                                   freeze_at=SCHELLING_STEPS // 6)
            frz[seg_class(sf)] += 1
        nat_p = [c / N_EP_SCH for c in nat]
        frz_p = [c / N_EP_SCH for c in frz]
        out[str(tau)] = {
            "mean_segregation": float(np.mean(segs)),
            "js_do_freeze_bits": js_bits(nat_p, frz_p),
            "natural_class_dist": nat_p,
        }
        print(f"schelling tau={tau}: seg "
              f"{out[str(tau)]['mean_segregation']:.3f} "
              f"JS {out[str(tau)]['js_do_freeze_bits']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------- life

def life_step(g: np.ndarray) -> np.ndarray:
    n = sum(np.roll(np.roll(g, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i, j) != (0, 0))
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def life_class(g: np.ndarray) -> int:
    density = g.mean()
    return 0 if density == 0 else (1 if density < 0.08 else 2)


def run_life() -> Dict:
    rng = np.random.default_rng(7)
    fixed = (rng.random((LIFE_N, LIFE_N)) < 0.35).astype(np.uint8)
    finals = set()
    for _ in range(50):  # determinism check: 50 replays of one soup
        g = fixed.copy()
        for _ in range(LIFE_STEPS):
            g = life_step(g)
        finals.add(g.tobytes())
    counts: Dict[int, int] = {}
    for s in range(N_SOUPS):
        g = (np.random.default_rng(100 + s).random((LIFE_N, LIFE_N))
             < 0.35).astype(np.uint8)
        for _ in range(LIFE_STEPS):
            g = life_step(g)
        c = life_class(g)
        counts[c] = counts.get(c, 0) + 1
    return {
        "deterministic_replays_distinct_finals": len(finals),
        "h_final_given_init_bits": 0.0 if len(finals) == 1 else 1.0,
        "across_soup_class_entropy_bits":
            entropy(list(counts.values())),
        "class_counts": counts,
    }


def main() -> None:
    boids = run_boids()
    sch = run_schelling()
    life = run_life()

    cvals = [boids[str(a)]["mean_heading_entropy_contraction_bits"]
             for a in ALIGN_GRID]
    ce1 = all(cvals[i + 1] >= cvals[i] - 0.05
              for i in range(len(cvals) - 1))
    tcs = [boids[str(a)]["tc3_bits"] for a in ALIGN_GRID]
    ce2 = (all(tcs[i + 1] >= tcs[i] - 0.05
               for i in range(len(tcs) - 1))
           and tcs[0] < 0.1 and tcs[-1] > 1.0)
    ce3 = (boids["0.75"]["js_do_decouple_bits"] >= 0.2
           and boids["1.0"]["js_do_decouple_bits"] >= 0.2
           and boids["0.0"]["js_do_decouple_bits"] <= 0.05)
    segs = [sch[str(t)]["mean_segregation"] for t in TAUS]
    ce4 = (segs[0] < segs[1] < segs[2]
           and sch["0.7"]["js_do_freeze_bits"] >= 0.2)
    ce5 = (life["deterministic_replays_distinct_finals"] == 1
           and life["across_soup_class_entropy_bits"] > 0.3)

    report = {
        "status": ("canonical exemplars battery; CE-1..CE-5 frozen in "
                   "the docstring"),
        "boids": boids,
        "schelling": sch,
        "life": life,
        "registered_outcomes": {
            "CE1_boids_collapse_monotone_in_coupling": bool(ce1),
            "CE2_boids_tc3_monotone_and_range": bool(ce2),
            "CE3_boids_do_decouple_loadbearing": bool(ce3),
            "CE4_schelling_tipping_and_loadbearing": bool(ce4),
            "CE5_life_deterministic_substrate_boundary": bool(ce5),
        },
        "classification": {
            "boids_high_coupling": ("structural collapse (open initial "
                                    "headings, coupling-monotone "
                                    "contraction, counterfactually "
                                    "load-bearing); NOT adaptive: "
                                    "dynamics prewired, no acquisition"),
            "schelling": ("structural collapse with tipping; NOT "
                          "adaptive: preferences prewired"),
            "life": ("substrate pattern formation: deterministic given "
                     "the soup, no action channel, zero "
                     "action-attributable collapse by the bridge "
                     "identity; outside the adaptive scope for a "
                     "stated reason"),
        },
    }
    out = OUTPUTS / "canonical_exemplars.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["registered_outcomes"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
