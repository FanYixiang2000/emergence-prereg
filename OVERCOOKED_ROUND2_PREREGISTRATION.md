# Preregistration DRAFT: independent held-out replication on Overcooked-AI (round 2)

STATUS: NOT EXECUTED (decision recorded 2026-07-19 in
EVIDENCE_SUFFICIENCY.md). No confirmatory seed was ever launched; no
round-2 verdict data exists. The layout-pair pilots (seeds 8951-8964,
disclosed in INVALID_DATA_REGISTRY.md) showed that most held-out pairs
train only one of the two layouts, so a replication there would
measure PPO training fragility rather than the criterion. Round 1
passed all five registered predictions; its exact-boundary 8/12
acceptance is reported as such in the manuscript. This draft is
retained for transparency and for any future replication by us or by
others.

Version: DRAFT (numbers were to be fixed from disclosed pilots; frozen
by external timestamp BEFORE any confirmatory seed)

## 1. Why a second, separate round

Round 1 (tag `v1.0-overcooked-prereg`) passed all five registered
predictions with the learned acceptance exactly at the registered 8/12
boundary. A sharp reviewer can say: "met exactly at the boundary; the
acceptance rate is uncertain." The correct response is not to extend
round 1 with more seeds (post-hoc extension after seeing 8/12 would be
sample-size chasing) but an INDEPENDENT replication:

- a NEW preregistration (this document, separately timestamped);
- a HELD-OUT layout pair (neither layout used in round 1);
- a NEW seed list, disjoint from round 1;
- the SAME thresholds, unchanged from the frozen criterion;
- results reported separately, never pooled with round 1.

This upgrades the claim from "one task pair passed" to "the protocol
transfers across task pairs within the same public benchmark, under
frozen thresholds, twice preregistered."

## 2. Disclosed design pilots (excluded from confirmation)

- pair_D (coordination_ring + counter_circuit), seed 8951, 3M steps.
- pair_E (coordination_ring + bottleneck), seed 8953, 3M steps.
- Outcomes recorded in `outputs/overcooked_mixed_pilot*.json` and the
  pilot logs; the chosen pair, trigger direction and any budget change
  are fixed in section 3 from these pilots BEFORE the freeze.

## 3. Frozen protocol (TO FIX FROM PILOTS)

Environment: `overcooked_ai_py`, horizon 400, layout pair
[TO FIX: pair_D or pair_E], contexts drawn uniformly per episode.

Training per seed: identical recipe to round 1
(`overcooked_confirmation.py::train_mixed`): self-play PPO, shared
parameters, community shaped rewards annealed to zero over 60% of
steps, sparse delivery reward untouched, [TO FIX: 5M or higher] steps.

Confirmation seeds: 78001..780NN ([TO FIX: N = 20] seeds; round-1
seeds 77001-77012 and pilot seeds excluded).

Candidate macro-structure, trigger, basins, contracts A and B,
systems per seed, thresholds: identical to round 1, copied unchanged.

## 4. Registered predictions (frozen at tag time)

    OC2-1  learned accepted on >= [TO FIX] / N seeds, with the exact
           binomial 95% lower confidence bound of the acceptance rate
           above 0.40.
    OC2-2  all 4N control verdicts are rejections.
    OC2-3  trigger direction matches the pilot sign on >= 80% of
           accepted seeds.
    OC2-4  every learned rejection routes through conditional
           selectivity (possibly with acquisition), not through
           potential/specificity/usefulness.
    OC2-5  learned usefulness do-contrast positive on >= N-2 seeds
           (exact one-sided sign test p < 0.01).
    OC2-6  contract-B twin rejections N/N.

Any miss is retained. Results reported separately from round 1.
