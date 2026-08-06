"""E1-C: confirmatory source-profile separation at matched product.

Registered in V2_ALIGNMENT_PREREGISTRATION.md (frozen before this
run). Fresh evaluation seeds (97_501 / 97_601) and fresh noise seed
(89_100); same frozen policy artifacts and estimator as E1-B.

E1C-1: C_env(learned) > C_env(noisy) with non-overlapping 95% CIs.
E1C-2: collapse_norm(learned) > collapse_norm(noisy), same criterion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

np.Inf = np.inf
import torch

import overcooked_criterion as oc
from overcooked_pilot import PolicyNet
from overcooked_genesis_comparison import LEARNED_CKPT
from overcooked_product_matched_genesis import NoisyScripted
from overcooked_source_profile_matched import certify

OUTPUTS = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    torch.set_num_threads(2)
    e1 = json.loads((OUTPUTS /
                     "overcooked_product_matched_genesis.json")
                    .read_text(encoding="utf-8"))
    eps_star = e1["eps_star"]

    noisy = NoisyScripted(eps_star, seed=89_100)
    row_noisy = certify(noisy, 97_501)
    print("noisy done", flush=True)
    net = PolicyNet()
    net.load_state_dict(torch.load(LEARNED_CKPT, weights_only=True,
                                   map_location="cpu"))
    net.eval()
    row_learned = certify(oc.TeamPolicy("net", net=net), 97_601)
    print("learned done", flush=True)

    def separated(key: str) -> bool:
        lo_l = row_learned[f"{key}_ci95"][0]
        hi_n = row_noisy[f"{key}_ci95"][1]
        return row_learned[key] > row_noisy[key] and lo_l > hi_n

    outcomes = {
        "E1C_1_env_separation": separated("C_env"),
        "E1C_2_total_separation": separated("collapse_norm"),
    }
    report = {
        "status": ("E1-C confirmatory profile separation; registered "
                   "with fresh seeds before running; falsification: "
                   "any CI overlap drops the claim"),
        "eps_star": eps_star,
        "noisy_scripted": row_noisy,
        "learned": row_learned,
        "registered_outcomes": outcomes,
    }
    out = OUTPUTS / "overcooked_profile_confirmatory.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {"val": round(row_learned[k], 4),
                          "ci": [round(x, 4) for x in
                                 row_learned[f"{k}_ci95"]],
                          "noisy_val": round(row_noisy[k], 4),
                          "noisy_ci": [round(x, 4) for x in
                                       row_noisy[f"{k}_ci95"]]}
                      for k in ("C_env", "collapse_norm")}, indent=2))
    print(json.dumps(outcomes, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
