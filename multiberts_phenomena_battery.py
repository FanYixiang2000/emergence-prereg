"""Phenomena battery on the public MultiBERTs series: abrupt vs gradual.

Registered in MULTIBERTS_PREREGISTRATION.md ("Registered extension:
phenomena battery") BEFORE any probe below was evaluated. Thresholds and
analysis are imported frozen; the forward pass, downloader, and vocabulary
come from multiberts_collapse_probe.

Four probe families on the same published checkpoints:

- reflexive:  subject-reflexive number agreement   (expected: emergent)
- determiner: demonstrative-noun number agreement  (expected: emergent)
- facts:      country-capital recall               (registered: NOT
              emergent -- this prediction FAILED; see the outcome note in
              the pre-registration: high-frequency facts are learned early)
- npi:        negative-polarity-item licensing     (registered follow-up
              R5: NOT emergent; documented as hard/late for BERT-type
              models in the BLiMP literature)

The point: on a system trained entirely by another lab, the criterion must
not only accept documented abrupt abilities but also reject at least one
documented hard/late ability -- separating emergence from accumulation
inside the same model, with the same frozen thresholds. Failures of our
auxiliary "which abilities are gradual" predictions are reported as
registered failures, never adjusted away.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

import torch

from grokking_collapse_bridge import THRESHOLDS, analyze_run, entropy_bits, kl_bits, verdict
from multiberts_collapse_probe import (
    CKPT_DIR,
    OUTPUTS,
    STEPS,
    BertWeights,
    Vocab,
    bert_forward,
    download_checkpoint,
)

SG_NOUNS = ["author", "pilot", "farmer", "teacher", "doctor", "manager",
            "guard", "driver", "customer"]
PL_NOUNS = ["authors", "pilots", "farmers", "teachers", "doctors",
            "managers", "guards", "drivers", "customers"]
REFLEXIVE_VERBS = ["blamed", "hurt", "taught", "helped", "trusted",
                   "described", "defended", "introduced"]

DET_NOUN_PAIRS = [("book", "books"), ("car", "cars"), ("house", "houses"),
                  ("dog", "dogs"), ("tree", "trees"), ("idea", "ideas"),
                  ("song", "songs"), ("game", "games"), ("story", "stories"),
                  ("picture", "pictures")]
DET_CONTEXTS = ["he likes", "she bought", "i remember", "they discussed",
                "we found", "you mentioned", "everyone loved"]

NPI_VPS = [["seen", "the", "ocean"], ["visited", "this", "city"],
           ["read", "that", "book"], ["heard", "this", "song"],
           ["won", "a", "prize"], ["tried", "this", "game"]]

CAPITALS = [("france", "paris"), ("italy", "rome"), ("england", "london"),
            ("germany", "berlin"), ("spain", "madrid"), ("russia", "moscow"),
            ("japan", "tokyo"), ("china", "beijing"), ("greece", "athens"),
            ("austria", "vienna"), ("ireland", "dublin"), ("norway", "oslo"),
            ("poland", "warsaw"), ("portugal", "lisbon"),
            ("sweden", "stockholm"), ("egypt", "cairo"), ("turkey", "ankara"),
            ("canada", "ottawa"), ("cuba", "havana"), ("scotland", "edinburgh")]


def build_families(vocab: Vocab) -> Dict[str, List[Dict]]:
    families: Dict[str, List[Dict]] = {
        "reflexive": [], "determiner": [], "facts": [], "npi": [],
    }

    for word in (SG_NOUNS + PL_NOUNS + REFLEXIVE_VERBS
                 + [w for p in DET_NOUN_PAIRS for w in p]
                 + [w for ctx in DET_CONTEXTS for w in ctx.split()]
                 + [w for pair in CAPITALS for w in pair]
                 + [w for vp in NPI_VPS for w in vp]
                 + ["himself", "herself", "themselves", "this", "these",
                    "no", "have", "ever", "never",
                    "the", "capital", "of", "is", "[CLS]", "[SEP]", "[MASK]"]):
        assert word in vocab.index, f"probe word not in vocab: {word}"

    him, her, them = vocab.id("himself"), vocab.id("herself"), vocab.id("themselves")
    for sg, pl in zip(SG_NOUNS, PL_NOUNS):
        for verb in REFLEXIVE_VERBS:
            families["reflexive"].append({
                "words": ["[CLS]", "the", sg, verb, "[MASK]", ".", "[SEP]"],
                # singular credit: best singular reflexive beats plural
                "correct_ids": [him, her], "wrong_ids": [them],
            })
            families["reflexive"].append({
                "words": ["[CLS]", "the", pl, verb, "[MASK]", ".", "[SEP]"],
                "correct_ids": [them], "wrong_ids": [him, her],
            })

    for noun_sg, noun_pl in DET_NOUN_PAIRS:
        for ctx in DET_CONTEXTS:
            base = ctx.split()
            families["determiner"].append({
                "words": ["[CLS]"] + base + ["this", "[MASK]", ".", "[SEP]"],
                "correct_ids": [vocab.id(noun_sg)], "wrong_ids": [vocab.id(noun_pl)],
            })
            families["determiner"].append({
                "words": ["[CLS]"] + base + ["these", "[MASK]", ".", "[SEP]"],
                "correct_ids": [vocab.id(noun_pl)], "wrong_ids": [vocab.id(noun_sg)],
            })

    ever, never = vocab.id("ever"), vocab.id("never")
    for noun_pl in PL_NOUNS:
        for vp in NPI_VPS:
            families["npi"].append({
                "words": ["[CLS]", "no", noun_pl, "have", "[MASK]"] + vp + [".", "[SEP]"],
                "correct_ids": [ever], "wrong_ids": [never],
            })
            families["npi"].append({
                "words": ["[CLS]", "the", noun_pl, "have", "[MASK]"] + vp + [".", "[SEP]"],
                "correct_ids": [never], "wrong_ids": [ever],
            })

    # fixed wrong capital: shift by one (derangement over the list)
    for i, (country, capital) in enumerate(CAPITALS):
        wrong_capital = CAPITALS[(i + 1) % len(CAPITALS)][1]
        families["facts"].append({
            "words": ["[CLS]", "the", "capital", "of", country, "is",
                      "[MASK]", ".", "[SEP]"],
            "correct_ids": [vocab.id(capital)], "wrong_ids": [vocab.id(wrong_capital)],
        })
    return families


def family_batches(families: Dict[str, List[Dict]], vocab: Vocab):
    batches = {}
    for name, probes in families.items():
        by_len: Dict[int, List[Dict]] = {}
        for probe in probes:
            by_len.setdefault(len(probe["words"]), []).append(probe)
        groups = []
        for _, group in sorted(by_len.items()):
            ids = torch.tensor([[vocab.id(w) for w in p["words"]] for p in group])
            mask_pos = torch.tensor([p["words"].index("[MASK]") for p in group])
            groups.append((ids, mask_pos, group))
        batches[name] = groups
    return batches


def evaluate(w: BertWeights, batches) -> Dict[str, Dict[str, object]]:
    out = {}
    with torch.no_grad():
        for name, groups in batches.items():
            probs_all, acc_all = [], []
            for ids, mask_pos, group in groups:
                logits, _ = bert_forward(w, ids)
                rows = torch.arange(len(ids))
                sel = logits[rows, mask_pos]
                probs_all.append(torch.softmax(sel, dim=1))
                for r, probe in enumerate(group):
                    best_correct = max(float(sel[r, i]) for i in probe["correct_ids"])
                    best_wrong = max(float(sel[r, i]) for i in probe["wrong_ids"])
                    acc_all.append(1.0 if best_correct > best_wrong else 0.0)
            out[name] = {
                "probs": torch.cat(probs_all),
                "acc": sum(acc_all) / len(acc_all),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiBERTs phenomena battery.")
    parser.add_argument("--steps", type=int, nargs="*", default=STEPS)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    tag = "" if args.model_seed == 0 else f"_seed{args.model_seed}"

    torch.set_num_threads(16)
    vocab = Vocab(CKPT_DIR / "vocab.txt")
    families = build_families(vocab)
    batches = family_batches(families, vocab)
    for name, probes in families.items():
        print(f"{name}: {len(probes)} probes", flush=True)

    rows_by_family: Dict[str, List[Dict[str, float]]] = {n: [] for n in families}
    p0: Dict[str, torch.Tensor] = {}
    skipped: List[int] = []
    for step in args.steps:
        t0 = time.time()
        try:
            prefix = download_checkpoint(step, args.model_seed)
        except Exception as exc:
            print(f"step {step}: DOWNLOAD FAILED ({exc}); skipped", flush=True)
            skipped.append(step)
            continue
        w = BertWeights(prefix)
        result = evaluate(w, batches)
        parts = [f"step {step:8d}"]
        for name in families:
            probs = result[name]["probs"]
            if name not in p0:
                p0[name] = probs.clone()
            h_bits = entropy_bits(probs)
            c_bits = kl_bits(probs, p0[name])
            acc = result[name]["acc"]
            rows_by_family[name].append({
                "run": name, "epoch": step, "train_acc": acc, "test_acc": acc,
                "test_entropy_bits": h_bits, "collapse_bits": c_bits,
                "embedding_jump": 0.0,
            })
            parts.append(f"{name} acc {acc:.3f} H {h_bits:.2f}")
        print(" | ".join(parts) + f" [{time.time()-t0:.0f}s]", flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(CKPT_DIR / f"seed_{args.model_seed}_step_{step}",
                          ignore_errors=True)

    summary: Dict[str, object] = {
        "thresholds": THRESHOLDS, "skipped_steps": skipped, "runs": {},
    }
    all_rows: List[Dict[str, float]] = []
    for name, rows in rows_by_family.items():
        all_rows.extend(rows)
        stats = analyze_run(rows)
        v = verdict(stats, prespecified=False)
        summary["runs"][name] = {"stats": stats, "verdict": v}
        failed = ";".join(k for k, ok in v["passes"].items() if not ok) or "-"
        print(f"{name}: emergent={v['emergent']} failed={failed} "
              f"(window {stats['window_epoch']}, acc gain "
              f"{stats['usefulness_acc_gain']:+.3f})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"multiberts_phenomena_timeseries{tag}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / f"multiberts_phenomena_summary{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / f'multiberts_phenomena_summary{tag}.json'}")


if __name__ == "__main__":
    main()
