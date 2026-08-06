"""Tail-gradualism rejection test on the public MultiBERTs series.

Registered in MULTIBERTS_PREREGISTRATION.md ("Registered follow-up:
tail-gradualism rejection test") BEFORE any probe below was evaluated on
any checkpoint. Thresholds and analysis imported frozen from the grokking
bridge; forward pass, downloader and vocabulary from the collapse probe.

The over-acceptance test: every ability probed so far on this public
model came out emergent. Here the gradual candidates are selected by the
variable the external literature actually ties to slow accrual --
FREQUENCY (Kandpal et al. 2023 long-tail knowledge; Chang & Bergen 2022
frequency-ordered word acquisition). Three families, same frozen
criterion:

- head_facts: high-frequency country capitals (registered: emergent,
  replicating the R3 outcome)
- tail_facts: capitals with zipf <= 3.4, both directions (registered:
  NOT emergent, route usefulness)
- tail_words: definitional cloze with rare in-vocab targets, frequency-
  matched distractors (registered: NOT emergent, route usefulness)
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
from multiberts_phenomena_battery import CAPITALS as HEAD_CAPITALS

# Capital zipf <= 3.4 (wordfreq), country and capital single-wordpiece.
TAIL_CAPITALS = [
    ("albania", "tirana"), ("slovenia", "ljubljana"), ("armenia", "yerevan"),
    ("slovakia", "bratislava"), ("estonia", "tallinn"), ("georgia", "tbilisi"),
    ("lithuania", "vilnius"), ("uruguay", "montevideo"), ("colombia", "bogota"),
    ("croatia", "zagreb"), ("latvia", "riga"), ("azerbaijan", "baku"),
    ("tunisia", "tunis"), ("jordan", "amman"), ("nepal", "kathmandu"),
    ("belarus", "minsk"), ("switzerland", "bern"), ("romania", "bucharest"),
    ("bangladesh", "dhaka"), ("venezuela", "caracas"), ("libya", "tripoli"),
    ("pakistan", "islamabad"), ("serbia", "belgrade"), ("taiwan", "taipei"),
    ("finland", "helsinki"), ("vietnam", "hanoi"), ("afghanistan", "kabul"),
]

# Definitional cloze, rare in-vocab targets (all zipf <= 3.3, verified
# with wordfreq), written blind to checkpoint behavior. Distractor =
# target of the paired probe from the same stratum (frequency-matched
# by construction, |zipf gap| <= 0.35 within each pair).
TAIL_WORD_PROBES = [
    ("a narrow inlet of the sea between steep cliffs is a [MASK] .", "fjord", "steppe"),
    ("a dry grassy plain in central asia is a [MASK] .", "steppe", "fjord"),
    ("a low wall along the edge of a roof or bridge is a [MASK] .", "parapet", "viaduct"),
    ("a long bridge that carries a railway over a valley is a [MASK] .", "viaduct", "parapet"),
    ("a building where logs are cut into boards is a [MASK] .", "sawmill", "windmill"),
    ("a mill powered by the wind is a [MASK] .", "windmill", "sawmill"),
    ("a dark volcanic glass used for blades is [MASK] .", "obsidian", "cinder"),
    ("a small piece of burned coal or wood is a [MASK] .", "cinder", "obsidian"),
    ("a fast sailing ship with two tall wooden poles is a [MASK] .", "schooner", "frigate"),
    ("a fast warship used to escort other ships is a [MASK] .", "frigate", "schooner"),
    ("a tall bird that hunts fish in shallow water is a [MASK] .", "heron", "trident"),
    ("a spear with three points is a [MASK] .", "trident", "heron"),
    ("a deep wide ditch filled with water around a castle is a [MASK] .", "moat", "crypt"),
    ("an underground room beneath a church where the dead are buried is a [MASK] .", "crypt", "moat"),
    ("a deep narrow valley carved by running water is a [MASK] .", "ravine", "dune"),
    ("a hill of sand shaped by the wind is a [MASK] .", "dune", "ravine"),
    ("a person who studies stars and planets is an [MASK] .", "astronomer", "blacksmith"),
    ("a person who shapes iron with a hammer is a [MASK] .", "blacksmith", "astronomer"),
    ("the flat piece at the back of a boat used for steering is the [MASK] .", "rudder", "keel"),
    ("the long beam along the bottom of a ship is the [MASK] .", "keel", "rudder"),
    ("the hidden home of a wild beast is its [MASK] .", "lair", "pantry"),
    ("a small room where food is stored is a [MASK] .", "pantry", "lair"),
    ("the wide mouth of a river where it meets the sea is an [MASK] .", "estuary", "causeway"),
    ("a raised road across wet ground or water is a [MASK] .", "causeway", "estuary"),
    ("a green forest plant with long divided leaves and no flowers is a [MASK] .", "fern", "quiver"),
    ("a case for carrying arrows is a [MASK] .", "quiver", "fern"),
    ("a tall asian temple tower with many levels is a [MASK] .", "pagoda", "mausoleum"),
    ("a grand building that holds the tombs of the dead is a [MASK] .", "mausoleum", "pagoda"),
    ("a light spear thrown in an athletic contest is a [MASK] .", "javelin", "sabre"),
    ("a curved sword used by cavalry is a [MASK] .", "sabre", "javelin"),
    ("a channel built to carry water across a valley is an [MASK] .", "aqueduct", "spire"),
    ("the tall pointed top of a church tower is a [MASK] .", "spire", "aqueduct"),
    ("the floor of a fireplace where the fire burns is the [MASK] .", "hearth", "vial"),
    ("a small glass bottle for medicine is a [MASK] .", "vial", "hearth"),
    ("heavy material carried by a ship to keep it steady is [MASK] .", "ballast", "armory"),
    ("a building where weapons are stored is an [MASK] .", "armory", "ballast"),
]


def build_families(vocab: Vocab) -> Dict[str, List[Dict]]:
    families: Dict[str, List[Dict]] = {
        "head_facts": [], "tail_facts": [], "tail_words": [],
    }

    for i, (country, capital) in enumerate(HEAD_CAPITALS):
        wrong = HEAD_CAPITALS[(i + 1) % len(HEAD_CAPITALS)][1]
        families["head_facts"].append({
            "words": ["[CLS]", "the", "capital", "of", country, "is",
                      "[MASK]", ".", "[SEP]"],
            "correct_ids": [vocab.id(capital)], "wrong_ids": [vocab.id(wrong)],
        })

    for i, (country, capital) in enumerate(TAIL_CAPITALS):
        wrong_cap = TAIL_CAPITALS[(i + 1) % len(TAIL_CAPITALS)][1]
        wrong_country = TAIL_CAPITALS[(i + 1) % len(TAIL_CAPITALS)][0]
        families["tail_facts"].append({
            "words": ["[CLS]", "the", "capital", "of", country, "is",
                      "[MASK]", ".", "[SEP]"],
            "correct_ids": [vocab.id(capital)], "wrong_ids": [vocab.id(wrong_cap)],
        })
        families["tail_facts"].append({
            "words": ["[CLS]", capital, "is", "the", "capital", "of",
                      "[MASK]", ".", "[SEP]"],
            "correct_ids": [vocab.id(country)], "wrong_ids": [vocab.id(wrong_country)],
        })

    for sentence, target, distractor in TAIL_WORD_PROBES:
        words = ["[CLS]"] + sentence.split() + ["[SEP]"]
        families["tail_words"].append({
            "words": words,
            "correct_ids": [vocab.id(target)], "wrong_ids": [vocab.id(distractor)],
        })

    for name, probes in families.items():
        for probe in probes:
            for word in probe["words"]:
                assert word in vocab.index, f"{name}: not in vocab: {word}"
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


def evaluate(w: BertWeights, batches, metric: str) -> Dict[str, Dict[str, object]]:
    out = {}
    with torch.no_grad():
        for name, groups in batches.items():
            probs_all, acc_all = [], []
            for ids, mask_pos, group in groups:
                logits, _ = bert_forward(w, ids)
                rows = torch.arange(len(ids))
                sel = logits[rows, mask_pos]
                probs_all.append(torch.softmax(sel, dim=1))
                argmax = sel.argmax(dim=1)
                for r, probe in enumerate(group):
                    if metric == "top1":
                        # Registered amendment T5-T7: absolute recall over
                        # the full vocabulary, the metric family on which
                        # gradual frequency-ordered acquisition is
                        # externally documented.
                        acc_all.append(1.0 if int(argmax[r]) in probe["correct_ids"]
                                       else 0.0)
                    else:
                        best_correct = max(float(sel[r, i]) for i in probe["correct_ids"])
                        best_wrong = max(float(sel[r, i]) for i in probe["wrong_ids"])
                        acc_all.append(1.0 if best_correct > best_wrong else 0.0)
            out[name] = {
                "probs": torch.cat(probs_all),
                "acc": sum(acc_all) / len(acc_all),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiBERTs tail-gradualism test.")
    parser.add_argument("--steps", type=int, nargs="*", default=STEPS)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    tag = "" if args.model_seed == 0 else f"_seed{args.model_seed}"

    torch.set_num_threads(32)
    vocab = Vocab(CKPT_DIR / "vocab.txt")
    families = build_families(vocab)
    batches = family_batches(families, vocab)
    for name, probes in families.items():
        print(f"{name}: {len(probes)} probes", flush=True)

    run_names = [n for n in families] + [f"{n}_top1" for n in families]
    rows_by_family: Dict[str, List[Dict[str, float]]] = {n: [] for n in run_names}
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
        results = {"pair": evaluate(w, batches, "pair"),
                   "top1": evaluate(w, batches, "top1")}
        parts = [f"step {step:8d}"]
        for name in families:
            for metric, suffix in (("pair", ""), ("top1", "_top1")):
                run = name + suffix
                result = results[metric]
                probs = result[name]["probs"]
                if run not in p0:
                    p0[run] = probs.clone()
                rows_by_family[run].append({
                    "run": run, "epoch": step,
                    "train_acc": result[name]["acc"],
                    "test_acc": result[name]["acc"],
                    "test_entropy_bits": entropy_bits(probs),
                    "collapse_bits": kl_bits(probs, p0[run]),
                    "embedding_jump": 0.0,
                })
            parts.append(f"{name} pair {results['pair'][name]['acc']:.3f} "
                         f"top1 {results['top1'][name]['acc']:.3f}")
        print(" | ".join(parts) + f" [{time.time()-t0:.0f}s]", flush=True)
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
        final_acc = rows[-1]["test_acc"]
        # T4 scope check: informative rejection = acquired-but-gradual.
        summary["runs"][name] = {
            "stats": stats, "verdict": v, "final_acc": final_acc,
            "t4_reading": ("acquired_gradual" if final_acc >= 0.6 and not v["emergent"]
                           else "never_acquired" if not v["emergent"]
                           else "emergent"),
        }
        failed = ";".join(k for k, ok in v["passes"].items() if not ok) or "-"
        print(f"{name}: emergent={v['emergent']} failed={failed} "
              f"final_acc={final_acc:.3f} (window {stats['window_epoch']}, "
              f"acc gain {stats['usefulness_acc_gain']:+.3f})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"multiberts_tail_timeseries{tag}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / f"multiberts_tail_summary{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / f'multiberts_tail_summary{tag}.json'}")


if __name__ == "__main__":
    main()
