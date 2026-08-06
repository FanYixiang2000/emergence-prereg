"""Decoder tail-gradualism rejection test on the public Pythia series.

Registered in PYTHIA_PREREGISTRATION.md ("Registered follow-up: decoder
tail-gradualism rejection test") BEFORE any probe below was evaluated on
any checkpoint. Thresholds and analysis imported frozen from the
grokking bridge; downloader, entropy/KL and checkpoint grid from the
Pythia collapse probe; head/tail item lists imported unchanged from the
MultiBERTs tail test (written blind to any checkpoint behavior).

The over-acceptance test on the decoder side: the criterion must REJECT
abilities the external literature ties to slow frequency-driven accrual
(Kandpal et al. 2023 long-tail knowledge; Chang & Bergen 2022
frequency-ordered acquisition) while still ACCEPTING high-frequency
factual recall on the same checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from grokking_collapse_bridge import THRESHOLDS, analyze_run, verdict
from multiberts_phenomena_battery import CAPITALS as HEAD_CAPITALS
from multiberts_tail_gradualism import TAIL_CAPITALS, TAIL_WORD_PROBES
from pythia_collapse_probe import (
    CKPT_DIR,
    OUTPUTS,
    STEPS,
    download_checkpoint,
    download_shared,
    entropy_bits,
    kl_bits,
)

HERE = Path(__file__).resolve().parent
MIN_ITEMS = 10


def title(word: str) -> str:
    return word[0].upper() + word[1:]


def build_families(tok) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """Amendment A1: completions may be multi-token; scored by mean
    per-token log-probability (eval-harness acc_norm convention)."""
    families: Dict[str, List[Dict]] = {
        "head_facts": [], "tail_facts": [], "tail_words": [],
    }
    drops: Dict[str, int] = {k: 0 for k in families}

    for name, caps in (("head_facts", HEAD_CAPITALS),
                       ("tail_facts", TAIL_CAPITALS)):
        for i, (country, capital) in enumerate(caps):
            wrong = caps[(i + 1) % len(caps)][1]
            families[name].append({
                "prefix_ids": tok.encode(f"The capital of {title(country)} is"),
                "correct_ids": tok.encode(" " + title(capital)),
                "wrong_ids": tok.encode(" " + title(wrong)),
            })

    for cloze, target, distractor in TAIL_WORD_PROBES:
        # "... is a [MASK] ." -> prefix "... is a"; article kept as written.
        prefix_text = title(cloze.split(" [MASK]")[0])
        families["tail_words"].append({
            "prefix_ids": tok.encode(prefix_text),
            "correct_ids": tok.encode(" " + target),
            "wrong_ids": tok.encode(" " + distractor),
        })
    return families, drops


def mean_logprob(model, prefix_ids: List[int], completion_ids: List[int]) -> Tuple[float, torch.Tensor]:
    """Returns (mean per-token log2-prob of completion, next-token probs
    at the first completion position)."""
    device = next(model.parameters()).device
    ids = torch.tensor([prefix_ids + completion_ids], device=device)
    logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    n_prefix = len(prefix_ids)
    total = 0.0
    for j, tok_id in enumerate(completion_ids):
        total += float(logp[n_prefix - 1 + j, tok_id])
    first_probs = torch.softmax(logits[n_prefix - 1], dim=-1)
    return total / len(completion_ids), first_probs.cpu()


def evaluate(model, families) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    with torch.no_grad():
        for fam, items in families.items():
            probs_all, ok_all = [], []
            for it in items:
                lp_c, first_probs = mean_logprob(model, it["prefix_ids"], it["correct_ids"])
                lp_w, _ = mean_logprob(model, it["prefix_ids"], it["wrong_ids"])
                probs_all.append(first_probs.unsqueeze(0))
                ok_all.append(1.0 if lp_c > lp_w else 0.0)
            result[fam] = {
                "probs": torch.cat(probs_all),
                "acc": float(sum(ok_all) / len(ok_all)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pythia tail-gradualism test.")
    parser.add_argument("--steps", type=int, nargs="*", default=STEPS)
    parser.add_argument("--size", type=str, default="160m")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"),
                        default="float32")
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    torch.set_num_threads(32)
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    model_id = f"EleutherAI/pythia-{args.size}"
    ckpt_dir = HERE / ("external_pythia" if args.size == "160m"
                       else f"external_pythia_{args.size}")
    tag = "" if args.size == "160m" else f"_{args.size}"
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    device = torch.device(args.device)

    download_shared(model_id, ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    families, drops = build_families(tok)
    underpowered = [f for f, items in families.items() if len(items) < MIN_ITEMS]
    for fam, items in families.items():
        print(f"{fam}: {len(items)} items ({drops[fam]} dropped)", flush=True)
    if underpowered:
        print(f"UNDERPOWERED (prediction void): {underpowered}", flush=True)

    rows_by_fam: Dict[str, List[Dict[str, float]]] = {f: [] for f in families}
    p0: Dict[str, torch.Tensor] = {}
    skipped: List[int] = []
    baseline_step = args.steps[0]
    for step in args.steps:
        t0 = time.time()
        try:
            ckpt = download_checkpoint(step, model_id, ckpt_dir)
        except Exception as exc:
            if step == baseline_step:
                raise RuntimeError(
                    f"baseline checkpoint step {baseline_step} failed"
                ) from exc
            print(f"step {step}: DOWNLOAD FAILED ({exc}); skipped", flush=True)
            skipped.append(step)
            continue
        model = GPTNeoXForCausalLM.from_pretrained(ckpt, dtype=dtype).to(device)
        model.eval()
        result = evaluate(model, families)
        del model
        parts = []
        for fam, res in result.items():
            if fam not in p0:
                p0[fam] = res["probs"].clone()
            h_bits = entropy_bits(res["probs"])
            c_bits = kl_bits(res["probs"], p0[fam])
            rows_by_fam[fam].append({
                "run": fam, "epoch": step, "train_acc": res["acc"],
                "test_acc": res["acc"], "test_entropy_bits": h_bits,
                "collapse_bits": c_bits, "embedding_jump": 0.0,
            })
            parts.append(f"{fam} acc {res['acc']:.3f} H {h_bits:5.2f} C {c_bits:5.2f}")
        print(f"step {step:8d} | " + " | ".join(parts) +
              f" [{time.time()-t0:.0f}s]", flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(ckpt_dir / f"step{step}", ignore_errors=True)

    summary: Dict[str, object] = {
        "thresholds": THRESHOLDS, "skipped_steps": skipped,
        "drops": drops, "underpowered": underpowered, "runs": {},
    }
    all_rows: List[Dict[str, float]] = []
    for fam, rows in rows_by_fam.items():
        all_rows.extend(rows)
        stats = analyze_run(rows)
        v = verdict(stats, prespecified=False)
        summary["runs"][fam] = {"stats": stats, "verdict": v,
                                "n_items": len(families[fam])}
        failed = ";".join(k for k, ok in v["passes"].items() if not ok) or "-"
        print(f"{fam}: emergent={v['emergent']} failed={failed}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"pythia_tail_timeseries{tag}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / f"pythia_tail_summary{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / f'pythia_tail_summary{tag}.json'}")


if __name__ == "__main__":
    main()
