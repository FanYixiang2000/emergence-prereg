"""Pythia decoder-side collapse probe (public checkpoint series).

Protocol frozen in PYTHIA_PREREGISTRATION.md BEFORE any checkpoint
beyond step0 was evaluated. Thresholds and analysis are imported
unchanged from grokking_collapse_bridge (the same frozen instrument
used for grokking, transformer grokking, induction heads, and
MultiBERTs).

Pipeline (disk-bounded, mirrors multiberts_collapse_probe.py): for each
published step, download the checkpoint from the configured Hugging Face
endpoint, evaluate the next-token distribution at the critical verb position
of every probe, record entropy / KL-from-step0 / minimal-pair accuracy for the
three registered conditions, delete the checkpoint, continue. Analysis
identical to the MultiBERTs probe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from grokking_collapse_bridge import THRESHOLDS, analyze_run, verdict

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
CKPT_DIR = HERE / "external_pythia"

MODEL = "EleutherAI/pythia-160m"
BASE = os.environ.get("PYTHIA_BASE_URL", "https://huggingface.co")

STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
         1000, 2000, 4000, 8000, 16000, 32000, 64000,
         96000, 128000, 143000]


# download

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) curl-compatible research fetcher"}


def fetch_ranged(url: str, dest: Path, n_threads: int = 16,
                 chunk: int = 16 * 2 ** 20) -> None:
    n_threads = int(os.environ.get("PYTHIA_DOWNLOAD_THREADS", n_threads))
    chunk = int(os.environ.get("PYTHIA_DOWNLOAD_CHUNK", chunk))
    timeout = float(os.environ.get("PYTHIA_RANGE_TIMEOUT", 120))
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    ranges = [(o, min(o + chunk, size) - 1) for o in range(0, size, chunk)]
    print(
        f"download {dest.name}: {size / 2**30:.2f} GiB, "
        f"{len(ranges)} ranges, {n_threads} threads",
        flush=True,
    )

    def get(rng: Tuple[int, int]) -> Tuple[int, bytes]:
        last_exc: Exception | None = None
        for attempt in range(8):
            try:
                r = urllib.request.Request(
                    url, headers={**UA, "Range": f"bytes={rng[0]}-{rng[1]}"})
                data = urllib.request.urlopen(r, timeout=timeout).read()
                expected = rng[1] - rng[0] + 1
                if len(data) != expected:
                    raise IOError(f"range size mismatch {len(data)} != {expected}")
                return rng[0], data
            except Exception as exc:
                last_exc = exc
                time.sleep(min(60, 3 * (attempt + 1)))
        raise IOError(f"range {rng[0]}-{rng[1]} failed") from last_exc

    # Resume journal: offsets of ranges already written to the tmp file
    # survive process restarts, so an interrupted 13 GB download does
    # not start over.
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    journal = dest.with_suffix(dest.suffix + ".ranges.done")
    done_offsets: set[int] = set()
    if tmp.exists() and tmp.stat().st_size == size and journal.exists():
        done_offsets = {int(line) for line in
                        journal.read_text().split() if line.strip()}
    else:
        with tmp.open("wb") as f:
            f.truncate(size)
        journal.write_text("")
    todo = [rng for rng in ranges if rng[0] not in done_offsets]
    if done_offsets:
        print(f"  {dest.name}: resuming, {len(done_offsets)}/"
              f"{len(ranges)} ranges already done", flush=True)
    with ThreadPoolExecutor(n_threads) as pool:
        futures = [pool.submit(get, rng) for rng in todo]
        with tmp.open("r+b") as f, journal.open("a") as jf:
            for done, future in enumerate(as_completed(futures), start=1):
                offset, data = future.result()
                f.seek(offset)
                f.write(data)
                f.flush()
                jf.write(f"{offset}\n")
                jf.flush()
                if done == 1 or done == len(futures) or done % 25 == 0:
                    print(
                        f"  {dest.name}: {done}/{len(futures)} ranges",
                        flush=True,
                    )
    if tmp.stat().st_size != size:
        raise IOError(f"size mismatch {tmp.stat().st_size} != {size}")
    tmp.replace(dest)
    journal.unlink(missing_ok=True)


def fetch_bytes(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())


def url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60):
            return True
    except Exception:
        return False


def local_checkpoint_complete(dest: Path) -> bool:
    """True when a pre-downloaded checkpoint needs no network access."""
    config = dest / "config.json"
    if not (config.exists() and config.stat().st_size > 0):
        return False
    for index_name in ("model.safetensors.index.json",
                       "pytorch_model.bin.index.json"):
        index_path = dest / index_name
        if index_path.exists() and index_path.stat().st_size > 0:
            shards = sorted(set(json.loads(
                index_path.read_text(encoding="utf-8"))
                ["weight_map"].values()))
            return all((dest / s).exists() and (dest / s).stat().st_size > 0
                       for s in shards)
    return any((dest / single).exists()
               and (dest / single).stat().st_size > 0
               for single in ("model.safetensors", "pytorch_model.bin"))


def download_checkpoint(step: int, model: str = MODEL,
                        ckpt_dir: Path = CKPT_DIR) -> Path:
    rev = f"step{step}"
    dest = ckpt_dir / rev
    dest.mkdir(parents=True, exist_ok=True)
    if local_checkpoint_complete(dest):
        return dest
    n_attempts = int(os.environ.get("PYTHIA_CKPT_ATTEMPTS", 3))
    for attempt in range(n_attempts):
        try:
            for fname in ("config.json",):
                out = dest / fname
                if out.exists() and out.stat().st_size > 0:
                    continue
                fetch_bytes(f"{BASE}/{model}/resolve/{rev}/{fname}", out)
            downloaded = False
            for index_name in ("model.safetensors.index.json",
                               "pytorch_model.bin.index.json"):
                index_path = dest / index_name
                if not (index_path.exists()
                        and index_path.stat().st_size > 0):
                    index_url = f"{BASE}/{model}/resolve/{rev}/{index_name}"
                    if not url_exists(index_url):
                        continue
                    fetch_bytes(index_url, index_path)
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shard_names = sorted(set(index["weight_map"].values()))
                for fname in shard_names:
                    out = dest / fname
                    if out.exists() and out.stat().st_size > 0:
                        continue
                    fetch_ranged(f"{BASE}/{model}/resolve/{rev}/{fname}", out)
                downloaded = True
                break
            if not downloaded:
                for single in ("model.safetensors", "pytorch_model.bin"):
                    if not url_exists(f"{BASE}/{model}/resolve/{rev}/{single}"):
                        continue
                    out = dest / single
                    if not out.exists() or out.stat().st_size == 0:
                        fetch_ranged(f"{BASE}/{model}/resolve/{rev}/{single}", out)
                    downloaded = True
                    break
            if not downloaded:
                raise IOError(f"no weight object found for {rev}")
            return dest
        except Exception as exc:
            print(f"  {rev}: attempt {attempt + 1}/{n_attempts} failed "
                  f"({exc}); retrying", flush=True)
            if attempt == n_attempts - 1:
                raise
            time.sleep(60)
    return dest


def download_shared(model: str = MODEL, ckpt_dir: Path = CKPT_DIR) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "config.json"):
        out = ckpt_dir / fname
        if out.exists() and out.stat().st_size > 0:
            continue
        req = urllib.request.Request(
            f"{BASE}/{model}/resolve/main/{fname}", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())


# probes

NOUNS = [("author", "authors"), ("pilot", "pilots"), ("farmer", "farmers"),
         ("teacher", "teachers"), ("doctor", "doctors"), ("manager", "managers"),
         ("guard", "guards"), ("driver", "drivers"), ("customer", "customers")]

VERBS = [("is", "are"), ("was", "were"), ("has", "have"),
         ("does", "do"), ("likes", "like"), ("writes", "write"),
         ("knows", "know"), ("sees", "see")]


def single_token_id(tok, word: str) -> Optional[int]:
    ids = tok.encode(" " + word)
    return ids[0] if len(ids) == 1 else None


def build_probes(tok) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """Prefix probes; both verb forms must be single tokens (with leading
    space). Returns (probes, dropped_verb_pairs)."""
    dropped: List[Tuple[str, str]] = []
    verb_ids: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for sg, pl in VERBS:
        a, b = single_token_id(tok, sg), single_token_id(tok, pl)
        if a is None or b is None:
            dropped.append((sg, pl))
        else:
            verb_ids[(sg, pl)] = (a, b)
    probes: List[Dict] = []
    for noun_sg, noun_pl in NOUNS:
        for (verb_sg, verb_pl), (id_sg, id_pl) in verb_ids.items():
            for number in ("sg", "pl"):
                subj = noun_sg if number == "sg" else noun_pl
                attractor = noun_pl if number == "sg" else noun_sg
                correct, wrong = (id_sg, id_pl) if number == "sg" else (id_pl, id_sg)
                for template, prefix in (
                    ("simple", f"The {subj}"),
                    ("attractor", f"The {subj} of the {attractor}"),
                ):
                    probes.append({
                        "prefix_ids": tok.encode(prefix),
                        "correct_id": correct,
                        "wrong_id": wrong,
                        "template": template,
                    })
    return probes, dropped


def probe_batches(probes: List[Dict]):
    by_len: Dict[int, List[Dict]] = {}
    for p in probes:
        by_len.setdefault(len(p["prefix_ids"]), []).append(p)
    batches = []
    for length, group in sorted(by_len.items()):
        ids = torch.tensor([p["prefix_ids"] for p in group])
        correct = torch.tensor([p["correct_id"] for p in group])
        wrong = torch.tensor([p["wrong_id"] for p in group])
        attractor = torch.tensor([p["template"] == "attractor" for p in group])
        batches.append((ids, correct, wrong, attractor))
    return batches


def entropy_bits(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float((-(p * p.log2()).sum(dim=1)).mean())


def kl_bits(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return float((p * (p.log2() - q.log2())).sum(dim=1).mean())


def evaluate_checkpoint(model, batches, perm: torch.Tensor,
                        rand_targets: List[torch.Tensor]) -> Dict:
    device = next(model.parameters()).device
    perm = perm.to(device)
    all_probs = []
    acc_agree, acc_attr, acc_rand, acc_perm = [], [], [], []
    with torch.no_grad():
        for bi, (ids, correct, wrong, attractor) in enumerate(batches):
            ids = ids.to(device)
            correct = correct.to(device)
            wrong = wrong.to(device)
            attractor = attractor.to(device)
            logits = model(ids).logits[:, -1, :].float()
            rows = torch.arange(len(ids))
            all_probs.append(torch.softmax(logits, dim=1))
            ok = (logits[rows, correct] > logits[rows, wrong]).float()
            acc_agree.append(ok)
            acc_attr.append(ok[attractor])
            rt = rand_targets[bi].to(device)
            acc_rand.append((logits[rows, rt[:, 0]] > logits[rows, rt[:, 1]]).float())
            sel_perm = logits[:, perm]
            acc_perm.append((sel_perm[rows, correct] > sel_perm[rows, wrong]).float())
    return {
        "probs": torch.cat(all_probs).cpu(),
        "acc_agreement": float(torch.cat(acc_agree).mean()),
        "acc_attractor": float(torch.cat(acc_attr).mean()),
        "acc_random_target": float(torch.cat(acc_rand).mean()),
        "acc_shuffled_vocab": float(torch.cat(acc_perm).mean()),
    }


# main

def main() -> None:
    parser = argparse.ArgumentParser(description="Pythia decoder collapse probe.")
    parser.add_argument("--steps", type=int, nargs="*", default=STEPS)
    parser.add_argument("--size", type=str, default="160m",
                        help="Pythia model size (160m, 410m, 1b, ...)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"),
                        default="float32")
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    model_id = f"EleutherAI/pythia-{args.size}"
    ckpt_dir = HERE / ("external_pythia" if args.size == "160m"
                       else f"external_pythia_{args.size}")
    tag = "" if args.size == "160m" else f"_{args.size}"

    torch.set_num_threads(32)
    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    device = torch.device(args.device)

    download_shared(model_id, ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    probes, dropped = build_probes(tok)
    batches = probe_batches(probes)
    vocab_size = len(tok)
    gen = torch.Generator().manual_seed(20260708)
    perm = torch.randperm(vocab_size, generator=gen)
    rand_targets = [
        torch.randint(1000, vocab_size, (len(b[0]), 2), generator=gen)
        for b in batches
    ]
    print(f"{len(probes)} probes in {len(batches)} length-batches; "
          f"dropped verb pairs: {dropped or 'none'}", flush=True)

    rows_by_cond: Dict[str, List[Dict[str, float]]] = {
        "pythia_agreement": [], "pythia_random_target": [],
        "shuffled_vocab": [],
    }
    p0: Optional[torch.Tensor] = None
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
            print(f"step {step}: DOWNLOAD FAILED after retries ({exc}); skipped",
                  flush=True)
            skipped.append(step)
            continue
        model = GPTNeoXForCausalLM.from_pretrained(ckpt, dtype=dtype).to(device)
        model.eval()
        result = evaluate_checkpoint(model, batches, perm, rand_targets)
        del model
        if p0 is None:
            p0 = result["probs"].clone()
        h_bits = entropy_bits(result["probs"])
        c_bits = kl_bits(result["probs"], p0)
        for cond, acc in (
            ("pythia_agreement", result["acc_agreement"]),
            ("pythia_random_target", result["acc_random_target"]),
            ("shuffled_vocab", result["acc_shuffled_vocab"]),
        ):
            rows_by_cond[cond].append({
                "run": cond, "epoch": step, "train_acc": acc, "test_acc": acc,
                "test_entropy_bits": h_bits, "collapse_bits": c_bits,
                "embedding_jump": 0.0,
            })
        print(f"step {step:8d} H {h_bits:6.2f} C {c_bits:6.2f} "
              f"agree {result['acc_agreement']:.3f} "
              f"(attractor {result['acc_attractor']:.3f}) "
              f"rand {result['acc_random_target']:.3f} "
              f"perm {result['acc_shuffled_vocab']:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(ckpt_dir / f"step{step}", ignore_errors=True)

    summary: Dict[str, object] = {
        "thresholds": THRESHOLDS, "skipped_steps": skipped,
        "dropped_verb_pairs": dropped, "download_base": BASE, "runs": {},
    }
    all_rows: List[Dict[str, float]] = []
    for cond, rows in rows_by_cond.items():
        all_rows.extend(rows)
        stats = analyze_run(rows)
        v = verdict(stats, prespecified=False)
        summary["runs"][cond] = {"stats": stats, "verdict": v}
        failed = ";".join(k for k, ok in v["passes"].items() if not ok) or "-"
        print(f"{cond}: emergent={v['emergent']} failed={failed}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"pythia_collapse_timeseries{tag}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / f"pythia_collapse_summary{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / f'pythia_collapse_summary{tag}.json'}")


if __name__ == "__main__":
    main()
