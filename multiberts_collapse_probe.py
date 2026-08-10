"""Possibility collapse across a PUBLIC checkpoint series (MultiBERTs).

Protocol, probes, thresholds, and predictions are frozen in
MULTIBERTS_PREREGISTRATION.md. The target system is Google Research's
MultiBERTs seed_0 intermediate checkpoints (Sellam et al., ICLR 2022):
a 110M-parameter BERT-base MLM trained, checkpointed, and published by an
external lab. We did not train it and cannot influence it: this is the
zero-authorial-control validation.

Pipeline (disk-bounded): for each published step, download the TF
checkpoint (~440 MB) with parallel range requests, read the weights with
tf.train.load_checkpoint, rebuild the exact BERT-base forward pass in
PyTorch, evaluate the fixed probe batteries, then delete the checkpoint.

Measured per checkpoint (identical to the grokking bridge; thresholds
imported frozen from grokking_collapse_bridge.THRESHOLDS):

    H_k   = mean predictive entropy at the masked verb position (bits)
    C_k   = mean KL(P_k || P_0) at that position, P_0 = published step 0
    acc_k = minimal-pair accuracy (correct verb form vs number-mismatched)
    J_k   = mean L2 jump of final-layer embeddings at the mask position

Conditions (see the pre-registration for predictions P1-P4):
    multiberts_agreement    -> expected emergent
    multiberts_random_target-> expected fails usefulness
    shuffled_vocab          -> identical entropy/KL by construction,
                               expected fails usefulness
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from grokking_collapse_bridge import THRESHOLDS, analyze_run, entropy_bits, kl_bits, verdict

OUTPUTS = Path(__file__).resolve().parent / "outputs"
CKPT_DIR = Path(__file__).resolve().parent / "external_checkpoints"
BASE_URL_TEMPLATE = "https://storage.googleapis.com/multiberts/public/intermediates/seed_{seed}"

STEPS = [0, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000, 140_000,
         160_000, 180_000, 200_000, 300_000, 400_000, 500_000, 600_000,
         700_000, 800_000, 900_000, 1_000_000, 1_100_000, 1_200_000,
         1_300_000, 1_400_000, 1_500_000, 1_600_000, 1_700_000, 1_800_000,
         1_900_000, 2_000_000]

# download

def fetch_ranged(url: str, dest: Path, n_threads: int = 32,
                 chunk: int = 8_000_000, retries: int = 4) -> None:
    req = urllib.request.Request(url, method="HEAD")
    size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    ranges = [(i, min(i + chunk, size) - 1) for i in range(0, size, chunk)]

    def fetch(rng: Tuple[int, int]) -> bytes:
        for attempt in range(retries):
            try:
                r = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"})
                return urllib.request.urlopen(r, timeout=120).read()
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    with cf.ThreadPoolExecutor(n_threads) as ex:
        parts = list(ex.map(fetch, ranges))
    with dest.open("wb") as f:
        for part in parts:
            f.write(part)


def download_checkpoint(step: int, model_seed: int) -> Path:
    base_url = BASE_URL_TEMPLATE.format(seed=model_seed)
    step_dir = CKPT_DIR / f"seed_{model_seed}_step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("index", "data-00000-of-00001"):
        dest = step_dir / f"bert.ckpt.{suffix}"
        if not dest.exists():
            fetch_ranged(f"{base_url}/step_{step}/bert.ckpt.{suffix}", dest)
    return step_dir / "bert.ckpt"


# BERT forward

class BertWeights:
    """Reads the original TF BERT checkpoint into torch tensors."""

    def __init__(self, ckpt_prefix: Path):
        import tensorflow as tf  # local import; only needed at read time
        reader = tf.train.load_checkpoint(str(ckpt_prefix))
        self.t = {}
        for name in reader.get_variable_to_shape_map():
            if name.startswith(("bert/", "cls/predictions/")):
                self.t[name] = torch.from_numpy(np.array(reader.get_tensor(name)))

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.t[name]


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + 1e-12) * gamma + beta


def bert_forward(w: BertWeights, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (mlm_logits, final_hidden) for a batch of token id rows."""
    b, s = ids.shape
    h = (w["bert/embeddings/word_embeddings"][ids]
         + w["bert/embeddings/position_embeddings"][:s]
         + w["bert/embeddings/token_type_embeddings"][0])
    h = layer_norm(h, w["bert/embeddings/LayerNorm/gamma"],
                   w["bert/embeddings/LayerNorm/beta"])
    n_heads, d_head = 12, 64
    for layer in range(12):
        p = f"bert/encoder/layer_{layer}"
        q = (h @ w[f"{p}/attention/self/query/kernel"] + w[f"{p}/attention/self/query/bias"])
        k = (h @ w[f"{p}/attention/self/key/kernel"] + w[f"{p}/attention/self/key/bias"])
        v = (h @ w[f"{p}/attention/self/value/kernel"] + w[f"{p}/attention/self/value/bias"])
        q = q.view(b, s, n_heads, d_head).transpose(1, 2)
        k = k.view(b, s, n_heads, d_head).transpose(1, 2)
        v = v.view(b, s, n_heads, d_head).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d_head), dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(b, s, 768)
        ctx = ctx @ w[f"{p}/attention/output/dense/kernel"] + w[f"{p}/attention/output/dense/bias"]
        h = layer_norm(h + ctx, w[f"{p}/attention/output/LayerNorm/gamma"],
                       w[f"{p}/attention/output/LayerNorm/beta"])
        mid = h @ w[f"{p}/intermediate/dense/kernel"] + w[f"{p}/intermediate/dense/bias"]
        mid = torch.nn.functional.gelu(mid)
        out = mid @ w[f"{p}/output/dense/kernel"] + w[f"{p}/output/dense/bias"]
        h = layer_norm(h + out, w[f"{p}/output/LayerNorm/gamma"],
                       w[f"{p}/output/LayerNorm/beta"])
    t = h @ w["cls/predictions/transform/dense/kernel"] + w["cls/predictions/transform/dense/bias"]
    t = torch.nn.functional.gelu(t)
    t = layer_norm(t, w["cls/predictions/transform/LayerNorm/gamma"],
                   w["cls/predictions/transform/LayerNorm/beta"])
    logits = t @ w["bert/embeddings/word_embeddings"].T + w["cls/predictions/output_bias"]
    return logits, h


# probes

class Vocab:
    def __init__(self, path: Path):
        self.tokens = path.read_text(encoding="utf-8").splitlines()
        self.index = {tok: i for i, tok in enumerate(self.tokens)}

    def id(self, token: str) -> int:
        return self.index[token]


NOUNS = [("author", "authors"), ("pilot", "pilots"), ("farmer", "farmers"),
         ("teacher", "teachers"), ("doctor", "doctors"), ("manager", "managers"),
         ("guard", "guards"), ("driver", "drivers"), ("customer", "customers")]

# (singular_form, plural_form, template_suffix)
VERBS = [("is", "are", "very happy ."),
         ("was", "were", "very tired ."),
         ("has", "have", "a lot of money ."),
         ("does", "do", "not want to leave ."),
         ("likes", "like", "the new movie ."),
         ("writes", "write", "letters every week ."),
         ("knows", "know", "the answer ."),
         ("sees", "see", "the ocean .")]


def build_probes(vocab: Vocab) -> List[Dict]:
    """288 minimal pairs: 9 nouns x 8 verb pairs x 2 numbers x 2 templates."""
    for word in [w for pair in NOUNS for w in pair] + \
                [w for v in VERBS for w in v[:2]] + \
                ["the", "of", "[CLS]", "[SEP]", "[MASK]"]:
        assert word in vocab.index, f"probe word not in vocab: {word}"
    probes = []
    for noun_sg, noun_pl in NOUNS:
        for verb_sg, verb_pl, suffix in VERBS:
            for number in ("sg", "pl"):
                subj = noun_sg if number == "sg" else noun_pl
                attractor = noun_pl if number == "sg" else noun_sg
                correct = verb_sg if number == "sg" else verb_pl
                wrong = verb_pl if number == "sg" else verb_sg
                for template, words in (
                    ("simple", ["the", subj, "[MASK]"] + suffix.split()),
                    ("attractor", ["the", subj, "of", "the", attractor,
                                   "[MASK]"] + suffix.split()),
                ):
                    probes.append({
                        "words": ["[CLS]"] + words + ["[SEP]"],
                        "correct_id": vocab.id(correct),
                        "wrong_id": vocab.id(wrong),
                        "template": template,
                    })
    return probes


def probe_batches(probes: List[Dict], vocab: Vocab):
    """Groups probes by length -> (ids tensor, mask position, correct, wrong)."""
    by_len: Dict[int, List[Dict]] = {}
    for probe in probes:
        by_len.setdefault(len(probe["words"]), []).append(probe)
    batches = []
    for length, group in sorted(by_len.items()):
        ids = torch.tensor([[vocab.id(w) for w in p["words"]] for p in group])
        mask_pos = torch.tensor([p["words"].index("[MASK]") for p in group])
        correct = torch.tensor([p["correct_id"] for p in group])
        wrong = torch.tensor([p["wrong_id"] for p in group])
        attractor = torch.tensor([p["template"] == "attractor" for p in group])
        batches.append((ids, mask_pos, correct, wrong, attractor))
    return batches


def evaluate_checkpoint(w: BertWeights, batches, perm: torch.Tensor,
                        rand_targets: List[torch.Tensor]):
    """Returns per-condition (probs at mask, accuracy) plus embeddings."""
    all_probs, all_emb = [], []
    acc_agree, acc_attr, acc_rand, acc_perm = [], [], [], []
    with torch.no_grad():
        for bi, (ids, mask_pos, correct, wrong, attractor) in enumerate(batches):
            logits, hidden = bert_forward(w, ids)
            rows = torch.arange(len(ids))
            sel = logits[rows, mask_pos]                     # [n, vocab]
            all_probs.append(torch.softmax(sel, dim=1))
            emb = hidden[rows, mask_pos]
            all_emb.append(emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8))
            ok = (sel[rows, correct] > sel[rows, wrong]).float()
            acc_agree.append(ok)
            acc_attr.append(ok[attractor])
            rt = rand_targets[bi]
            acc_rand.append((sel[rows, rt[:, 0]] > sel[rows, rt[:, 1]]).float())
            sel_perm = sel[:, perm]
            acc_perm.append((sel_perm[rows, correct] > sel_perm[rows, wrong]).float())
    probs = torch.cat(all_probs)
    emb = torch.cat(all_emb)
    return {
        "probs": probs,
        "emb": emb,
        "acc_agreement": float(torch.cat(acc_agree).mean()),
        "acc_attractor": float(torch.cat(acc_attr).mean()),
        "acc_random_target": float(torch.cat(acc_rand).mean()),
        "acc_shuffled_vocab": float(torch.cat(acc_perm).mean()),
    }


# main

def main() -> None:
    parser = argparse.ArgumentParser(description="MultiBERTs collapse probe.")
    parser.add_argument("--steps", type=int, nargs="*", default=STEPS)
    parser.add_argument("--model_seed", type=int, default=0,
                        help="MultiBERTs intermediate seed (0-4 published)")
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    tag = "" if args.model_seed == 0 else f"_seed{args.model_seed}"

    torch.set_num_threads(16)
    vocab = Vocab(CKPT_DIR / "vocab.txt")
    probes = build_probes(vocab)
    batches = probe_batches(probes, vocab)
    gen = torch.Generator().manual_seed(20260706)
    perm = torch.randperm(len(vocab.tokens), generator=gen)
    rand_targets = [
        torch.randint(1000, len(vocab.tokens), (len(b[0]), 2), generator=gen)
        for b in batches
    ]
    print(f"{len(probes)} probes in {len(batches)} length-batches", flush=True)

    rows_by_cond: Dict[str, List[Dict[str, float]]] = {
        "multiberts_agreement": [], "multiberts_random_target": [],
        "shuffled_vocab": [],
    }
    p0: torch.Tensor | None = None
    prev_emb: torch.Tensor | None = None
    skipped: List[int] = []
    for step in args.steps:
        t0 = time.time()
        try:
            prefix = download_checkpoint(step, args.model_seed)
        except Exception as exc:
            print(f"step {step}: DOWNLOAD FAILED after retries ({exc}); skipped",
                  flush=True)
            skipped.append(step)
            continue
        w = BertWeights(prefix)
        result = evaluate_checkpoint(w, batches, perm, rand_targets)
        if p0 is None:
            p0 = result["probs"].clone()
        jump = (float((result["emb"] - prev_emb).norm(dim=1).mean())
                if prev_emb is not None else 0.0)
        prev_emb = result["emb"].clone()
        h_bits = entropy_bits(result["probs"])
        c_bits = kl_bits(result["probs"], p0)
        for cond, acc in (
            ("multiberts_agreement", result["acc_agreement"]),
            ("multiberts_random_target", result["acc_random_target"]),
            ("shuffled_vocab", result["acc_shuffled_vocab"]),
        ):
            rows_by_cond[cond].append({
                "run": cond, "epoch": step, "train_acc": acc, "test_acc": acc,
                "test_entropy_bits": h_bits, "collapse_bits": c_bits,
                "embedding_jump": jump,
            })
        print(f"step {step:8d} H {h_bits:6.2f} C {c_bits:6.2f} "
              f"agree {result['acc_agreement']:.3f} "
              f"(attractor {result['acc_attractor']:.3f}) "
              f"rand {result['acc_random_target']:.3f} "
              f"perm {result['acc_shuffled_vocab']:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(CKPT_DIR / f"seed_{args.model_seed}_step_{step}",
                          ignore_errors=True)

    summary: Dict[str, object] = {
        "thresholds": THRESHOLDS, "skipped_steps": skipped, "runs": {},
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
    with (args.output_dir / f"multiberts_collapse_timeseries{tag}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    (args.output_dir / f"multiberts_collapse_summary{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / f'multiberts_collapse_summary{tag}.json'}")


if __name__ == "__main__":
    main()
