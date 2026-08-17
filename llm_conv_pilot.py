"""LLM-CONV pilot gate: can a population of LLM instances form a
symbol convention in context, and is the outcome seed-dependent?

Registered in METHOD_BASELINE_FIXEDTIME_PREREGISTRATION.md (frozen
before this file was written). Pilot conversations are excluded from
any confirmatory set.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUTS = Path(__file__).resolve().parent / "outputs"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYMBOLS = ("zib", "kem", "rop", "dax", "fen")
N_AGENTS = 4
N_ROUNDS = 30
N_CONVS = 8
TEMP = 0.8
TOP_P = 0.95
UNANIMITY_STREAK = 3


def prompt_for(agent: int, history: list[list[str]]) -> str:
    lines = [
        f"You are agent {agent + 1} in a group of {N_AGENTS}.",
        "Each round, every agent simultaneously picks one word from "
        f"this list: {', '.join(SYMBOLS)}.",
        "The group's goal is for all agents to end up picking the "
        "same word. There is no externally correct word; any shared "
        "word is equally good.",
    ]
    if history:
        lines.append("Choice history (round: agent1 agent2 agent3 agent4):")
        for t, row in enumerate(history, 1):
            lines.append(f"  round {t}: {' '.join(row)}")
    else:
        lines.append("This is the first round; there is no history.")
    lines.append("Reply with exactly one word from the list and "
                 "nothing else.")
    return "\n".join(lines)


def parse_symbol(text: str) -> str | None:
    low = text.lower()
    hits = [s for s in SYMBOLS if s in low]
    return hits[0] if len(hits) == 1 else None


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:1")
    model.eval()

    convs = []
    for c in range(N_CONVS):
        torch.manual_seed(1_000 + c)
        history: list[list[str]] = []
        parse_failures = 0
        converged_at = None
        for t in range(N_ROUNDS):
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": prompt_for(a, history)}],
                    tokenize=False, add_generation_prompt=True)
                for a in range(N_AGENTS)
            ]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      padding_side="left").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8,
                                     do_sample=True, temperature=TEMP,
                                     top_p=TOP_P,
                                     pad_token_id=tok.eos_token_id)
            row = []
            for a in range(N_AGENTS):
                text = tok.decode(out[a][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                sym = parse_symbol(text)
                if sym is None:
                    parse_failures += 1
                    sym = SYMBOLS[0]
                row.append(sym)
            history.append(row)
            if (converged_at is None and t >= UNANIMITY_STREAK - 1
                    and all(len(set(r)) == 1
                            for r in history[t - UNANIMITY_STREAK + 1:])
                    and len({r[0] for r in
                             history[t - UNANIMITY_STREAK + 1:]}) == 1):
                converged_at = t + 1
        final = (history[-1][0]
                 if converged_at is not None and len(set(history[-1])) == 1
                 else None)
        convs.append({"conv": c, "history": history,
                      "converged_at": converged_at,
                      "final_symbol": final,
                      "parse_failures": parse_failures})
        print(f"conv {c}: converged_at={converged_at} final={final} "
              f"parse_failures={parse_failures}", flush=True)

    converged = [c for c in convs if c["converged_at"] is not None]
    finals = {c["final_symbol"] for c in converged if c["final_symbol"]}
    outcomes = {
        "G1_converged": f"{len(converged)}/{N_CONVS}",
        "G1_pass": bool(len(converged) >= 6),
        "G2_distinct_final_symbols": sorted(finals),
        "G2_pass": bool(len(finals) >= 2),
    }
    (OUTPUTS / "llm_conv_pilot.json").write_text(json.dumps({
        "status": ("LLM-CONV pilot gate; conversations excluded from "
                   "any confirmatory set"),
        "config": {"model": MODEL, "symbols": SYMBOLS,
                   "n_agents": N_AGENTS, "n_rounds": N_ROUNDS,
                   "n_convs": N_CONVS, "temperature": TEMP,
                   "top_p": TOP_P, "unanimity_streak": UNANIMITY_STREAK},
        "conversations": convs,
        "registered_outcomes": outcomes}, indent=2), encoding="utf-8")
    print(json.dumps(outcomes, indent=2))
    print("Wrote llm_conv_pilot.json")


if __name__ == "__main__":
    main()
