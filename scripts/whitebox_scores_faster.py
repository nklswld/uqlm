# whitebox_scores.py
# Optimierte White-Box-Scoring-Pipeline:
# - nutzt generate() mit use_cache (KV-Cache) -> deutlich schneller
# - berechnet LNTP, Avg-NLL, Token-Entropy, Min-Token-Prob
# - schreibt Ergebnisse als Excel (.xlsx)

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------- Utility: Scores -----------------------------------------------------
def lntp(logprobs):
    """Length-normalized token probability (geometrisches Mittel)."""
    if not logprobs:
        return float("nan")
    return math.exp(sum(logprobs) / len(logprobs))

def avg_nll(logprobs):
    """Durchschnittliche Negative Log-Likelihood (niedriger = sicherer)."""
    if not logprobs:
        return float("nan")
    return -sum(logprobs) / len(logprobs)

def min_token_prob(logprobs):
    """Kleinste Token-Wahrscheinlichkeit (zeigt 'unsicherstes' Token)."""
    if not logprobs:
        return float("nan")
    return math.exp(min(logprobs))

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

# -------- IO ------------------------------------------------------------------
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"id": obj.get("id"), "prompt": obj.get("prompt")})
    return rows

# -------- Main ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="White-Box scoring to Excel (fast KV-cache)")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HF-Model-ID (z.B. mistralai/Mistral-7B-Instruct-v0.3 oder meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--input", type=str, required=True, help="JSONL mit Feldern id,prompt (eine Zeile pro Prompt)")
    parser.add_argument("--out", type=str, required=True, help="Pfad zur Excel-Datei (.xlsx)")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy (empfohlen für stabile Scores)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None, help="optional: Anzahl CPU-Threads für Torch/BLAS")
    args = parser.parse_args()

    # ---- Performance-Settings (CPU) ------------------------------------------
    # Tokenizers-Parallelisierung vermeiden (spart Overhead)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Threading (optional von CLI steuerbar)
    if args.threads and args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)

    # ---- Laden von Tokenizer & Modell ----------------------------------------
    print(f"[INFO] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # CPU reicht; float32 ist stabil. (GPU/CUDA optional separat installieren)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=None
    )
    model.eval()

    # ---- Prompts laden --------------------------------------------------------
    data = read_jsonl(args.input)
    if not data:
        raise ValueError(f"Keine Prompts in: {args.input}")

    results = []
    t_start = time.time()

    for item in tqdm(data, desc="Scoring", ncols=80):
        qid = item["id"]
        prompt = item["prompt"]

        # Eingabe vorbereiten
        enc = tokenizer(prompt, return_tensors="pt")

        # ---- Generation mit KV-Cache + Scores (schnell) ----------------------
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0.0),
                temperature=max(args.temperature, 1e-8),
                return_dict_in_generate=True,
                output_scores=True,   # liefert Logits je Schritt
                use_cache=True        # KV-Cache aktiv -> großer Speedup
            )
        dt_ms = int((time.time() - t0) * 1000)

        # neue Tokens & Antwort
        gen_ids = gen_out.sequences            # [1, input_len + new_len]
        input_len = enc["input_ids"].shape[1]
        new_ids = gen_ids[:, input_len:]       # [1, new_len]
        reply = tokenizer.decode(new_ids[0], skip_special_tokens=True)

        # ---- White-Box: Logprobs & Entropie pro Schritt ----------------------
        step_logprobs = []
        entropies = []
        # gen_out.scores ist eine Liste von logits (Tensor [1, vocab]) pro Step
        for i, step_logits in enumerate(gen_out.scores):
            # Log-Softmax -> logprobs über Vokabular
            lp = torch.log_softmax(step_logits, dim=-1)         # [1, vocab]
            chosen = int(new_ids[0, i].item())                   # tatsächlich gewähltes Token
            step_logprobs.append(lp[0, chosen].item())

            probs = torch.softmax(step_logits, dim=-1)
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
            entropies.append(ent)

        # ---- Scores aggregieren ----------------------------------------------
        score_lntp = lntp(step_logprobs)
        score_avg_nll = avg_nll(step_logprobs)
        score_token_entropy = mean(entropies)
        score_min_token_prob = min_token_prob(step_logprobs)

        results.append({
            "id": qid,
            "prompt": prompt,
            "generated_text": reply.strip(),
            "new_tokens": int(new_ids.shape[1]),
            "latency_ms": dt_ms,
            "score_lntp": score_lntp,
            "score_avg_nll": score_avg_nll,
            "score_token_entropy": score_token_entropy,
            "score_min_token_prob": score_min_token_prob,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        })

    # ---- Excel schreiben ------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_excel(out_path, index=False)
    print(f"\n✅ Done in {int(time.time()-t_start)}s. Wrote results to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
