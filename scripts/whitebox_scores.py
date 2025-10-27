import argparse, json, math, time, os
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def lntp(logprobs):
    if not logprobs: return float("nan")
    return math.exp(sum(logprobs) / len(logprobs))

def avg_nll(logprobs):
    if not logprobs: return float("nan")
    return -sum(logprobs) / len(logprobs)

def min_token_prob(logprobs):
    if not logprobs: return float("nan")
    return math.exp(min(logprobs))

def mean_entropy(entropies):
    if not entropies: return float("nan")
    return sum(entropies) / len(entropies)

def token_entropy_from_logits(last_step_logits_list):
    """list of logits tensors [vocab] for each generated step -> average entropy"""
    ents = []
    for logits in last_step_logits_list:
        probs = torch.softmax(logits, dim=-1)
        # clamp for numerical stability
        e = -(probs * (probs.clamp_min(1e-12)).log()).sum().item()
        ents.append(e)
    return mean_entropy(ents)

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            items.append({"id": obj.get("id"), "prompt": obj.get("prompt")})
    return items

def main():
    parser = argparse.ArgumentParser(description="White-Box scoring to Excel")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HF model id (start klein; danach z.B. mistralai/Mistral-7B-Instruct-v0.3 oder meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--input", type=str, required=True, help="JSONL mit Feldern id,prompt")
    parser.add_argument("--out", type=str, required=True, help="Pfad zur Excel-Datei (.xlsx)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy (empfohlen für stabile Scores)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # CPU-Setup reicht für den Check; dtype float32 stabil
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map=None)
    model.eval()

    data = read_jsonl(args.input)
    results = []

    for item in tqdm(data, desc="Scoring", ncols=80):
        qid = item["id"]
        prompt = item["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        generated = input_ids.clone()
        step_logprobs = []
        step_last_logits = []

        t0 = time.time()
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                out = model(generated)
                logits = out.logits[:, -1, :]  # [1, vocab]
                # Log-Softmax für logprob des gewählten Tokens
                log_probs = torch.log_softmax(logits, dim=-1)

                if args.temperature == 0.0:
                    next_token = torch.argmax(log_probs, dim=-1, keepdim=True)  # greedy
                else:
                    # sampling als Option
                    probs = torch.softmax(logits / max(1e-8, args.temperature), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                # logprob des tatsächlich gewählten Tokens
                step_logprobs.append(log_probs.gather(1, next_token).item())
                step_last_logits.append(logits.squeeze(0).cpu())

                generated = torch.cat([generated, next_token], dim=1)

        dt = time.time() - t0
        new_tokens = generated.shape[1] - input_ids.shape[1]
        reply = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)

        # White-Box Scores
        score_lntp = lntp(step_logprobs)
        score_avg_nll = avg_nll(step_logprobs)
        score_token_entropy = token_entropy_from_logits(step_last_logits)
        score_min_token_prob = min_token_prob(step_logprobs)

        results.append({
            "id": qid,
            "prompt": prompt,
            "generated_text": reply.strip(),
            "new_tokens": new_tokens,
            "latency_ms": int(dt * 1000),
            "score_lntp": score_lntp,
            "score_avg_nll": score_avg_nll,
            "score_token_entropy": score_token_entropy,
            "score_min_token_prob": score_min_token_prob,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        })

    df = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Excel schreiben
    df.to_excel(out_path, index=False)
    print(f"\n✅ Done. Wrote results to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
