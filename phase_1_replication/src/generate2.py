# phase_1_replication/src/generate.py
# HF-only generation with white-box signals:
# - token_probs from generation scores
# - hidden_last_mean from final layer hidden states (gen tokens)
# - attn_entropy from full-seq attentions (mean over layers/heads/tokens)

import os
import json
import math
import argparse
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(path_or_name: str, dtype: str = "bfloat16") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(path_or_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path_or_name,
        torch_dtype=DTYPE_MAP[dtype],
        device_map="auto",
    )
    # ensure pad token (some instruct models don't set it)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(mdl.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id
    return tok, mdl


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def compute_token_probs_from_scores(scores: List[torch.Tensor], gen_ids: torch.Tensor) -> List[float]:
    # scores: list length T of [batch=1, vocab], or stacked -> handle both
    if isinstance(scores, list):
        step_scores = torch.stack(scores, dim=0)  # [T, 1, vocab]
        step_scores = step_scores[:, 0, :]       # [T, vocab]
    else:
        # already a tensor [T, vocab]
        step_scores = scores
    probs = softmax_stable(step_scores.float(), dim=-1)      # [T, vocab]
    tok_idx = gen_ids.detach().to(step_scores.device)
    tok_probs = probs[torch.arange(len(tok_idx)), tok_idx].detach().cpu().tolist()
    return [float(max(1e-12, p)) for p in tok_probs]


def attention_entropy_from_forward(attentions: Tuple[torch.Tensor, ...]) -> float:
    """
    attentions: tuple(L) of tensors [bs, heads, seq, seq]
    We compute mean entropy over (layers, heads, query-tokens).
    """
    if not attentions:
        return float("nan")
    ents = []
    for layer_attn in attentions:
        A = layer_attn[0].float().clamp(1e-12, 1.0)  # [heads, seq, seq]
        ent = -(A * torch.log(A)).sum(dim=-1)       # [heads, seq]
        ents.append(ent.mean().item())
    return float(np.mean(ents))


def forward_hidden_and_attn(
    mdl: AutoModelForCausalLM,
    tok: AutoTokenizer,
    full_ids: torch.Tensor,
    full_mask: torch.Tensor,
) -> Tuple[np.ndarray, float]:
    """
    Runs a forward pass to get hidden states & attentions on the full sequence.
    Returns:
      hidden_last_mean: mean over final-layer hidden states of generated tokens (H,)
      attn_entropy: mean entropy over all layers/heads/tokens
    """
    with torch.no_grad():
        out = mdl(
            input_ids=full_ids,
            attention_mask=full_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    # hidden_states: tuple(L+1) of [bs, seq, H]; take final layer
    last_h = out.hidden_states[-1][0]  # [seq, H]
    # We'll select generated slice outside (need prompt/gen boundary)
    attn_entropy = attention_entropy_from_forward(out.attentions)
    return last_h.detach().cpu().numpy(), attn_entropy


def run(args):
    seed_all(args.seed)
    tok, mdl = load_model(args.model, dtype=args.dtype)

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(data_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            qid = ex.get("id", str(n_written))
            prompt = ex["prompt"]
            gold = ex.get("gold")

            # Encode prompt
            enc = tok(prompt, return_tensors="pt")
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            # Generate deterministically
            gen_out = mdl.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=False,      # we'll get attentions from a separate forward
                output_hidden_states=False,   # same
            )

            full_seq = gen_out.sequences[0]                      # [seq_full]
            gen_ids = full_seq[prompt_len:]                      # generated token ids
            pred_text = tok.decode(gen_ids, skip_special_tokens=True)

            # token-level probabilities from generation scores
            token_probs = compute_token_probs_from_scores(gen_out.scores, gen_ids)

            # Full forward for hidden states & attentions on the full sequence
            full_ids = full_seq.unsqueeze(0).to(mdl.device)      # [1, seq]
            full_mask = torch.ones_like(full_ids, dtype=torch.long, device=mdl.device)
            last_h_all, attn_entropy = forward_hidden_and_attn(mdl, tok, full_ids, full_mask)

            # Mean-pool final-layer hidden states over generated region
            if len(gen_ids) > 0:
                gen_hidden = last_h_all[prompt_len:, :]          # [T_gen, H]
                hidden_last_mean = gen_hidden.mean(axis=0).astype(np.float32).tolist()
            else:
                # empty generation edge-case
                hidden_last_mean = last_h_all[-1, :].astype(np.float32).tolist()

            rec = {
                "id": qid,
                "prompt": prompt,
                "gold": gold,
                "pred_text": pred_text,
                "token_probs": token_probs,
                "hidden_last_mean": hidden_last_mean,
                "attn_entropy": float(attn_entropy),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[OK] wrote {n_written} rows -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model name or local path on THIS machine/VM")
    ap.add_argument("--data_path", type=str, required=True, help="JSONL with {id, prompt, gold}")
    ap.add_argument("--out_path", type=str, required=True, help="Where to write generated JSONL")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)
