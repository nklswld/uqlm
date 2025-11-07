import torch, numpy as np
from typing import Dict, Sequence

@torch.no_grad()
def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    p = torch.clamp(p, eps, 1.0)
    return -(p * p.log()).sum(dim=dim)

@torch.no_grad()
def score_option(tokenizer, model, prompt: str, option: str,
                 mid_layers: Sequence[int] = tuple(range(12, 20))) -> Dict[str, float]:
    # 1) Prompt und Prompt+Option getrennt tokenisieren → saubere Span-Grenzen
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    enc_full   = tokenizer(prompt + " " + option, return_tensors="pt")

    input_ids = enc_full["input_ids"].to(model.device)
    attn_mask = enc_full["attention_mask"].to(model.device)

    out = model(input_ids=input_ids, attention_mask=attn_mask)

    # 2) Logprobs über Option-Token (Teacher forcing)
    # logits für nächste Token (verschoben)
    logits = out.logits[:, :-1, :]                      # [B, T-1, V]
    labels = input_ids[:, 1:]                           # [B, T-1]
    # Option-Start: Länge des Prompt-Encodings
    opt_start = enc_prompt["input_ids"].shape[1]
    opt_len   = input_ids.shape[1] - opt_start
    # Slice über die Option
    opt_logits = logits[:, opt_start-1: opt_start-1 + opt_len, :]   # logits die die Option-Token „vorhersagen“
    opt_labels = labels[:, opt_start-1: opt_start-1 + opt_len]

    # length-normalized token logprob (LNTP)
    token_logp = torch.log_softmax(opt_logits, dim=-1).gather(2, opt_labels.unsqueeze(-1)).squeeze(-1)
    lntp = token_logp.mean().item()
    max_token_prob = torch.softmax(opt_logits, dim=-1).max(dim=-1).values.max().item()

    # 3) Hidden State am letzten Option-Token
    end_idx = opt_start + opt_len - 1
    last_hidden = out.hidden_states[-1][:, end_idx, :]  # [B, D]
    hidden_norm = last_hidden.norm(dim=-1).item()
    hidden_var  = last_hidden.var(dim=-1, unbiased=False).item()

    # 4) Attention-Entropie (mittlere Layer) am Endtoken
    entropies = []
    for L in mid_layers:
        A = out.attentions[L][:, :, end_idx, :input_ids.shape[1]]   # [B, heads, seq]
        P = torch.softmax(A, dim=-1)
        H = _entropy(P, dim=-1).mean().item()
        entropies.append(H)
    attn_entropy = float(np.mean(entropies))

    return {
        "lntp": lntp,
        "max_token_prob": max_token_prob,
        "hidden_norm": hidden_norm,
        "hidden_var": hidden_var,
        "attn_entropy": attn_entropy,
        "opt_len": int(opt_len),
    }
