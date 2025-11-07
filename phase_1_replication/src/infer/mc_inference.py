# src/infer/mc_inference.py
import torch
import numpy as np
from typing import Dict, Sequence

@torch.no_grad()
def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    p = torch.clamp(p, eps, 1.0)
    return -(p * p.log()).sum(dim=dim)

@torch.no_grad()
def score_letter(tokenizer, model, prompt: str, letter: str,
                 mid_layers: Sequence[int] = tuple(range(12, 20))) -> Dict[str, float]:
    """
    Scort NUR die Wahrscheinlichkeit des Antwort-Buchstabens (A/B/…).
    LNTP = log p(letter | prompt)
    (Hidden/Attention werden nicht hier berechnet; das macht decision_state_from_prompt)
    """
    # Prompt und Prompt+Letter separat tokenisieren
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    enc_full   = tokenizer(prompt + " " + letter, return_tensors="pt")

    input_ids = enc_full["input_ids"].to(model.device)
    attn_mask = enc_full["attention_mask"].to(model.device)

    out = model(input_ids=input_ids, attention_mask=attn_mask)

    # Logprob des Letter-Tokens (genau ein Schritt)
    logits = out.logits[:, :-1, :]   # prädiziert nächstes Token
    labels = input_ids[:, 1:]

    opt_start = enc_prompt["input_ids"].shape[1]  # Position des Letter-Tokens in input_ids
    log_probs = torch.log_softmax(logits[:, opt_start-1:opt_start, :], dim=-1)
    lntp = log_probs.gather(2, labels[:, opt_start-1:opt_start].unsqueeze(-1)).squeeze(-1).mean().item()

    return {"lntp": float(lntp)}

@torch.no_grad()
def decision_state_from_prompt(tokenizer, model, prompt: str,
                               mid_layers: Sequence[int] = tuple(range(20, 30))) -> Dict[str, float]:
    """
    Läuft NUR über den Prompt (ohne Letter) und gibt den Decision-State
    (letztes Token des Prompts) zurück – einmal pro Item.

    Returned keys:
      - decision_hidden_norm
      - decision_hidden_var
      - decision_attn_entropy
      - decision_hidden_feat (np.array, vorletzte Hidden-Schicht)
    """
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    input_ids = enc_prompt["input_ids"].to(model.device)
    attn_mask = enc_prompt["attention_mask"].to(model.device)

    out = model(input_ids=input_ids, attention_mask=attn_mask)

    # Index des letzten Prompt-Tokens (vor dem Letter)
    decision_idx = input_ids.shape[1] - 1

    # Vorletzte Hidden-Schicht ist oft stabiler als die letzte
    h_prev = out.hidden_states[-2][:, decision_idx, :]    # [1, D]
    hidden_norm = h_prev.norm(dim=-1).item()
    hidden_var  = h_prev.var(dim=-1, unbiased=False).item()

    # Attention-Entropie (falls verfügbar)
    if hasattr(out, "attentions") and out.attentions is not None:
        ents = []
        T = input_ids.shape[1]
        for L in mid_layers:
            A = out.attentions[L][:, :, decision_idx, :T]  # [B, heads, K]
            P = torch.softmax(A, dim=-1)
            H = _entropy(P, dim=-1).mean().item()
            ents.append(H)
        attn_entropy = float(np.mean(ents)) if ents else float("nan")
    else:
        attn_entropy = float("nan")

    # rohes Feature (für Mini-Probe)
    h_feat = h_prev.squeeze(0).detach().cpu().numpy()     # [D]

    return {
        "decision_hidden_norm": float(hidden_norm),
        "decision_hidden_var":  float(hidden_var),
        "decision_attn_entropy": float(attn_entropy),
        "decision_hidden_feat": h_feat,
    }
