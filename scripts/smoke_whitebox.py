"""
Smoke test for White-Box setup (Transformers).
- Lädt ein Modell
- Führt einen Forward-Pass mit hidden_states/attentions aus
- Speichert eine einfache Entropie-Grafik
"""

import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------
# CONFIG
# ------------------------------
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"   # ändere hier bei Bedarf
PROMPT = "Briefly explain the function of the kidneys in maintaining homeostasis."
OUT_PATH = Path("artifacts/plots/entropy_smoketest.png")

# Ordner sicherstellen
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Stabilere Numerik (optional)
torch.set_float32_matmul_precision("high")

# ------------------------------
# LOAD MODEL & TOKENIZER
# ------------------------------
print(f"Loading model: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Für Mistral: 'attn_implementation="eager"' vermeidet die SDPA-Warnung bei output_attentions=True
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)

# White-Box-Ausgaben aktivieren
model.config.output_hidden_states = True
model.config.output_attentions = True

# ------------------------------
# FORWARD PASS
# ------------------------------
inputs = tok(PROMPT, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs)

# Sanity-Prints
print(f"Hidden states: {len(out.hidden_states)} layers")
print(f"Attentions: {len(out.attentions)} layers")
print(f"Logits shape: {tuple(out.logits.shape)}")

# ------------------------------
# TOKEN-LEVEL ENTROPY
# ------------------------------
logits = out.logits[0]                   # (T, V)
probs = torch.softmax(logits, dim=-1)
entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=-1).cpu()  # (T,)

tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(10, 4))
plt.plot(entropy.numpy(), marker="o")
plt.title(f"Token-Level Entropy • {MODEL_ID}")
plt.xlabel("Token position")
plt.ylabel("Entropy")

# Viele Tokens? Dann die xticks ausdünnen:
if len(tokens) <= 40:
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
else:
    step = max(1, len(tokens) // 40)
    shown = [t if i % step == 0 else "" for i, t in enumerate(tokens)]
    plt.xticks(range(len(tokens)), shown, rotation=90, fontsize=7)

plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=150)
plt.close()

print(f"✅ Entropy plot saved to {OUT_PATH.as_posix()}")
