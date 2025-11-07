import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

def load_model(model_name: str = MODEL_NAME, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Mistral nutzt meist kein pad_token -> auf eos setzen (nur fürs Scoring/Batching hilfreich)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    mdl.eval()
    return tok, mdl
