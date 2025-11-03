import os, json, random, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

SNAP = r"C:\Users\nikla\.cache\huggingface\hub\models--mistralai--Mistral-7B-Instruct-v0.3\snapshots\0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
MODEL_ID = Path(SNAP).resolve().as_posix()  

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "arc_c", "validation.jsonl")
OUT  = os.path.join(os.path.dirname(__file__), "..", "outputs", "arc_c_llama3_val.jsonl")

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f: yield json.loads(line)

def build_prompt_arc(ex):
    q = ex["question"]
    choices = ex["choices"]["text"] if "choices" in ex else ex["choices"]
    letters = ["A","B","C","D","E","F"]
    opts = "\n".join(f"{letters[i]}. {c}" for i,c in enumerate(choices))
    return f"Question: {q}\nOptions:\n{opts}\nAnswer (give only the letter):"

def gold_letter(ex):
    return ex["answerKey"]

if __name__ == "__main__":
    set_seed(42)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    ).eval()

    # >>> WICHTIG: pad_token_id sauber setzen (einmalig, verhindert den Fallback bei jedem generate)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    # <<<

    with open(OUT, "w", encoding="utf-8") as wf:
        for ex in tqdm(load_jsonl(DATA), total=5):  # für Smoke-Test
            prompt = build_prompt_arc(ex)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=8,            # kurz halten, wir wollen nur den Buchstaben
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tok.pad_token_id  # >>> explizit übergeben
                )

            # Decoding des erzeugten Texts
            gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
            pred_txt = tok.decode(gen_ids, skip_special_tokens=True).strip()

            # pro Schritt: Logits -> für Scorer später
            step_scores = [s[0].float().cpu().tolist() for s in out.scores]
            rec = {
                "id": ex.get("id"),
                "prompt": prompt,
                "pred_text": pred_txt,
                "gold": gold_letter(ex),
                "gen_token_ids": gen_ids.tolist(),
                "step_scores": step_scores
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✓ geschrieben: {OUT}")
