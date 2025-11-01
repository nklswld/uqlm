import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Modell auswählen:
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # z. B. Mistral, Llama, Qwen, Meditron

# Testprompts
PROMPTS = [
    "Explain the role of mitochondria in human cells.",
    "Summarize the causes and consequences of the French Revolution in two sentences.",
    "What are the ethical risks of using large language models in medicine?"
]

# -------------------------------
print(f"🚀 Loading model: {MODEL_ID}")
t0 = time.time()

tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

load_time = time.time() - t0
print(f"✅ Model loaded in {load_time:.2f} seconds\n")

# -------------------------------
for i, prompt in enumerate(PROMPTS, 1):
    print(f"🧠 Prompt {i}: {prompt}")
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    t1 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150)
    gen_time = time.time() - t1

    result = tok.decode(output[0], skip_special_tokens=True)
    print(f"⏱ Generation time: {gen_time:.2f}s | Tokens: {len(result.split())}")
    print(f"💬 Output:\n{result}\n{'-'*80}\n")
