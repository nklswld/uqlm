from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True) # important for offline use
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain why the sky appears blue."
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=150)

print(tok.decode(output[0], skip_special_tokens=True))
