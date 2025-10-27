# scripts/prefetch_models.py
from pathlib import Path
import inspect
from huggingface_hub import snapshot_download, __version__ as hf_ver

MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", None),  # ggf. Access auf HF-Seite akzeptieren
    ("mistralai/Mistral-7B-Instruct-v0.3", None),
    ("Qwen/Qwen2.5-7B-Instruct", None),
    ("epfl-llm/meditron-7b", None),
]

print("huggingface_hub version:", hf_ver)
print(">>> Prefetching models into local HF cache ...")

# Dynamisch herausfinden, welche Parameter snapshot_download unterstützt
sig = inspect.signature(snapshot_download)
params = sig.parameters

def download(repo_id, rev):
    kw = {}
    # bevorzugt allow_regex, sonst allow_patterns, sonst ignore_patterns, sonst nichts
    if "allow_regex" in params:
        kw["allow_regex"] = [r".*"]
    elif "allow_patterns" in params:
        kw["allow_patterns"] = ["*"]
    elif "ignore_patterns" in params:
        kw["ignore_patterns"] = []  # nix ignorieren -> alles laden
    # local_files_only=False erzwingt Download, falls noch nicht im Cache
    path = snapshot_download(
        repo_id=repo_id,
        revision=rev,
        local_files_only=False,
        **kw
    )
    return path

for repo_id, rev in MODELS:
    tag = f"{repo_id}" + (f" @ {rev}" if rev else "")
    print(f"- {tag}")
    try:
        local_path = download(repo_id, rev)
        print(f"  cached at: {local_path}")
    except Exception as e:
        print(f"  ❌ failed: {e}")
        print("  Hint: Ist das Modell 'gated'? Dann auf der HF-Seite Zugriff akzeptieren und eingeloggt sein: `hf auth login`")

print("\n✅ Done. Danach kannst du mit local_files_only=True offline laden.")
