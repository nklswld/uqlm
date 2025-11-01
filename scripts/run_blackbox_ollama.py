# scripts/run_blackbox_ollama.py
import asyncio
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Bevorzugt neue Lib, sonst Fallback:
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama  # deprecated, aber ok

from uqlm import BlackBoxUQ

# ---- Konfiguration ----
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")         # z.B. "llama3:8b" oder "mistral"
NUM_RESPONSES = int(os.getenv("NUM_RESPONSES", "2"))

PROMPTS = [
    "Erkläre kurz, was semantische Entropie ist.",
    "Nenne drei Fakten über den Planeten Mars mit Quellenangaben."
]

async def main():
    llm = ChatOllama(model=MODEL, temperature=0.3)
    bbuq = BlackBoxUQ(llm=llm, scorers=["semantic_negentropy"], use_best=True)

    dfs = []
    # Fortschrittsbalken: 1 Schritt je Prompt
    for prompt in tqdm(PROMPTS, desc="UQLM (BlackBox)"):
        # pro Prompt generieren & scoren
        results = await bbuq.generate_and_score(prompts=[prompt], num_responses=NUM_RESPONSES)
        df = results.to_df()
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Terminal-Vorschau
    cols_summary = [c for c in ["prompt", "response", "semantic_negentropy"] if c in df_all.columns]
    print("\nVorschau:")
    print(df_all[cols_summary].head() if cols_summary else df_all.head())

    # Excel-Export in ./results/ mit Zeitstempel
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = MODEL.replace(":", "_").replace("/", "_")
    out_path = os.path.join("results", f"ollama_blackbox_{safe_model}_{ts}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_all.to_excel(xw, index=False, sheet_name="raw")
        if cols_summary:
            df_all[cols_summary].to_excel(xw, index=False, sheet_name="summary")

    print(f"\n✅ Ergebnisse gespeichert: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
