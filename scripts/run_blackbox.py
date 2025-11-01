# run_blackbox.py
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# ✳️ Wähle DEIN LLM:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# # ODER Vertex AI (wenn du das nutzt):
# from langchain_google_vertexai import ChatVertexAI
# llm = ChatVertexAI(model="gemini-1.5-flash-002", temperature=0.7)

from uqlm import BlackBoxUQ

prompts = [
    "Nenne drei Fakten über den Planeten Mars mit Quellenangaben.",
    "Erkläre kurz, was semantische Entropie ist."
]

async def main():
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["semantic_negentropy"],  # guter Allround-Start
        use_best=True                     # gleich die 'sicherste' Antwort wählen
    )

    results = await bbuq.generate_and_score(prompts=prompts, num_responses=3)
    df = results.to_df()

    # ---- robuste Anzeige im Terminal ----
    print("\nVerfügbare Spalten:", list(df.columns))
    candidate_resp_cols = ["chosen_response", "best_response", "mitigated_response", "response", "final_response"]
    candidate_conf_cols = ["confidence", "confidence_score", "score", "uq_score"]

    resp_col = next((c for c in candidate_resp_cols if c in df.columns), None)
    conf_col = next((c for c in candidate_conf_cols if c in df.columns), None)

    cols_to_show = ["prompt"] + ([resp_col] if resp_col else []) + ([conf_col] if conf_col else [])
    print("\nVorschau:")
    print(df[cols_to_show].head() if cols_to_show else df.head())

    # ---- Excel-Export in ./results/ ----
    from datetime import datetime
    import pandas as pd

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden
    path = os.path.join(results_dir, f"results_blackbox_{ts}.xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="raw")
        if cols_to_show:
            df[cols_to_show].to_excel(xw, index=False, sheet_name="summary")

    print(f"\n✅ Ergebnisse gespeichert: {path}")

if __name__ == "__main__":
    asyncio.run(main())
