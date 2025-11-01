# scripts/run_blackbox_fake.py
import asyncio
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_community.chat_models.fake import FakeListChatModel
from uqlm import BlackBoxUQ

# --- Neu: Wrapper mit temperature -------------------
class FakeWithTemp(FakeListChatModel):
    # UQLM erwartet dieses Attribut
    temperature: float = 0.3
# ----------------------------------------------------

NUM_RESPONSES = int(os.getenv("NUM_RESPONSES", "3"))

PROMPTS = [
    "Nenne drei Fakten über den Planeten Mars mit Quellenangaben.",
    "Erkläre kurz, was semantische Entropie ist."
]

def build_fake_responses(prompts, num_responses):
    total_calls = len(prompts) * (1 + num_responses)
    pool = [
        "Antwort A – eher sicher klingend.",
        "Antwort B – alternative Formulierung.",
        "Antwort C – widersprüchlich zu A.",
        "Antwort D – knapp, evtl. unpräzise.",
        "Antwort E – ausführlicher, aber ähnlich zu A."
    ]
    responses = [pool[i % len(pool)] for i in range(total_calls)]
    return responses

async def main():
    fake_responses = build_fake_responses(PROMPTS, NUM_RESPONSES)

    # Wichtig: die neue Klasse verwenden
    llm = FakeWithTemp(responses=fake_responses)

    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["semantic_negentropy"],
        use_best=True
    )

    results = await bbuq.generate_and_score(prompts=PROMPTS, num_responses=NUM_RESPONSES)
    df = results.to_df()

    cols_summary = [c for c in ["prompt", "response", "semantic_negentropy"] if c in df.columns]
    print("\nVorschau:")
    print(df[cols_summary].head() if cols_summary else df.head())

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("results", f"fake_blackbox_{ts}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="raw")
        if cols_summary:
            df[cols_summary].to_excel(xw, index=False, sheet_name="summary")
    print(f"\n✅ Ergebnisse gespeichert: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
