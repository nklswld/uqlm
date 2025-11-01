# scripts/run_uqlm.py
import os
import sys
import argparse
import asyncio
from datetime import datetime
from typing import List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --- UQLM ---
from uqlm import BlackBoxUQ

# --- optionale Backends importieren (lazy / robust) ---
def import_ollama():
    try:
        from langchain_ollama import ChatOllama
    except Exception:
        # Fallback für ältere Setups
        from langchain_community.chat_models import ChatOllama
    return ChatOllama

def import_openai():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI

def import_fake_with_temp():
    from langchain_community.chat_models.fake import FakeListChatModel

    class FakeWithTemp(FakeListChatModel):
        # UQLM erwartet u.a. temperature
        temperature: float = 0.3

    return FakeWithTemp

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run UQLM Black-Box scorers with fake|ollama|openai backends."
    )
    p.add_argument("--backend", choices=["fake", "ollama", "openai"], default="fake")
    p.add_argument("--model", default=None,
                   help="Ollama/OpenAI Modell (z.B. mistral, llama3, gpt-4o-mini). Optional für fake.")
    p.add_argument("--prompts", default="benchmarks/prompts_demo.csv",
                   help="CSV mit Spalte 'prompt'. Fällt zurück auf Demo-Prompts, wenn Datei fehlt.")
    p.add_argument("--num-responses", type=int, default=3,
                   help="Kandidaten pro Prompt (black-box).")
    p.add_argument("--scorers", default="semantic_negentropy",
                   help="Kommagetrennte Liste, z.B. 'semantic_negentropy,noncontradiction'.")
    p.add_argument("--outdir", default="results",
                   help="Basis-Ausgabeverzeichnis.")
    return p.parse_args()

# -------------- Utilities --------------
def load_prompts(path: str) -> List[str]:
    if not os.path.exists(path):
        return [
            "Nenne drei Fakten über den Planeten Mars mit Quellenangaben.",
            "Erkläre kurz, was semantische Entropie ist."
        ]
    try:
        df = pd.read_csv(path)
        if "prompt" not in df.columns:
            # Falls jemand ; als Trennzeichen nutzt:
            df = pd.read_csv(path, sep=";")
        if "prompt" not in df.columns:
            raise ValueError("Spalte 'prompt' fehlt.")
        return [p for p in df["prompt"].dropna().astype(str).tolist() if p.strip()]
    except Exception:
        # Fallback: Datei als Text lesen, erste Zeile 'prompt' ggf. entfernen
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines and lines[0].lower() == "prompt":
            lines = lines[1:]
        return lines

def build_fake_responses(num_prompts: int, num_responses: int) -> List[str]:
    # UQLM ruft pro Prompt: 1x Original + num_responses Kandidaten.
    # Wir geben großzügig doppelt so viele Antworten, um batching o.ä. abzudecken.
    total = num_prompts * (1 + num_responses) * 2
    pool = [
        "Antwort A – eher sicher klingend.",
        "Antwort B – alternative Formulierung.",
        "Antwort C – widersprüchlich zu A.",
        "Antwort D – knapp, evtl. unpräzise.",
        "Antwort E – ausführlicher, aber ähnlich zu A."
    ]
    return [pool[i % len(pool)] for i in range(total)]

def summarize_cols(df: pd.DataFrame, scorer_names: List[str]) -> List[str]:
    cols = ["prompt"]
    if "response" in df.columns:
        cols.append("response")
    # mögliche Score-Spalten (je nach UQLM-Version)
    for s in scorer_names:
        if s in df.columns and s not in cols:
            cols.append(s)
    # generische Kandidaten
    for generic in ["confidence", "score", "uq_score"]:
        if generic in df.columns and generic not in cols:
            cols.append(generic)
    return cols

# -------------- Main logic --------------
async def run():
    args = parse_args()

    backend = args.backend
    model = args.model

    prompts = load_prompts(args.prompts)
    if not prompts:
        print("❌ Keine Prompts gefunden.")
        sys.exit(1)

    scorer_list = [s.strip() for s in args.scorers.split(",") if s.strip()]

    # --- LLM wählen ---
    if backend == "fake":
        FakeWithTemp = import_fake_with_temp()
        llm = FakeWithTemp(responses=build_fake_responses(len(prompts), args.num_responses))
        model_name = "fake"
    elif backend == "ollama":
        ChatOllama = import_ollama()
        model = model or os.getenv("OLLAMA_MODEL", "mistral")
        llm = ChatOllama(model=model, temperature=0.3)
        model_name = model
    else:  # openai
        ChatOpenAI = import_openai()
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # OPENAI_API_KEY muss gesetzt sein
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY fehlt in deiner Umgebung (.env).")
            sys.exit(1)
        llm = ChatOpenAI(model=model, temperature=0.3)
        model_name = model

    # --- UQLM ausführen ---
    bbuq = BlackBoxUQ(llm=llm, scorers=scorer_list, use_best=True)
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=args.num_responses)
    df = results.to_df()

    # --- Ausgabe ---
    cols_summary = summarize_cols(df, scorer_list)
    print("\nVorschau:")
    try:
        print(df[cols_summary].head())
    except Exception:
        print(df.head())

    # --- Export ---
    subdir = os.path.join(args.outdir, backend)
    os.makedirs(subdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = (model_name or backend).replace(":", "_").replace("/", "_")
    out_path = os.path.join(subdir, f"uqlm_{backend}_{safe_model}_{ts}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="raw")
        try:
            df[cols_summary].to_excel(xw, index=False, sheet_name="summary")
        except Exception:
            pass

    print(f"\n✅ Ergebnisse gespeichert: {out_path}")

if __name__ == "__main__":
    asyncio.run(run())
