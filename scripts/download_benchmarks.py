"""
Benchmark Download Script for UQLM
Author: Niklas Wild
Purpose: Download and verify medical benchmark datasets (MedQA, MedNLI, PubMedQA)
"""

from datasets import load_dataset
import os
import humanize
import random

# === Funktion zum Ausgeben der Cache-Größe ===
def get_cache_size():
    path = os.path.expanduser("~/.cache/huggingface/datasets")
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return humanize.naturalsize(total_size)

# === Funktion zum Laden und Anzeigen eines Beispiels ===
def load_and_preview(name, *args):
    print(f"\n📥 Lade Datensatz: {name} ...")
    dataset = load_dataset(name, *args)
    print(dataset)
    # Zeige ein zufälliges Beispiel aus dem Trainingssplit
    if 'train' in dataset:
        sample = random.choice(dataset['train'])
        print("\n🧩 Beispiel aus dem Trainingssplit:")
        for k, v in list(sample.items())[:5]:  # zeige nur die ersten 5 Schlüssel
            print(f"  {k}: {str(v)[:200]}")  # kürze lange Texte
    return dataset

# === Hauptprogramm ===
if __name__ == "__main__":
    print("🚀 Starte Download medizinischer Benchmarks...\n")

    # 1. MedQA (alias MedMCQA)
    medqa = load_and_preview("medmcqa")

    # 2. MedNLI
    mednli = load_and_preview("mednli_matias")

    # 3. PubMedQA (gelabelte Variante)
    pubmedqa = load_and_preview("pubmed_qa", "pqa_labeled")

    print("\n📦 Aktuelle Gesamtgröße des Hugging Face Dataset-Caches:", get_cache_size())
    print("\n✅ Download abgeschlossen. Alle Datensätze erfolgreich geladen!")
