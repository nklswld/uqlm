import os
from datasets import load_dataset

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(BASE, exist_ok=True)

def save_split(ds, tgt_dir, split_name):
    path = os.path.join(tgt_dir, f"{split_name}.jsonl")
    ds.to_json(path, lines=True, force_ascii=False)

def get_arc():
    ds = load_dataset("ai2_arc", "ARC-Challenge")
    tgt = os.path.join(BASE, "arc_c"); os.makedirs(tgt, exist_ok=True)
    for split in ("train","validation","test"):
        if split in ds: save_split(ds[split], tgt, split)

def get_truthfulqa_mc():
    ds = load_dataset("truthful_qa", "multiple_choice")
    tgt = os.path.join(BASE, "truthfulqa_mc"); os.makedirs(tgt, exist_ok=True)
    for split in ds.keys(): save_split(ds[split], tgt, split)

if __name__ == "__main__":
    get_arc(); get_truthfulqa_mc()
    print("✓ Datasets in experiments/phase1_general/data/")
