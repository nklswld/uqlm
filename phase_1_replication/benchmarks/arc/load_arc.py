from datasets import load_dataset

def load_arc(split="test", subset="ARC-Challenge"):
    return load_dataset("allenai/ai2_arc", subset)[split]
