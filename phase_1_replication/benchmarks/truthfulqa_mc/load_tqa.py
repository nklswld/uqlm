from datasets import load_dataset

def load_truthfulqa_mc(split="validation"):
    return load_dataset("truthfulqa/truthful_qa", "multiple_choice")[split]
