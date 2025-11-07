def prompt_arc(q, options):
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    # kurze, neutrale Instruktion – wir scoren nur, nicht generieren
    return f"Question: {q}\nOptions:\n{opts}\nAnswer (just the letter):"

def prompt_tqa_mc(q, options):
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
    return f"Question: {q}\nOptions:\n{opts}\nAnswer (just the letter):"
