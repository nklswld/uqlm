import numpy as np

def select_option_by_lntp(option_scores):
    # option_scores: Liste von dicts (eine pro Option)
    lntps = np.array([s["lntp"] for s in option_scores])
    idx = int(lntps.argmax())
    return idx, float(lntps[idx])

def extract_conf(option_scores, idx, key):
    return float(option_scores[idx][key])  # z.B. "hidden_norm", "attn_entropy"
