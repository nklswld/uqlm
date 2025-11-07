import numpy as np

def select_by_lntp(option_scores):
    lntps = np.array([s["lntp"] for s in option_scores])
    idx = int(lntps.argmax())
    return idx, float(lntps[idx])

def norm01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return ((x - lo) / (hi - lo + 1e-12)).tolist()
