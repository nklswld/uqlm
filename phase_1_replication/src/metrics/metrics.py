import numpy as np
from sklearn.metrics import roc_auc_score

def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs); labels = np.asarray(labels).astype(int)
    bins = np.linspace(0, 1, n_bins+1)
    inds = np.digitize(probs, bins) - 1
    ece_val = 0.0
    for b in range(n_bins):
        mask = inds == b
        if not mask.any(): continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece_val += (mask.sum()/len(probs)) * abs(acc - conf)
    return float(ece_val)

def reliability_points(probs, labels, n_bins=10):
    probs = np.asarray(probs); labels = np.asarray(labels).astype(int)
    bins = np.linspace(0, 1, n_bins+1)
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (probs >= lo) & (probs < hi) if b < n_bins-1 else (probs >= lo) & (probs <= hi)
        if mask.any():
            xs.append(probs[mask].mean()); ys.append(labels[mask].mean()); ws.append(mask.sum())
    return xs, ys, ws

def auroc(scores, labels, higher_is_better=True):
    s = np.asarray(scores)
    y = np.asarray(labels).astype(int)
    if not higher_is_better: s = -s
    return float(roc_auc_score(y, s))
