import numpy as np

def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs); labels = np.asarray(labels).astype(int)
    bins = np.linspace(0,1,n_bins+1)
    inds = np.digitize(probs, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = inds == b
        if not m.any(): continue
        conf = probs[m].mean(); acc = labels[m].mean()
        e += (m.sum()/len(probs)) * abs(acc - conf)
    return float(e)

def reliability_bins(probs, labels, n_bins=10):
    probs = np.asarray(probs); labels = np.asarray(labels).astype(int)
    bins = np.linspace(0,1,n_bins+1)
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        m = (probs>=lo)&(probs<(hi if b<n_bins-1 else hi+1e-12))
        if m.any():
            xs.append(probs[m].mean()); ys.append(labels[m].mean()); ws.append(m.sum())
    return xs, ys, ws
