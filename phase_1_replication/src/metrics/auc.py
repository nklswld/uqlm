import numpy as np
from sklearn.metrics import roc_auc_score

def auroc(scores, labels, higher_is_better=True):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(int)
    if not higher_is_better: s = -s
    return float(roc_auc_score(y, s))
