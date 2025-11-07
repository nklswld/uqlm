# phase_1_replication/src/evaluate.py
import json
import argparse
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from scipy.stats import spearmanr, pearsonr

LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)

def extract_letter(ans: str):
    """
    Extrahiert den ersten einzelnen Buchstaben A-H aus dem Modelloutput.
    Akzeptiert Varianten wie 'C', 'Answer: C', 'C)' etc.
    """
    if not ans:
        return None
    m = LETTER_RE.search(ans.strip())
    return m.group(1).upper() if m else None

def ece(conf, correct, bins=10):
    """
    Expected Calibration Error (isotone binning per fixed-width bins).
    conf: [N] in [0,1]
    correct: [N] in {0,1}
    """
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=int)
    edges = np.linspace(0, 1, bins + 1)
    n = len(y)
    err = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < (edges[i+1] if i < bins - 1 else edges[i+1] + 1e-9))
        if not np.any(m):
            continue
        acc = y[m].mean()
        c = conf[m].mean()
        err += (m.sum() / n) * abs(acc - c)
    return float(err)

def load_preds(generated_jsonl: str, score_jsonl: str, score_key: str):
    """
    Joint über 'id'; korrekte Antwort = extrahierter Buchstabe == gold.
    """
    gold = {}
    correct = {}
    with open(generated_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            g = ex.get("gold")
            pred_letter = extract_letter(ex.get("pred_text", ""))
            is_correct = (pred_letter is not None and isinstance(g, str) and pred_letter == g.strip().upper())
            gold[ex["id"]] = g
            correct[ex["id"]] = 1 if is_correct else 0

    scores = {}
    with open(score_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if score_key not in r:
                raise KeyError(f"score_key '{score_key}' not in record for id={r.get('id')}")
            scores[r["id"]] = float(r[score_key])

    ids = [i for i in gold.keys() if i in scores]
    y = np.array([correct[i] for i in ids], dtype=int)
    s = np.array([scores[i] for i in ids], dtype=float)
    return ids, y, s

def compute_metrics(y: np.ndarray, s: np.ndarray):
    uniq = np.unique(y)
    auroc = float(roc_auc_score(y, s)) if len(uniq) > 1 else float("nan")
    auprc = float(average_precision_score(y, s)) if len(uniq) > 1 else float("nan")
    return {
        "N": int(len(y)),
        "Acc": float(accuracy_score(y, y)),  # y==y ist die Task-Accuracy; identisch zu y.mean()
        "ECE_10": ece(s, y, bins=10),
        "ECE_20": ece(s, y, bins=20),
        "AUROC": auroc,
        "AUPRC": auprc,
        "Spearman": float(spearmanr(y, s, nan_policy="omit").correlation),
        "Pearson": float(pearsonr(y, s)[0]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True, help="generated.jsonl aus generate.py")
    ap.add_argument("--scores", required=True, help="scores_*.jsonl (lntp/actcab/attn)")
    ap.add_argument("--score_key", required=True, help="score_lntp | score_mtp | score_actcab | score_attn")
    ap.add_argument("--out", required=True, help="Pfad für metrics.json")
    args = ap.parse_args()

    ids, y, s = load_preds(args.generated, args.scores, args.score_key)
    metrics = compute_metrics(y, s)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as g:
        json.dump(metrics, g, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
