import json, math, re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# Pfade
# =========================
IN_FILE   = Path("experiments/phase1_general/outputs/arc_c_llama3_val.jsonl")
OUT_XLSX  = Path("experiments/phase1_general/outputs/arc_c_llama3_val_lntp.xlsx")

# =========================
# Helpers
# =========================
LETTERS = set("ABCDEF")
_letter_regex = re.compile(r"\b([A-F])\b", re.IGNORECASE)

def _ensure_list(x):
    """Falls Felder als JSON-String (z. B. via Excel) vorliegen, zurück in Python-Listen parsen."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return x if isinstance(x, list) else []

def logsumexp(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    m = np.max(a)
    return m + math.log(np.exp(a - m).sum())

def lntp_logmean(step_scores, token_ids):
    """Geometrisches Mittel der generierten Token-Wahrscheinlichkeiten (stabil im Log-Raum)."""
    step_scores = _ensure_list(step_scores)
    token_ids   = _ensure_list(token_ids)
    if not step_scores or not token_ids:
        return 0.0

    T = min(len(step_scores), len(token_ids))
    logps = []
    for t in range(T):
        logits = step_scores[t]
        tid    = token_ids[t]
        if not isinstance(logits, list) or len(logits) == 0:
            continue
        if not isinstance(tid, int) or tid < 0 or tid >= len(logits):
            continue
        lse  = logsumexp(logits)
        logp = float(logits[tid]) - lse
        if np.isfinite(logp):
            logps.append(logp)

    if not logps:
        return 0.0
    p = math.exp(sum(logps) / len(logps))
    return max(1e-12, min(1.0 - 1e-12, p))

def parse_letter(txt):
    if not txt:
        return ""
    s = str(txt).strip()
    m = _letter_regex.search(s)
    if m:
        return m.group(1).upper()
    for ch in s.upper():
        if ch in LETTERS:
            return ch
    return ""

def ece(probs, labels, n_bins=10):
    probs  = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    N = len(labels)
    for i in range(n_bins):
        left, right = bins[i], bins[i+1]
        mask = (probs >= left) & (probs < right if i < n_bins - 1 else probs <= right)
        if mask.any():
            conf = probs[mask].mean()
            acc  = labels[mask].mean()
            e += (mask.sum() / N) * abs(acc - conf)
    return float(e)

def safe_auroc(y_true_bin, scores):
    try:
        return roc_auc_score(y_true_bin, scores)
    except Exception:
        return float("nan")

def safe_ap(y_true_bin, scores):
    try:
        return average_precision_score(y_true_bin, scores)
    except Exception:
        return float("nan")

# =========================
# Hauptlogik
# =========================
def main():
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    y_true = []   # 1 = korrekt, 0 = falsch
    confs  = []   # LNTP

    with open(IN_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Score → Excel"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            step_scores   = r.get("step_scores", [])
            gen_token_ids = r.get("gen_token_ids", [])
            conf          = lntp_logmean(step_scores, gen_token_ids)

            predL = r.get("pred_letter") or parse_letter(r.get("pred_text", ""))
            goldL = r.get("gold", "")
            correct = int(predL == goldL) if predL and goldL else 0

            # Für Metriken
            y_true.append(correct)
            confs.append(conf)

            # Zeile für Excel (kompakte, gut lesbare Spalten)
            rows.append({
                "id": r.get("id"),
                "pred_letter": predL,
                "gold": goldL,
                "correct": correct,
                "score_lntp": conf,
                "pred_text": r.get("pred_text", ""),
                "prompt": r.get("prompt", ""),
                # komplexe Felder als JSON-String
                "gen_token_ids": json.dumps(_ensure_list(gen_token_ids), ensure_ascii=False),
                "step_scores":   json.dumps(_ensure_list(step_scores), ensure_ascii=False),
            })

    if len(rows) == 0:
        print("Keine Daten gefunden – prüfe IN_FILE.")
        return

    # Metriken
    acc   = float(sum(y_true)) / len(y_true)
    ece10 = ece(confs, y_true, n_bins=10)
    ece20 = ece(confs, y_true, n_bins=20)

    y_err  = [1 - y for y in y_true]   # 1 = Fehler
    uncert = [1.0 - c for c in confs]  # höhere Unsicherheit → eher Fehler
    auroc  = safe_auroc(y_err, uncert)
    ap     = safe_ap(y_err, uncert)

    # DataFrame + Metrikblatt
    df = pd.DataFrame(rows)

    # In eine Excel-Datei mit zwei Sheets schreiben
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xlw:
        df.to_excel(xlw, index=False, sheet_name="scores")
        met = pd.DataFrame([
            {"metric": "samples",         "value": len(y_true)},
            {"metric": "accuracy",        "value": acc},
            {"metric": "ece@10",          "value": ece10},
            {"metric": "ece@20",          "value": ece20},
            {"metric": "auroc(error↑)",   "value": auroc},
            {"metric": "pr_auc(error↑)",  "value": ap},
        ])
        met.to_excel(xlw, index=False, sheet_name="metrics")

    print(f"✓ Excel geschrieben: {OUT_XLSX}")
    print(f"Samples: {len(y_true)}  |  Acc: {acc:.3f}  |  ECE10/ECE20: {ece10:.3f}/{ece20:.3f}  |  AUROC: {auroc:.3f}  |  PR-AUC: {ap:.3f}")

if __name__ == "__main__":
    main()