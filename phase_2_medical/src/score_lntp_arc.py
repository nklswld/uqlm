import json, math, re, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# === Pfade ===
IN_FILE  = Path("experiments/phase1_general/outputs/arc_c_llama3_val.jsonl")
OUT_FILE = Path("experiments/phase1_general/outputs/arc_c_llama3_val_lntp.jsonl")

LETTERS = set(list("ABCDEF"))

# ---------- Utils ----------

def _ensure_list(x):
    """Falls aus Excel/CSV JSON-Strings kommen, in Python-Objekte back-parsen."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return x if isinstance(x, list) else []

def logsumexp(a):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a)
    # falls a leer, -inf zurückgeben
    if a.size == 0 or not np.isfinite(m):
        return -np.inf
    return m + math.log(np.exp(a - m).sum())

def lntp_logmean(step_scores, token_ids):
    """
    Geometrisches Mittel der generierten Token-Wahrscheinlichkeiten im Log-Raum.
    Rückgabe: LNTP in [0,1].
    """
    step_scores = _ensure_list(step_scores)
    token_ids   = _ensure_list(token_ids)

    if not step_scores or not token_ids:
        return 0.0

    T = min(len(step_scores), len(token_ids))
    logps = []

    for t in range(T):
        logits = step_scores[t]
        tid = token_ids[t]

        # Guard: leere Logits oder falscher tid -> überspringen
        if not isinstance(logits, list) or len(logits) == 0:
            continue
        if not isinstance(tid, int) or tid < 0 or tid >= len(logits):
            continue

        lse = logsumexp(logits)
        # numerisch robuster: log p = logit_tid - logsumexp(logits)
        logp = float(logits[tid]) - lse
        # harte Caps vermeiden NaN bei exotischen Fällen
        if not np.isfinite(logp):
            continue
        logps.append(logp)

    if not logps:
        return 0.0

    mean_logp = sum(logps) / len(logps)
    # zurück nach [0,1]
    p = math.exp(mean_logp)
    # clamp minimal, um ECE-Nan zu vermeiden
    return max(1e-12, min(1.0 - 1e-12, p))

_letter_regex = re.compile(r"\b([A-F])\b", re.IGNORECASE)

def parse_letter(txt):
    if not txt:
        return ""
    s = str(txt).strip()
    m = _letter_regex.search(s)
    if m:
        return m.group(1).upper()
    # fallback: erstes A-F im String
    for ch in s.upper():
        if ch in LETTERS:
            return ch
    return ""

def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=float)
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

# ---------- Streaming Load, Score, Write ----------

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

y_true = []   # 1 = korrekt, 0 = falsch
confs  = []   # LNTP
preds  = []   # vorhergesagter Buchstabe (optional für Debug)

with open(IN_FILE, "r", encoding="utf-8") as rf, open(OUT_FILE, "w", encoding="utf-8") as wf:
    for line in tqdm(rf, desc="Scoring LNTP (stream)"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        step_scores = r.get("step_scores", [])
        gen_token_ids = r.get("gen_token_ids", [])
        conf = lntp_logmean(step_scores, gen_token_ids)

        predL = r.get("pred_letter") or parse_letter(r.get("pred_text", ""))
        goldL = r.get("gold", "")
        correct = int(predL == goldL) if predL and goldL else 0

        y_true.append(correct)
        confs.append(conf)
        preds.append(predL)

        r["score_lntp"]  = conf
        r["pred_letter"] = predL
        r["correct"]     = correct

        wf.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Metrics ----------
if len(y_true) == 0:
    print("Keine Daten gelesen.")
    sys.exit(0)

acc = float(sum(y_true)) / len(y_true)

ece10 = ece(confs, y_true, n_bins=10)
ece20 = ece(confs, y_true, n_bins=20)

# Für Halluzinationserkennung: 1 = Fehler
y_err  = [1 - y for y in y_true]
uncert = [1.0 - c for c in confs]

auroc = safe_auroc(y_err, uncert)
ap    = safe_ap(y_err, uncert)

print(f"Samples:          {len(y_true)}")
print(f"Accuracy:         {acc:.3f}")
print(f"ECE@10 / ECE@20:  {ece10:.3f} / {ece20:.3f}")
print(f"AUROC (error↑):   {auroc:.3f}")
print(f"PR-AUC (error↑):  {ap:.3f}")
print(f"→ Augmentierte Datei: {OUT_FILE}")
