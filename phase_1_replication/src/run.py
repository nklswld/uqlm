# src/run.py
import os
import json
import argparse
import numpy as np
import pandas as pd

from models.loaders import load_model
from benchmarks.arc.load_arc import load_arc
from benchmarks.truthfulqa_mc.load_tqa import load_truthfulqa_mc
from src.prompting.mc_prompting import prompt_arc, prompt_tqa_mc
from src.infer.mc_inference import score_letter, decision_state_from_prompt
from src.metrics.calibration import ece
from src.metrics.auc import auroc

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


# -----------------------------
# Helpers
# -----------------------------
def safe_has_attn(df: pd.DataFrame) -> bool:
    return ("conf_attn_entropy" in df.columns) and (not np.all(np.isnan(df["conf_attn_entropy"]).astype(bool)))

def fit_probe(X: np.ndarray, y: np.ndarray):
    """Linear-Probe (ACTCAB-Light) auf Decision-Hidden-Features -> P(correct)."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(max_iter=200, solver="lbfgs").fit(Xs, y)
    def predict_proba(Z):
        Zs = sc.transform(Z)
        return clf.predict_proba(Zs)[:, 1]
    return predict_proba

def oof_probe_scores(emb: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Out-of-Fold-Probabilities (stratifiziert), robust bei kleinen Klassen."""
    n = len(y)
    oof_p = np.full(n, np.nan, dtype=float)
    classes = np.unique(y)
    if classes.size < 2:
        return oof_p
    counts = np.bincount(y)
    min_class = int(counts.min())
    k = max(2, min(5, min_class))  # mind. 2, höchstens 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(emb, y):
        probe = fit_probe(emb[tr_idx], y[tr_idx])
        oof_p[te_idx] = probe(emb[te_idx])
    return oof_p

def sample_exact_indices(n_total: int, n_want: int, seed: int = 42):
    """Ziehe EXAKT n_want eindeutige Indizes ohne Zurücklegen (reproduzierbar)."""
    n = int(min(n_total, n_want))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n, replace=False)).tolist()


# -----------------------------
# Core runner (MCQ Letter-Scoring + Decision-State, Exact-N Sampling)
# -----------------------------
def _run_mcq(all_records,
             make_prompt,
             get_gt_from_record,
             model_name="mistralai/Mistral-7B-Instruct-v0.3",
             items: int = 300,
             bench_name: str = "arc",
             seed: int = 42):
    """
    - all_records: vollständige Dataset-Sequenz (HF Dataset / Liste)
    - items: EXAKTE Anzahl zu evaluierender Items (ohne Zurücklegen)
    """
    # ---- Exact-N Sampling (repro) ----
    n_total = len(all_records)
    picked = sample_exact_indices(n_total, items, seed=seed)

    # Save which items we used (for full reproducibility)
    os.makedirs("outputs/runs", exist_ok=True)
    with open(f"outputs/runs/{bench_name}_items_used.json", "w") as f:
        json.dump({"total": n_total, "used_n": len(picked), "indices": picked, "seed": seed}, f, indent=2)

    tok, mdl = load_model(model_name)

    rows = []
    dec_feats = []

    for i in picked:
        r = all_records[i]

        # Frage/Optionen/GT extrahieren
        if bench_name == "arc":
            q = r["question"]
            opts = r["choices"]["text"]
        else:
            q = r["question"]
            opts = r["mc1_targets"]["choices"]

        gt = get_gt_from_record(r)
        prompt = make_prompt(q, opts)

        # 1x Decision-State aus Prompt (vor Letter)
        dec = decision_state_from_prompt(tok, mdl, prompt)
        dec_hidden_norm  = dec["decision_hidden_norm"]
        dec_hidden_var   = dec["decision_hidden_var"]
        dec_attn_entropy = dec["decision_attn_entropy"]
        dec_hidden_feat  = dec["decision_hidden_feat"]

        # Letter-Scoring: LNTP je Antwort-Buchstabe
        letters = [chr(65 + k) for k in range(len(opts))]
        option_scores = [score_letter(tok, mdl, prompt, L) for L in letters]
        lntps = np.array([s["lntp"] for s in option_scores], dtype=float)
        pred_idx = int(np.nanargmax(lntps))
        conf_lntp = float(lntps[pred_idx])

        rows.append({
            "orig_idx": i,  # originaler Datensatzindex (gesampelt)
            "gt": gt,
            "pred": pred_idx,
            "correct": int(pred_idx == gt),
            "conf_lntp": conf_lntp,
            "conf_hidden_norm": dec_hidden_norm,
            "conf_hidden_var":  dec_hidden_var,
            "conf_attn_entropy": dec_attn_entropy,
        })
        dec_feats.append(dec_hidden_feat)

    # DataFrame + Embeddings
    df  = pd.DataFrame(rows)
    emb = np.stack(dec_feats)
    y   = df["correct"].to_numpy().astype(int)

    # Shuffle for stability (but deterministic given seed)
    rng = np.random.default_rng(seed)
    order = np.arange(len(df))
    rng.shuffle(order)
    df  = df.iloc[order].reset_index(drop=True)
    emb = emb[order]
    y   = y[order]

    # ---- OOF-Probe ----
    oof_p = oof_probe_scores(emb, y, seed=seed)
    df["p_hidden_probe"] = oof_p
    tst = df[df["p_hidden_probe"].notna()].reset_index(drop=True)

    # ---- Metrics (auf OOF-Test) ----
    metrics = {
        "n_total_available": int(n_total),
        "n_used": int(len(df)),
        "seed": int(seed),
        "acc": float(df["correct"].mean()),
        "auroc_logit_raw":  auroc(tst["conf_lntp"],        tst["correct"], True) if len(tst) else float("nan"),
        "auroc_hidden_raw": auroc(tst["conf_hidden_norm"], tst["correct"], True) if len(tst) else float("nan"),
        "auroc_hidden_probe": auroc(tst["p_hidden_probe"],  tst["correct"], True) if len(tst) else float("nan"),
        "ece_hidden_probe_10": ece(tst["p_hidden_probe"],   tst["correct"], 10) if len(tst) else float("nan"),
    }
    if safe_has_attn(df) and len(tst):
        metrics["auroc_attn_raw"] = auroc(-tst["conf_attn_entropy"], tst["correct"], True)
        attn_min, attn_max = tst["conf_attn_entropy"].min(), tst["conf_attn_entropy"].max()
        p_attn = 1.0 - (tst["conf_attn_entropy"] - attn_min) / (attn_max - attn_min + 1e-12)
        metrics["ece_attn_10"] = ece(p_attn, tst["correct"], 10)

    _save_outputs(bench_name, df, metrics)
    print(json.dumps(metrics, indent=2))
    return df, metrics


# -----------------------------
# Save helper
# -----------------------------
def _save_outputs(name, df, metrics):
    os.makedirs("outputs/runs", exist_ok=True)
    df.to_parquet(f"outputs/runs/{name}.parquet", index=False)
    with open(f"outputs/runs/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Saved: outputs/runs/{name}.parquet & {name}_metrics.json")


# -----------------------------
# Public functions
# -----------------------------
def run_arc(items=300, model_name="mistralai/Mistral-7B-Instruct-v0.3", seed=42):
    ds = load_arc(split="test", subset="ARC-Challenge")
    def get_gt(r):  # "A".."D" -> 0..3
        return ord(r["answerKey"]) - 65
    return _run_mcq(
        all_records=ds,
        make_prompt=prompt_arc,
        get_gt_from_record=get_gt,
        model_name=model_name,
        items=items,
        bench_name="arc",
        seed=seed,
    )

def run_tqa(items=300, model_name="mistralai/Mistral-7B-Instruct-v0.3", seed=42):
    ds = load_truthfulqa_mc(split="validation")
    def get_gt(r):
        return int(np.argmax(r["mc1_targets"]["labels"]))
    return _run_mcq(
        all_records=ds,
        make_prompt=prompt_tqa_mc,
        get_gt_from_record=get_gt,
        model_name=model_name,
        items=items,
        bench_name="tqa_mc",
        seed=seed,
    )


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 replication runner (Exact-N sampling + decision-state OOF probe)")
    parser.add_argument("--benchmark", type=str, default="both", choices=["arc", "tqa", "both"])
    parser.add_argument("--items", type=int, default=300, help="EXAKTE Anzahl der zu evaluierenden Items (random sampling, no replacement)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling/OOF Seed (Reproduzierbarkeit)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="HF Modell-ID")
    args = parser.parse_args()

    if args.benchmark in ("arc", "both"):
        print("\n[Run] ARC-Challenge (exact-N sampling + decision-state OOF probe)")
        run_arc(items=args.items, model_name=args.model, seed=args.seed)

    if args.benchmark in ("tqa", "both"):
        print("\n[Run] TruthfulQA-MC (exact-N sampling + decision-state OOF probe)")
        run_tqa(items=args.items, model_name=args.model, seed=args.seed)
