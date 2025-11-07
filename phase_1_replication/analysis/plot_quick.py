# analysis/plot_quick.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------
# Helpers
# -----------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def reliability_bins(probs, labels, n_bins=10):
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i+1] if i < n_bins-1 else probs <= bins[i+1])
        if m.sum() == 0:
            continue
        xs.append(probs[m].mean())
        ys.append(labels[m].mean())
        ws.append(m.sum() / len(probs))
    return np.array(xs), np.array(ys), np.array(ws)

def plot_reliability(probs, labels, title, out_path):
    xs, ys, ws = reliability_bins(probs, labels, n_bins=10)
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    # Punktgröße ~ Bin-Gewicht
    sizes = (ws * 600) + 5
    plt.scatter(xs, ys, s=sizes)
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc(scores, labels, title, out_path, higher_is_better=True):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if not higher_is_better:
        scores = -scores
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def platt_calibrate(scores, labels):
    """Nur für Visualisierung: 1D Platt-Scaling (LogReg) -> p(correct)."""
    x = np.asarray(scores, float).reshape(-1, 1)
    y = np.asarray(labels, int)
    sc = StandardScaler().fit(x)
    xs = sc.transform(x)
    clf = LogisticRegression(solver="lbfgs").fit(xs, y)
    def cal(z):
        z = np.asarray(z, float).reshape(-1, 1)
        return clf.predict_proba(sc.transform(z))[:, 1]
    return cal

# -----------------------
# Main
# -----------------------
def do_dataset(name):
    df = pd.read_parquet(f"outputs/runs/{name}.parquet")
    out_dir = ensure_dir("outputs/figs")

    # Test-/OOF-Sicht: dort, wo p_hidden_probe verfügbar ist
    has_probe = "p_hidden_probe" in df.columns
    if has_probe and df["p_hidden_probe"].notna().any():
        tst = df[df["p_hidden_probe"].notna()].reset_index(drop=True)
    else:
        tst = df.copy()

    y = tst["correct"].astype(int).to_numpy()

    # -------- ROC: Logit / Hidden raw / Hidden probe / Attention raw --------
    plot_roc(
        scores=tst["conf_lntp"],
        labels=y,
        title=f"{name.upper()} – ROC (Logit raw)",
        out_path=f"{out_dir}/{name}_roc_logit_raw.png",
        higher_is_better=True,
    )

    plot_roc(
        scores=tst["conf_hidden_norm"],
        labels=y,
        title=f"{name.upper()} – ROC (Hidden raw)",
        out_path=f"{out_dir}/{name}_roc_hidden_raw.png",
        higher_is_better=True,
    )

    if has_probe:
        plot_roc(
            scores=tst["p_hidden_probe"],
            labels=y,
            title=f"{name.upper()} – ROC (Hidden probe, OOF)",
            out_path=f"{out_dir}/{name}_roc_hidden_probe.png",
            higher_is_better=True,
        )

    if "conf_attn_entropy" in tst.columns:
        plot_roc(
            scores=tst["conf_attn_entropy"],
            labels=y,
            title=f"{name.upper()} – ROC (Attention entropy ↓)",
            out_path=f"{out_dir}/{name}_roc_attn.png",
            higher_is_better=False,  # Entropie niedrig = sicher
        )

    # -------- Reliability: Hidden probe (direkt), Logit (optional kalibriert), Attention (monoton skaliert) --------
    # Hidden-Probe ist bereits p(correct)
    if has_probe:
        plot_reliability(
            probs=tst["p_hidden_probe"],
            labels=y,
            title=f"{name.upper()} – Reliability (Hidden probe, OOF)",
            out_path=f"{out_dir}/{name}_rel_hidden_probe.png",
        )

    # Für Logit bauen wir fürs Diagramm eine kleine Platt-Kalibrierung (nur Visualisierung)
    cal_logit = platt_calibrate(tst["conf_lntp"], y)
    p_logit = cal_logit(tst["conf_lntp"])
    plot_reliability(
        probs=p_logit,
        labels=y,
        title=f"{name.upper()} – Reliability (Logit, Platt vis.)",
        out_path=f"{out_dir}/{name}_rel_logit_platt.png",
    )

    # Attention: monotone Min-Max zu p(correct) (nur anschaulich)
    if "conf_attn_entropy" in tst.columns:
        att = tst["conf_attn_entropy"].to_numpy(float)
        p_attn = 1.0 - (att - att.min()) / (att.max() - att.min() + 1e-12)
        plot_reliability(
            probs=p_attn,
            labels=y,
            title=f"{name.upper()} – Reliability (Attention, monotone)",
            out_path=f"{out_dir}/{name}_rel_attn.png",
        )

    print(f"[OK] Plots geschrieben nach {out_dir}/ (Prefix: {name}_)")

def main():
    parser = argparse.ArgumentParser(description="Plot ROC & Reliability for ARC and TruthfulQA-MC")
    parser.add_argument("--dataset", type=str, default="all", choices=["arc", "tqa_mc", "all"])
    args = parser.parse_args()

    if args.dataset in ("arc", "all"):
        do_dataset("arc")
    if args.dataset in ("tqa_mc", "all"):
        do_dataset("tqa_mc")

if __name__ == "__main__":
    main()
