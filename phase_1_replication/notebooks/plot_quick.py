# notebooks/plot_quick.py
import json, pandas as pd, matplotlib.pyplot as plt, numpy as np, sys
from src.metrics.calibration import reliability_bins
from sklearn.metrics import RocCurveDisplay, roc_curve, auc

def reliability_plot(probs, labels, title, out):
    xs, ys, ws = reliability_bins(probs, labels, n_bins=10)
    plt.figure()
    plt.plot([0,1],[0,1],"--",linewidth=1)
    sizes = [w*1.5 for w in ws]
    plt.scatter(xs, ys, s=sizes)
    plt.xlabel("Predicted confidence"); plt.ylabel("Empirical accuracy")
    plt.title(title); plt.grid(alpha=.3)
    plt.savefig(out, dpi=150); plt.close()

def roc_plot(scores, labels, title, out, higher_is_better=True):
    s = np.array(scores); y = np.array(labels).astype(int)
    if not higher_is_better: s = -s
    fpr, tpr, _ = roc_curve(y, s)
    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr,tpr)).plot()
    plt.title(title); plt.grid(alpha=.3)
    plt.savefig(out, dpi=150); plt.close()

def do_one(name):
    df = pd.read_parquet(f"outputs/runs/{name}.parquet")
    # Reliability: benutze die bereits normalisierten p_* Felder
    reliability_plot(df["p_logit"],  df["correct"], f"{name.upper()} – Reliability (Logit)",  f"outputs/figs/{name}_rel_logit.png")
    reliability_plot(df["p_hidden"], df["correct"], f"{name.upper()} – Reliability (Hidden)", f"outputs/figs/{name}_rel_hidden.png")
    if "p_attn" in df:
        reliability_plot(df["p_attn"],  df["correct"], f"{name.upper()} – Reliability (Attn)",  f"outputs/figs/{name}_rel_attn.png")

    # ROC: rohe Scores
    roc_plot(df["conf_lntp"],        df["correct"], f"{name.upper()} – ROC (Logit)",  f"outputs/figs/{name}_roc_logit.png",  True)
    roc_plot(df["conf_hidden_norm"], df["correct"], f"{name.upper()} – ROC (Hidden)", f"outputs/figs/{name}_roc_hidden.png", True)
    if "conf_attn_entropy" in df:
        roc_plot(df["conf_attn_entropy"], df["correct"], f"{name.upper()} – ROC (Attn↓)", f"outputs/figs/{name}_roc_attn.png", False)

if __name__ == "__main__":
    import os; os.makedirs("outputs/figs", exist_ok=True)
    for name in ["arc","tqa_mc"]:
        do_one(name)
    print("Plots gespeichert unter outputs/figs/")
