import os, json, numpy as np, pandas as pd
from tqdm import tqdm

from models.loaders import load_model
from benchmarks.arc.load_arc import load_arc
from benchmarks.truthfulqa_mc.load_tqa import load_truthfulqa_mc
from src.prompting.mc_prompting import prompt_arc, prompt_tqa_mc
from src.infer.mc_inference import score_option
from src.scoring.basic import select_by_lntp, norm01
from src.metrics.calibration import ece
from src.metrics.auc import auroc

def run_arc(max_items=None):
    tok, mdl = load_model()
    ds = load_arc(split="test", subset="ARC-Challenge")

    rows = []
    for i, r in enumerate(tqdm(ds, desc="ARC-Challenge")):
        if max_items and i >= max_items: break
        q = r["question"]; opts = r["choices"]["text"]; gt = ord(r["answerKey"])-65
        prompt = prompt_arc(q, opts)

        option_scores = [score_option(tok, mdl, prompt, opt) for opt in opts]
        pred_idx, conf_lntp = select_by_lntp(option_scores)

        rows.append({
            "idx": i, "gt": gt, "pred": pred_idx, "correct": int(pred_idx==gt),
            "conf_lntp": conf_lntp,
            "conf_hidden_norm": option_scores[pred_idx]["hidden_norm"],
            "conf_hidden_var":  option_scores[pred_idx]["hidden_var"],
            "conf_attn_entropy": option_scores[pred_idx]["attn_entropy"],
        })

    df = pd.DataFrame(rows)
    # ECE braucht [0,1] – einfache monotone Skalierung je Scorefamilie
    df["p_logit"]  = norm01(df["conf_lntp"].values)
    df["p_hidden"] = norm01(df["conf_hidden_norm"].values)
    df["p_attn"]   = norm01((-df["conf_attn_entropy"]).values)  # geringe Entropie = hohe Sicherheit

    metrics = {
        "acc": float(df["correct"].mean()),
        "ece_logit_10": ece(df["p_logit"], df["correct"], 10),
        "ece_hidden_10": ece(df["p_hidden"], df["correct"], 10),
        "ece_attn_10": ece(df["p_attn"], df["correct"], 10),
        "auroc_logit": auroc(df["conf_lntp"], df["correct"], True),
        "auroc_hidden": auroc(df["conf_hidden_norm"], df["correct"], True),
        "auroc_attn": auroc(-df["conf_attn_entropy"], df["correct"], True),
    }
    os.makedirs("outputs/runs", exist_ok=True)
    df.to_parquet("outputs/runs/arc.parquet")
    with open("outputs/runs/arc_metrics.json","w") as f: json.dump(metrics, f, indent=2)
    print(metrics)

def run_tqa(max_items=None):
    tok, mdl = load_model()
    ds = load_truthfulqa_mc(split="validation")

    rows = []
    for i, r in enumerate(tqdm(ds, desc="TruthfulQA-MC")):
        if max_items and i >= max_items: break
        q = r["question"]; opts = r["mc1_targets"]["choices"]; gt = int(np.argmax(r["mc1_targets"]["labels"]))
        prompt = prompt_tqa_mc(q, opts)

        option_scores = [score_option(tok, mdl, prompt, opt) for opt in opts]
        pred_idx, conf_lntp = select_by_lntp(option_scores)

        rows.append({
            "idx": i, "gt": gt, "pred": pred_idx, "correct": int(pred_idx==gt),
            "conf_lntp": conf_lntp,
            "conf_hidden_norm": option_scores[pred_idx]["hidden_norm"],
            "conf_hidden_var":  option_scores[pred_idx]["hidden_var"],
            "conf_attn_entropy": option_scores[pred_idx]["attn_entropy"],
        })

    df = pd.DataFrame(rows)
    df["p_logit"]  = norm01(df["conf_lntp"].values)
    df["p_hidden"] = norm01(df["conf_hidden_norm"].values)
    df["p_attn"]   = norm01((-df["conf_attn_entropy"]).values)

    metrics = {
        "acc": float(df["correct"].mean()),
        "ece_logit_10": ece(df["p_logit"], df["correct"], 10),
        "ece_hidden_10": ece(df["p_hidden"], df["correct"], 10),
        "ece_attn_10": ece(df["p_attn"], df["correct"], 10),
        "auroc_logit": auroc(df["conf_lntp"], df["correct"], True),
        "auroc_hidden": auroc(df["conf_hidden_norm"], df["correct"], True),
        "auroc_attn": auroc(-df["conf_attn_entropy"], df["correct"], True),
    }
    os.makedirs("outputs/runs", exist_ok=True)
    df.to_parquet("outputs/runs/tqa_mc.parquet")
    with open("outputs/runs/tqa_mc_metrics.json","w") as f: json.dump(metrics, f, indent=2)
    print(metrics)

if __name__ == "__main__":
    run_arc(max_items=500)      # starte mit kleineren Subsets
    run_tqa(max_items=420)
