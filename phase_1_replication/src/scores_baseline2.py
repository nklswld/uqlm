# phase_1_replication/src/scores_baseline.py
import json, math, argparse
from pathlib import Path

def lntp(token_probs):
    logs = [math.log(max(1e-12,p)) for p in token_probs]
    return math.exp(sum(logs)/len(logs)) if logs else 1e-12

def mtp(token_probs):
    return float(max(1e-12, min(token_probs))) if token_probs else 1e-12

def run(args):
    out = []
    with open(args.inputs, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            tp = ex["token_probs"]
            out.append({
                "id": ex["id"],
                "gold": ex.get("gold"),
                "pred_text": ex["pred_text"],
                "score_lntp": lntp(tp),
                "score_mtp": mtp(tp)
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as g:
        for r in out: g.write(json.dumps(r)+"\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
