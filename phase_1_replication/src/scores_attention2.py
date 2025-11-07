# phase_1_replication/src/scores_attention.py
import json, argparse
def run(args):
    out=[]
    with open(args.inputs,"r",encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            # niedrigere Entropie => mehr Sicherheit? Du kannst das invertieren:
            attn_ent = float(ex["attn_entropy"])
            conf = 1.0 / (1.0 + attn_ent)   # simple monotone map
            out.append({"id": ex["id"], "score_attn": conf})
    with open(args.out,"w",encoding="utf-8") as g:
        for r in out: g.write(json.dumps(r)+"\n")
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
